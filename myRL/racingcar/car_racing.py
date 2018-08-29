import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import colorize, seeding

import pyglet
from pyglet import gl

from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import random

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discreet control is reasonable in this environment as well, on/off discretisation is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles in track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position, gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 800
WINDOW_H = 800

SCALE       = 6.0        # Track scale
TRACK_RAD   = 900/SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD   = 2000/SCALE # Game over boundary
FPS         = 50
ZOOM        = 2.7        # Camera zoom
ZOOM_FOLLOW = True       # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

EPISODES = 50000

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile: return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__: return
        if begin:
            obj.tiles.add(tile)
            #print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0/len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            #print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)

class CarRacing(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second' : FPS
    }

    def __init__(self):
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0

        self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
        self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        random_flag = True
        CHECKPOINTS = 12
        if random_flag == True:
            random_flag = False
            # Create checkpoints
            checkpoints = []
            for c in range(CHECKPOINTS):
                alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
                rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
                if c==0:
                    alpha = 0
                    rad = 1.5*TRACK_RAD
                if c==CHECKPOINTS-1:
                    alpha = 2*math.pi*c/CHECKPOINTS
                    self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                    rad = 1.5*TRACK_RAD
                checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

                #checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )
            #print "\n".join(str(h) for h in checkpoints)
            #self.road_poly = [ (    # uncomment this to see checkpoints
            #    [ (tx,ty) for a,tx,ty in checkpoints ],
            #    (0.7,0.7,0.9) ) ]
            
            self.road = []

            # Go from one checkpoint to another to create track
            x, y, beta = 1.5*TRACK_RAD, 0, 0
            dest_i = 0
            laps = 0
            track = []
            no_freeze = 2500
            visited_other_side = False
            while 1:
                alpha = math.atan2(y, x)
                if visited_other_side and alpha > 0:
                    laps += 1
                    visited_other_side = False
                if alpha < 0:
                    visited_other_side = True
                    alpha += 2*math.pi
                while True: # Find destination from checkpoints
                    failed = True
                    while True:
                        dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                        if alpha <= dest_alpha:
                            failed = False
                            break
                        dest_i += 1
                        if dest_i % len(checkpoints) == 0: break
                    if not failed: break
                    alpha -= 2*math.pi
                    continue
                r1x = math.cos(beta)
                r1y = math.sin(beta)
                p1x = -r1y
                p1y = r1x
                dest_dx = dest_x - x  # vector towards destination
                dest_dy = dest_y - y
                proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
                while beta - alpha >  1.5*math.pi: beta -= 2*math.pi
                while beta - alpha < -1.5*math.pi: beta += 2*math.pi
                prev_beta = beta
                proj *= SCALE
                if proj >  0.3: beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
                if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001*proj))
                x += p1x*TRACK_DETAIL_STEP
                y += p1y*TRACK_DETAIL_STEP
                track.append( (alpha,prev_beta*0.5 + beta*0.5,x,y) )
                if laps > 4: break
                no_freeze -= 1
                if no_freeze==0: break
            #print "\n".join([str(t) for t in enumerate(track)])

            # Find closed loop range i1..i2, first loop should be ignored, second is OK
            i1, i2 = -1, -1
            i = len(track)
            while True:
                i -= 1
                if i==0: return False  # Failed
                pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
                if pass_through_start and i2==-1:
                    i2 = i
                elif pass_through_start and i1==-1:
                    i1 = i
                    break
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2-i1))
            assert i1!=-1
            assert i2!=-1

            track = track[i1:i2-1]

            first_beta = track[0][1]
            first_perp_x = math.cos(first_beta)
            first_perp_y = math.sin(first_beta)
            # Length of perpendicular jump to put together head and tail
            well_glued_together = np.sqrt(
                np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
                np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
            if well_glued_together > TRACK_DETAIL_STEP:
                return False

            # Red-white border on hard turns
            border = [False]*len(track)
            for i in range(len(track)):
                good = True
                oneside = 0
                for neg in range(BORDER_MIN_COUNT):
                    beta1 = track[i-neg-0][1]
                    beta2 = track[i-neg-1][1]
                    good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                    oneside += np.sign(beta1 - beta2)
                good &= abs(oneside) == BORDER_MIN_COUNT
                border[i] = good
            for i in range(len(track)):
                for neg in range(BORDER_MIN_COUNT):
                    border[i-neg] |= border[i]

            # Create tiles
            for i in range(len(track)):
                alpha1, beta1, x1, y1 = track[i]
                alpha2, beta2, x2, y2 = track[i-1]
                road1_l = (x1 - TRACK_WIDTH*math.cos(beta1), y1 - TRACK_WIDTH*math.sin(beta1))
                road1_r = (x1 + TRACK_WIDTH*math.cos(beta1), y1 + TRACK_WIDTH*math.sin(beta1))
                road2_l = (x2 - TRACK_WIDTH*math.cos(beta2), y2 - TRACK_WIDTH*math.sin(beta2))
                road2_r = (x2 + TRACK_WIDTH*math.cos(beta2), y2 + TRACK_WIDTH*math.sin(beta2))
                t = self.world.CreateStaticBody( fixtures = fixtureDef(
                    shape=polygonShape(vertices=[road1_l, road1_r, road2_r, road2_l])
                    ))
                t.userData = t
                c = 0.01*(i%3)
                t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
                t.road_visited = False
                t.road_friction = 1.0
                t.fixtures[0].sensor = True
                self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
                self.road.append(t)
                if border[i]:
                    side = np.sign(beta2 - beta1)
                    b1_l = (x1 + side* TRACK_WIDTH        *math.cos(beta1), y1 + side* TRACK_WIDTH        *math.sin(beta1))
                    b1_r = (x1 + side*(TRACK_WIDTH+BORDER)*math.cos(beta1), y1 + side*(TRACK_WIDTH+BORDER)*math.sin(beta1))
                    b2_l = (x2 + side* TRACK_WIDTH        *math.cos(beta2), y2 + side* TRACK_WIDTH        *math.sin(beta2))
                    b2_r = (x2 + side*(TRACK_WIDTH+BORDER)*math.cos(beta2), y2 + side*(TRACK_WIDTH+BORDER)*math.sin(beta2))
                    self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else (1,0,0) ))
            self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.human_render = False

        while True:
            success = self._create_track()
            if success: break
            print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.steer(-action[0]) # 방향
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            #self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count==len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('0000', font_size=36,
                x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet

        zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   # Animate zoom first second
        zoom_state  = ZOOM*SCALE*STATE_W/WINDOW_W
        zoom_video  = ZOOM*SCALE*VIDEO_W/WINDOW_W
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
            WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")

        arr = None
        win = self.viewer.window
        if mode != 'state_pixels':
            win.switch_to()
            win.dispatch_events()
        if mode=="rgb_array" or mode=="state_pixels":
            win.clear()
            t = self.transform
            if mode=='rgb_array':
                VP_W = VIDEO_W
                VP_H = VIDEO_H
            else:
                VP_W = STATE_W
                VP_H = STATE_H
            gl.glViewport(0, 0, VP_W, VP_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)  # TODO: find why 2x needed, wtf
            image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            arr = arr.reshape(VP_H, VP_W, 4)
            arr = arr[::-1, :, 0:3]

        if mode=="rgb_array" and not self.human_render: # agent can call or not call env.render() itself when recording video.
            win.flip()

        if mode=='human':
            self.human_render = True
            win.clear()
            t = self.transform
            gl.glViewport(0, 0, WINDOW_W, WINDOW_H)
            t.enable()
            self.render_road()
            for geom in self.viewer.onetime_geoms:
                geom.render()
            t.disable()
            self.render_indicators(WINDOW_W, WINDOW_H)
            win.flip()

        self.viewer.onetime_geoms = []
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W/40.0
        h = H/40.0
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        vertical_ind(5, 0.02*true_speed, (1,1,1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        horiz_ind(20, -10.0*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(30, -0.8*self.car.hull.angularVelocity, (1,0,0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()

class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False

        self.state_size = (96, 96, 3)
        self.action_size = action_size
        self.epsilon = 1
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        self.memory = deque(maxlen = 400000)
        self.no_op_steps = 30

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_writer = tf.summary.FileWriter('./RacingCar_DQN', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./RacingCar_DQN.h5")

    def optimizer(self):
        a = K.placeholder(shape = (None,), dtype = 'int32')
        y = K.placeholder(shape = (None,), dtype = 'float32')

        prediction = self.model.output
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis = 1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        optimizer = RMSprop(lr = 0.00025, epsilon = 0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates = updates)

        return train

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides = (4, 4), activation = 'relu', input_shape = self.state_size))
        model.add(Conv2D(64, (4, 4), strides = (2, 2), activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(512, activation = 'relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))

    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
        mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        actions, rewards, deads = [], [], []


        for i in range((self.batch_size, )):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            deads.append(mini_batch[i][4])
        
        target_value = self.target_model.predict(next_history)

        for i in range(self.batch_size):
            if deads[i]:
                target[i] = rewards[i]
            else:
                target[i] = rewards[i] + self.discount_factor * np.amax(target_value[i])
        loss = self.optimizer([history, actions, target])
        self.avg_loss += loss[0]

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

def pre_processing(observe):
    processed_observe = np.uint8(resize(rgb2gray(observe), (96, 96), mode = 'constant') * 255)
    return processed_observe

def Real_action(index, a):
    if index == 0:
        a[0] = -1.0
    if index == 1:
        a[0] = +1.0
    if index == 2:
        a[1] = +1.0
    if index == 3:
        a[2] = +0.8
    return a


if __name__=="__main__":
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )

    agent = DQNAgent(action_size = 4) # 0 : 왼쪽, 1 : 오른쪽, 2 : 위, 3 : 아래
    scores, episodes, global_step = [], [], 0
    
    def key_press(k, mod): # 왼쪽, 오른쪽 방향키 : a[0]이 음수이면 왼쪽, 양수이면 오른쪽 / 위 방향키 : a[1]이 양수 / 아래 방향키(브레이크) : a[2]가 양수
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0
    env = CarRacing()
    env.render()
    record_video = False
    if record_video:
        env.monitor.start('/tmp/video-test', force=True)
    #env.viewer.window.on_key_press = key_press
    #env.viewer.window.on_key_release = key_release
    env.reset()
    for _ in range(random.randint(1, agent.no_op_steps)):
        observe, _, _, _ = env.step(a)
    state = pre_processing(observe)
    history = np.stack((state, state, state), axis = 2)
    history = np.reshape([history], (1, 96, 96, 3))    
    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        
        while True:
            dead = False
            action = agent.get_action(history)
            Real_action(action, a)
            a[1] = 1.0
            observe, r, done, info = env.step(a)
            print(a)
            for i in range(len(a)):
                a[i] = 0

            env.render()
            #print(s)
            #print(np.shape(s)) # state는 (96, 96, 3)
            next_state = pre_processing(observe)
            #print(np.shape(next_state))
            next_state = np.reshape([next_state], (1, 96, 96, 1))
            next_history = np.append(next_state, history[:,:,:,:2], axis = 3)
          #  print(np.shape(history))
            agent.avg_q_max += np.max(agent.model.predict(np.float32(history / 255.))[0])

            total_reward += r
            agent.append_sample(history, action, total_reward, next_history, dead)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            if steps % agent.update_target_rate == 0:
                agent.update_target_model()
            #history = next_history
           # print(np.shape(next_history))

            if steps % 200 == 0 or done:
                if steps > agent.train_start:
                    stats = [total_reward, agent.avg_q_max / float(steps), steps, agent.avg_loss / float(steps)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict = {agent.summary_placeholders[i]: float(stats[i])})
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, steps + 1)
                
                agent.avg_loss, agent.avg_loss = 0, 0
                agent.model.save_weights("./RacingCar_DQN_model.h5")
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                #print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            if not record_video: # Faster, but you can as well call env.render() every time to play full window.
                env.render()
            if done or restart: break
    env.close()