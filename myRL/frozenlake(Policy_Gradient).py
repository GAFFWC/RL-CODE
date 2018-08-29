import gym
from gym.envs.registration import register
from colorama import init
import readchar
import sys
import numpy as np
import random
init(autoreset=True)    # Reset the terminal mode to display ansi color
from collections import defaultdict
import os
import copy
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import pylab
from keras import backend as K

EPISODES = 2500

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
                         # Show the initial board


class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = [0, 1, 2, 3]
        self.action_size = len(self.action_space)
        self.state_size = 26
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.states, self.actions, self.rewards = [], [], []

        self.model.load_weights('./policy_gradient_trained.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(30, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'softmax'))
        model.summary()
        return model

    def build_optimizer(self):
        action = K.placeholder(shape = [None, 4])
        discounted_rewards = K.placeholder(shape = [None, ])
        action_prob = K.sum(action * self.model.output, axis = 1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr = self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates = updates)

        return train



    def get_action(self, state):
            state = np.float32(state)
            state = np.reshape(state, [1, 26])			
            policy = self.model.predict(state)
            #print(policy[0])
            return np.random.choice(self.action_size, 1, p = policy[0])

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]    
            discounted_rewards[t] = running_add
        return discounted_rewards


    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        #print(self.states, self.actions, discounted_rewards)
        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []   

cnt = 1
path = []


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

reward = [[0.00] * 8 for _ in range(8)]

reward[6][1] = -1.0
reward[6][4] = -1.0
reward[6][6] = -1.0
reward[7][3] = -1.0
reward[5][2] = -1.0
reward[5][6] = -1.0
reward[5][1] = -1.0
reward[4][3] = -1.0
reward[2][3] = -1.0
reward[3][5] = -1.0
reward[3][3] = -1.0
reward[7][7] = 1.0
    
def state_after_action(state, action_index):
    row = state[0]
    col = state[1]
    if arrow_keys[action_index] == 0: #LEFT
        col = max(col - 1, 0)
    elif arrow_keys[action_index] == 1: # down
        row = min(row+1, 7)
    elif arrow_keys[action_index] == 2: # right
        col = min(col + 1, 7)
    elif arrow_keys[action_index] == 3: # up
        row = max(row - 1, 0)
    return int(row), int(col)

def append_state(state):
    state.append(abs(2 - state[0]))
    state.append(abs(3 - state[1]))
    state.append(abs(3 - state[0]))
    state.append(abs(3 - state[1]))
    state.append(abs(3 - state[0]))
    state.append(abs(5 - state[1]))
    state.append(abs(4 - state[0]))
    state.append(abs(3 - state[1]))
    state.append(abs(5 - state[0]))
    state.append(abs(1 - state[1]))
    state.append(abs(5 - state[0]))
    state.append(abs(2 - state[1]))
    state.append(abs(5 - state[0]))
    state.append(abs(6 - state[1]))
    state.append(abs(6 - state[0]))
    state.append(abs(1 - state[1]))
    state.append(abs(6 - state[0]))
    state.append(abs(4 - state[1]))
    state.append(abs(6 - state[0]))
    state.append(abs(6 - state[1]))
    state.append(abs(7 - state[0]))
    state.append(abs(3 - state[1]))
    state.append(7)
    state.append(7)
if __name__ == "__main__":
    agent = DeepSARSAgent()
    global_step = 0
    temp = [0, 0]
    scores, episodes = [], []
    for e in range(4000):
        d = False
        score = 0
        state = [0, 0, 2, 3, 3, 3, 3, 5, 4, 3, 5, 1, 5, 2, 5, 6, 6, 1, 6, 4, 6, 6, 7, 3, 7, 7]
        time_stamp = 0.0
        #print(state)

        while True:
            time_stamp -= 0.005
            global_step += 1
            flag = 0
            next_state =  [0, 0]
            action = agent.get_action(state)
            print(action[0])
            n, r, d, info = env.step(action[0])
            #print(n)
            #print(n, r, d, info)
            if action[0] == 0: #LEFT
                next_state[0], next_state[1] = state_after_action(state, '\x1b[D')
                if state[1] == 0:
                    flag = 1
                    rew = -0.25
            elif action[0] == 1: #DOWN
                next_state[0], next_state[1]  = state_after_action(state, '\x1b[B')    
                if state[0] == 7:
                    flag = 1
                    rew = -0.25   
            elif action[0] == 2: #RIGHT
                next_state[0], next_state[1]  = state_after_action(state, '\x1b[C')
                if state[1] == 7:
                    flag = 1
                    rew = -0.25          
            else: #UP
                next_state[0], next_state[1]  = state_after_action(state,  '\x1b[A')
                if state[0] == 0:
                    flag = 1
                    rew = -0.25
            
            #print(next_state)
            state = np.reshape(state, [1, 26])
            append_state(next_state)
	
            if flag == 0:
                rew = reward[next_state[0]][next_state[1]]
            
            next_action = agent.get_action(next_state)
            #print(state, action[0], rew)
            rew += time_stamp
            agent.append_sample(state, action, rew)
            next_state = np.reshape(next_state, [26, 1])	
            state = next_state
            #print(state)
            score += rew
            #state = copy.deepcopy(next_state)
            #print(state)
            os.system('cls')
            env.render()
            print("Count : ", e)
            if d:
                agent.train_model()
                
                #print(state)
                #env.render()
                #print(target)
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./policy-gradient.png")
                #print(state)
                #print("episode:", e, " score:", score, "global_step", global_step, " epsilon:", agent.epsilon)
                env.reset()
                break
        agent.model.save_weights("./policy_gradient.h5")
           
        



