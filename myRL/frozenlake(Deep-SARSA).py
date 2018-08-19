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

        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.model.load_weights('./deep_sarsa_trained.h5')

    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(30, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = np.float32(state)
            state = np.reshape(state, [1, 26])			
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        state = np.float32(state)
        state = np.reshape(state, [1, 26])
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        next_state = np.reshape(next_state, [1, 26])
        

        if done:
            target[action] = reward
            #print(target)
        else:
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, 4])
        self.model.fit(state, target, epochs = 1, verbose = 0)
		

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

if __name__ == "__main__":
    agent = DeepSARSAgent()
    global_step = 0
    temp = [0, 0]
    scores, episodes = [], []
    for e in range(2500):
        d = False
        score = 0
        state = [0, 0, 2, 3, 3, 3, 3, 5, 4, 3, 5, 1, 5, 2, 5, 6, 6, 1, 6, 4, 6, 6, 7, 3, 7, 7]
       
        #print(state)

        while True:
            global_step += 1
            next_state =  [0, 0, 2, 3, 3, 3, 3, 5, 4, 3, 5, 1, 5, 2, 5, 6, 6, 1, 6, 4, 6, 6, 7, 3, 7, 7]
            action = agent.get_action(state)
            #print(action)
            n, r, d, info = env.step(action)
            #print(n)
            #print(n, r, d, info)
            if action == 0:
                next_state[0], next_state[1] = state_after_action(state, '\x1b[D')
            elif action == 1:
                next_state[0], next_state[1]  = state_after_action(state, '\x1b[B')          
            elif action == 2:
                next_state[0], next_state[1]  = state_after_action(state, '\x1b[C')           
            else:
                next_state[0], next_state[1]  = state_after_action(state,  '\x1b[A')
            
            print(next_state)
            state = np.reshape(state, [1, 26])
            rew = reward[next_state[0]][next_state[1]]
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, rew, next_state, next_action, d)
            state = next_state
            #print(state)
            score += rew
            state = copy.deepcopy(next_state)
            #print(state)
            os.system('cls')
            env.render()
            if d:

                #print(state)
                #env.render()
                #print(target)
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./deep-sarsa.png")
                #print(state)
                print("episode:", e, " score:", score, "global_step", global_step, " epsilon:", agent.epsilon)
                env.reset()
                break
            agent.model.save_weights("./deep_sarsa.h5")
           
        



