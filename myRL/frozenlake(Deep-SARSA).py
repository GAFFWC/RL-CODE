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
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        
        if self.load_model:
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
            q_values = self.model.predic(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]

        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, 5])
        self.model.fit(state, target, epochs = 1, verbose = 0)

env.render()    
state = [0, 0]
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


if __name__ == "__main__":
    agent = DeepSARSAgent()
    global_step = 0
    scores, episodes = [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = [0, 0]
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./deep-sarsa.png")
                print("episode:", e, " score:", score, "global_step", global_step, " epsilon:", agent.epsilon)

        if e%100 == 0:
            agent.model.save_weights("./deep_sarsa.h5")
           
        



