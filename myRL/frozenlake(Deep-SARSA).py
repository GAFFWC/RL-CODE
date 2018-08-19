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

class DeepSARSAAgent:
    def __init__(self):
        self.load_model = False
        self.action_space = [0, 1, 2, 3, 4]
        self.action_size = len(self.action_space)
        self.state_size = 8 * 8
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(30, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss='mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def get_action



