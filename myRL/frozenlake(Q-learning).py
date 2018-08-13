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



discount_factor = 0.8 # set discount factor
learning_rate = 1 / 64
epsilon = 0.1
q_table = [[[0.0, 0.0, 0.0, 0.0] for _ in range(8)] for _ in range(8)]


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

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
                         # Show the initial board

def learn(state, action, reward, next_state, next_action):
    #print(action)
    #print(state)
    current_q = q_table[state[0]][state[1]][action]
    next_state_q = q_table[next_state[0]][next_state[1]][next_action]
    new_q = current_q + learning_rate * (reward + discount_factor * next_state_q - current_q)
    #print(new_q)
    q_table[state[0]][state[1]][action] = new_q

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
    return (int(row), int(col))

 
def get_all_states():
    states = []
    for i in range(8):
        for j in range(8):
            states.append([i, j])
    return states

def get_reward(state, action):
    next_state = state_after_action(state, action)
    return reward[next_state[0]][next_state[1]]    	
    

def get_action(state):
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1, 2, 3])
    else:
        #print(state)
        #print(type(state))
        state_action = q_table[state[0]][state[1]]
        #print(state_action)
        action = arg_max(state_action)
    return action

def arg_max(state_action):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return random.choice(max_index_list)

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



for _ in range(1000):
    while True:
        action = get_action(state)
        s, r, done, info = env.step(action)
        if action == 0:
            next_state = state_after_action(state, '\x1b[D')
            path.append('LEFT')
        elif action == 1:
            next_state = state_after_action(state, '\x1b[B')
            path.append('DOWN')
        elif action == 2:
            next_state = state_after_action(state, '\x1b[C')
            path.append('RIGHT')
        else:
            next_state = state_after_action(state,  '\x1b[A')
            path.append('UP')		
        #print(next_state)
        rew = reward[next_state[0]][next_state[1]]
        next_action = get_action(next_state)
        learn(state, action, rew, next_state, next_action)
        state = next_state
        action = next_action
        print(q_table)
        os.system('cls')
        env.render()
        if rew != 0.0:
            path = []
            state = [0, 0]
            env.reset()
            print(q_table)
            break
          



