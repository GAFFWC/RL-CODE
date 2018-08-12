import gym
from gym.envs.registration import register
from colorama import init
import readchar
import sys
import numpy as np
import random
init(autoreset=True)    # Reset the terminal mode to display ansi color

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


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

def value_iteration():
    global value_table
    next_value_table = [[0.00] * 8 for _ in range(8)]
    for state in get_all_states():
        if state == [7, 7]:
            next_value_table[state[0]][state[1]] = 0.0
            continue
        value_list = []

        for action in arrow_keys:
                next_state = state_after_action(state, action)
                rew = get_reward(state, action)
                next_value = get_value(next_state)
                value_list.append((rew + discount_factor * next_value))
        next_value_table[state[0]][state[1]] = round(max(value_list), 2)
    value_table = next_value_table       
    	
def get_value(state):
    return round(value_table[state[0]][state[1]], 2)

def get_all_states():
    states = []
    for i in range(8):
        for j in range(8):
            states.append([i, j])
    return states

def get_reward(state, action):
    next_state = state_after_action(state, action)
    return reward[next_state[0]][next_state[1]]    	
    
value_table = [[0.00] * 8 for _ in range(8)] # initialize value function of mat 8x8
policy_table = [[[0.25, 0.25, 0.25, 0.25]] * 4 for _ in range(4)] # initialize policy of mat 8x8 (25% x 4)
discount_factor = 0.8 # set discount factor
reward = [[0.00] * 8 for _ in range(8)]


reward[6][1] = -3.0
reward[6][4] = -3.0
reward[6][6] = -3.0
reward[7][3] = -3.0
reward[5][2] = -3.0
reward[5][6] = -3.0
reward[5][1] = -3.0
reward[4][3] = -3.0
reward[2][3] = -3.0
reward[3][5] = -3.0
reward[3][3] = -3.0
reward[7][7] = 10.0

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
                         # Show the initial board

def get_action(state):
    action_list = []
    max_value = -99999

    if state == [7, 7]:
        return []
    for action in arrow_keys:
        next_state = state_after_action(state, action)
        rew = get_reward(state, action)
        next_value = get_value(next_state)
        value = rew + discount_factor * next_value

        if value > max_value:
            action_list.clear()
            action_list.append(action)
            max_value = value
        elif value == max_value:
            action_list.append(action)
    return action_list

for _ in range(20):
    value_iteration()
env.render()    
state = [0, 0]
cnt = 1
path = []
while True:
    temp = get_action(state)
    action = temp[random.randrange(100)%len(temp)]
    #print(value_table)
    s, r, done, info = env.step(arrow_keys[action])
    env.render()
    if action == '\x1b[D':
        state = state_after_action(state, '\x1b[D')
        path.append('LEFT')
    elif action == '\x1b[B':
        state = state_after_action(state, '\x1b[B')
        path.append('DOWN')
    elif action == '\x1b[C':
        state = state_after_action(state, '\x1b[C')
        path.append('RIGHT')
    else:
        state = state_after_action(state,  '\x1b[A')
        path.append('UP')
    if r == 1.0:
        print("\n<Learned Path> : ", path)
        break
    else:
        cnt += 1