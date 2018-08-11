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
        row = min(row+1, 3)
    elif arrow_keys[action_index] == 2: # right
        col = min(col + 1, 3)
    elif arrow_keys[action_index] == 3: # up
        row = max(row - 1, 0)
    return (int(row), int(col))

def value_iteration():
    next_value_table = [[0.00] * 4 for _ in range(4)]
    for state in get_all_states():
        if state == [3, 3]:
            next_value_table[state[0]][state[1]] = 0.0
            continue
        value_list = []

    for action in arrow_keys:
            next_state = state_after_action(state, action)
            rew = get_reward(state, action)
            next_value = get_value(next_state)
            value_list.append((rew + discount_factor * next_value))
    next_value_table[state[0]][state[1]] = round(max(value_list), 2)       
    	
def get_value(state):
    return round(value_table[state[0]][state[1]], 2)

def get_all_states():
    return [[0,0], [0,1], [0,2], [0,3],[1,0], [1,1], [1,2], [1,3],[2,0], [2,1], [2,2], [2,3],[3,0], [3,1], [3,2], [3,3]]

def get_reward(state, action):
    next_state = state_after_action(state, action)
    return reward[next_state[0]][next_state[1]]    	
    
value_table = [[0.00] * 4 for _ in range(4)] # initialize value function of mat 4x4
policy_table = [[[0.25, 0.25, 0.25, 0.25]] * 4 for _ in range(4)] # initialize policy of mat 4x4 (25% x 4)
discount_factor = 0.8 # set discount factor
reward = [[0.00] * 4 for _ in range(4)]


reward[1][1] = -3.0
reward[1][3] = -3.0
reward[2][3] = -3.0
reward[3][0] = -3.0
reward[3][3] = 10.0

arrow_keys = {
    '\x1b[A' : UP,
    '\x1b[B' : DOWN,
    '\x1b[C' : RIGHT,
    '\x1b[D' : LEFT
}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')        # is_slippery False
                         # Show the initial board

def get_action(state):
    action_list = []
    max_value = -99999

    if state == [3, 3]:
        return []
    for action in arrow_keys:
        next_state = state_after_action(state, action)
        rew = get_reward(state, action)
        next_value = get_value(next_state)
        value = rew + discount_factor * next_value

        if value > max_value
            action_list.clear()
            action_list.append(action)
            max_value = value
        elif value == max_value:
            action_list.append(action)
    return action_list

print("Iteration times : 5\n")
env.render()    
state = [0, 0]
cnt = 1
path = []
while True:
    action = get_action(state)
    s, r, done, info = env.step(action)
    env.render()
    if action == 0:
        state = state_after_action(state, '\x1b[D')
        path.append('LEFT')
    elif action == 1:
        state = state_after_action(state, '\x1b[B')
        path.append('DOWN')
    elif action == 2:
        state = state_after_action(state, '\x1b[C')
        path.append('RIGHT')
    else:
        state = state_after_action(state,  '\x1b[A')
        path.append('UP')
    if r == 1.0:
        print("\nPath : ", path)
        break
    else:
        cnt += 1

print(value_table)