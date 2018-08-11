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


def policy_evaluation():
    global value_table
    next_value_table = value_table # next value function initialize
    for state in get_all_states():
        value = 0.00
        if state == [3, 3]:
            next_value_table[state[0]][state[1]] = 0.0
            continue
        for action in arrow_keys:
            next_state = state_after_action(state, action)
            rew = get_reward(state, action)
            #print(rew)
            next_value = get_value(next_state)
            value += get_policy(state)[arrow_keys[action]] * (rew + discount_factor * next_value)
            next_value_table[state[0]][state[1]] = round(value, 2)
			
    value_table = next_value_table	

def policy_improvement():
    global policy_table
    next_policy = policy_table
    for state in get_all_states():
        if state == [3, 3]:
            continue
        value = -99999
        max_index = []

        result = [0.0, 0.0, 0.0, 0.0]
		
        for index, action in enumerate(arrow_keys):
            next_state = state_after_action(state, action)
            rew = get_reward(state, action)
            next_value = get_value(next_state)
            temp = rew + discount_factor * next_value	
            if temp == value:
                max_index.append(index)	
            elif temp > value:
                value = temp
                max_index.clear()
                max_index.append(index)
				
        prob = 1 / len(max_index)
        for index in max_index:
            result[index] = prob
      		
        next_policy[state[0]][state[1]] = result
	
    policy_table = next_policy
	

			

	
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

def get_policy(state):
    global policy_table
    if state == [3, 3]:
        return 0.0
    return policy_table[state[0]][state[1]]


def get_action(state):
    random_pick = random.randrange(100) / 100
    policy = get_policy(state)
    policy_sum = 0.0
    for index, value in enumerate(policy):
        policy_sum += value
        if random_pick < policy_sum:
            return index

#print("\nIteration times : 5")
env.render()    
for _ in range(5):
    policy_evaluation()
    policy_improvement()
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
        print("\n<Learned Path> : ", path)
        break
    else:
        cnt += 1

