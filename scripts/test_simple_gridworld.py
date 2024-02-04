import sys
sys.path.insert(0,'./src/')

import time
from numpy import random

from world_model import SimpleGridworld
from environments.very_simple_gridworlds import make_simple_gridworld
import pylab as plt

def move_randomly(env):
    state, delta, terminated, _, info = env.reset()
    for t in range(1000):
        actions = env.possible_actions(state)
        action = random.choice(actions)
        print(t, state, delta, terminated, _, info, actions, action)
        state, delta, terminated, _, info = env.step(action)
        if terminated:
            print(t, state, delta, terminated)
            print("Goal reached!")
            break

# test creation of a simple gridworld:
env, aleph0 = make_simple_gridworld(gw = "GW5") #, render_mode = "human")
env.reset()
move_randomly(env)
env.render()
#time.sleep(5)
env.close()

# run around a random grid until the agent reaches the goal:
grid = [
    [   
        random.choice([' ', ' ', ' ', '#', '#', ',', '^', '~'])
        for x in range(11)
    ]
    for y in range(11)
]
grid[2][2] = "A"
grid[8][8] = "G"
delta_grid = [
    [
        ' ' if grid[y][x] == '#' else random.choice([' ','M','P'], p=[0.4,0.3,0.3])
        for x in range(11)
    ]
    for y in range(11)
]
print(grid)
print(delta_grid)
env = SimpleGridworld(grid = grid, delta_grid = delta_grid, cell_code2delta = {'M':-1, 'P':1}, render_mode = "human", fps=1)
move_randomly(env)
env.close()
