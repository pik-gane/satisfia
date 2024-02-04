import sys
sys.path.insert(0,'./src/')

from numpy import random
from world_model import SimpleGridworld
import pylab as plt

# run around a random grid until the agent reaches the goal:
grid = [
    [   
        random.choice([" ", "#"], p=[0.8,0.2])
        for x in range(11)
    ]
    for y in range(11)
]
grid[2][2] = "A"
grid[8][8] = "G"
delta_grid = [
    [
        random.choice([' ','M','P'], p=[0.4,0.3,0.3])
        for x in range(11)
    ]
    for y in range(11)
]
print(grid)
print(delta_grid)
env = SimpleGridworld(grid = grid, delta_grid = delta_grid, cell_code2delta = {'M':-1, 'P':1}, render_mode = "human")
state, delta, terminated, _, info = env.reset()
for t in range(1000):
    actions = env.possible_actions(state)
    action = random.choice(actions)
    print(t, state, delta, terminated, _, info, actions, action)
    state, delta, terminated, _, info = env.step(action)
    if terminated:
        print("Goal reached!")
        break
env.close()
