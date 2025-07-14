#\!/usr/bin/env python3
"""
Debug human movement issues.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_human_movement():
    """Debug human movement from (1,3)"""
    print("=== Debug Human Movement ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Initial state:")
    print(f"  Human position: {env.agent_positions['human_0']}")
    print(f"  Human direction: {env.agent_dirs['human_0']}")
    print(f"  Human goal: {env.human_goals['human_0']}")
    
    # Check surrounding positions
    human_pos = env.agent_positions['human_0']
    print(f"\nSurrounding positions from {human_pos}:")
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    dir_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    
    for direction, (dx, dy) in deltas.items():
        target_pos = (human_pos[0] + dx, human_pos[1] + dy)
        if (0 <= target_pos[0] < env.grid_size and 
            0 <= target_pos[1] < env.grid_size):
            grid_char = env.grid[target_pos[0], target_pos[1]]
            is_valid = env._is_valid_pos(target_pos)
            print(f"  {dir_names[direction]} {target_pos}: grid='{grid_char}', valid={is_valid}")
        else:
            print(f"  {dir_names[direction]} {target_pos}: out of bounds")
    
    # Show grid with coordinates
    print(f"\nGrid layout:")
    print("  ", end="")
    for j in range(env.grid_size):
        print(f"{j}", end="")
    print()
    
    for i in range(env.grid_size):
        print(f"{i} ", end="")
        for j in range(env.grid_size):
            if (i, j) == env.agent_positions['human_0']:
                print('H', end='')
            elif (i, j) == tuple(env.human_goals['human_0']):
                print('G', end='')
            else:
                print(env.grid[i, j], end='')
        print()

if __name__ == "__main__":
    debug_human_movement()
EOF < /dev/null
