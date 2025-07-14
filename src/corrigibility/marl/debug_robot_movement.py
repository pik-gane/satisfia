#!/usr/bin/env python3
"""
Debug why robot can't move after picking up key.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_robot_movement():
    """Debug robot movement after key pickup"""
    print("=== Debug Robot Movement ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Initial state:")
    print(f"  Robot position: {env.agent_positions['robot_0']}")
    print(f"  Robot direction: {env.agent_dirs['robot_0']}")
    print(f"  Grid at (1,1): '{env.grid[1, 1]}'")
    print(f"  Grid at (1,2): '{env.grid[1, 2]}'")
    print(f"  Grid at (2,1): '{env.grid[2, 1]}'")
    print(f"  Grid at (2,2): '{env.grid[2, 2]}'")
    
    # Step 1: Robot picks up key
    print(f"\nStep 1: Robot turns and picks up key")
    env.agent_dirs['robot_0'] = 1  # Face right
    actions = {"human_0": 0, "robot_0": 3}  # pickup
    obs, rewards, terms, truncs, _ = env.step(actions)
    
    print(f"  After pickup:")
    print(f"    Robot position: {env.agent_positions['robot_0']}")
    print(f"    Robot direction: {env.agent_dirs['robot_0']}")
    print(f"    Grid at (1,2): '{env.grid[1, 2]}'")
    print(f"    Robot has keys: {env.robot_has_keys}")
    
    # Step 2: Try to move right to (1,2)
    print(f"\nStep 2: Robot tries to move right")
    actions = {"human_0": 0, "robot_0": 2}  # move forward
    obs, rewards, terms, truncs, _ = env.step(actions)
    
    print(f"  After move attempt:")
    print(f"    Robot position: {env.agent_positions['robot_0']}")
    print(f"    Robot direction: {env.agent_dirs['robot_0']}")
    
    if env.agent_positions['robot_0'] == (1, 2):
        print(f"    ✅ Robot moved successfully!")
    else:
        print(f"    ❌ Robot didn't move")
        
        # Check what's blocking movement
        robot_pos = env.agent_positions['robot_0']
        robot_dir = env.agent_dirs['robot_0']
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = deltas[robot_dir]
        target_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
        
        print(f"    Target position: {target_pos}")
        if (0 <= target_pos[0] < env.grid_size and 
            0 <= target_pos[1] < env.grid_size):
            print(f"    Grid at target: '{env.grid[target_pos[0], target_pos[1]]}'")
            
            # Check if another agent is there
            other_agents = {aid: pos for aid, pos in env.agent_positions.items() if aid != 'robot_0'}
            agent_at_target = any(tuple(pos) == target_pos for pos in other_agents.values())
            print(f"    Agent at target: {agent_at_target}")
            if agent_at_target:
                agent_id = next(aid for aid, pos in other_agents.items() if tuple(pos) == target_pos)
                print(f"      Agent: {agent_id}")
        else:
            print(f"    Target position out of bounds")

def debug_environment_state():
    """Debug the environment state in detail"""
    print("\n=== Debug Environment State ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Grid size: {env.grid_size}")
    print(f"Grid layout:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            print(f"'{env.grid[i, j]}'", end="")
        print()
    
    print(f"\nAgent positions: {env.agent_positions}")
    print(f"Agent directions: {env.agent_dirs}")
    print(f"Keys: {env.keys}")
    print(f"Doors: {env.doors}")
    
    # Test _is_valid_pos function
    print(f"\nTesting _is_valid_pos:")
    test_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (0, 1), (1, 0)]
    for pos in test_positions:
        if (0 <= pos[0] < env.grid_size and 0 <= pos[1] < env.grid_size):
            is_valid = env._is_valid_pos(pos)
            grid_char = env.grid[pos[0], pos[1]]
            print(f"  {pos}: valid={is_valid}, grid='{grid_char}'")
        else:
            print(f"  {pos}: out of bounds")

if __name__ == "__main__":
    debug_environment_state()
    debug_robot_movement()