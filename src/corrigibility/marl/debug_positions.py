#!/usr/bin/env python3
"""
Debug exact positions and create working sequence.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_positions():
    """Debug exact positions in the environment"""
    print("=== Debug Positions ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Human position: {env.agent_positions['human_0']}")
    print(f"Robot position: {env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    print(f"Keys: {[(k['pos'], k['color']) for k in env.keys]}")
    print(f"Doors: {[(d['pos'], d['color'], d['is_locked']) for d in env.doors]}")
    
    # Print the grid with coordinates
    print("\nGrid with coordinates:")
    print("  ", end="")
    for j in range(env.grid_size):
        print(f"{j}", end="")
    print()
    
    for i in range(env.grid_size):
        print(f"{i} ", end="")
        for j in range(env.grid_size):
            if (i, j) == env.agent_positions['human_0']:
                print('H', end='')
            elif (i, j) == env.agent_positions['robot_0']:
                print('R', end='')
            elif (i, j) == tuple(env.human_goals['human_0']):
                print('G', end='')
            elif any((i, j) == tuple(k['pos']) for k in env.keys):
                print('K', end='')
            elif any((i, j) == tuple(d['pos']) for d in env.doors):
                print('D', end='')
            else:
                print(env.grid[i, j], end='')
        print()

def test_robot_movement():
    """Test robot movement to door"""
    print("\n=== Test Robot Movement ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    # Robot starts at (1,1), needs to go to door at (2,2)
    # Path: (1,1) -> (2,1) -> (2,2)
    
    sequence = [
        # Step 1: Robot picks up key (already tested this works)
        {"human_0": 0, "robot_0": 1},  # Robot: turn right to face key
        {"human_0": 0, "robot_0": 3},  # Robot: pickup key
        # Step 2: Robot moves to door position
        {"human_0": 0, "robot_0": 1},  # Robot: turn right to face down
        {"human_0": 0, "robot_0": 2},  # Robot: move down to (2,1)
        {"human_0": 0, "robot_0": 1},  # Robot: turn right to face door
        {"human_0": 0, "robot_0": 2},  # Robot: move right to (2,2)
        {"human_0": 0, "robot_0": 5},  # Robot: toggle door
        # Step 3: Human moves through door
        {"human_0": 2, "robot_0": 0},  # Human: move down, Robot: turn left
        {"human_0": 2, "robot_0": 0},  # Human: move down through door, Robot: continue
        {"human_0": 1, "robot_0": 0},  # Human: turn right to face goal, Robot: continue
        {"human_0": 2, "robot_0": 0},  # Human: move right to goal, Robot: continue
    ]
    
    for step, actions in enumerate(sequence):
        print(f"\nStep {step}: Actions = {actions}")
        print(f"  Before: Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}, Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After: Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}, Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Rewards: {rewards}")
        
        # Check if human reached goal
        if tuple(env.agent_positions['human_0']) == tuple(env.human_goals['human_0']):
            print("  🎉 GOAL REACHED!")
            return True
            
        if any(terms.values()) or any(truncs.values()):
            print("  Episode ended")
            break
    
    return False

if __name__ == "__main__":
    debug_positions()
    success = test_robot_movement()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")