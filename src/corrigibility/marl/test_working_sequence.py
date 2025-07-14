#!/usr/bin/env python3
"""
Test a working sequence that should achieve the goal.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def test_working_sequence():
    """Test a sequence that should work to reach the goal"""
    print("=== Testing Working Sequence ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Initial directions: Human={env.agent_dirs['human_0']}, Robot={env.agent_dirs['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    # Print the grid
    print("\nGrid layout:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) == env.agent_positions['human_0']:
                print('H', end='')
            elif (i, j) == env.agent_positions['robot_0']:
                print('R', end='')
            elif (i, j) == tuple(env.human_goals['human_0']):
                print('G', end='')
            else:
                print(env.grid[i, j], end='')
        print()
    
    # Working sequence:
    # Robot starts at (1,1) facing down (dir=2), needs to go to key at (1,2)
    # Key is to the right, so robot needs to turn to face right (dir=1)
    # Then move to key, pick it up, go to door at (2,2), and open it
    
    sequence = [
        # Robot: turn from down (2) to right (1) to face key
        {"human_0": 0, "robot_0": 0},  # Human: turn left, Robot: turn left (2->1)
        {"human_0": 0, "robot_0": 2},  # Human: turn left, Robot: move right to key
        {"human_0": 0, "robot_0": 3},  # Human: turn left, Robot: pickup key
        # Robot: turn from right (1) to down (2) to face door
        {"human_0": 0, "robot_0": 1},  # Human: turn left, Robot: turn right (1->2)
        {"human_0": 0, "robot_0": 2},  # Human: turn left, Robot: move down to door
        {"human_0": 0, "robot_0": 5},  # Human: turn left, Robot: toggle door
        # Robot: move away from door, Human: go through door
        {"human_0": 0, "robot_0": 2},  # Human: turn left, Robot: move away
        {"human_0": 2, "robot_0": 2},  # Human: move down through door, Robot: continue
        {"human_0": 1, "robot_0": 2},  # Human: turn right to face goal, Robot: continue
        {"human_0": 2, "robot_0": 2},  # Human: move right to goal, Robot: continue
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

def test_direction_system():
    """Test understanding of direction system"""
    print("\n=== Testing Direction System ===")
    print("Direction mappings:")
    print("  0: Up (-1, 0)")
    print("  1: Right (0, 1)")
    print("  2: Down (1, 0)")
    print("  3: Left (0, -1)")
    print("\nAction mappings:")
    print("  0: Turn left")
    print("  1: Turn right")
    print("  2: Move forward")
    print("  3: Pickup (robot only)")
    print("  4: Drop (robot only)")
    print("  5: Toggle (robot only)")

if __name__ == "__main__":
    test_direction_system()
    success = test_working_sequence()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")