#!/usr/bin/env python3
"""
Debug environment mechanics in detail.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corrigibility.marl.env import CustomEnvironment, Actions


def test_human_movement():
    """Test human movement mechanics systematically."""
    
    print("=== DETAILED HUMAN MOVEMENT TEST ===")
    
    env = CustomEnvironment(map_name="simple_map")
    
    # Test human movement in all directions
    for trial in range(3):
        obs = env.reset()
        print(f"\nTrial {trial + 1}:")
        print(f"Initial state: human_0 at {env.agent_positions['human_0']}, facing direction {env.agent_dirs.get('human_0', 'unknown')}")
        
        # Check what's around the human
        human_pos = env.agent_positions['human_0']
        print(f"Environment around human at {human_pos}:")
        for dr, dc, direction in [(-1, 0, "north"), (1, 0, "south"), (0, -1, "west"), (0, 1, "east")]:
            r, c = human_pos[0] + dr, human_pos[1] + dc
            if 0 <= r < len(env.map_layout) and 0 <= c < len(env.map_layout[0]):
                cell = env.map_layout[r][c]
                print(f"  {direction}: {cell}")
            else:
                print(f"  {direction}: OUT_OF_BOUNDS")
        
        # Try each action and see what happens
        print("\nTesting actions:")
        for action_idx, action in enumerate(Actions):
            if action_idx >= 7:  # Only test the basic actions
                break
                
            # Reset to test each action independently
            env.reset()
            initial_pos = env.agent_positions['human_0']
            initial_dir = env.agent_dirs.get('human_0', 0)
            
            # Apply action
            actions = {'human_0': action_idx, 'robot_0': 6}  # Robot does nothing
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            final_pos = env.agent_positions['human_0']
            final_dir = env.agent_dirs.get('human_0', 0)
            reward = rewards['human_0']
            
            print(f"  {action.name} ({action_idx}): "
                  f"pos {initial_pos}→{final_pos}, "
                  f"dir {initial_dir}→{final_dir}, "
                  f"reward {reward:.2f}")
        
        # Also test a sequence of moves
        env.reset()
        print(f"\nTesting sequence: turn_right, forward, forward")
        actions_sequence = [1, 2, 2]  # turn_right, forward, forward
        for i, action in enumerate(actions_sequence):
            pos_before = env.agent_positions['human_0']
            dir_before = env.agent_dirs.get('human_0', 0)
            
            actions = {'human_0': action, 'robot_0': 6}
            obs, rewards, terms, truncs, infos = env.step(actions)
            
            pos_after = env.agent_positions['human_0']
            dir_after = env.agent_dirs.get('human_0', 0)
            
            print(f"  Step {i+1}: {Actions(action).name} -> "
                  f"pos {pos_before}→{pos_after}, dir {dir_before}→{dir_after}")


def check_doors_and_keys():
    """Check door and key mechanics."""
    
    print("\n=== DOOR AND KEY MECHANICS ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    print("Doors:")
    for door in env.doors:
        print(f"  {door}")
    
    print("Keys:")
    for key in env.keys:
        print(f"  {key}")
    
    print("Robot inventory:", getattr(env, 'robot_has_keys', 'No inventory attribute'))
    
    # Check if human can pick up key
    env.reset()
    human_pos = env.agent_positions['human_0']
    print(f"\nHuman at {human_pos}")
    
    # Check if there's a key at human position or adjacent
    for key in env.keys:
        key_pos = key['pos']
        distance = abs(human_pos[0] - key_pos[0]) + abs(human_pos[1] - key_pos[1])
        print(f"Key at {key_pos}, distance from human: {distance}")


def test_goal_reachability():
    """Test if the goal is actually reachable."""
    
    print("\n=== GOAL REACHABILITY TEST ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    human_pos = env.agent_positions['human_0']
    goal_pos = (3, 2)  # From the debug output
    
    print(f"Human starts at: {human_pos}")
    print(f"Goal is at: {goal_pos}")
    
    # Check path from human to goal
    print("Path analysis:")
    print("  Need to go from (1,3) to (3,2)")
    print("  That's: 2 steps down, 1 step left")
    
    # Check each cell in potential path
    path_cells = [(1, 3), (2, 3), (3, 3), (3, 2)]
    for i, (r, c) in enumerate(path_cells):
        if 0 <= r < len(env.map_layout) and 0 <= c < len(env.map_layout[0]):
            cell = env.map_layout[r][c]
            print(f"  Step {i}: ({r},{c}) = '{cell}'")
        else:
            print(f"  Step {i}: ({r},{c}) = OUT_OF_BOUNDS")


if __name__ == "__main__":
    test_human_movement()
    check_doors_and_keys()
    test_goal_reachability()