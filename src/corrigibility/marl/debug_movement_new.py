#!/usr/bin/env python3
"""
Debug script to understand movement in the environment.
"""

import sys
import os


import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def debug_movement():
    """Debug movement in the simple map"""
    print("=== Debugging Movement ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    print(f"Map layout: {map_layout}")
    print(f"Map metadata: {map_metadata}")
    
    # Initialize environment
    env.reset()
    print(f"Agent positions: {env.agent_positions}")
    print(f"Agent directions: {env.agent_dirs}")
    print(f"Grid size: {env.grid_size}")
    
    print("\nGrid contents:")
    for r in range(env.grid_size):
        row = ""
        for c in range(env.grid_size):
            if (r, c) in env.agent_positions.values():
                if (r, c) == env.agent_positions.get('robot_0'):
                    row += "R"
                elif (r, c) == env.agent_positions.get('human_0'):
                    row += "H"
                else:
                    row += "?"
            else:
                row += env.grid[r, c]
        print(f"Row {r}: {row}")
    
    # Test different movements for robot
    print("\n=== Testing Robot Movement ===")
    robot_id = "robot_0"
    robot_pos = env.agent_positions[robot_id]
    robot_dir = env.agent_dirs[robot_id]
    
    print(f"Robot at {robot_pos}, facing direction {robot_dir}")
    
    # Check all possible front positions for different directions
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
    for direction, (dx, dy) in deltas.items():
        front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
        is_valid = env._is_valid_pos(front_pos) if 0 <= front_pos[0] < env.grid_size and 0 <= front_pos[1] < env.grid_size else False
        grid_char = env.grid[front_pos] if 0 <= front_pos[0] < env.grid_size and 0 <= front_pos[1] < env.grid_size else "OUT_OF_BOUNDS"
        print(f"  Direction {direction}: front_pos={front_pos}, valid={is_valid}, grid_char='{grid_char}'")
    
    # Test actual movement
    print("\n=== Testing Actual Movement ===")
    
    # Try turning left
    print("Step 1: Turn left")
    actions = {robot_id: 0}  # turn left
    env.step(actions)
    print(f"  Robot pos: {env.agent_positions[robot_id]}, dir: {env.agent_dirs[robot_id]}")
    
    # Try moving forward
    print("Step 2: Move forward")
    actions = {robot_id: 2}  # forward
    env.step(actions)
    print(f"  Robot pos: {env.agent_positions[robot_id]}, dir: {env.agent_dirs[robot_id]}")
    
    # Test human movement too
    print("\n=== Testing Human Movement ===")
    human_id = "human_0"
    human_pos = env.agent_positions[human_id]
    human_dir = env.agent_dirs[human_id]
    
    print(f"Human at {human_pos}, facing direction {human_dir}")
    
    # Try moving forward
    print("Step 3: Human move forward")
    actions = {human_id: 2}  # forward
    env.step(actions)
    print(f"  Human pos: {env.agent_positions[human_id]}, dir: {env.agent_dirs[human_id]}")
    
    # Test manual goal-directed movement
    print("\n=== Testing Goal-Directed Movement ===")
    goal = (3, 2)  # From metadata
    print(f"Goal position: {goal}")
    
    # Try different human actions to reach goal
    for i in range(5):
        human_pos = env.agent_positions[human_id]
        human_dir = env.agent_dirs[human_id]
        print(f"\nStep {i+4}: Human at {human_pos}, dir {human_dir}")
        
        # Simple heuristic: turn towards goal
        dx = goal[0] - human_pos[0]
        dy = goal[1] - human_pos[1]
        
        # Determine target direction
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        print(f"  Target direction: {target_dir}, current direction: {human_dir}")
        
        # Choose action
        if human_dir == target_dir:
            action = 2  # Move forward
            print(f"  Action: Move forward")
        else:
            # Calculate turn direction
            left_turn = (human_dir - 1) % 4
            right_turn = (human_dir + 1) % 4
            
            if left_turn == target_dir:
                action = 0  # Turn left
                print(f"  Action: Turn left")
            elif right_turn == target_dir:
                action = 1  # Turn right
                print(f"  Action: Turn right")
            else:
                action = 0  # Turn left arbitrarily
                print(f"  Action: Turn left (arbitrary)")
        
        actions = {human_id: action}
        env.step(actions)
        print(f"  New pos: {env.agent_positions[human_id]}, new dir: {env.agent_dirs[human_id]}")
        
        # Check if reached goal
        if tuple(env.agent_positions[human_id]) == goal:
            print(f"  GOAL REACHED!")
            break


if __name__ == "__main__":
    debug_movement()