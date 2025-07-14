#!/usr/bin/env python3
"""
Find a valid path for human to reach the goal.
"""

import numpy as np
from collections import deque

from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def find_path_bfs(env, start, goal):
    """Find shortest path using BFS"""
    if start == goal:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    # Direction deltas: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        (current, path) = queue.popleft()
        
        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            
            # Check bounds
            if (0 <= next_pos[0] < env.grid_size and 
                0 <= next_pos[1] < env.grid_size and
                next_pos not in visited):
                
                # Check if position is valid (walkable)
                if env._is_valid_pos(next_pos):
                    new_path = path + [next_pos]
                    
                    if next_pos == goal:
                        return new_path
                    
                    queue.append((next_pos, new_path))
                    visited.add(next_pos)
    
    return None  # No path found


def analyze_human_paths():
    """Analyze possible paths for human to reach goal"""
    print("=== HUMAN PATH ANALYSIS ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    human_start = (1, 3)  # Human starting position
    goal = (3, 2)         # Goal position
    
    print(f"Human starts at: {human_start}")
    print(f"Goal at: {goal}")
    
    # Find path without considering the door
    print(f"\nFinding path ignoring door constraint...")
    path = find_path_bfs(env, human_start, goal)
    
    if path:
        print(f"Path found: {path}")
        print(f"Path length: {len(path) - 1} steps")
        
        # Check each step
        print(f"\nPath analysis:")
        for i, pos in enumerate(path):
            char = env.grid[pos] if pos != human_start else 'H'
            is_valid = env._is_valid_pos(pos)
            print(f"  Step {i}: {pos} - char='{char}', valid={is_valid}")
    else:
        print(f"No path found!")
    
    # Check all walkable positions
    print(f"\nAll walkable positions:")
    walkable = []
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            pos = (r, c)
            if env._is_valid_pos(pos):
                walkable.append(pos)
                char = env.grid[pos]
                print(f"  {pos}: '{char}'")
    
    print(f"\nTotal walkable positions: {len(walkable)}")
    
    return path


def test_robot_human_coordination():
    """Test if robot and human can coordinate to solve puzzle"""
    print(f"\n=== ROBOT-HUMAN COORDINATION TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    print(f"Initial positions:")
    print(f"  Robot: {env.agent_positions[robot_id]}")
    print(f"  Human: {env.agent_positions[human_id]}")
    
    # Strategy: Robot moves out of the way, human finds path to goal
    
    # Step 1: Robot gets key
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})
    env.step({robot_id: 3})  # pickup key
    print(f"\nRobot picked up key: {list(env.robot_has_keys)}")
    
    # Step 2: Robot moves out of the way to (3,1)
    # Robot needs to go (1,1) -> (3,1)
    # Path: down to (2,1) [wall], so go around: (1,1) -> (1,3) -> (3,3) -> (3,1)
    # But (1,3) has human! Robot needs to coordinate with human movement
    
    print(f"\nRobot trying to reach (3,1) to be adjacent to goal...")
    
    # Robot faces down to try moving down
    while env.agent_dirs[robot_id] != 2:  # Face down
        env.step({robot_id: 1})
    env.step({robot_id: 2})  # Try to move down
    
    robot_pos_after_move = env.agent_positions[robot_id]
    print(f"Robot position after trying to move down: {robot_pos_after_move}")
    
    if robot_pos_after_move == (1, 1):
        print("Robot cannot move down - blocked by wall at (2,1)")
        
        # Try moving right to where human is
        while env.agent_dirs[robot_id] != 1:  # Face right
            env.step({robot_id: 1})
        env.step({robot_id: 2})  # Try to move right
        
        robot_pos_after_move = env.agent_positions[robot_id]
        print(f"Robot position after trying to move right: {robot_pos_after_move}")
        
        if robot_pos_after_move == (1, 1):
            print("Robot cannot move right - blocked by human or key position")
    
    # The insight: Robot and human need to swap positions!
    print(f"\nKey insight: Robot and human need to swap positions")
    print(f"- Human at (1,3) needs to move to (1,1)")
    print(f"- Robot at (1,1) needs to move to path that leads to (3,1) or (3,2)")
    
    # Test human movement to (1,1)
    print(f"\nTesting human movement...")
    
    # Human faces left to move to (1,2) - but that has key!
    while env.agent_dirs[human_id] != 3:  # Face left
        env.step({human_id: 0})
    env.step({human_id: 2})  # Try to move left
    
    human_pos_after_move = env.agent_positions[human_id]
    print(f"Human position after trying to move left: {human_pos_after_move}")
    
    if human_pos_after_move == (1, 3):
        print("Human cannot move left - blocked by key position (1,2)")
    
    return False


if __name__ == "__main__":
    path = analyze_human_paths()
    test_robot_human_coordination()