#!/usr/bin/env python3
"""
Test the correct sequence for solving the puzzle.
"""

import numpy as np

from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def test_correct_sequence():
    """Test the correct sequence with proper adjacency"""
    print("=== CORRECT SEQUENCE TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    print(f"Initial state:")
    print(f"  Robot: pos={env.agent_positions[robot_id]}, dir={env.agent_dirs[robot_id]}")
    print(f"  Human: pos={env.agent_positions[human_id]}, dir={env.agent_dirs[human_id]}")
    print(f"  Keys: {[k['pos'] for k in env.keys]}")
    print(f"  Doors: {[(d['pos'], d['is_locked']) for d in env.doors]}")
    
    # Robot at (1,1), key at (1,2), door at (2,2), human at (1,3), goal at (3,2)
    
    # Step 1: Robot faces right to face key (robot starts facing down=2)
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})  # turn right
    print(f"\nStep 1: Robot faces right to be adjacent to key")
    print(f"  Robot now facing: {env.agent_dirs[robot_id]} (1=right)")
    
    # Step 2: Robot picks up key (robot is adjacent to key)
    env.step({robot_id: 3})  # pickup
    print(f"\nStep 2: Robot picks up key")
    print(f"  Robot has keys: {list(env.robot_has_keys)}")
    print(f"  Keys remaining: {len(env.keys)}")
    
    # Step 3: Robot faces down to face door
    while env.agent_dirs[robot_id] != 2:  # Face down
        env.step({robot_id: 1})  # turn right
    print(f"\nStep 3: Robot faces down to be adjacent to door")
    print(f"  Robot now facing: {env.agent_dirs[robot_id]} (2=down)")
    
    # Step 4: Robot moves to be adjacent to door at (2,2)
    # Robot needs to be at (2,1) to be adjacent to door at (2,2)
    # Robot is at (1,1), so move down to (2,1)
    env.step({robot_id: 2})  # forward
    print(f"\nStep 4: Robot moves to be adjacent to door")
    print(f"  Robot moved to: {env.agent_positions[robot_id]}")
    
    # Step 5: Robot faces right to face the door
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})  # turn right
    print(f"\nStep 5: Robot faces right to face door")
    print(f"  Robot now facing: {env.agent_dirs[robot_id]} (1=right)")
    
    # Step 6: Robot opens door
    env.step({robot_id: 5})  # toggle
    print(f"\nStep 6: Robot opens door")
    print(f"  Door state: {env.doors[0]}")
    print(f"  Door is locked: {env.doors[0]['is_locked']}")
    print(f"  Door is open: {env.doors[0]['is_open']}")
    
    # Step 7: Human moves to goal
    print(f"\nStep 7: Human moves to goal")
    print(f"  Human at: {env.agent_positions[human_id]}")
    print(f"  Goal at: (3, 2)")
    
    # Human needs to go from (1,3) to (3,2) through the now-open door at (2,2)
    # Path: (1,3) -> (1,2) -> (2,2) -> (3,2)
    
    # Face left to move to (1,2)
    while env.agent_dirs[human_id] != 3:  # Face left
        env.step({human_id: 0})  # turn left
    env.step({human_id: 2})  # forward
    print(f"  Human moved to: {env.agent_positions[human_id]}")
    
    # Face down to move to (2,2) through door
    while env.agent_dirs[human_id] != 2:  # Face down
        env.step({human_id: 1})  # turn right
    env.step({human_id: 2})  # forward
    print(f"  Human moved to: {env.agent_positions[human_id]}")
    
    # Continue down to goal at (3,2)
    env.step({human_id: 2})  # forward
    print(f"  Human moved to: {env.agent_positions[human_id]}")
    
    # Check success
    final_pos = env.agent_positions[human_id]
    goal_pos = (3, 2)
    success = tuple(final_pos) == goal_pos
    
    print(f"\nFinal result:")
    print(f"  Human final position: {final_pos}")
    print(f"  Goal position: {goal_pos}")
    print(f"  Success: {success}")
    
    return success


def test_movement_constraints():
    """Test what positions agents can and cannot move to"""
    print("\n=== MOVEMENT CONSTRAINTS TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print("Grid layout:")
    for r in range(env.grid_size):
        row = ""
        for c in range(env.grid_size):
            pos = (r, c)
            if pos == env.agent_positions.get('robot_0'):
                row += "R"
            elif pos == env.agent_positions.get('human_0'):
                row += "H"
            else:
                char = env.grid[r, c]
                if char == ' ':
                    row += "."
                else:
                    row += char
        print(f"  {row}")
    
    # Test each position
    print("\nPosition validity:")
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            pos = (r, c)
            is_valid = env._is_valid_pos(pos)
            char = env.grid[r, c]
            print(f"  ({r},{c}): valid={is_valid}, char='{char}'")
    
    # Test key position specifically
    key_pos = (1, 2)
    print(f"\nKey position {key_pos}:")
    print(f"  Valid for movement: {env._is_valid_pos(key_pos)}")
    print(f"  Grid character: '{env.grid[key_pos]}'")
    
    # Test door position
    door_pos = (2, 2)
    print(f"\nDoor position {door_pos}:")
    print(f"  Valid for movement: {env._is_valid_pos(door_pos)}")
    print(f"  Grid character: '{env.grid[door_pos]}'")


if __name__ == "__main__":
    test_movement_constraints()
    success = test_correct_sequence()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ CORRECT SEQUENCE WORKS!")
        print("The puzzle is solvable with proper coordination.")
    else:
        print("❌ SEQUENCE FAILED")
        print("Need to debug the movement mechanics.")
    print(f"{'='*50}")