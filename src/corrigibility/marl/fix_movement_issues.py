#!/usr/bin/env python3
"""
Debug and fix movement issues in the environment.
"""

import numpy as np

from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def debug_movement_step_by_step():
    """Debug movement step by step"""
    print("=== MOVEMENT DEBUG ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Initial state:")
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    print(f"  Human: pos={env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
    print(f"  Keys: {[k['pos'] for k in env.keys]}")
    print(f"  Doors: {[(d['pos'], d['is_locked']) for d in env.doors]}")
    
    # Direction mapping: 0=up, 1=right, 2=down, 3=left
    dir_names = {0: "up", 1: "right", 2: "down", 3: "left"}
    action_names = {0: "turn_left", 1: "turn_right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
    
    print(f"\nDirection mappings: {dir_names}")
    print(f"Action mappings: {action_names}")
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    # Test 1: Robot turns
    print(f"\n--- Test 1: Robot Turns ---")
    initial_robot_dir = env.agent_dirs[robot_id]
    print(f"Robot initial direction: {initial_robot_dir} ({dir_names[initial_robot_dir]})")
    
    # Turn left
    env.step({robot_id: 0})
    new_robot_dir = env.agent_dirs[robot_id]
    print(f"After turn_left: {new_robot_dir} ({dir_names[new_robot_dir]})")
    
    # Turn right
    env.step({robot_id: 1})
    new_robot_dir = env.agent_dirs[robot_id]
    print(f"After turn_right: {new_robot_dir} ({dir_names[new_robot_dir]})")
    
    # Test 2: Robot movement to key
    print(f"\n--- Test 2: Robot Movement to Key ---")
    robot_pos = env.agent_positions[robot_id]
    key_pos = env.keys[0]['pos']
    
    print(f"Robot at {robot_pos}, key at {key_pos}")
    print(f"Robot facing {dir_names[env.agent_dirs[robot_id]]}")
    
    # Robot needs to face right (direction 1) to face the key
    while env.agent_dirs[robot_id] != 1:
        env.step({robot_id: 1})  # turn right
        print(f"  Turned to face {dir_names[env.agent_dirs[robot_id]]}")
    
    # Now try to move forward to the key
    print(f"Robot now facing {dir_names[env.agent_dirs[robot_id]]}, trying to move forward...")
    
    # Check what's in front
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    robot_dir = env.agent_dirs[robot_id]
    dx, dy = deltas[robot_dir]
    front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
    
    print(f"  Front position would be: {front_pos}")
    print(f"  Is valid: {env._is_valid_pos(front_pos)}")
    print(f"  Grid char at front: '{env.grid[front_pos]}'")
    
    # Try to move forward
    env.step({robot_id: 2})
    new_robot_pos = env.agent_positions[robot_id]
    print(f"  After move: robot at {new_robot_pos}")
    
    # Test 3: Key pickup
    print(f"\n--- Test 3: Key Pickup ---")
    robot_pos = env.agent_positions[robot_id]
    robot_dir = env.agent_dirs[robot_id]
    
    # Check if robot is adjacent to key
    dx, dy = deltas[robot_dir]
    front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
    
    print(f"Robot at {robot_pos}, facing {dir_names[robot_dir]}")
    print(f"Front position: {front_pos}")
    print(f"Key positions: {[tuple(k['pos']) for k in env.keys]}")
    print(f"Is key in front: {any(tuple(k['pos']) == front_pos for k in env.keys)}")
    
    # Try pickup
    keys_before = len(env.keys)
    robot_keys_before = len(env.robot_has_keys)
    
    env.step({robot_id: 3})  # pickup
    
    keys_after = len(env.keys)
    robot_keys_after = len(env.robot_has_keys)
    
    print(f"Before pickup: {keys_before} keys in world, robot has {robot_keys_before}")
    print(f"After pickup: {keys_after} keys in world, robot has {robot_keys_after}")
    print(f"Robot keys: {list(env.robot_has_keys)}")
    
    # Test 4: Human movement
    print(f"\n--- Test 4: Human Movement ---")
    human_pos = env.agent_positions[human_id]
    human_dir = env.agent_dirs[human_id]
    
    print(f"Human at {human_pos}, facing {dir_names[human_dir]}")
    
    # Check what's in front of human
    dx, dy = deltas[human_dir]
    front_pos = (human_pos[0] + dx, human_pos[1] + dy)
    
    print(f"Human front position would be: {front_pos}")
    print(f"Is valid: {env._is_valid_pos(front_pos)}")
    print(f"Grid char at front: '{env.grid[front_pos]}'")
    
    # Try human turn
    env.step({human_id: 0})  # turn left
    new_human_dir = env.agent_dirs[human_id]
    print(f"After turn_left: human facing {dir_names[new_human_dir]}")
    
    # Try human move
    env.step({human_id: 2})  # forward
    new_human_pos = env.agent_positions[human_id]
    print(f"After move: human at {new_human_pos}")
    
    # Check if there's a path for human to goal
    print(f"\n--- Test 5: Path Analysis ---")
    goal_pos = (3, 2)
    human_pos = env.agent_positions[human_id]
    
    print(f"Human at {human_pos}, goal at {goal_pos}")
    print(f"Door state: {[(d['pos'], d['is_locked'], d['is_open']) for d in env.doors]}")
    
    # Check if position (2,2) is passable (where the door is)
    door_pos = (2, 2)
    print(f"Door position {door_pos} valid: {env._is_valid_pos(door_pos)}")
    print(f"Grid at door: '{env.grid[door_pos]}'")


def test_corrected_sequence():
    """Test a corrected sequence with proper understanding"""
    print(f"\n=== CORRECTED SEQUENCE TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    print("Sequence: Robot gets key -> Robot opens door -> Human reaches goal")
    
    # Step 1: Robot faces the key (robot at (1,1), key at (1,2))
    # Robot starts facing down (2), needs to face right (1)
    print(f"\nStep 1: Robot turns to face key")
    print(f"  Robot at {env.agent_positions[robot_id]}, direction {env.agent_dirs[robot_id]}")
    
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})  # turn right
    print(f"  Robot now facing right: {env.agent_dirs[robot_id]}")
    
    # Step 2: Robot moves to key position
    print(f"\nStep 2: Robot moves to key")
    env.step({robot_id: 2})  # forward
    print(f"  Robot moved to {env.agent_positions[robot_id]}")
    
    # Step 3: Robot picks up key
    print(f"\nStep 3: Robot picks up key")
    env.step({robot_id: 3})  # pickup
    print(f"  Robot has keys: {list(env.robot_has_keys)}")
    print(f"  Keys in world: {len(env.keys)}")
    
    # Step 4: Robot faces door (robot at (1,2), door at (2,2))
    # Robot needs to face down (2)
    print(f"\nStep 4: Robot turns to face door")
    while env.agent_dirs[robot_id] != 2:  # Face down
        env.step({robot_id: 1})  # turn right
    print(f"  Robot now facing down: {env.agent_dirs[robot_id]}")
    
    # Step 5: Robot moves to be adjacent to door
    print(f"\nStep 5: Robot moves to door")
    env.step({robot_id: 2})  # forward
    print(f"  Robot moved to {env.agent_positions[robot_id]}")
    
    # Step 6: Robot opens door
    print(f"\nStep 6: Robot opens door")
    door_before = env.doors[0]['is_locked']
    env.step({robot_id: 5})  # toggle
    door_after = env.doors[0]['is_locked']
    print(f"  Door locked before: {door_before}, after: {door_after}")
    print(f"  Door state: {env.doors[0]}")
    
    # Step 7: Human moves to goal
    print(f"\nStep 7: Human moves to goal")
    human_start_pos = env.agent_positions[human_id]
    goal_pos = (3, 2)
    
    print(f"  Human at {human_start_pos}, goal at {goal_pos}")
    
    # Human needs to go from (1,3) to (3,2)
    # Path: (1,3) -> (2,3) -> (2,2) -> (3,2)
    # But (2,3) is wall, so human needs to go (1,3) -> (1,2) -> (2,2) -> (3,2)
    
    # First, human faces left to go to (1,2)
    while env.agent_dirs[human_id] != 3:  # Face left
        env.step({human_id: 0})  # turn left
    print(f"    Human facing left: {env.agent_dirs[human_id]}")
    
    # Move left
    env.step({human_id: 2})  # forward
    print(f"    Human moved to {env.agent_positions[human_id]}")
    
    # Face down to go through door
    while env.agent_dirs[human_id] != 2:  # Face down
        env.step({human_id: 1})  # turn right
    print(f"    Human facing down: {env.agent_dirs[human_id]}")
    
    # Move through door
    env.step({human_id: 2})  # forward
    print(f"    Human moved to {env.agent_positions[human_id]}")
    
    # Move to goal
    env.step({human_id: 2})  # forward
    final_pos = env.agent_positions[human_id]
    print(f"    Human final position: {final_pos}")
    
    # Check success
    success = tuple(final_pos) == goal_pos
    print(f"\nGoal reached: {success}")
    
    return success


if __name__ == "__main__":
    debug_movement_step_by_step()
    success = test_corrected_sequence()
    
    print(f"\n{'='*50}")
    if success:
        print("✅ GOAL IS REACHABLE!")
        print("The sequence works. Issues are likely in:")
        print("- Action selection during training")
        print("- Learning rate / exploration parameters")
        print("- State representation")
    else:
        print("❌ GOAL IS NOT REACHABLE")
        print("Fundamental environment or movement issues")
    print(f"{'='*50}")