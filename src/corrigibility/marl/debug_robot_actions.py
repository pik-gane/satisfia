#!/usr/bin/env python3
"""
Debug robot actions to understand what's blocking movement.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_robot_actions():
    """Debug robot actions one by one"""
    print("=== Debug Robot Actions ===")
    
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
    print(f"  Keys: {[(k['pos'], k['color']) for k in env.keys]}")
    print(f"  Grid at robot position (1,1): {env.grid[1, 1]}")
    print(f"  Grid at key position (1,2): {env.grid[1, 2]}")
    
    # Test each action individually
    test_actions = [
        ("Turn left", 0),
        ("Turn right", 1), 
        ("Move forward", 2),
        ("Pickup", 3),
        ("Drop", 4),
        ("Toggle", 5),
    ]
    
    for action_name, action_id in test_actions:
        print(f"\n--- Testing {action_name} (action {action_id}) ---")
        
        # Reset environment
        env.reset()
        
        print(f"Before: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
        print(f"Keys: {[(k['pos'], k['color']) for k in env.keys]}")
        
        # Execute action
        actions = {"human_0": 0, "robot_0": action_id}
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"After: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
        print(f"Keys: {[(k['pos'], k['color']) for k in env.keys]}")
        print(f"Robot has keys: {env.robot_has_keys}")
        print(f"Rewards: {rewards}")
        
        # Check if anything changed
        if env.agent_positions['robot_0'] != (1, 1):
            print("  ✅ Robot moved!")
        elif len(env.robot_has_keys) > 0:
            print("  ✅ Robot picked up key!")
        elif env.agent_dirs['robot_0'] != 2:  # Initial direction
            print("  ✅ Robot turned!")
        else:
            print("  ❌ No change")

def test_movement_directions():
    """Test movement in each direction"""
    print("\n=== Test Movement Directions ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    directions = [
        ("Up", 0),
        ("Right", 1),
        ("Down", 2),
        ("Left", 3),
    ]
    
    for dir_name, dir_id in directions:
        print(f"\n--- Testing movement {dir_name} (direction {dir_id}) ---")
        
        # Reset and set robot direction
        env.reset()
        env.agent_dirs['robot_0'] = dir_id
        
        print(f"Before: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
        
        # Try to move forward
        actions = {"human_0": 0, "robot_0": 2}  # Move forward
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"After: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
        
        if env.agent_positions['robot_0'] != (1, 1):
            print("  ✅ Robot moved!")
        else:
            print("  ❌ Robot blocked")
            # Check what's in that direction
            deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dx, dy = deltas[dir_id]
            target_pos = (1 + dx, 1 + dy)
            if (0 <= target_pos[0] < env.grid_size and 
                0 <= target_pos[1] < env.grid_size):
                print(f"    Target position {target_pos} has: {env.grid[target_pos[0], target_pos[1]]}")
            else:
                print(f"    Target position {target_pos} is out of bounds")

if __name__ == "__main__":
    debug_robot_actions()
    test_movement_directions()