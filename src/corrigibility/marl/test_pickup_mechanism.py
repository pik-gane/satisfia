#!/usr/bin/env python3
"""
Test the pickup mechanism to understand how it works.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def test_pickup_mechanism():
    """Test pickup mechanism in detail"""
    print("=== Test Pickup Mechanism ===")
    
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
    print(f"  Robot has keys: {env.robot_has_keys}")
    
    # Calculate front position for each direction
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    robot_pos = env.agent_positions['robot_0']
    
    for direction in range(4):
        dx, dy = deltas[direction]
        front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
        
        # Check if there's a key at front position
        key_at_front = any(tuple(k['pos']) == front_pos for k in env.keys)
        
        print(f"\nDirection {direction}: front_pos = {front_pos}")
        print(f"  Key at front position: {key_at_front}")
        if key_at_front:
            key_info = next(k for k in env.keys if tuple(k['pos']) == front_pos)
            print(f"  Key info: {key_info}")
            
            # Test pickup from this direction
            print(f"  Testing pickup from direction {direction}")
            
            # Reset and set robot direction
            env.reset()
            env.agent_dirs['robot_0'] = direction
            
            print(f"  Before pickup: Robot dir={env.agent_dirs['robot_0']}, Keys={len(env.keys)}, Robot has={env.robot_has_keys}")
            
            # Try pickup
            actions = {"human_0": 0, "robot_0": 3}  # pickup
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            print(f"  After pickup: Robot dir={env.agent_dirs['robot_0']}, Keys={len(env.keys)}, Robot has={env.robot_has_keys}")
            
            if len(env.robot_has_keys) > 0:
                print(f"  ✅ PICKUP SUCCESSFUL!")
                return True
            else:
                print(f"  ❌ Pickup failed")
    
    return False

def test_successful_cooperation():
    """Test the complete cooperation sequence"""
    print("\n=== Test Successful Cooperation ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Initial positions: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
    print(f"Initial directions: Robot={env.agent_dirs['robot_0']}, Human={env.agent_dirs['human_0']}")
    
    # Step 1: Robot turns to face key (from direction 2 to direction 1)
    print(f"\nStep 1: Robot turns to face key")
    actions = {"human_0": 0, "robot_0": 0}  # Robot: turn left (2->1)
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"  After turn: Robot dir={env.agent_dirs['robot_0']}")
    
    # Step 2: Robot picks up key
    print(f"\nStep 2: Robot picks up key")
    actions = {"human_0": 0, "robot_0": 3}  # Robot: pickup
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"  After pickup: Keys={len(env.keys)}, Robot has={env.robot_has_keys}")
    
    if len(env.robot_has_keys) > 0:
        print(f"  ✅ Key picked up successfully!")
        
        # Step 3: Robot turns to face door (from direction 1 to direction 2)
        print(f"\nStep 3: Robot turns to face door")
        actions = {"human_0": 0, "robot_0": 1}  # Robot: turn right (1->2)
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After turn: Robot dir={env.agent_dirs['robot_0']}")
        
        # Step 4: Robot moves to door
        print(f"\nStep 4: Robot moves to door")
        actions = {"human_0": 0, "robot_0": 2}  # Robot: move forward
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After move: Robot pos={env.agent_positions['robot_0']}")
        
        # Step 5: Robot opens door
        print(f"\nStep 5: Robot opens door")
        actions = {"human_0": 0, "robot_0": 5}  # Robot: toggle
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After toggle: Doors={[(d['pos'], d['is_open']) for d in env.doors]}")
        
        door_open = any(d['is_open'] for d in env.doors)
        if door_open:
            print(f"  ✅ Door opened successfully!")
            
            # Step 6: Human moves through door to goal
            print(f"\nStep 6: Human moves through door")
            # Human needs to turn and move to goal
            goal = env.human_goals['human_0']
            print(f"  Human goal: {goal}")
            
            # Human moves down through door
            actions = {"human_0": 2, "robot_0": 0}  # Human: move forward
            obs, rewards, terms, truncs, _ = env.step(actions)
            print(f"  After move: Human pos={env.agent_positions['human_0']}")
            
            # Human turns to face goal
            actions = {"human_0": 1, "robot_0": 0}  # Human: turn right
            obs, rewards, terms, truncs, _ = env.step(actions)
            print(f"  After turn: Human dir={env.agent_dirs['human_0']}")
            
            # Human moves to goal
            actions = {"human_0": 2, "robot_0": 0}  # Human: move forward
            obs, rewards, terms, truncs, _ = env.step(actions)
            print(f"  After move: Human pos={env.agent_positions['human_0']}")
            
            # Check if goal reached
            if tuple(env.agent_positions['human_0']) == tuple(goal):
                print(f"  🎉 GOAL REACHED!")
                return True
            else:
                print(f"  ❌ Goal not reached")
        else:
            print(f"  ❌ Door not opened")
    else:
        print(f"  ❌ Key not picked up")
    
    return False

if __name__ == "__main__":
    pickup_success = test_pickup_mechanism()
    if pickup_success:
        cooperation_success = test_successful_cooperation()
        print(f"\nFinal result: {'SUCCESS' if cooperation_success else 'FAILED'}")
    else:
        print(f"\nPickup mechanism failed - cannot proceed to cooperation test")