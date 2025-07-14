#!/usr/bin/env python3
"""
Test robot cooperation to help human reach goal.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corrigibility.marl.env import CustomEnvironment, Actions


def test_robot_key_pickup():
    """Test if robot can pick up key and open door."""
    
    print("=== ROBOT COOPERATION TEST ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    print(f"Initial positions:")
    print(f"  Robot: {env.agent_positions['robot_0']}")
    print(f"  Human: {env.agent_positions['human_0']}")
    print(f"  Key: {env.keys[0]['pos']}")
    print(f"  Door: {env.doors[0]['pos']}, locked: {env.doors[0]['is_locked']}")
    
    # Test robot movement toward key
    print(f"\n=== ROBOT MOVING TO KEY ===")
    
    # Robot should move from (1,1) to (1,2) to get the key
    robot_pos = env.agent_positions['robot_0']
    key_pos = env.keys[0]['pos']
    
    print(f"Robot needs to move from {robot_pos} to {key_pos}")
    
    # Test robot actions
    step = 0
    max_steps = 10
    
    while step < max_steps:
        robot_pos = env.agent_positions['robot_0']
        print(f"\nStep {step + 1}: Robot at {robot_pos}")
        
        # If robot is at key position, try to pick it up
        if robot_pos == key_pos:
            print("Robot is at key position, trying to pick up...")
            actions = {'robot_0': 3, 'human_0': 6}  # pickup, human does nothing
        else:
            # Move robot toward key (east)
            print("Robot moving east toward key...")
            actions = {'robot_0': 2, 'human_0': 6}  # forward, human does nothing
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        print(f"  Robot moved to: {env.agent_positions['robot_0']}")
        print(f"  Robot inventory: {getattr(env, 'robot_has_keys', 'No inventory')}")
        print(f"  Rewards: {rewards}")
        
        # Check if key was picked up
        key_still_there = any(key['pos'] == key_pos for key in env.keys)
        print(f"  Key still at {key_pos}: {key_still_there}")
        
        step += 1
        
        # If robot picked up key, try to open door
        if not key_still_there:
            print(f"\n=== ROBOT OPENING DOOR ===")
            door_pos = env.doors[0]['pos']
            
            # Move robot to door position
            while env.agent_positions['robot_0'] != door_pos and step < max_steps:
                robot_pos = env.agent_positions['robot_0']
                print(f"Step {step + 1}: Robot at {robot_pos}, moving to door at {door_pos}")
                
                # Simple pathfinding: move toward door
                if robot_pos[0] < door_pos[0]:
                    # Move south
                    actions = {'robot_0': 2, 'human_0': 6}  # forward (assuming facing south)
                elif robot_pos[1] > door_pos[1]:
                    # Move west - need to turn first
                    actions = {'robot_0': 0, 'human_0': 6}  # turn left
                else:
                    # Try to move forward
                    actions = {'robot_0': 2, 'human_0': 6}
                
                obs, rewards, terms, truncs, infos = env.step(actions)
                print(f"  Robot moved to: {env.agent_positions['robot_0']}")
                step += 1
            
            # Now try to toggle/open the door
            if env.agent_positions['robot_0'] == door_pos:
                print("Robot at door, trying to toggle...")
                actions = {'robot_0': 5, 'human_0': 6}  # toggle
                obs, rewards, terms, truncs, infos = env.step(actions)
                
                print(f"Door state after toggle: locked={env.doors[0]['is_locked']}, open={env.doors[0]['is_open']}")
            
            break
        
        if any(terms.values()) or any(truncs.values()):
            break


def test_human_movement_after_door_open():
    """Test if human can move after door is opened."""
    
    print(f"\n=== TESTING HUMAN MOVEMENT AFTER DOOR OPENING ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    # Manually open the door to test human movement
    env.doors[0]['is_locked'] = False
    env.doors[0]['is_open'] = True
    
    print(f"Door manually opened: locked={env.doors[0]['is_locked']}, open={env.doors[0]['is_open']}")
    
    # Now test if human can move
    human_pos = env.agent_positions['human_0']
    print(f"Human at: {human_pos}")
    
    # Test human movement south toward the open door
    for step in range(5):
        human_pos = env.agent_positions['human_0']
        print(f"\nStep {step + 1}: Human at {human_pos}")
        
        # Try different movement actions
        if step == 0:
            action = 0  # turn left
        elif step == 1:
            action = 2  # forward
        elif step == 2:
            action = 2  # forward
        else:
            action = 2  # forward
        
        actions = {'human_0': action, 'robot_0': 6}
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        new_pos = env.agent_positions['human_0']
        print(f"  {Actions(action).name} -> moved to {new_pos}")
        
        if new_pos != human_pos:
            print("  ✅ Human moved!")
        else:
            print("  ❌ Human didn't move")


if __name__ == "__main__":
    test_robot_key_pickup()
    test_human_movement_after_door_open()