#!/usr/bin/env python3
"""
Debug why the robot isn't learning to pick up the key and open the door.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_robot_actions_step_by_step():
    """Debug robot actions step by step to see what's happening"""
    print("=== Debug Robot Key Pickup and Door Opening ===")
    
    # Get map and create environment
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
    print(f"  Robot has keys: {env.robot_has_keys}")
    print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
    
    print(f"\nGrid layout:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) == env.agent_positions['robot_0']:
                print('R', end='')
            elif (i, j) == env.agent_positions['human_0']:
                print('H', end='')
            else:
                print(env.grid[i, j], end='')
        print()
    
    # Test the exact sequence that should work
    print(f"\n=== Testing Robot Key Pickup Sequence ===")
    
    # Step 1: Robot should turn right to face key
    print(f"\nStep 1: Robot turn right to face key")
    print(f"  Before: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    actions = {"robot_0": 1, "human_0": 0}  # Robot turn right, human turn left
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"  After: Robot pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    print(f"  Robot facing direction: {env.agent_dirs['robot_0']} (should be 1 for right)")
    
    # Step 2: Robot should pickup key
    print(f"\nStep 2: Robot pickup key")
    print(f"  Before pickup: Keys={[k['pos'] for k in env.keys]}, Robot has={env.robot_has_keys}")
    actions = {"robot_0": 3, "human_0": 0}  # Robot pickup, human turn left
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"  After pickup: Keys={[k['pos'] for k in env.keys]}, Robot has={env.robot_has_keys}")
    print(f"  Pickup successful: {len(env.robot_has_keys) > 0}")
    print(f"  Rewards: {rewards}")
    
    if len(env.robot_has_keys) > 0:
        print(f"  ✅ Robot successfully picked up key!")
        
        # Step 3: Robot move to door area
        print(f"\nStep 3: Robot move right to key position")
        print(f"  Before move: Robot pos={env.agent_positions['robot_0']}")
        actions = {"robot_0": 2, "human_0": 0}  # Robot move forward
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After move: Robot pos={env.agent_positions['robot_0']}")
        
        # Step 4: Robot turn to face door
        print(f"\nStep 4: Robot turn right to face door")
        print(f"  Before turn: Robot dir={env.agent_dirs['robot_0']}")
        actions = {"robot_0": 1, "human_0": 0}  # Robot turn right to face down
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After turn: Robot dir={env.agent_dirs['robot_0']} (should be 2 for down)")
        
        # Step 5: Robot should open door
        print(f"\nStep 5: Robot open door")
        print(f"  Before door open: Doors={[(d['pos'], d['is_open']) for d in env.doors]}")
        actions = {"robot_0": 5, "human_0": 0}  # Robot toggle door
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"  After door action: Doors={[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Door opened: {any(d['is_open'] for d in env.doors)}")
        print(f"  Rewards: {rewards}")
        
        if any(d['is_open'] for d in env.doors):
            print(f"  ✅ Robot successfully opened door!")
            
            # Step 6: Human can now move to goal
            print(f"\nStep 6: Human move toward goal")
            goal = env.human_goals['human_0']
            print(f"  Human goal: {goal}")
            print(f"  Before: Human pos={env.agent_positions['human_0']}")
            
            # Human moves down
            actions = {"robot_0": 0, "human_0": 2}  # Human move forward
            obs, rewards, terms, truncs, _ = env.step(actions)
            print(f"  After move down: Human pos={env.agent_positions['human_0']}")
            
            # Human moves down through door
            actions = {"robot_0": 0, "human_0": 2}  # Human move forward
            obs, rewards, terms, truncs, _ = env.step(actions)
            print(f"  After move through door: Human pos={env.agent_positions['human_0']}")
            
            # Check if human reached goal
            if tuple(env.agent_positions['human_0']) == tuple(goal):
                print(f"  🎉 HUMAN REACHED GOAL!")
                return True
            else:
                print(f"  Human still needs to reach goal at {goal}")
        else:
            print(f"  ❌ Robot failed to open door")
    else:
        print(f"  ❌ Robot failed to pick up key")
    
    return False

def debug_reward_structure():
    """Debug the reward structure to understand what the agents are learning"""
    print(f"\n=== Debug Reward Structure ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    goal = env.human_goals['human_0']
    
    print(f"Testing different actions and their rewards:")
    
    # Test robot key pickup
    print(f"\n1. Robot pickup key:")
    env.reset()
    env.agent_dirs['robot_0'] = 1  # Face right toward key
    actions = {"robot_0": 3, "human_0": 0}
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"   Actions: {actions}")
    print(f"   Rewards: {rewards}")
    print(f"   Keys picked up: {len(env.robot_has_keys) > 0}")
    
    # Test robot door toggle
    print(f"\n2. Robot door toggle (with key):")
    # Set up robot with key at door
    env.reset()
    env.robot_has_keys.add('blue')  # Give robot the key
    env.keys = []  # Remove key from environment
    env.agent_positions['robot_0'] = (1, 2)  # Move robot to door area
    env.agent_dirs['robot_0'] = 2  # Face down toward door
    actions = {"robot_0": 5, "human_0": 0}
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"   Actions: {actions}")
    print(f"   Rewards: {rewards}")
    print(f"   Door opened: {any(d['is_open'] for d in env.doors)}")
    
    # Test human reaching goal
    print(f"\n3. Human reaching goal:")
    env.reset()
    # Open the door manually
    for door in env.doors:
        door['is_open'] = True
    env.agent_positions['human_0'] = (2, 2)  # Position human near goal
    actions = {"robot_0": 0, "human_0": 2}
    obs, rewards, terms, truncs, _ = env.step(actions)
    print(f"   Actions: {actions}")
    print(f"   Rewards: {rewards}")
    print(f"   Human at goal: {tuple(env.agent_positions['human_0']) == tuple(goal)}")

if __name__ == "__main__":
    success = debug_robot_actions_step_by_step()
    debug_reward_structure()
    print(f"\nManual sequence success: {'SUCCESS' if success else 'FAILED'}")