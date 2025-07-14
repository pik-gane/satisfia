#!/usr/bin/env python3
"""
Test manual cooperation sequence to verify the environment works for the simple map.
"""

import sys
import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment, Actions

def test_manual_cooperation():
    """Test manual cooperation sequence"""
    print("=== Test Manual Cooperation ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    # Print the grid to understand the layout
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
    
    # Print all open positions
    print("\nOpen positions:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if env.grid[i, j] == ' ':
                print(f"  ({i}, {j}): empty")
    
    print(f"\nDoor position: {env.doors[0]['pos']}")
    print(f"Goal position: {env.human_goals['human_0']}")
    
    # Manual cooperation sequence:
    # Robot starts at (1,1) facing down (2), Human starts at (1,3) facing down (2)
    # Robot needs to turn right to face key at (1,2), then move forward and pick up key
    # Human needs to turn left to face key position, then move there, then down to door
    # Robot needs to move to door and open it, then human can move through
    
    cooperation_sequence = [
        # Step 1: Robot turns left to face right (direction 1)
        {'human_0': Actions.done, 'robot_0': Actions.turn_left},
        # Step 2: Robot moves right to key position (1,2) - but can't because key is there
        {'human_0': Actions.done, 'robot_0': Actions.forward},
        # Step 3: Robot picks up key (from (1,1) facing right, front_pos is (1,2))
        {'human_0': Actions.done, 'robot_0': Actions.pickup},
        # Step 4: Robot moves right to key position (1,2) - now possible since key is gone
        {'human_0': Actions.done, 'robot_0': Actions.forward},
        # Step 5: Robot turns right to face down (direction 2)
        {'human_0': Actions.done, 'robot_0': Actions.turn_right},
        # Step 6: Robot opens door (robot at (1,2) facing down, front_pos is (2,2))
        {'human_0': Actions.done, 'robot_0': Actions.toggle},
        # Step 7: Human turns right to face left (direction 3)
        {'human_0': Actions.turn_right, 'robot_0': Actions.done},
        # Step 8: Human moves left to robot position (1,2) - but robot is there!
        {'human_0': Actions.forward, 'robot_0': Actions.done},
        # Step 9: Robot moves down to (2,2) - through the door
        {'human_0': Actions.done, 'robot_0': Actions.forward},
        # Step 10: Robot moves down to (3,2) - to the goal
        {'human_0': Actions.done, 'robot_0': Actions.forward},
        # Step 11: Robot moves left to (3,1) to get out of the way
        {'human_0': Actions.done, 'robot_0': Actions.turn_left},
        # Step 12: Robot moves left to (3,1)
        {'human_0': Actions.done, 'robot_0': Actions.forward},
        # Step 13: Human moves left to robot's old position (1,2)
        {'human_0': Actions.forward, 'robot_0': Actions.done},
        # Step 14: Human turns left to face down (direction 2)
        {'human_0': Actions.turn_left, 'robot_0': Actions.done},
        # Step 15: Human moves down to door position (2,2)
        {'human_0': Actions.forward, 'robot_0': Actions.done},
        # Step 16: Human moves down through door to goal (3,2)
        {'human_0': Actions.forward, 'robot_0': Actions.done},
    ]
    
    goal_reached = False
    for step, actions in enumerate(cooperation_sequence):
        print(f"\nStep {step}: Actions={actions}")
        
        # Debug positions before step
        print(f"  Before: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Directions: Human={env.agent_dirs['human_0']}, Robot={env.agent_dirs['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has keys: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Directions: Human={env.agent_dirs['human_0']}, Robot={env.agent_dirs['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has keys: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Rewards: {rewards}")
        
        # Check if human reached goal
        human_pos = env.agent_positions['human_0']
        goal_pos = env.human_goals['human_0']
        if tuple(human_pos) == tuple(goal_pos):
            print("GOAL REACHED!")
            goal_reached = True
            break
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("Episode ended")
            break
    
    return goal_reached

if __name__ == "__main__":
    success = test_manual_cooperation()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")