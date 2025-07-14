#!/usr/bin/env python3
"""
Simple debug test to verify movement works in the environment.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def test_manual_movement():
    """Test manual movement to ensure environment works"""
    print("=== Testing Manual Movement ===")
    
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
    
    # Print the grid
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
    
    # Test specific actions
    test_actions = [
        {"human_0": 2, "robot_0": 2},  # Both forward
        {"human_0": 0, "robot_0": 1},  # Human left, Robot right
        {"human_0": 2, "robot_0": 2},  # Both forward
    ]
    
    for step, actions in enumerate(test_actions):
        print(f"\nStep {step}: Actions = {actions}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Rewards: {rewards}")
        print(f"  Terms: {terms}, Truncs: {truncs}")
        
        if any(terms.values()) or any(truncs.values()):
            print("  Episode ended")
            break
    
    return True

def test_greedy_policy():
    """Test if greedy action selection is working"""
    print("\n=== Testing Greedy Policy ===")
    
    # Test different Q-value scenarios
    test_cases = [
        [1.0, 2.0, 3.0],  # Action 2 should be selected
        [5.0, 1.0, 2.0],  # Action 0 should be selected  
        [0.0, 0.0, 0.0],  # Should pick action 0 (first max)
        [-1.0, -0.5, -2.0],  # Action 1 should be selected
    ]
    
    for i, q_values in enumerate(test_cases):
        q_vals = np.array(q_values)
        action = np.argmax(q_vals)
        print(f"Test {i}: Q-values={q_vals}, Selected action={action}")
    
    return True

def test_environment_state():
    """Test if environment state is changing properly"""
    print("\n=== Testing Environment State ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    # Get initial observation
    initial_obs = env.observe('human_0')
    print(f"Initial human observation: {initial_obs}")
    
    # Try moving human forward
    actions = {"human_0": 2, "robot_0": 2}
    obs, rewards, terms, truncs, _ = env.step(actions)
    
    # Get new observation
    new_obs = env.observe('human_0')
    print(f"After forward action observation: {new_obs}")
    print(f"Observation changed: {not np.array_equal(initial_obs, new_obs)}")
    
    return True

if __name__ == "__main__":
    test_manual_movement()
    test_greedy_policy()  
    test_environment_state()