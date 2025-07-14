#!/usr/bin/env python3
"""
Debug the environment to understand why agents aren't moving.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def debug_environment_state():
    """Debug the environment state and constraints"""
    print("=== Debugging Environment State ===")
    
    map_layout, map_metadata = get_map()
    print("Map layout:")
    for i, row in enumerate(map_layout):
        print(f"Row {i}: {row}")
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"\nAfter reset:")
    print(f"Agent positions: {env.agent_positions}")
    print(f"Human goals: {env.human_goals}")
    
    # Check if environment has basic attributes
    print(f"\nEnvironment attributes:")
    print(f"Has grid: {hasattr(env, 'grid')}")
    print(f"Has action_spaces: {hasattr(env, 'action_spaces')}")
    
    if hasattr(env, 'action_spaces'):
        print(f"Action spaces: {env.action_spaces}")
    
    # Test if step function works at all
    print(f"\nTesting step function...")
    actions = {"robot_0": 0, "human_0": 0}
    
    try:
        result = env.step(actions)
        print(f"Step result type: {type(result)}")
        print(f"Step result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        obs, rewards, terms, truncs, info = result
        print(f"Observations: {type(obs)}")
        print(f"Rewards: {rewards}")
        print(f"Terminated: {terms}")
        print(f"Truncated: {truncs}")
        print(f"Info: {info}")
        
        print(f"Agent positions after step: {env.agent_positions}")
        
    except Exception as e:
        print(f"Error in step: {e}")
        import traceback
        traceback.print_exc()

def test_simple_movements():
    """Test if any action causes movement"""
    print(f"\n=== Testing All Actions for Movement ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Test robot actions
    print("Testing robot actions:")
    for action in range(6):
        env.reset()
        initial_robot = env.agent_positions['robot_0'].copy()
        initial_human = env.agent_positions['human_0'].copy()
        
        actions = {"robot_0": action, "human_0": 0}
        env.step(actions)
        
        robot_moved = not np.array_equal(initial_robot, env.agent_positions['robot_0'])
        human_moved = not np.array_equal(initial_human, env.agent_positions['human_0'])
        
        print(f"  Robot action {action}: Robot moved: {robot_moved}, Human moved: {human_moved}")
        if robot_moved:
            print(f"    Robot: {initial_robot} -> {env.agent_positions['robot_0']}")
        if human_moved:
            print(f"    Human: {initial_human} -> {env.agent_positions['human_0']}")
    
    # Test human actions
    print("\nTesting human actions:")
    for action in range(3):
        env.reset()
        initial_robot = env.agent_positions['robot_0'].copy()
        initial_human = env.agent_positions['human_0'].copy()
        
        actions = {"robot_0": 0, "human_0": action}
        env.step(actions)
        
        robot_moved = not np.array_equal(initial_robot, env.agent_positions['robot_0'])
        human_moved = not np.array_equal(initial_human, env.agent_positions['human_0'])
        
        print(f"  Human action {action}: Robot moved: {robot_moved}, Human moved: {human_moved}")
        if robot_moved:
            print(f"    Robot: {initial_robot} -> {env.agent_positions['robot_0']}")
        if human_moved:
            print(f"    Human: {initial_human} -> {env.agent_positions['human_0']}")

if __name__ == "__main__":
    debug_environment_state()
    test_simple_movements()