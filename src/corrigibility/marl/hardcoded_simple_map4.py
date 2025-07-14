#!/usr/bin/env python3
"""
Hardcoded solution for simple_map4 to understand exact mechanics and achieve 100% success.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_hardcoded_solution():
    """Test a hardcoded solution to understand the exact mechanics"""
    print("=== Testing Hardcoded Solution for Simple Map 4 ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    successes = 0
    total_tests = 10
    
    for test in range(total_tests):
        print(f"\n--- Test {test + 1} ---")
        env.reset()
        
        print(f"Initial positions: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
        print(f"Goal: {env.human_goals['human_0']}")
        
        # Hardcoded sequence based on our analysis
        actions_sequence = [
            # Step 1-2: Robot moves to key
            {"robot_0": 2, "human_0": 0},  # Robot: forward, Human: turn left  
            {"robot_0": 2, "human_0": 0},  # Robot: forward to (2,2), Human: turn left
            
            # Step 3: Robot picks up key
            {"robot_0": 3, "human_0": 0},  # Robot: pickup key, Human: turn left
            
            # Step 4-5: Robot moves to door
            {"robot_0": 1, "human_0": 0},  # Robot: turn right, Human: turn left  
            {"robot_0": 2, "human_0": 0},  # Robot: forward to door, Human: turn left
            
            # Step 6: Robot opens door
            {"robot_0": 5, "human_0": 0},  # Robot: toggle door, Human: turn left
            
            # Step 7-8: Human moves to goal
            {"robot_0": 0, "human_0": 2},  # Robot: turn left, Human: forward
            {"robot_0": 0, "human_0": 2},  # Robot: turn left, Human: forward to goal
        ]
        
        success = False
        for step, actions in enumerate(actions_sequence):
            print(f"  Step {step + 1}: Robot action {actions['robot_0']}, Human action {actions['human_0']}")
            
            env.step(actions)
            
            robot_pos = env.agent_positions['robot_0']
            human_pos = env.agent_positions['human_0']
            goal_pos = env.human_goals['human_0']
            
            print(f"    -> Robot={robot_pos}, Human={human_pos}")
            
            # Check for success
            if tuple(human_pos) == tuple(goal_pos):
                print(f"    🎉 SUCCESS! Human reached goal at step {step + 1}")
                success = True
                successes += 1
                break
        
        if not success:
            print(f"    ❌ Failed to reach goal")
    
    success_rate = successes / total_tests
    print(f"\n=== Results ===")
    print(f"Success rate: {success_rate:.1%} ({successes}/{total_tests})")
    
    return success_rate

def test_environment_mechanics():
    """Test environment mechanics to understand constraints"""
    print("\n=== Testing Environment Mechanics ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
    
    # Test robot movement
    print("\nTesting robot movement:")
    
    # Robot forward from (1,1) -> should go to (2,1)
    actions = {"robot_0": 2, "human_0": 0}
    env.step(actions)
    print(f"Robot forward: {env.agent_positions['robot_0']} (expected: (2,1))")
    
    # Robot right from (2,1) -> should face east
    actions = {"robot_0": 1, "human_0": 0}  
    env.step(actions)
    print(f"Robot turn right: {env.agent_positions['robot_0']}")
    
    # Robot forward -> should go to (2,2) where key is
    actions = {"robot_0": 2, "human_0": 0}
    env.step(actions)
    print(f"Robot forward to key: {env.agent_positions['robot_0']} (expected: (2,2))")
    
    # Robot pickup key
    actions = {"robot_0": 3, "human_0": 0}
    result = env.step(actions)
    print(f"Robot pickup key: {env.agent_positions['robot_0']}")
    
    # Test human movement
    print(f"\nTesting human movement from {env.agent_positions['human_0']}:")
    
    # Human forward from (1,3) -> should go to (2,3) but door might be closed
    actions = {"robot_0": 0, "human_0": 2}
    env.step(actions)
    print(f"Human forward: {env.agent_positions['human_0']} (testing if door blocks)")
    
    return env

if __name__ == "__main__":
    # First test environment mechanics
    env = test_environment_mechanics()
    
    # Then test hardcoded solution
    success_rate = test_hardcoded_solution()
    
    if success_rate >= 0.8:
        print("\n✅ Hardcoded solution works! Can use this to guide learning.")
    else:
        print("\n❌ Hardcoded solution failed. Need to understand environment better.")