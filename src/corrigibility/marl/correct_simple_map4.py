#!/usr/bin/env python3
"""
Correct approach for simple_map4 with proper directional navigation.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def understand_directions():
    """Understand the direction system"""
    print("=== Understanding Direction System ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial directions: {env.agent_dirs}")
    print(f"Robot dir={env.agent_dirs['robot_0']}, Human dir={env.agent_dirs['human_0']}")
    print(f"Direction meanings: 0=North, 1=East, 2=South, 3=West")
    
    print(f"\nRobot starts at {env.agent_positions['robot_0']} facing direction {env.agent_dirs['robot_0']} (South)")
    print(f"Human starts at {env.agent_positions['human_0']} facing direction {env.agent_dirs['human_0']} (East)")
    
    return env

def test_correct_robot_navigation():
    """Test correct robot navigation to key and door"""
    print(f"\n=== Correct Robot Navigation ===")
    
    env = understand_directions()
    
    print(f"\nGoal: Get robot from (1,1) to key at (2,2)")
    print(f"Robot currently faces South (dir=2), so forward goes to (2,1)")
    
    # Step 1: Robot forward to (2,1)
    actions = {"robot_0": 2, "human_0": 0}
    env.step(actions)
    print(f"Step 1 - Robot forward: {env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}")
    
    # Step 2: Robot turn right to face East (toward key)
    actions = {"robot_0": 1, "human_0": 0}
    env.step(actions)
    print(f"Step 2 - Robot turn right: {env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']} (should be 1=East)")
    
    # Step 3: Robot forward to (2,2) - key position
    actions = {"robot_0": 2, "human_0": 0}
    env.step(actions)
    print(f"Step 3 - Robot forward: {env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']} (should be at key)")
    
    if env.agent_positions['robot_0'] == (2, 2):
        print("✅ Robot reached key position!")
        
        # Step 4: Robot pickup key
        actions = {"robot_0": 3, "human_0": 0}
        env.step(actions)
        print(f"Step 4 - Robot pickup: {env.agent_positions['robot_0']}")
        
        # Step 5: Robot turn right to face South (toward door)
        actions = {"robot_0": 1, "human_0": 0}
        env.step(actions)
        print(f"Step 5 - Robot turn right: {env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']} (should be 2=South)")
        
        # Step 6: Robot forward to (2,3) - door position  
        actions = {"robot_0": 2, "human_0": 0}
        env.step(actions)
        print(f"Step 6 - Robot forward: {env.agent_positions['robot_0']} (should be at door)")
        
        if env.agent_positions['robot_0'] == (2, 3):
            print("✅ Robot reached door position!")
            
            # Step 7: Robot toggle door
            actions = {"robot_0": 5, "human_0": 0}
            env.step(actions)
            print(f"Step 7 - Robot toggle door: {env.agent_positions['robot_0']}")
            print("🔓 Door should now be open!")
            
            return env, True
        else:
            print("❌ Robot failed to reach door")
            return env, False
    else:
        print("❌ Robot failed to reach key")
        return env, False

def test_human_navigation(env):
    """Test human navigation to goal after door is opened"""
    print(f"\n=== Human Navigation to Goal ===")
    
    print(f"Human starts at {env.agent_positions['human_0']} facing {env.agent_dirs['human_0']} (East)")
    print(f"Goal is at (3,3)")
    print(f"Human needs to go: (1,3) -> (2,3) -> (3,3)")
    
    # Step 1: Human forward to (2,3) - through opened door
    actions = {"robot_0": 0, "human_0": 2}
    env.step(actions)
    print(f"Step 1 - Human forward: {env.agent_positions['human_0']} (should be (2,3))")
    
    # Step 2: Human forward to (3,3) - goal
    actions = {"robot_0": 0, "human_0": 2}
    env.step(actions)
    print(f"Step 2 - Human forward: {env.agent_positions['human_0']} (should be (3,3) - GOAL)")
    
    if env.agent_positions['human_0'] == (3, 3):
        print("🎉 SUCCESS! Human reached goal!")
        return True
    else:
        print("❌ Human failed to reach goal")
        return False

def run_complete_solution():
    """Run the complete solution"""
    print(f"\n=== Complete Solution Test ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    successes = 0
    total_tests = 5
    
    for test in range(total_tests):
        print(f"\n--- Test {test + 1} ---")
        env.reset()
        
        # Complete sequence
        sequence = [
            # Robot to key
            {"robot_0": 2, "human_0": 0},  # Robot forward to (2,1)
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face East
            {"robot_0": 2, "human_0": 0},  # Robot forward to (2,2) - key
            {"robot_0": 3, "human_0": 0},  # Robot pickup key
            # Robot to door
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face South  
            {"robot_0": 2, "human_0": 0},  # Robot forward to (2,3) - door
            {"robot_0": 5, "human_0": 0},  # Robot toggle door
            # Human to goal
            {"robot_0": 0, "human_0": 2},  # Human forward to (2,3)
            {"robot_0": 0, "human_0": 2},  # Human forward to (3,3) - goal
        ]
        
        success = False
        for step, actions in enumerate(sequence):
            env.step(actions)
            
            # Check for success after each step
            if env.agent_positions['human_0'] == (3, 3):
                print(f"  🎉 SUCCESS at step {step + 1}!")
                success = True
                successes += 1
                break
        
        if not success:
            print(f"  ❌ Failed - final human position: {env.agent_positions['human_0']}")
    
    success_rate = successes / total_tests
    print(f"\n=== FINAL RESULTS ===")
    print(f"Success rate: {success_rate:.1%} ({successes}/{total_tests})")
    
    return success_rate

if __name__ == "__main__":
    # Test navigation step by step
    env, robot_success = test_correct_robot_navigation()
    
    if robot_success:
        human_success = test_human_navigation(env)
        
        if human_success:
            print(f"\n✅ Manual solution works!")
            # Test multiple times
            final_rate = run_complete_solution()
            
            if final_rate >= 0.8:
                print(f"\n🎉 SOLUTION FOUND! {final_rate:.1%} success rate")
                print("Ready to implement this in the learning algorithm!")
            else:
                print(f"\n⚠️ Inconsistent results: {final_rate:.1%}")
        else:
            print(f"\n❌ Human navigation failed")
    else:
        print(f"\n❌ Robot navigation failed")