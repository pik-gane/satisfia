#!/usr/bin/env python3
"""
Test the complete cooperation sequence that should achieve 100% success.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def test_complete_cooperation():
    """Test the complete cooperation sequence"""
    print("=== Complete Cooperation Test ===")
    
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
    print(f"  Goal: {env.human_goals['human_0']}")
    print(f"  Keys: {[k['pos'] for k in env.keys]}")
    print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
    
    # Working sequence
    steps = [
        # Phase 1: Robot gets key
        ("Robot turn left to face key", {"human_0": 0, "robot_0": 0}),
        ("Robot pickup key", {"human_0": 0, "robot_0": 3}),
        ("Robot move to key position", {"human_0": 0, "robot_0": 2}),
        
        # Phase 2: Robot opens door from adjacent position
        ("Robot turn right to face door", {"human_0": 0, "robot_0": 1}),
        ("Robot open door from adjacent position", {"human_0": 0, "robot_0": 5}),
        
        # Phase 3: Human goes through opened door to goal
        ("Human move down", {"human_0": 2, "robot_0": 0}),
        ("Human move down through door", {"human_0": 2, "robot_0": 0}),
        ("Human turn right to face goal", {"human_0": 1, "robot_0": 0}),
        ("Human move right to goal", {"human_0": 2, "robot_0": 0}),
    ]
    
    for step_num, (description, actions) in enumerate(steps):
        print(f"\nStep {step_num + 1}: {description}")
        print(f"  Actions: {actions}")
        print(f"  Before: Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}, Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After: Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}, Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Rewards: {rewards}")
        
        # Check if goal reached
        goal = env.human_goals['human_0']
        if tuple(env.agent_positions['human_0']) == tuple(goal):
            print(f"  🎉 GOAL REACHED!")
            return True
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print(f"  Episode ended")
            break
    
    return False

def test_multiple_trials():
    """Test multiple trials to verify consistency"""
    print("\n=== Multiple Trials Test ===")
    
    success_count = 0
    num_trials = 5
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        success = test_complete_cooperation()
        if success:
            success_count += 1
            print(f"Trial {trial + 1}: SUCCESS")
        else:
            print(f"Trial {trial + 1}: FAILED")
    
    success_rate = success_count / num_trials
    print(f"\nOverall Results:")
    print(f"  Successful trials: {success_count}/{num_trials}")
    print(f"  Success rate: {success_rate:.1%}")
    
    return success_rate

if __name__ == "__main__":
    # Test single trial first
    single_success = test_complete_cooperation()
    
    if single_success:
        # Test multiple trials
        overall_success_rate = test_multiple_trials()
        print(f"\nFinal Assessment:")
        if overall_success_rate >= 1.0:
            print(f"✅ PERFECT: 100% success rate achieved!")
        elif overall_success_rate >= 0.8:
            print(f"🟡 GOOD: {overall_success_rate:.1%} success rate")
        else:
            print(f"❌ POOR: {overall_success_rate:.1%} success rate")
    else:
        print(f"\nSingle trial failed - debugging needed")