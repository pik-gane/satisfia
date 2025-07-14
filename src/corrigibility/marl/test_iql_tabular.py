#!/usr/bin/env python3
"""
Test script to validate IQL timescale algorithm works in tabular case
for all simple maps and ensures human consistently reaches goal.
"""

import sys
import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment


def test_tabular_case_on_map(map_name, get_map_func, num_trials=5):
    """Test tabular IQL on a specific map"""
    print(f"\n=== Testing {map_name} ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_map_func()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Setup agent IDs and action spaces
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),  # [turn_left, turn_right, forward, pickup, drop, toggle]
        "human_0": list(range(3))   # [turn_left, turn_right, forward]
    }
    
    # Get possible human goals
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)  # Uniform distribution over goals
    
    success_count = 0
    
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}")
        
        # Create IQL algorithm instance (tabular case)
        iql = TwoPhaseTimescaleIQL(
            alpha_m=0.1,
            alpha_e=0.1,
            alpha_r=0.1,
            alpha_p=0.1,
            gamma_h=0.9,
            gamma_r=0.9,
            beta_r_0=5.0,
            G=G,
            mu_g=mu_g,
            action_space_dict=action_space_dict,
            robot_agent_ids=robot_agent_ids,
            human_agent_ids=human_agent_ids,
            network=False,  # Use tabular case
            env=env
        )
        
        # Train the algorithm
        print(f"    Training...")
        iql.train(
            environment=env,
            phase1_episodes=1000,
            phase2_episodes=1000,
            render=False
        )
        
        # Test if human reaches goal consistently
        print(f"    Testing goal reaching...")
        goal_reached_count = 0
        test_episodes = 20
        
        for episode in range(test_episodes):
            env.reset()
            goal = env.human_goals[human_agent_ids[0]]
            
            for step in range(200):  # Max steps
                actions = {}
                
                # Get actions from trained algorithm - use same policy as end of training
                for hid in human_agent_ids:
                    state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                    # Use the same policy as used in training phase 2 (cautious model)
                    actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
                
                for rid in robot_agent_ids:
                    state_r = iql.get_full_state(env, rid)
                    # Use deterministic policy for testing (no exploration)
                    actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
                
                # Execute actions
                obs, rewards, terms, truncs, _ = env.step(actions)
                
                # Check if human reached goal
                human_pos = env.agent_positions[human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    goal_reached_count += 1
                    break
                
                # Check if episode ended
                if any(terms.values()) or any(truncs.values()):
                    break
        
        goal_success_rate = goal_reached_count / test_episodes
        print(f"    Goal success rate: {goal_success_rate:.2%}")
        
        # Consider trial successful if human reaches goal >80% of the time
        if goal_success_rate > 0.8:
            success_count += 1
            print(f"    Trial {trial + 1}: SUCCESS")
        else:
            print(f"    Trial {trial + 1}: FAILED")
    
    overall_success_rate = success_count / num_trials
    print(f"  Overall success rate: {overall_success_rate:.2%}")
    
    return overall_success_rate > 0.6  # At least 60% of trials should succeed


def main():
    """Run tests on all simple maps"""
    print("Testing IQL Timescale Algorithm - Tabular Case")
    print("=" * 50)
    
    maps_to_test = [
        ("Simple Map 1", get_simple_map),
        ("Simple Map 2", get_simple_map2),
        ("Simple Map 3", get_simple_map3),
        ("Simple Map 4", get_simple_map4),
    ]
    
    results = {}
    
    for map_name, get_map_func in maps_to_test:
        try:
            success = test_tabular_case_on_map(map_name, get_map_func, num_trials=3)
            results[map_name] = success
        except Exception as e:
            print(f"Error testing {map_name}: {e}")
            results[map_name] = False
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for map_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{map_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed! IQL tabular case works correctly.")
    else:
        print("\n❌ Some tests failed. Need to fix issues.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)