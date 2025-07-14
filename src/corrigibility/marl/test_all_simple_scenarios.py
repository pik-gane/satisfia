#!/usr/bin/env python3
"""
Test the fixed tabular IQL algorithm on all simple scenarios with visualization.
"""

import sys
import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment

def test_scenario(map_name, get_map_func, num_trials=3, visualize=True):
    """Test tabular IQL on a specific map scenario"""
    print(f"\n{'='*60}")
    print(f"Testing {map_name}")
    print(f"{'='*60}")
    
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
    
    # Print scenario info
    env.reset()
    print(f"Scenario: {map_metadata.get('description', 'No description')}")
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    if visualize:
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
    
    success_count = 0
    trial_results = []
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
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
        print(f"Training...")
        iql.train(
            environment=env,
            phase1_episodes=500,
            phase2_episodes=500,
            render=False
        )
        
        # Test if human reaches goal consistently
        print(f"Testing goal reaching...")
        goal_reached_count = 0
        test_episodes = 10
        
        for episode in range(test_episodes):
            env.reset()
            goal = env.human_goals[human_agent_ids[0]]
            
            goal_reached_this_episode = False
            for step in range(100):  # Max steps
                actions = {}
                
                # Get actions from trained algorithm - use same policy as end of training
                for hid in human_agent_ids:
                    state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                    actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
                
                for rid in robot_agent_ids:
                    state_r = iql.get_full_state(env, rid)
                    actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
                
                # Execute actions
                obs, rewards, terms, truncs, _ = env.step(actions)
                
                # Check if human reached goal
                human_pos = env.agent_positions[human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    goal_reached_count += 1
                    goal_reached_this_episode = True
                    break
                
                # Check if episode ended
                if any(terms.values()) or any(truncs.values()):
                    break
            
            if visualize and episode == 0:  # Show first test episode
                print(f"  Episode {episode + 1}: {'SUCCESS' if goal_reached_this_episode else 'FAILED'}")
        
        goal_success_rate = goal_reached_count / test_episodes
        print(f"Goal success rate: {goal_success_rate:.1%}")
        trial_results.append(goal_success_rate)
        
        # Consider trial successful if human reaches goal >80% of the time
        if goal_success_rate >= 0.8:
            success_count += 1
            print(f"Trial {trial + 1}: SUCCESS (≥80% goal rate)")
        else:
            print(f"Trial {trial + 1}: FAILED (<80% goal rate)")
    
    overall_success_rate = success_count / num_trials
    avg_goal_rate = np.mean(trial_results)
    
    print(f"\n{map_name} Results:")
    print(f"  Trials successful: {success_count}/{num_trials} ({overall_success_rate:.1%})")
    print(f"  Average goal success rate: {avg_goal_rate:.1%}")
    print(f"  Individual trial rates: {[f'{r:.1%}' for r in trial_results]}")
    
    return {
        'scenario': map_name,
        'trials_successful': success_count,
        'total_trials': num_trials,
        'trial_success_rate': overall_success_rate,
        'avg_goal_rate': avg_goal_rate,
        'individual_rates': trial_results
    }

def main():
    """Run tests on all simple maps"""
    print("Testing Fixed Tabular IQL Algorithm - All Simple Scenarios")
    print("="*70)
    
    scenarios = [
        ("Simple Map 1", get_simple_map),
        ("Simple Map 2", get_simple_map2),
        ("Simple Map 3", get_simple_map3),
        ("Simple Map 4", get_simple_map4),
    ]
    
    results = []
    
    for map_name, get_map_func in scenarios:
        try:
            result = test_scenario(map_name, get_map_func, num_trials=2, visualize=True)
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {map_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scenario': map_name,
                'trials_successful': 0,
                'total_trials': 2,
                'trial_success_rate': 0.0,
                'avg_goal_rate': 0.0,
                'individual_rates': [0.0, 0.0],
                'error': str(e)
            })
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    total_scenarios = len(scenarios)
    successful_scenarios = sum(1 for r in results if r['trial_success_rate'] >= 0.5)
    
    print(f"\nScenario Success Overview:")
    print(f"  Scenarios with ≥50% trial success: {successful_scenarios}/{total_scenarios}")
    
    print(f"\nDetailed Results:")
    for result in results:
        if 'error' in result:
            print(f"  {result['scenario']}: ERROR - {result['error']}")
        else:
            print(f"  {result['scenario']}:")
            print(f"    Trial success: {result['trials_successful']}/{result['total_trials']} ({result['trial_success_rate']:.1%})")
            print(f"    Avg goal rate: {result['avg_goal_rate']:.1%}")
            print(f"    Individual: {result['individual_rates']}")
    
    # Overall assessment
    avg_goal_rates = [r['avg_goal_rate'] for r in results if 'error' not in r]
    overall_avg = np.mean(avg_goal_rates) if avg_goal_rates else 0.0
    
    print(f"\nOverall Assessment:")
    print(f"  Average goal success rate across all scenarios: {overall_avg:.1%}")
    
    if overall_avg >= 0.8:
        print("  ✅ EXCELLENT: Algorithm achieves high success rates!")
    elif overall_avg >= 0.5:
        print("  ⚠️  MODERATE: Algorithm shows some success but needs improvement")
    else:
        print("  ❌ POOR: Algorithm needs significant fixes")
    
    return results

if __name__ == "__main__":
    results = main()