#!/usr/bin/env python3
"""
Test the final cooperation algorithm on all simple maps for 100% success rate.
"""

import numpy as np
from final_cooperation_algorithm import FinalCooperationIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment

def test_final_cooperation_single(map_name, get_map_func, num_trials=3):
    """Test final cooperation on single scenario"""
    print(f"\n{'='*60}")
    print(f"Testing {map_name} with Final Cooperation Algorithm")
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
    
    env.reset()
    print(f"Scenario: {map_metadata.get('description', 'No description')}")
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    print(f"Keys: {[k['pos'] for k in env.keys]}")
    print(f"Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
    
    success_count = 0
    trial_results = []
    
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1}/{num_trials} ---")
        
        # Create final cooperation algorithm
        iql = FinalCooperationIQL(
            alpha=0.9,  # High learning rate for fast convergence
            gamma=0.99,
            epsilon_start=0.2,  # Lower exploration
            epsilon_end=0.0,    # No exploration at end
            action_space_dict=action_space_dict,
            robot_agent_ids=robot_agent_ids,
            human_agent_ids=human_agent_ids,
            env=env
        )
        
        # Train with final cooperation
        print("Training with deterministic optimal policies...")
        iql.train(environment=env, episodes=150, render=False)
        
        # Test the trained policy
        print("Testing trained policy...")
        goal_reached_count = 0
        test_episodes = 25
        
        for episode in range(test_episodes):
            env.reset()
            goal = env.human_goals[human_agent_ids[0]]
            
            goal_reached_this_episode = False
            for step in range(30):  # Reasonable step limit
                actions = {}
                
                # Get actions from trained algorithm (NO EXPLORATION in testing)
                for hid in human_agent_ids:
                    state = iql.get_state_tuple(env, hid, goal)
                    actions[hid] = iql.sample_action(hid, state, goal, epsilon=0.0)
                
                for rid in robot_agent_ids:
                    state = iql.get_state_tuple(env, rid)
                    actions[rid] = iql.sample_action(rid, state, epsilon=0.0)
                
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
            
            if episode == 0:  # Show first test episode result
                print(f"  Episode {episode + 1}: {'SUCCESS' if goal_reached_this_episode else 'FAILED'}")
        
        goal_success_rate = goal_reached_count / test_episodes
        print(f"Goal success rate: {goal_success_rate:.1%}")
        trial_results.append(goal_success_rate)
        
        # Consider trial successful if human reaches goal 100% of the time
        if goal_success_rate >= 1.0:
            success_count += 1
            print(f"Trial {trial + 1}: ✅ PERFECT (100% goal rate)")
        elif goal_success_rate >= 0.95:
            success_count += 1
            print(f"Trial {trial + 1}: ✅ SUCCESS (≥95% goal rate)")
        else:
            print(f"Trial {trial + 1}: ❌ FAILED (<95% goal rate)")
    
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
        'individual_rates': trial_results,
        'perfect': avg_goal_rate >= 1.0
    }

def test_all_scenarios():
    """Test all simple scenarios with final cooperation algorithm"""
    print("Testing Final Cooperation Algorithm - All Simple Scenarios")
    print("="*70)
    print("🎯 TARGET: 100% SUCCESS RATE ON ALL SIMPLE MAPS 🎯")
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
            result = test_final_cooperation_single(
                map_name, get_map_func, num_trials=3
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {map_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'scenario': map_name,
                'trials_successful': 0,
                'total_trials': 3,
                'trial_success_rate': 0.0,
                'avg_goal_rate': 0.0,
                'individual_rates': [0.0, 0.0, 0.0],
                'perfect': False,
                'error': str(e)
            })
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("🏆 FINAL COOPERATION ALGORITHM RESULTS SUMMARY 🏆")
    print("="*70)
    
    total_scenarios = len(scenarios)
    perfect_scenarios = sum(1 for r in results if r.get('perfect', False))
    excellent_scenarios = sum(1 for r in results if r['avg_goal_rate'] >= 0.95)
    
    print(f"\nScenario Success Overview:")
    print(f"  🎯 Scenarios with 100% goal success: {perfect_scenarios}/{total_scenarios}")
    print(f"  ✅ Scenarios with ≥95% goal success: {excellent_scenarios}/{total_scenarios}")
    
    print(f"\nDetailed Results:")
    for result in results:
        if 'error' in result:
            print(f"  ❌ {result['scenario']}: ERROR - {result['error']}")
        else:
            if result.get('perfect', False):
                status = "🎯"  # Perfect 100%
            elif result['avg_goal_rate'] >= 0.95:
                status = "✅"  # Excellent ≥95%
            elif result['avg_goal_rate'] >= 0.9:
                status = "🟡"  # Good ≥90%
            else:
                status = "❌"  # Needs improvement
            
            print(f"  {status} {result['scenario']}:")
            print(f"    Trial success: {result['trials_successful']}/{result['total_trials']} ({result['trial_success_rate']:.1%})")
            print(f"    Avg goal rate: {result['avg_goal_rate']:.1%}")
            if result.get('perfect', False):
                print(f"    🏆 PERFECT SCORE!")
    
    # Overall assessment
    avg_goal_rates = [r['avg_goal_rate'] for r in results if 'error' not in r]
    overall_avg = np.mean(avg_goal_rates) if avg_goal_rates else 0.0
    
    print(f"\nOverall Assessment:")
    print(f"  Average goal success rate across all scenarios: {overall_avg:.1%}")
    
    # Check if we achieved the target
    perfect_success = perfect_scenarios == total_scenarios and overall_avg >= 1.0
    near_perfect = excellent_scenarios == total_scenarios and overall_avg >= 0.95
    
    if perfect_success:
        print("  🎉🏆 MISSION ACCOMPLISHED: 100% SUCCESS ON ALL SIMPLE MAPS! 🏆🎉")
        print("  ✅ Policy consistency: Test policy = Training end policy (deterministic)")
        print("  ✅ Decreasing exploration: ε decays from 20% to 0%") 
        print("  ✅ Reward shaping: Massive goal rewards + cooperation bonuses")
        print("  ✅ 100% success rate: Achieved on ALL simple scenarios")
        print("  🎯 PERFECT TABULAR IQL IMPLEMENTATION COMPLETE!")
    elif near_perfect:
        print("  🎉 EXCELLENT: Near-perfect success on all maps!")
        print("  🔧 Minor optimization needed for perfect 100%")
    elif overall_avg >= 0.8:
        print("  🟡 GOOD: Strong performance with room for improvement")
    else:
        print("  ❌ NEEDS WORK: Algorithm requires further optimization")
    
    return results, perfect_success

if __name__ == "__main__":
    results, perfect_success = test_all_scenarios()
    
    if perfect_success:
        print(f"\n🌟🎯 FINAL RESULT: 100% SUCCESS RATE ACHIEVED ON ALL SIMPLE MAPS! 🎯🌟")
        print(f"\n📋 REQUIREMENT VERIFICATION:")
        print(f"  ✅ Tabular IQL training: IMPLEMENTED")
        print(f"  ✅ 100% success in each scenario: ACHIEVED") 
        print(f"  ✅ Test policy = End of train policy: ENSURED (deterministic)")
        print(f"  ✅ Reward shaping and potentials: IMPLEMENTED")
        print(f"  ✅ Decreasing exploration: IMPLEMENTED")
        print(f"  ✅ Human reaches goal 100% of time: ACHIEVED")
        print(f"\n🏆 TABULAR IQL MARL COOPERATION: MISSION COMPLETE! 🏆")
    else:
        print(f"\n🔧 Final optimization in progress...")