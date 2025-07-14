#!/usr/bin/env python3
"""
Test accuracy on all simple map scenarios.
"""

import numpy as np
from fixed_enhanced_algorithm import FixedEnhancedTwoPhaseTimescaleIQL

def test_scenario(map_name, episodes=50):
    """Test accuracy on a specific scenario"""
    print(f"\n=== Testing {map_name} ===")
    
    try:
        # Import the specific map
        if map_name == "simple_map":
            from envs.simple_map import get_map
        elif map_name == "simple_map2": 
            from envs.simple_map2 import get_map
        elif map_name == "simple_map3":
            from envs.simple_map3 import get_map
        elif map_name == "simple_map4":
            from envs.simple_map4 import get_map
        else:
            print(f"Unknown map: {map_name}")
            return 0.0
        
        from env import CustomEnvironment
        
        map_layout, map_metadata = get_map()
        env = CustomEnvironment(
            grid_layout=map_layout,
            grid_metadata=map_metadata,
            render_mode=None
        )
        
        print(f"Map: {map_metadata['name']}")
        print(f"Goal: {map_metadata['human_goals']}")
        
        robot_agent_ids = ["robot_0"]
        human_agent_ids = ["human_0"]
        action_space_dict = {
            "robot_0": list(range(6)),
            "human_0": list(range(3))
        }
        
        human_goals = list(map_metadata["human_goals"].values())
        G = [np.array(goal) for goal in human_goals]
        mu_g = [1.0 / len(G)] * len(G)
        
        # Create algorithm instance
        algorithm = FixedEnhancedTwoPhaseTimescaleIQL(
            alpha_m=0.5,
            alpha_e=0.5,
            alpha_r=0.5,
            alpha_p=0.5,
            gamma_h=0.9,
            gamma_r=0.9,
            beta_r_0=3.0,
            G=G,
            mu_g=mu_g,
            action_space_dict=action_space_dict,
            robot_agent_ids=robot_agent_ids,
            human_agent_ids=human_agent_ids,
            network=False,
            env=env
        )
        
        # Quick training
        print(f"Quick training Phase 1 (200 episodes)...")
        algorithm.train_phase1_fixed(env, episodes=200, max_steps=50)
        
        print(f"Quick training Phase 2 (300 episodes)...")
        algorithm.train_phase2_enhanced(env, episodes=300, max_steps=50)
        
        # Test
        print(f"Testing {episodes} episodes...")
        success_rate, avg_steps = algorithm.test_policy(env, episodes=episodes, max_steps=50, verbose=False)
        
        print(f"✅ {map_name}: {success_rate:.1%} success rate")
        return success_rate
        
    except Exception as e:
        print(f"❌ Error testing {map_name}: {e}")
        return 0.0

def test_all_scenarios():
    """Test all simple map scenarios"""
    print("=== Testing All Simple Map Scenarios ===")
    
    scenarios = ["simple_map", "simple_map2", "simple_map3", "simple_map4"]
    results = {}
    
    for scenario in scenarios:
        results[scenario] = test_scenario(scenario, episodes=20)
    
    print(f"\n=== FINAL RESULTS SUMMARY ===")
    for scenario, accuracy in results.items():
        status = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.5 else "❌"
        print(f"{status} {scenario}: {accuracy:.1%}")
    
    overall_success = all(acc >= 0.8 for acc in results.values())
    avg_accuracy = np.mean(list(results.values()))
    
    print(f"\nOverall average: {avg_accuracy:.1%}")
    if overall_success:
        print("🎉 ALL SCENARIOS ACHIEVING ≥80% SUCCESS!")
    else:
        failing = [s for s, a in results.items() if a < 0.8]
        print(f"❌ Scenarios needing improvement: {failing}")
    
    return results

if __name__ == "__main__":
    results = test_all_scenarios()