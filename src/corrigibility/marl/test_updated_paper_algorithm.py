#!/usr/bin/env python3
"""
Test the updated paper-based algorithm on all simple maps
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from src.corrigibility.marl.corrected_iql_paper_algorithm import PaperBasedTwoPhaseIQL
from src.corrigibility.marl.env import CustomEnvironment
from src.corrigibility.marl.envs.simple_map import get_map as get_simple_map
from src.corrigibility.marl.envs.simple_map2 import get_map as get_simple_map2
from src.corrigibility.marl.envs.simple_map3 import get_map as get_simple_map3
from src.corrigibility.marl.envs.simple_map4 import get_map as get_simple_map4

def test_algorithm_on_map(map_name, get_map_func, num_test_episodes=20):
    """Test the updated algorithm on a specific map"""
    print(f"\n=== Testing {map_name} ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_map_func()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Environment setup
    robot_agent_ids = ['robot_0']
    human_agent_ids = ['human_0']
    action_space_dict = {
        'robot_0': list(range(6)),  # 6 actions for robot
        'human_0': list(range(3))   # 3 actions for human
    }
    
    # Get possible goals
    human_goals = list(map_metadata["human_goals"].values())
    goals = [np.array(goal) for goal in human_goals]
    goal_probs = [1.0 / len(goals)] * len(goals)
    
    # Initialize algorithm with updated parameters
    algorithm = PaperBasedTwoPhaseIQL(
        alpha_m=0.3,
        alpha_r=0.3,
        beta_h=1.0,      # Lower for more exploration
        beta_r=1.0,      # Lower for more exploration
        nu_h=0.3,        # Higher habitual mixing
        G=goals,
        mu_g=goal_probs,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        env=env
    )
    
    # Training
    print("Training Phase 1...")
    algorithm.train_phase1(env, episodes=200, max_steps=30)
    
    print("Training Phase 2...")
    algorithm.train_phase2(env, episodes=200, max_steps=30)
    
    # Testing
    print(f"Testing on {num_test_episodes} episodes...")
    successful_episodes = 0
    
    for episode in range(num_test_episodes):
        env.reset()
        goal = goals[0]
        
        for step in range(50):
            actions = {}
            
            # Get deterministic actions for testing
            for agent_id in robot_agent_ids + human_agent_ids:
                state = algorithm.get_state_tuple(env, agent_id, goal if agent_id in human_agent_ids else None)
                actions[agent_id] = algorithm.sample_action(agent_id, state, goal if agent_id in human_agent_ids else None, epsilon=0.0)
            
            # Execute actions
            obs, rewards, terms, truncs, _ = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            
            # Check if goal reached
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                successful_episodes += 1
                if episode < 3:
                    print(f"  Episode {episode}: Goal reached in {step + 1} steps!")
                break
            
            if done:
                break
    
    success_rate = successful_episodes / num_test_episodes
    print(f"Success rate: {success_rate:.1%} ({successful_episodes}/{num_test_episodes})")
    
    return success_rate

def main():
    """Test algorithm on all simple maps"""
    maps = [
        ('simple_map', get_simple_map),
        ('simple_map2', get_simple_map2),
        ('simple_map3', get_simple_map3),
        ('simple_map4', get_simple_map4)
    ]
    
    overall_success = True
    results = {}
    
    for map_name, get_map_func in maps:
        try:
            success_rate = test_algorithm_on_map(map_name, get_map_func)
            results[map_name] = success_rate
            
            if success_rate < 1.0:
                overall_success = False
                print(f"❌ {map_name}: {success_rate:.1%} (FAILED)")
            else:
                print(f"✅ {map_name}: {success_rate:.1%} (PASSED)")
                
        except Exception as e:
            print(f"❌ {map_name}: ERROR - {str(e)}")
            overall_success = False
            results[map_name] = 0.0
    
    print(f"\n=== FINAL RESULTS ===")
    for map_name, success_rate in results.items():
        status = "✅ PASS" if success_rate == 1.0 else "❌ FAIL"
        print(f"{map_name}: {success_rate:.1%} {status}")
    
    if overall_success:
        print("\n🎉 ALL TESTS PASSED! Algorithm achieves 100% success on all simple maps.")
    else:
        print("\n❌ Some tests failed. Algorithm needs further improvement.")
    
    return overall_success

if __name__ == "__main__":
    main()