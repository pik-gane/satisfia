#!/usr/bin/env python3
"""
Quick test script to validate IQL tabular case functionality.
"""

import sys
import os


import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def test_quick():
    """Quick test of the IQL algorithm"""
    print("=== Quick IQL Test ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
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
    mu_g = [1.0 / len(G)] * len(G)
    
    print(f"Human goals: {human_goals}")
    
    # Create IQL algorithm instance (tabular case)
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.2,
        alpha_e=0.2,
        alpha_r=0.2,
        alpha_p=0.2,
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
    
    # Train the algorithm with fewer episodes
    print("Training...")
    iql.train(
        environment=env,
        phase1_episodes=50,
        phase2_episodes=100,
        render=False
    )
    
    # Test if human reaches goal
    print("Testing goal reaching...")
    goal_reached_count = 0
    test_episodes = 10
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        print(f"\nEpisode {episode + 1}: Goal at {goal}")
        print(f"Initial positions: {env.agent_positions}")
        
        for step in range(50):  # Reduced max steps
            actions = {}
            
            # Get actions from trained algorithm
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            print(f"  Step {step + 1}: actions={actions}, pos={env.agent_positions}")
            
            # Execute actions
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                goal_reached_count += 1
                print(f"  GOAL REACHED at step {step + 1}!")
                break
            
            # Check if episode ended
            if any(terms.values()) or any(truncs.values()):
                print(f"  Episode ended at step {step + 1}")
                break
        else:
            print(f"  Episode timed out")
    
    goal_success_rate = goal_reached_count / test_episodes
    print(f"\nGoal success rate: {goal_success_rate:.2%}")
    
    if goal_success_rate > 0:
        print("✅ SUCCESS: Human can reach goal with trained robot!")
    else:
        print("❌ FAILED: Human unable to reach goal")
    
    return goal_success_rate > 0


if __name__ == "__main__":
    success = test_quick()
    sys.exit(0 if success else 1)