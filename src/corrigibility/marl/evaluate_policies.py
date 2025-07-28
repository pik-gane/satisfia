#!/usr/bin/env python3
"""
Simple evaluation script to test if humans consistently reach their goals.
"""

import sys
import numpy as np
from env import GridEnvironment
from trained_agent import TrainedAgent
from envs.map_loader import load_map

def evaluate_policies(q_values_path, map_name, num_trials=20, max_steps=200):
    """
    Evaluate the trained policies by running multiple episodes
    and checking if humans reach their goals.
    """
    print(f"Evaluating policies from {q_values_path} on {map_name}")
    print(f"Running {num_trials} trials with max {max_steps} steps each")
    
    # Load the environment and trained agent
    try:
        env = GridEnvironment(map_name=map_name)
        env.max_steps = max_steps
    except Exception as e:
        print(f"Error loading map {map_name}: {e}")
        return False
    
    try:
        trained_agent = TrainedAgent(q_values_path)
    except Exception as e:
        print(f"Error loading trained agent from {q_values_path}: {e}")
        return False
    
    success_counts = []
    human_goal_reached_counts = []
    
    for trial in range(num_trials):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        episode_done = False
        
        # Track goal reached status for each human
        humans_reached_goals = {agent_id: False for agent_id in env.human_agent_ids}
        
        while not episode_done and step_count < max_steps:
            # Get actions from the trained agent
            actions = {}
            for agent_id, agent_obs in obs.items():
                action = trained_agent.choose_action(agent_obs, agent_id)
                actions[agent_id] = action
            
            # Take a step
            obs, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())
            step_count += 1
            episode_done = any(terminations.values()) or any(truncations.values())
            
            # Check if any human reached their goal this step
            for agent_id in humans_reached_goals:
                if agent_id in infos and infos[agent_id].get('goal_reached', False):
                    humans_reached_goals[agent_id] = True
        
        # Count how many humans reached their goals
        humans_that_reached_goals = sum(humans_reached_goals.values())
        total_humans = len(humans_reached_goals)
        
        success_counts.append(humans_that_reached_goals)
        human_goal_reached_counts.append(humans_reached_goals)
        
        print(f"Trial {trial+1:2d}: {humans_that_reached_goals}/{total_humans} humans reached goals, "
              f"steps: {step_count:3d}, total_reward: {total_reward:7.2f}")
    
    # Calculate statistics
    success_rate = np.mean(success_counts) / total_humans if total_humans > 0 else 0
    all_humans_success_rate = np.mean([count == total_humans for count in success_counts])
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS:")
    print(f"{'='*60}")
    print(f"Average humans reaching goals: {np.mean(success_counts):.2f}/{total_humans} ({success_rate*100:.1f}%)")
    print(f"Trials where ALL humans reached goals: {all_humans_success_rate*100:.1f}%")
    print(f"Individual human success rates:")
    
    # Calculate per-human success rates
    if human_goal_reached_counts:
        human_ids = list(human_goal_reached_counts[0].keys())
        for human_id in human_ids:
            individual_success = np.mean([trial[human_id] for trial in human_goal_reached_counts])
            print(f"  {human_id}: {individual_success*100:.1f}%")
    
    # Return True if success rate is good enough (e.g., >80% of humans reach goals)
    return success_rate > 0.8

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_policies.py <q_values_path> <map_name> [num_trials]")
        sys.exit(1)
    
    q_values_path = sys.argv[1]
    map_name = sys.argv[2]
    num_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    success = evaluate_policies(q_values_path, map_name, num_trials)
    
    if success:
        print(f"\n✅ SUCCESS: Policies are working well!")
        sys.exit(0)
    else:
        print(f"\n❌ NEEDS IMPROVEMENT: Policies need more training or tuning.")
        sys.exit(1)
