#!/usr/bin/env python3
"""
Test script to verify that trained tabular Q-learning agents can reach goals using simple_map2.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corrigibility.marl.env import CustomEnvironment
from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL


def test_goal_reaching_with_working_map(model_path="trained_tabular_model.pkl", num_episodes=10, max_steps=200):
    """Test if the trained agent can reach goals using simple_map2 which allows movement."""
    
    # Create environment with working map
    env = CustomEnvironment(map_name="simple_map2")
    
    # Create action space dictionary
    action_space_dict = {}
    for agent_id in env.possible_agents:
        action_space_dict[agent_id] = list(range(env.action_space(agent_id).n))
    
    # Get human goals and create goal distribution
    env.reset()
    human_goals_dict = env.get_all_possible_human_goals()
    
    # Extract goal positions as a list
    human_goals_list = []
    for human_id in env.human_agent_ids:
        if human_id in human_goals_dict:
            human_goals_list.append(human_goals_dict[human_id])
    
    if not human_goals_list:
        human_goals_list = [(3, 3)]  # Default goal position for simple_map2
    
    # Create uniform distribution over goals
    mu_g = np.ones(len(human_goals_list)) / len(human_goals_list)
    
    print(f"Environment: simple_map2")
    print(f"Available goals: {human_goals_list}")
    
    # Show map layout
    env.reset()
    print(f"\nMap layout:")
    for i, row in enumerate(env.map_layout):
        print(f"  {i}: {' '.join(row)}")
    
    print(f"\nAgent initial positions: {env.agent_positions}")
    
    # Initialize IQL algorithm
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.1,
        alpha_r=0.01,
        alpha_p=0.1,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=human_goals_list,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=env.robot_agent_ids,
        human_agent_ids=env.human_agent_ids,
        network=False,
    )
    
    # Try to load trained model, if it fails, train a quick one
    try:
        print(f"\nTrying to load trained model from {model_path}")
        iql.load_models(model_path)
        print("✅ Loaded existing trained model")
    except:
        print("❌ No trained model found, training a quick one...")
        print("Phase 1 training...")
        iql.train_phase1(env, episodes=100, max_steps=50)
        print("Phase 2 training...")  
        iql.train_phase2(env, episodes=100, max_steps=50)
        print("✅ Quick training completed")
    
    # Test goal reaching
    success_count = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        
        # Set goals for humans
        current_goals = {}
        for hid in env.human_agent_ids:
            goal_idx = np.random.choice(len(iql.G), p=iql.mu_g)
            current_goals[hid] = iql.state_to_tuple(iql.G[goal_idx])
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Human goals: {current_goals}")
        
        # Check initial positions
        initial_positions = {}
        for agent_id in env.possible_agents:
            initial_positions[agent_id] = env.agent_positions[agent_id]
        print(f"  Initial positions: {initial_positions}")
        
        while not done and step_count < max_steps:
            actions = {}
            
            # Get actions from the trained IQL agent
            for agent_id, agent_obs in obs.items():
                if agent_id in iql.robot_agent_ids:
                    state_r = iql.get_full_state(env, agent_id)
                    actions[agent_id] = iql.sample_robot_action_phase2(agent_id, state_r)
                elif agent_id in iql.human_agent_ids:
                    goal = current_goals[agent_id]
                    state_h = iql.get_human_state(env, agent_id, goal)
                    actions[agent_id] = iql.sample_human_action_phase1(agent_id, state_h)
            
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if any human reached their goal
            goal_reached = False
            for hid in env.human_agent_ids:
                current_pos = env.agent_positions[hid]
                target_goal = current_goals[hid]
                if current_pos == target_goal:
                    print(f"  🎯 GOAL REACHED! {hid} reached goal {target_goal} at step {step_count + 1}")
                    goal_reached = True
                    success_count += 1
                    break
            
            done = any(terminations.values()) or any(truncations.values()) or goal_reached
            step_count += 1
            
            # Print progress every 20 steps
            if step_count % 20 == 0:
                current_positions = {aid: env.agent_positions[aid] for aid in env.possible_agents}
                print(f"  Step {step_count}: {current_positions}")
        
        total_steps += step_count
        
        # Final positions
        final_positions = {}
        for agent_id in env.possible_agents:
            final_positions[agent_id] = env.agent_positions[agent_id]
        
        print(f"  Final positions: {final_positions}")
        print(f"  Episode completed in {step_count} steps")
        
        # Check if goal was reached
        goal_reached_final = False
        for hid in env.human_agent_ids:
            if final_positions[hid] == current_goals[hid]:
                goal_reached_final = True
                if step_count == max_steps:  # Didn't count earlier
                    success_count += 1
                break
        
        if goal_reached_final:
            print(f"  ✅ Success!")
        else:
            print(f"  ❌ Goal not reached")
    
    # Summary
    success_rate = success_count / num_episodes
    avg_steps = total_steps / num_episodes
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"Episodes: {num_episodes}")
    print(f"Successes: {success_count}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average steps per episode: {avg_steps:.1f}")
    print(f"="*50)
    
    return success_rate, avg_steps


if __name__ == "__main__":
    success_rate, avg_steps = test_goal_reaching_with_working_map()
    
    # Verify reasonable performance
    if success_rate >= 0.3:  # At least 30% success rate for this test
        print("✅ PASS: Trained agent demonstrates goal-reaching behavior")
    else:
        print("❌ FAIL: Trained agent does not reliably reach goals")
    
    if avg_steps <= 150:  # Reasonable efficiency
        print("✅ PASS: Agent reaches goals efficiently")
    else:
        print("⚠️  WARNING: Agent takes many steps to reach goals")