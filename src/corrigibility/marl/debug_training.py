#!/usr/bin/env python3
"""
Debug script to understand training and environment dynamics.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corrigibility.marl.env import CustomEnvironment
from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL


def debug_environment():
    """Debug the environment to understand the layout and dynamics."""
    
    print("=== ENVIRONMENT DEBUG ===")
    
    # Create environment
    env = CustomEnvironment(map_name="simple_map")
    obs = env.reset()
    
    print(f"Map layout:")
    for i, row in enumerate(env.map_layout):
        print(f"  {i}: {' '.join(row)}")
    
    print(f"\nAgent positions: {env.agent_positions}")
    print(f"Human agent IDs: {env.human_agent_ids}")
    print(f"Robot agent IDs: {env.robot_agent_ids}")
    
    # Check human goals
    human_goals = env.get_all_possible_human_goals()
    print(f"Human goals: {human_goals}")
    
    # Check action spaces
    for agent_id in env.possible_agents:
        action_space = env.action_space(agent_id)
        print(f"{agent_id} action space: {action_space.n} actions")
    
    # Test basic movement
    print(f"\n=== TESTING BASIC MOVEMENT ===")
    initial_pos = env.agent_positions.copy()
    print(f"Initial positions: {initial_pos}")
    
    # Try moving human right (action 2 = forward, facing right)
    # First, let's see what actions are available
    from corrigibility.marl.env import Actions
    print(f"Available actions: {list(Actions)}")
    
    # Test different actions for human
    for action_val in range(7):  # Actions 0-6
        env.reset()
        initial_positions = env.agent_positions.copy()
        
        # Human action, robot no-op
        actions = {'human_0': action_val, 'robot_0': 6}  # 6 = no-op/done
        
        obs, rewards, terms, truncs, infos = env.step(actions)
        final_positions = env.agent_positions.copy()
        
        print(f"Action {action_val} ({Actions(action_val).name}): "
              f"human moved from {initial_positions['human_0']} to {final_positions['human_0']}")


def debug_training_simple():
    """Debug training with very simple setup."""
    
    print("\n=== TRAINING DEBUG ===")
    
    # Create environment
    env = CustomEnvironment(map_name="simple_map")
    
    # Create action space dictionary
    action_space_dict = {}
    for agent_id in env.possible_agents:
        action_space_dict[agent_id] = list(range(env.action_space(agent_id).n))
    
    # Simple goals
    human_goals_list = [(3, 2)]
    mu_g = np.ones(len(human_goals_list)) / len(human_goals_list)
    
    # Initialize IQL algorithm with higher learning rates
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.5,  # Higher learning rate
        alpha_e=0.5,  # Higher learning rate 
        alpha_r=0.1,  # Higher learning rate
        alpha_p=0.1,
        gamma_h=0.9,  # Lower discount for quicker learning
        gamma_r=0.9,
        beta_r_0=5.0,
        G=human_goals_list,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=env.robot_agent_ids,
        human_agent_ids=env.human_agent_ids,
        network=False,
    )
    
    print("Training with simple setup...")
    
    # Very focused Phase 1 training - just a few episodes to see if anything changes
    print("Phase 1 (short)...")
    iql.train_phase1(env, episodes=5, max_steps=20)
    
    print("Phase 2 (short)...")
    iql.train_phase2(env, episodes=5, max_steps=20)
    
    # Test immediately after training
    print("\n=== TESTING AFTER SIMPLE TRAINING ===")
    
    env.reset()
    human_id = env.human_agent_ids[0]
    robot_id = env.robot_agent_ids[0]
    goal = (3, 2)
    
    print(f"Initial positions: {env.agent_positions}")
    print(f"Target goal: {goal}")
    
    # Test human action selection
    state_h = iql.get_human_state(env, human_id, goal)
    print(f"Human state: {state_h}")
    
    # Check Q-values for human
    if state_h in iql.q_m[human_id]:
        q_values = iql.q_m[human_id][state_h]
        print(f"Human Q-values: {q_values}")
        print(f"Best action: {np.argmax(q_values)}")
    else:
        print("No Q-values found for this state")
    
    # Sample actions multiple times
    for i in range(5):
        action = iql.sample_human_action_phase1(human_id, state_h, epsilon=0.0)  # No exploration
        print(f"Human action sample {i+1}: {action}")


if __name__ == "__main__":
    debug_environment()
    debug_training_simple()