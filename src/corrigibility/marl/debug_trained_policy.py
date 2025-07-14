#!/usr/bin/env python3
"""
Debug the trained policy to understand why it's not working.
"""

import sys
import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_trained_policy():
    """Debug the trained policy step by step"""
    print("=== Debug Trained Policy ===")
    
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
    mu_g = [1.0 / len(G)] * len(G)  # Uniform distribution over goals
    
    # Create IQL algorithm instance (tabular case)
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.3,  # Higher learning rate
        alpha_e=0.3,
        alpha_r=0.3,
        alpha_p=0.3,
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
    print("Training...")
    iql.train(
        environment=env,
        phase1_episodes=200,  # Increased for better learning
        phase2_episodes=200,
        render=False
    )
    
    # Debug the Q-tables
    print(f"\nQ-table sizes:")
    print(f"Q_m (human cautious): {len(iql.q_m['human_0'])} states")
    print(f"Q_e (human effective): {len(iql.q_e['human_0'])} states")
    print(f"Q_r (robot): {len(iql.q_r['robot_0'])} states")
    
    # Show some Q-values
    print(f"\nSample Q_m values for human_0:")
    human_states = list(iql.q_m['human_0'].keys())[:5]
    for state in human_states:
        q_vals = iql.q_m['human_0'][state]
        print(f"  State {state}: {q_vals}")
    
    print(f"\nSample Q_r values for robot_0:")
    robot_states = list(iql.q_r['robot_0'].keys())[:5]
    for state in robot_states:
        q_vals = iql.q_r['robot_0'][state]
        print(f"  State {state}: {q_vals}")
    
    # Test the trained policy step by step
    print(f"\n=== Testing Trained Policy ===")
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    # Print the grid
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
    
    goal_reached = False
    prev_human_pos = None
    prev_robot_pos = None
    for step in range(20):  # Max 20 steps
        goal = env.human_goals['human_0']
        actions = {}
        
        # Human action
        state_h = iql.get_human_state(env, 'human_0', iql.state_to_tuple(goal))
        q_vals_h = iql.q_m['human_0'][state_h]
        action_h = iql.sample_human_action_phase1('human_0', state_h, epsilon=0.0)
        actions['human_0'] = action_h
        
        # Robot action  
        state_r = iql.get_full_state(env, 'robot_0')
        q_vals_r = iql.q_r['robot_0'][state_r]
        action_r = iql.sample_robot_action_phase2('robot_0', state_r, env, epsilon=0.0)
        actions['robot_0'] = action_r
        
        print(f"\nStep {step}:")
        print(f"  Human state: {state_h}")
        print(f"  Human Q-values: {q_vals_h}")
        print(f"  Human action: {action_h}")
        print(f"  Robot state: {state_r}")
        print(f"  Robot Q-values: {q_vals_r}")
        print(f"  Robot action: {action_r}")
        
        # Execute actions
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has keys: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Rewards: {rewards}")
        
        # Check if human reached goal
        human_pos = env.agent_positions['human_0']
        if tuple(human_pos) == tuple(goal):
            print("  GOAL REACHED!")
            goal_reached = True
            break
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("  Episode ended")
            break
        
        # Check for stuck behavior (no movement)
        if step > 0 and prev_human_pos == tuple(human_pos) and prev_robot_pos == tuple(env.agent_positions['robot_0']):
            print("  WARNING: Agents seem stuck!")
        
        prev_human_pos = tuple(human_pos)
        prev_robot_pos = tuple(env.agent_positions['robot_0'])
    
    if not goal_reached:
        print(f"\nGoal NOT reached in 20 steps")
    
    return goal_reached

if __name__ == "__main__":
    success = debug_trained_policy()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")