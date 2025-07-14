#!/usr/bin/env python3
"""
Debug the paper-based algorithm on a single scenario to verify correctness.
"""

import sys
import numpy as np
from corrected_iql_paper_algorithm import PaperBasedTwoPhaseIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment

def debug_paper_algorithm():
    """Debug the paper-based algorithm step by step"""
    print("=== Debug Paper-Based Algorithm ===")
    
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
    
    # Create paper-based IQL algorithm
    iql = PaperBasedTwoPhaseIQL(
        alpha_m=0.3,
        alpha_e=0.3,
        alpha_r=0.3,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_h=5.0,
        beta_r=2.0,
        nu_h=0.1,
        zeta=1.5,
        xi=1.0,
        eta=1.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        env=env
    )
    
    # Train with reduced episodes for debugging
    print("Training with paper-based algorithm...")
    iql.train(
        environment=env,
        phase1_episodes=200,
        phase2_episodes=200,
        render=False
    )
    
    # Debug the learned policies
    print(f"\nLearned Q-table sizes:")
    print(f"Q_m (human cautious): {len(iql.Q_m['human_0'])} states")
    print(f"Q_e (human effective): {len(iql.Q_e['human_0'])} states")
    print(f"Q_r (robot): {len(iql.Q_r['robot_0'])} states")
    
    # Show some sample values
    print(f"\nSample Q_m values:")
    for i, (state, q_vals) in enumerate(list(iql.Q_m['human_0'].items())[:3]):
        print(f"  State {i}: {q_vals}")
    
    print(f"\nSample Q_r values:")
    for i, (state, q_vals) in enumerate(list(iql.Q_r['robot_0'].items())[:3]):
        print(f"  State {i}: {q_vals}")
    
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
    goal = env.human_goals['human_0']
    
    for step in range(30):  # Max 30 steps
        actions = {}
        
        # Get actions from trained algorithm
        for hid in human_agent_ids:
            state_h = iql.get_state_tuple(env, hid, goal)
            if state_h in iql.pi_h[hid]:
                pi_h = iql.pi_h[hid][state_h]
            else:
                pi_h = np.ones(3) / 3
            action_h = iql.sample_action(hid, state_h, goal, epsilon=0.0)
            actions[hid] = action_h
        
        for rid in robot_agent_ids:
            state_r = iql.get_state_tuple(env, rid)
            if state_r in iql.pi_r[rid]:
                pi_r = iql.pi_r[rid][state_r]
            else:
                pi_r = np.ones(6) / 6
            action_r = iql.sample_action(rid, state_r, epsilon=0.0)
            actions[rid] = action_r
        
        print(f"\nStep {step}:")
        print(f"  Human π: {pi_h}, action: {action_h}")
        print(f"  Robot π: {pi_r}, action: {action_r}")
        
        # Execute actions
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has keys: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"  Rewards: {rewards}")
        
        # Check if human reached goal
        human_pos = env.agent_positions['human_0']
        if tuple(human_pos) == tuple(goal):
            print("  🎉 GOAL REACHED!")
            goal_reached = True
            break
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("  Episode ended")
            break
        
        # Check for progress (position changes)
        if step > 0:
            if tuple(prev_human_pos) == tuple(human_pos) and tuple(prev_robot_pos) == tuple(env.agent_positions['robot_0']):
                stuck_count += 1
                if stuck_count >= 5:
                    print("  ⚠️ Agents seem stuck - ending episode")
                    break
            else:
                stuck_count = 0
        else:
            stuck_count = 0
        
        prev_human_pos = tuple(human_pos)
        prev_robot_pos = tuple(env.agent_positions['robot_0'])
    
    if not goal_reached:
        print(f"\n❌ Goal NOT reached in 30 steps")
    
    return goal_reached

if __name__ == "__main__":
    success = debug_paper_algorithm()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILED'}")