#!/usr/bin/env python3
"""
Debug script to identify why IQL tabular case is failing.
"""

import sys
import os


import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def debug_simple_map():
    """Debug the simple map case"""
    print("=== Debugging Simple Map ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    print(f"Map layout: {map_layout}")
    print(f"Map metadata: {map_metadata}")
    
    # Initialize environment
    env.reset()
    print(f"Agent positions: {env.agent_positions}")
    print(f"Human goals: {env.human_goals}")
    print(f"Possible agents: {env.possible_agents}")
    
    # Setup agent IDs and action spaces
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(3)),  # [turn_left, turn_right, forward]
        "human_0": list(range(3))   # [turn_left, turn_right, forward]
    }
    
    # Get possible human goals
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    print(f"Human goals: {human_goals}")
    print(f"G: {G}")
    print(f"mu_g: {mu_g}")
    
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
    
    # Test state conversion
    print("\n=== Testing State Conversion ===")
    sample_obs = env.observe(human_agent_ids[0])
    print(f"Sample observation: {sample_obs}")
    state_tuple = iql.state_to_tuple(sample_obs)
    print(f"State tuple: {state_tuple}")
    full_state = iql.get_full_state(env, human_agent_ids[0])
    print(f"Full state: {full_state}")
    
    # Test goal-conditioned state
    goal = iql.state_to_tuple(G[0])
    human_state = iql.get_human_state(env, human_agent_ids[0], goal)
    print(f"Human state with goal: {human_state}")
    
    # Test action sampling
    print("\n=== Testing Action Sampling ===")
    action = iql.sample_human_action_phase1(human_agent_ids[0], human_state)
    print(f"Sampled action: {action}")
    
    # Test a few training episodes with detailed logging
    print("\n=== Testing Training Episodes ===")
    for ep in range(3):
        print(f"\n--- Episode {ep + 1} ---")
        env.reset()
        
        # Sample initial goal for human
        goal_idx = np.random.choice(len(G), p=mu_g)
        current_goal = iql.state_to_tuple(G[goal_idx])
        print(f"Selected goal: {current_goal}")
        
        for step in range(10):  # Short episode for debugging
            print(f"Step {step + 1}:")
            print(f"  Agent positions: {env.agent_positions}")
            
            actions = {}
            
            # Robot action (Phase 1): Block the human
            human_id = human_agent_ids[0]
            robot_id = robot_agent_ids[0]
            human_pos = env.agent_positions[human_id]
            robot_pos = env.agent_positions[robot_id]
            
            print(f"  Human pos: {human_pos}, Robot pos: {robot_pos}")
            print(f"  Human dir: {env.agent_dirs[human_id]}, Robot dir: {env.agent_dirs[robot_id]}")
            
            # Use the movement action method
            actions[robot_id] = iql._get_movement_action(env, robot_id, human_id)
            
            # Human action
            state_h = iql.get_human_state(env, human_id, current_goal)
            actions[human_id] = iql.sample_human_action_phase1(human_id, state_h)
            
            print(f"  Actions: {actions}")
            
            # Execute actions
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            
            print(f"  Rewards: {rewards}")
            print(f"  Terms: {terms}")
            print(f"  Truncs: {truncs}")
            print(f"  Done: {done}")
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_id]
            if tuple(human_pos) == tuple(current_goal):
                print(f"  GOAL REACHED!")
                break
            
            if done:
                print(f"  Episode ended")
                break


if __name__ == "__main__":
    debug_simple_map()