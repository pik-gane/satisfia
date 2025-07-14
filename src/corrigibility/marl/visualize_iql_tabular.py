#!/usr/bin/env python3
"""
Visualization script for IQL tabular case to see the trained agents in action.
"""

import sys
import os


import numpy as np
import time
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def visualize_trained_agents():
    """Visualize the trained IQL agents in action"""
    print("=== IQL Tabular Visualization ===")
    
    # Get map and create environment with rendering
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode="human"  # Enable visual rendering
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
    
    print(f"Map: {map_metadata['name']}")
    print(f"Human goal: {human_goals[0]}")
    
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
    
    # Train the algorithm
    print("\nTraining Phase 1 (Robot blocking human)...")
    iql.train_phase1(env, episodes=100, max_steps=50)
    
    print("\nTraining Phase 2 (Robot assisting human)...")
    iql.train_phase2(env, episodes=200, max_steps=50)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE - Starting Visualization")
    print("="*50)
    print("Press any key to continue through episodes...")
    print("Close the pygame window to exit")
    
    # Run visualization episodes
    for episode in range(5):
        print(f"\n--- Episode {episode + 1} ---")
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        print(f"Goal: {goal}")
        print(f"Initial positions: {env.agent_positions}")
        print(f"Keys available: {[k['pos'] for k in env.keys]}")
        print(f"Doors: {[(d['pos'], d['is_locked'], d['color']) for d in env.doors]}")
        
        # Render initial state
        env.render()
        input("Press Enter to start episode...")
        
        for step in range(100):  # Max steps per episode
            actions = {}
            
            # Get actions from trained algorithm
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            # Convert action numbers to action names for display
            action_names = {
                0: "turn_left", 1: "turn_right", 2: "forward", 
                3: "pickup", 4: "drop", 5: "toggle"
            }
            
            print(f"\nStep {step + 1}:")
            print(f"  Actions: {[(agent, action_names.get(int(action), str(action))) for agent, action in actions.items()]}")
            print(f"  Positions: {env.agent_positions}")
            print(f"  Directions: {env.agent_dirs}")
            print(f"  Robot has keys: {env.robot_has_keys}")
            
            # Execute actions
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            # Render the environment
            env.render()
            
            print(f"  Rewards: {rewards}")
            print(f"  Keys remaining: {[k['pos'] for k in env.keys]}")
            print(f"  Doors: {[(d['pos'], d['is_locked'], d['is_open']) for d in env.doors]}")
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                print(f"  🎉 GOAL REACHED! Human reached {goal}")
                env.render()
                input("Goal reached! Press Enter to continue to next episode...")
                break
            
            # Check if episode ended
            if any(terms.values()) or any(truncs.values()):
                print(f"  Episode ended (terms: {terms}, truncs: {truncs})")
                break
            
            # Wait a bit for visualization
            time.sleep(0.5)
            
        else:
            print(f"  Episode timed out after {step + 1} steps")
            input("Episode timed out. Press Enter to continue...")
    
    print("\nVisualization complete!")
    env.close()


def quick_test_without_rendering():
    """Quick test without rendering to verify functionality"""
    print("=== Quick Test (No Rendering) ===")
    
    # Get map and create environment without rendering
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
        "robot_0": list(range(6)),
        "human_0": list(range(3))
    }
    
    # Get possible human goals
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    # Create and train IQL algorithm
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.2, alpha_e=0.2, alpha_r=0.2, alpha_p=0.2,
        gamma_h=0.9, gamma_r=0.9, beta_r_0=5.0,
        G=G, mu_g=mu_g, action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids, human_agent_ids=human_agent_ids,
        network=False, env=env
    )
    
    print("Training...")
    iql.train(env, phase1_episodes=50, phase2_episodes=100)
    
    # Test goal reaching
    goal_reached = 0
    test_episodes = 10
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(50):
            actions = {}
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            env.step(actions)
            
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                goal_reached += 1
                break
    
    success_rate = goal_reached / test_episodes
    print(f"Success rate: {success_rate:.1%}")
    return success_rate > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize IQL tabular case")
    parser.add_argument("--no-render", action="store_true", help="Run without rendering for quick test")
    args = parser.parse_args()
    
    if args.no_render:
        success = quick_test_without_rendering()
        print("✅ Test passed!" if success else "❌ Test failed!")
    else:
        try:
            visualize_trained_agents()
        except Exception as e:
            print(f"Visualization failed (probably no display): {e}")
            print("Running quick test instead...")
            success = quick_test_without_rendering()
            print("✅ Test passed!" if success else "❌ Test failed!")