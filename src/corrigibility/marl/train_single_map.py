#!/usr/bin/env python3
"""
Train IQL algorithm on a single map and save checkpoints.
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime

from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment


def get_map_by_name(map_name):
    """Get map function by name"""
    map_functions = {
        "simple_map": get_simple_map,
        "simple_map2": get_simple_map2,
        "simple_map3": get_simple_map3,
        "simple_map4": get_simple_map4,
    }
    
    if map_name not in map_functions:
        raise ValueError(f"Unknown map: {map_name}. Available: {list(map_functions.keys())}")
    
    return map_functions[map_name]


def train_single_map(map_name, phase1_episodes=500, phase2_episodes=500, save_dir="checkpoints"):
    """Train IQL on a single map and save checkpoint"""
    print(f"=== Training IQL on {map_name} ===")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get map and create environment
    get_map_func = get_map_by_name(map_name)
    map_layout, map_metadata = get_map_func()
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    print(f"Map: {map_metadata['name']}")
    print(f"Size: {map_metadata['size']}")
    print(f"Human goals: {map_metadata['human_goals']}")
    
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
    
    print(f"Goals: {G}")
    print(f"Goal distribution: {mu_g}")
    
    # Create IQL algorithm instance (tabular case)
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.2,  # Higher learning rates for faster convergence
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
    
    # Training Phase 1
    print(f"\n--- Phase 1 Training ({phase1_episodes} episodes) ---")
    iql.train_phase1(env, episodes=phase1_episodes, max_steps=100)
    
    # Save Phase 1 checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    phase1_checkpoint = os.path.join(save_dir, f"{map_name}_phase1_{timestamp}.pkl")
    iql.save_models(phase1_checkpoint)
    print(f"Phase 1 checkpoint saved: {phase1_checkpoint}")
    
    # Training Phase 2
    print(f"\n--- Phase 2 Training ({phase2_episodes} episodes) ---")
    iql.train_phase2(env, episodes=phase2_episodes, max_steps=100)
    
    # Save final checkpoint
    final_checkpoint = os.path.join(save_dir, f"{map_name}_final_{timestamp}.pkl")
    iql.save_models(final_checkpoint)
    print(f"Final checkpoint saved: {final_checkpoint}")
    
    # Quick validation during training
    print(f"\n--- Quick Validation ---")
    success_count = 0
    test_episodes = 10
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(100):
            actions = {}
            
            # Get actions from trained algorithm
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            env.step(actions)
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                break
    
    success_rate = success_count / test_episodes
    print(f"Validation success rate: {success_rate:.1%}")
    
    # Save training metadata
    metadata = {
        "map_name": map_name,
        "map_metadata": map_metadata,
        "phase1_episodes": phase1_episodes,
        "phase2_episodes": phase2_episodes,
        "validation_success_rate": success_rate,
        "timestamp": timestamp,
        "phase1_checkpoint": phase1_checkpoint,
        "final_checkpoint": final_checkpoint,
        "action_space_dict": action_space_dict,
        "robot_agent_ids": robot_agent_ids,
        "human_agent_ids": human_agent_ids,
        "G": [g.tolist() for g in G],
        "mu_g": mu_g
    }
    
    metadata_file = os.path.join(save_dir, f"{map_name}_metadata_{timestamp}.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {metadata_file}")
    print(f"\nTraining complete for {map_name}")
    print(f"Final checkpoint: {final_checkpoint}")
    
    return final_checkpoint, metadata_file


def main():
    parser = argparse.ArgumentParser(description="Train IQL on a single map")
    parser.add_argument("map_name", choices=["simple_map", "simple_map2", "simple_map3", "simple_map4"],
                       help="Name of the map to train on")
    parser.add_argument("--phase1-episodes", type=int, default=500,
                       help="Number of Phase 1 episodes (default: 500)")
    parser.add_argument("--phase2-episodes", type=int, default=500,
                       help="Number of Phase 2 episodes (default: 500)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")
    
    args = parser.parse_args()
    
    try:
        checkpoint_path, metadata_path = train_single_map(
            args.map_name,
            args.phase1_episodes,
            args.phase2_episodes,
            args.save_dir
        )
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING SUCCESSFUL")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Metadata: {metadata_path}")
        print(f"\nTo test this model, run:")
        print(f"python test_trained_model.py {checkpoint_path} {args.map_name}")
        print(f"\nTo visualize this model, run:")
        print(f"python visualize_trained_model.py {checkpoint_path} {args.map_name}")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ TRAINING FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()