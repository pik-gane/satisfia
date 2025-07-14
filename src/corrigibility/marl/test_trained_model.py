#!/usr/bin/env python3
"""
Test a trained IQL model from checkpoint.
"""

import os
import sys
import json
import numpy as np
import argparse

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


def load_metadata(checkpoint_path):
    """Load metadata file corresponding to checkpoint"""
    # Try to find metadata file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Extract timestamp from checkpoint name
    if "_final_" in checkpoint_name:
        timestamp = checkpoint_name.split("_final_")[1].replace(".pkl", "")
        map_name = checkpoint_name.split("_final_")[0]
    elif "_phase1_" in checkpoint_name:
        timestamp = checkpoint_name.split("_phase1_")[1].replace(".pkl", "")
        map_name = checkpoint_name.split("_phase1_")[0]
    else:
        raise ValueError(f"Cannot parse checkpoint name: {checkpoint_name}")
    
    metadata_file = os.path.join(checkpoint_dir, f"{map_name}_metadata_{timestamp}.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    else:
        return None


def test_trained_model(checkpoint_path, map_name, num_episodes=20, max_steps=200, verbose=True):
    """Test a trained IQL model"""
    
    if verbose:
        print(f"=== Testing Trained Model ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Map: {map_name}")
    
    # Load metadata if available
    metadata = load_metadata(checkpoint_path)
    if metadata and verbose:
        print(f"Training info:")
        print(f"  Phase 1 episodes: {metadata.get('phase1_episodes', 'N/A')}")
        print(f"  Phase 2 episodes: {metadata.get('phase2_episodes', 'N/A')}")
        print(f"  Training timestamp: {metadata.get('timestamp', 'N/A')}")
        print(f"  Validation success rate: {metadata.get('validation_success_rate', 'N/A'):.1%}")
    
    # Get map and create environment
    get_map_func = get_map_by_name(map_name)
    map_layout, map_metadata = get_map_func()
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    if verbose:
        print(f"\nEnvironment: {map_metadata['name']}")
        print(f"Size: {map_metadata['size']}")
        print(f"Human goals: {map_metadata['human_goals']}")
    
    # Setup from metadata or defaults
    if metadata:
        robot_agent_ids = metadata["robot_agent_ids"]
        human_agent_ids = metadata["human_agent_ids"]
        action_space_dict = metadata["action_space_dict"]
        G = [np.array(g) for g in metadata["G"]]
        mu_g = metadata["mu_g"]
    else:
        robot_agent_ids = ["robot_0"]
        human_agent_ids = ["human_0"]
        action_space_dict = {
            "robot_0": list(range(6)),
            "human_0": list(range(3))
        }
        human_goals = list(map_metadata["human_goals"].values())
        G = [np.array(goal) for goal in human_goals]
        mu_g = [1.0 / len(G)] * len(G)
    
    # Create IQL algorithm instance
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.2, alpha_e=0.2, alpha_r=0.2, alpha_p=0.2,
        gamma_h=0.9, gamma_r=0.9, beta_r_0=5.0,
        G=G, mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Load trained model
    if verbose:
        print(f"\nLoading model from {checkpoint_path}...")
    
    try:
        iql.load_models(checkpoint_path)
        if verbose:
            print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 0.0, []
    
    # Test the model
    if verbose:
        print(f"\n--- Testing ({num_episodes} episodes) ---")
    
    success_count = 0
    episode_results = []
    
    for episode in range(num_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        episode_success = False
        episode_steps = 0
        
        if verbose and episode < 3:  # Show details for first 3 episodes
            print(f"\nEpisode {episode + 1}:")
            print(f"  Initial positions: {env.agent_positions}")
            print(f"  Goal: {goal}")
        
        for step in range(max_steps):
            actions = {}
            
            # Get actions from trained algorithm
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            # Execute actions
            env.step(actions)
            episode_steps += 1
            
            if verbose and episode < 3 and step < 10:  # Show first 10 steps of first 3 episodes
                action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
                readable_actions = {agent: action_names.get(int(action), str(action)) for agent, action in actions.items()}
                print(f"    Step {step + 1}: {readable_actions} -> {env.agent_positions}")
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                episode_success = True
                if verbose and episode < 3:
                    print(f"    🎉 GOAL REACHED at step {step + 1}!")
                break
        
        episode_results.append({
            "episode": episode + 1,
            "success": episode_success,
            "steps": episode_steps,
            "goal": goal
        })
        
        if verbose and episode < 3 and not episode_success:
            print(f"    ❌ Episode timed out after {episode_steps} steps")
    
    success_rate = success_count / num_episodes
    
    if verbose:
        print(f"\n--- Results ---")
        print(f"Success rate: {success_rate:.1%} ({success_count}/{num_episodes})")
        
        if success_count > 0:
            successful_episodes = [r for r in episode_results if r["success"]]
            avg_steps = np.mean([r["steps"] for r in successful_episodes])
            print(f"Average steps to goal (successful episodes): {avg_steps:.1f}")
        
        print(f"\nDetailed Results:")
        for result in episode_results[:10]:  # Show first 10 episodes
            status = "✅" if result["success"] else "❌"
            print(f"  Episode {result['episode']:2d}: {status} {result['steps']:3d} steps")
        
        if len(episode_results) > 10:
            print(f"  ... and {len(episode_results) - 10} more episodes")
    
    return success_rate, episode_results


def main():
    parser = argparse.ArgumentParser(description="Test a trained IQL model")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint (.pkl file)")
    parser.add_argument("map_name", choices=["simple_map", "simple_map2", "simple_map3", "simple_map4"],
                       help="Name of the map to test on")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of test episodes (default: 20)")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode (default: 200)")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    try:
        success_rate, results = test_trained_model(
            args.checkpoint_path,
            args.map_name,
            args.episodes,
            args.max_steps,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n{'='*50}")
            if success_rate > 0.5:
                print(f"✅ MODEL PERFORMS WELL")
                print(f"Success rate: {success_rate:.1%}")
            elif success_rate > 0.0:
                print(f"⚠️  MODEL PARTIALLY WORKING")
                print(f"Success rate: {success_rate:.1%}")
            else:
                print(f"❌ MODEL NOT WORKING")
                print(f"Success rate: {success_rate:.1%}")
                print("Consider:")
                print("- Training for more episodes")
                print("- Adjusting learning rates")
                print("- Checking environment constraints")
            print(f"{'='*50}")
        else:
            print(f"{success_rate:.3f}")
        
        sys.exit(0 if success_rate > 0.0 else 1)
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()