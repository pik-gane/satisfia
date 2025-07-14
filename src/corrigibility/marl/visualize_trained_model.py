#!/usr/bin/env python3
"""
Visualize a trained IQL model in action.
"""

import os
import sys
import json
import numpy as np
import argparse
import time

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


def print_grid_with_agents(env):
    """Print the grid with agent positions clearly marked"""
    print("\nCurrent Grid State:")
    for r in range(env.grid_size):
        row = ""
        for c in range(env.grid_size):
            pos = (r, c)
            if pos == env.agent_positions.get('robot_0'):
                row += "R"
            elif pos == env.agent_positions.get('human_0'):
                row += "H"
            else:
                char = env.grid[r, c]
                if char == ' ':
                    row += "."
                else:
                    row += char
        print(f"  {row}")


def visualize_trained_model(checkpoint_path, map_name, num_episodes=3, max_steps=100, render_mode=None, step_delay=1.0):
    """Visualize a trained IQL model in action"""
    
    print(f"=== Visualizing Trained Model ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Map: {map_name}")
    
    # Load metadata if available
    metadata = load_metadata(checkpoint_path)
    if metadata:
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
        render_mode=render_mode
    )
    
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
    print(f"\nLoading model from {checkpoint_path}...")
    
    try:
        iql.load_models(checkpoint_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Visualization
    print(f"\n{'='*60}")
    print(f"VISUALIZATION STARTING")
    print(f"{'='*60}")
    
    if render_mode == "human":
        print("Close the pygame window to exit")
        print("Press Enter to continue through episodes...")
    else:
        print("Text-based visualization")
        print("Press Enter to continue through steps...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        print(f"Goal: {goal}")
        print(f"Initial positions: {env.agent_positions}")
        print(f"Initial directions: {env.agent_dirs}")
        print(f"Keys available: {[k['pos'] for k in env.keys]}")
        print(f"Doors: {[(d['pos'], d['is_locked'], d['color']) for d in env.doors]}")
        
        if render_mode == "human":
            env.render()
            input("Press Enter to start episode...")
        else:
            print_grid_with_agents(env)
            if step_delay > 0:
                input("Press Enter to start episode...")
        
        episode_success = False
        
        for step in range(max_steps):
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
            
            readable_actions = {agent: action_names.get(int(action), str(action)) for agent, action in actions.items()}
            
            print(f"\n  Step {step + 1}:")
            print(f"    Actions: {readable_actions}")
            print(f"    Positions: {env.agent_positions}")
            print(f"    Directions: {env.agent_dirs}")
            print(f"    Robot keys: {list(env.robot_has_keys)}")
            
            # Execute actions
            obs, rewards, terms, truncs, _ = env.step(actions)
            
            print(f"    Rewards: {rewards}")
            print(f"    Keys remaining: {[k['pos'] for k in env.keys]}")
            print(f"    Doors: {[(d['pos'], d['is_locked'], d['is_open']) for d in env.doors]}")
            
            if render_mode == "human":
                env.render()
            else:
                print_grid_with_agents(env)
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                print(f"    🎉 GOAL REACHED!")
                episode_success = True
                if render_mode == "human":
                    env.render()
                    input("Goal reached! Press Enter to continue...")
                else:
                    if step_delay > 0:
                        input("Goal reached! Press Enter to continue...")
                break
            
            # Check if episode ended
            if any(terms.values()) or any(truncs.values()):
                print(f"    Episode ended (terms: {terms}, truncs: {truncs})")
                break
            
            # Wait for user input or delay
            if step_delay > 0:
                if render_mode == "human":
                    time.sleep(step_delay)
                else:
                    input("Press Enter for next step...")
            else:
                time.sleep(0.5)
        
        if not episode_success:
            print(f"  ❌ Episode timed out after {step + 1} steps")
            if step_delay > 0:
                input("Episode timed out. Press Enter to continue...")
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    
    if render_mode == "human":
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained IQL model")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint (.pkl file)")
    parser.add_argument("map_name", choices=["simple_map", "simple_map2", "simple_map3", "simple_map4"],
                       help="Name of the map to visualize")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to visualize (default: 3)")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode (default: 100)")
    parser.add_argument("--render", action="store_true",
                       help="Use pygame rendering (requires display)")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-advance steps without waiting for input")
    parser.add_argument("--delay", type=float, default=1.0,
                       help="Delay between steps in auto mode (default: 1.0)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)
    
    render_mode = "human" if args.render else None
    step_delay = 0 if args.auto else args.delay
    
    try:
        visualize_trained_model(
            args.checkpoint_path,
            args.map_name,
            args.episodes,
            args.max_steps,
            render_mode,
            step_delay
        )
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()