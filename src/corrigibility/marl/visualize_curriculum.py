#!/usr/bin/env python3
"""
Curriculum Model Visualization Tool

This script allows you to load trained models and visualize their behavior
in each curriculum environment. You can see how the robot helps humans
reach their goals in real-time.

Usage:
    # Visualize specific environment with trained model
    python visualize_curriculum.py --env 0 --checkpoint simple_checkpoints/env_01_simple_*.pkl
    
    # Visualize all environments in sequence
    python visualize_curriculum.py --all
    
    # Visualize with custom settings
    python visualize_curriculum.py --env 2 --episodes 5 --delay 500
"""

import argparse
import glob
import os
import time

from curriculum_envs import (
    get_all_env_names,
    get_curriculum_env,
    get_curriculum_map,
)
from env import CustomEnvironment as GridEnvironment
from trained_agent import TrainedAgent


def find_checkpoint(env_name):
    """Find the most recent checkpoint for an environment."""
    pattern = f"simple_checkpoints/{env_name}_*.pkl"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        print(f"❌ No checkpoints found for {env_name}")
        return None

    # Return the most recent checkpoint (assumes timestamp in filename)
    latest = sorted(checkpoints)[-1]
    print(f"📁 Found checkpoint: {latest}")
    return latest


def load_trained_agent(checkpoint_path):
    """Load a trained agent from checkpoint."""
    try:
        # Wrap the loaded model with TrainedAgent for a consistent API
        agent = TrainedAgent(checkpoint_path)
        print(f"✅ Loaded and wrapped trained agent from {checkpoint_path}")
        return agent
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return None


def visualize_environment(env_name, agent, episodes=3, delay=300, render=True):
    """Visualize agent behavior in an environment."""

    print(f"🎬 Visualizing {env_name} for {episodes} episodes...")

    # Load environment
    map_layout, metadata = get_curriculum_map(env_name)
    env = GridEnvironment(
        grid_layout=map_layout,
        grid_metadata=metadata,
        render_mode="human" if render else None,
    )

    successes = 0

    for episode in range(episodes):
        print(f"\n📺 Episode {episode + 1}/{episodes}")
        env.reset()
        done = False
        steps = 0
        max_steps = metadata.get("max_steps", 200)

        # Track initial positions
        print(f"🎯 Human goals: {metadata.get('human_goals', {})}")

        if render:
            env.render()
            time.sleep(delay / 1000.0)

        while not done and steps < max_steps:
            actions = {}

            # Robot actions
            for rid in agent.robot_agent_ids:
                state = agent.state_to_tuple(env.observe(rid))
                actions[rid] = agent.sample_robot_action_phase2(rid, state)

            # Human actions
            for hid in agent.human_agent_ids:
                state = agent.state_to_tuple(env.observe(hid))
                goal = agent.state_to_tuple(agent.G[0]) if agent.G else (0, 0)
                actions[hid] = agent.sample_human_action_phase2(hid, state, goal)

            # Environment step
            _, rewards, terms, truncs, _ = env.step(actions)
            done = any(terms.values()) or any(truncs.values())

            if render:
                env.render()
                time.sleep(delay / 1000.0)

            # Print action info
            action_str = ", ".join([f"{aid}={act}" for aid, act in actions.items()])
            reward_str = ", ".join(
                [f"{aid}={rew:.2f}" for aid, rew in rewards.items() if rew != 0]
            )
            if reward_str:
                print(f"  Step {steps}: Actions=[{action_str}] Rewards=[{reward_str}]")

            steps += 1

        # Check success
        episode_success = False
        if any(terms.values()) and not any(truncs.values()):
            if hasattr(env, "_humans_completed"):
                if len(env._humans_completed) == len(env.human_agent_ids):
                    episode_success = True
                    successes += 1

        status = "✅ SUCCESS" if episode_success else "❌ FAILED"
        print(f"  {status} in {steps} steps")

        if render and episode < episodes - 1:
            print("  Press Enter for next episode...")
            input()

    success_rate = successes / episodes
    print(
        f"\n📊 Results: {successes}/{episodes} episodes successful ({success_rate:.1%})"
    )
    return success_rate


def visualize_single_environment(
    env_index, checkpoint_path=None, episodes=3, delay=300
):
    """Visualize a single environment."""

    # Get environment info
    env_name, env_config = get_curriculum_env(env_index)
    print(f"\n🎯 Environment {env_index + 1}: {env_name}")
    print(f"📋 {env_config['description']}")

    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint(env_name)
        if checkpoint_path is None:
            return False

    # Load trained agent
    agent = load_trained_agent(checkpoint_path)
    if agent is None:
        return False

    # Visualize
    success_rate = visualize_environment(env_name, agent, episodes, delay)
    return success_rate > 0.5


def visualize_all_environments(episodes=2, delay=400):
    """Visualize all environments in sequence."""

    print("🎬 Visualizing All Curriculum Environments")
    print("=" * 60)

    env_names = get_all_env_names()
    results = {}

    for i, env_name in enumerate(env_names):
        print(f"\n{'='*60}")
        success = visualize_single_environment(i, episodes=episodes, delay=delay)
        results[env_name] = success

        if i < len(env_names) - 1:
            print("\nPress Enter to continue to next environment...")
            input()

    # Summary
    print(f"\n{'='*60}")
    print("📊 CURRICULUM VISUALIZATION SUMMARY")
    print(f"{'='*60}")

    for env_name, success in results.items():
        status = "✅ Working" if success else "❌ Needs Training"
        print(f"  {env_name}: {status}")


def list_available_checkpoints():
    """List all available checkpoints."""
    checkpoints = glob.glob("simple_checkpoints/*.pkl")
    if not checkpoints:
        print("❌ No checkpoints found in simple_checkpoints/")
        return

    print("📁 Available Checkpoints:")
    for checkpoint in sorted(checkpoints):
        size = os.path.getsize(checkpoint) / 1024  # KB
        mtime = time.ctime(os.path.getmtime(checkpoint))
        print(f"  {checkpoint} ({size:.1f} KB, {mtime})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Curriculum Model Visualization")
    parser.add_argument("--env", type=int, help="Environment index (0-4)")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--delay", type=int, default=300, help="Delay between steps (ms)"
    )
    parser.add_argument("--all", action="store_true", help="Visualize all environments")
    parser.add_argument(
        "--list", action="store_true", help="List available checkpoints"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable visual rendering"
    )

    args = parser.parse_args()

    if args.list:
        list_available_checkpoints()
        return

    if args.all:
        visualize_all_environments(episodes=args.episodes, delay=args.delay)
    elif args.env is not None:
        if args.env < 0 or args.env > 4:
            print("❌ Environment index must be 0-4")
            return

        success = visualize_single_environment(
            args.env,
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            delay=args.delay,
        )

        if success:
            print("✅ Visualization completed successfully!")
        else:
            print("❌ Visualization failed or model needs more training")
    else:
        print("❌ Please specify --env, --all, or --list")
        parser.print_help()


if __name__ == "__main__":
    main()
