#!/usr/bin/env python3
"""
Simple Curriculum Training Runner

This script provides a working curriculum training system that can be run
from the marl directory. It handles the environment sequence and weight
transfer between environments.

Usage:
    python simple_curriculum_runner.py [--start-env=0] [--single-env=N]
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from curriculum_envs import CURRICULUM_SEQUENCE, get_curriculum_env
from env import GridEnvironment
from envs.auto_env import generate_env_map


def create_agent_for_env(env_config, network=True):
    """Create an agent configured for the given environment."""
    # Import here to avoid issues with relative imports
    from curriculum_envs import get_curriculum_map
    from iql_timescale_algorithm import TwoPhaseTimescaleIQL

    # Check if using hardcoded map
    if env_config.get("use_hardcoded_map", False):
        # Get metadata from hardcoded map
        env_name = env_config["name"]
        map_layout, metadata = get_curriculum_map(env_name)

        # Extract agent info from metadata
        num_robots = metadata.get("num_robots", 1)
        num_humans = metadata.get("num_humans", 1)

        # Create agent IDs
        robot_agent_ids = [f"robot_{i}" for i in range(num_robots)]
        human_agent_ids = [f"human_{i}" for i in range(num_humans)]

        # Goals from hardcoded map
        human_goals = metadata.get("human_goals", {})
        if human_goals:
            goals = list(human_goals.values())
        else:
            # Fallback: find GG positions in map
            goals = []
            for r, row in enumerate(map_layout):
                for c, cell in enumerate(row):
                    if cell == "GG":
                        goals.append((r, c))
            if not goals:
                goals = [(len(map_layout) - 2, len(map_layout[0]) - 2)]  # Default goal
        goal_probs = [1.0 / len(goals)] * len(goals)
    else:
        # Use traditional gen_args approach
        gen_args = env_config["gen_args"]

        # Determine agent IDs
        robot_agent_ids = [f"robot_{i}" for i in range(gen_args["num_robots"])]
        human_agent_ids = [f"human_{i}" for i in range(gen_args["num_humans"])]

        # Goals
        goals = [
            (gen_args["width"] - 2, gen_args["height"] - 2)
        ]  # Default: near bottom-right
        goal_probs = [1.0]

    # Create action space (common for both hardcoded and generated environments)
    action_space_dict = {}
    for rid in robot_agent_ids:
        action_space_dict[rid] = [0, 1, 2, 3]  # up, right, down, left
    for hid in human_agent_ids:
        action_space_dict[hid] = [0, 1, 2, 3]

    return TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.2,
        alpha_r=0.01,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=goals,
        mu_g=goal_probs,
        p_g=0.0,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=network,
        state_dim=4,
        beta_h=5.0,
        nu_h=0.1,
    )


def transfer_agent_weights(source_agent, target_env_config):
    """Transfer weights from source agent to new agent for target environment."""

    # Create new agent for target environment
    target_agent = create_agent_for_env(target_env_config, network=source_agent.network)

    print("🔄 Transferring weights...")
    print(
        f"   Source: {len(source_agent.human_agent_ids)} humans, {len(source_agent.robot_agent_ids)} robots"
    )
    print(
        f"   Target: {len(target_agent.human_agent_ids)} humans, {len(target_agent.robot_agent_ids)} robots"
    )

    # Transfer robot weights
    if hasattr(source_agent, "robot_q_backend") and hasattr(
        target_agent, "robot_q_backend"
    ):
        if source_agent.network:
            # Neural network transfer
            if hasattr(source_agent.robot_q_backend, "networks"):
                target_agent.robot_q_backend.networks = (
                    source_agent.robot_q_backend.networks.copy()
                )
        else:
            # Tabular transfer
            if hasattr(source_agent.robot_q_backend, "q_tables"):
                target_agent.robot_q_backend.q_tables = (
                    source_agent.robot_q_backend.q_tables.copy()
                )

    # Transfer human weights (shared across all target humans)
    if hasattr(source_agent, "human_q_m_backend") and hasattr(
        target_agent, "human_q_m_backend"
    ):
        if source_agent.network:
            # Neural network transfer - share the first human's network with all target humans
            if (
                hasattr(source_agent.human_q_m_backend, "networks")
                and source_agent.human_agent_ids
                and source_agent.human_agent_ids[0]
                in source_agent.human_q_m_backend.networks
            ):

                source_network = source_agent.human_q_m_backend.networks[
                    source_agent.human_agent_ids[0]
                ]
                target_networks = {}
                for target_human_id in target_agent.human_agent_ids:
                    target_networks[target_human_id] = (
                        source_network  # Shared reference
                    )
                target_agent.human_q_m_backend.networks = target_networks

                # Transfer Q_e networks as well
                if (
                    hasattr(source_agent, "human_q_e_backend")
                    and hasattr(source_agent.human_q_e_backend, "networks")
                    and source_agent.human_agent_ids[0]
                    in source_agent.human_q_e_backend.networks
                ):

                    source_e_network = source_agent.human_q_e_backend.networks[
                        source_agent.human_agent_ids[0]
                    ]
                    target_e_networks = {}
                    for target_human_id in target_agent.human_agent_ids:
                        target_e_networks[target_human_id] = source_e_network
                    target_agent.human_q_e_backend.networks = target_e_networks
        else:
            # Tabular transfer - copy the first human's Q-table to all target humans
            if (
                hasattr(source_agent.human_q_m_backend, "q_tables")
                and source_agent.human_agent_ids
                and source_agent.human_agent_ids[0]
                in source_agent.human_q_m_backend.q_tables
            ):

                source_q_table = source_agent.human_q_m_backend.q_tables[
                    source_agent.human_agent_ids[0]
                ]
                target_q_tables = {}
                for target_human_id in target_agent.human_agent_ids:
                    target_q_tables[target_human_id] = source_q_table.copy()
                target_agent.human_q_m_backend.q_tables = target_q_tables

    print("✅ Weights transferred successfully")
    return target_agent


def evaluate_agent(agent, env, eval_episodes=10):
    """Evaluate agent performance."""
    successes = 0
    total_steps = 0
    total_robot_reward = 0.0
    total_human_reward = 0.0
    humans_reached_goal_count = 0

    for episode in range(eval_episodes):
        env.reset()
        done = False
        steps = 0
        episode_robot_reward = 0.0
        episode_human_reward = 0.0
        max_steps = getattr(env, "max_steps", 200)
        episode_success = False

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

            # Track rewards
            for rid in agent.robot_agent_ids:
                episode_robot_reward += rewards.get(rid, 0)
            for hid in agent.human_agent_ids:
                episode_human_reward += rewards.get(hid, 0)

            # Check if episode ended due to humans reaching goals (not truncation)
            if any(terms.values()) and not any(truncs.values()):
                # Check if all humans completed their goals
                if hasattr(env, "_humans_completed"):
                    if len(env._humans_completed) == len(env.human_agent_ids):
                        episode_success = True
                        humans_reached_goal_count += len(env._humans_completed)

            steps += 1

        # Success is based on humans reaching their goals
        if episode_success:
            successes += 1

        total_steps += steps
        total_robot_reward += episode_robot_reward
        total_human_reward += episode_human_reward

    success_rate = successes / eval_episodes
    avg_robot_reward = total_robot_reward / eval_episodes
    avg_human_reward = total_human_reward / eval_episodes
    avg_steps = total_steps / eval_episodes

    return success_rate, {
        "success_rate": success_rate,
        "avg_robot_reward": avg_robot_reward,
        "avg_human_reward": avg_human_reward,
        "avg_steps": avg_steps,
        "humans_reached_goals": humans_reached_goal_count / eval_episodes,
    }


def train_single_environment(agent, env_config, env_name):
    """Train agent on a single environment."""
    print(f"🏗️  Generating environment: {env_name}")

    # Generate environment using hardcoded map if available
    if env_config.get("use_hardcoded_map", False):
        from curriculum_envs import get_curriculum_map

        env_map, meta = get_curriculum_map(env_name)
        print(f"   🗺️  Using hardcoded map: {meta['size']}")
    else:
        # Fallback to procedural generation
        env_map, meta = generate_env_map(env_config["gen_args"])
        print("   🎲 Using procedural generation")

    env = GridEnvironment(grid_layout=env_map, grid_metadata=meta)

    # Get training configuration
    training_config = env_config["training_config"]
    phase1_episodes = training_config["phase1_episodes"]
    phase2_episodes = training_config["phase2_episodes"]
    success_threshold = training_config.get("success_threshold", 0.7)
    max_retries = training_config.get("max_retries", 2)

    print("🎯 Training configuration:")
    print(f"   Phase 1: {phase1_episodes} episodes")
    print(f"   Phase 2: {phase2_episodes} episodes")
    print(f"   Success threshold: {success_threshold}")
    print(f"   Max retries: {max_retries}")

    best_performance = 0.0
    best_agent = None

    for attempt in range(max_retries):
        print(f"\n🔄 Training attempt {attempt + 1}/{max_retries}")

        # Phase 1: Human model learning
        print("📚 Phase 1: Learning human models...")
        agent.train_phase1(env, phase1_episodes)

        # Phase 2: Robot policy learning
        print("🤖 Phase 2: Learning robot policy...")
        agent.train_phase2(env, phase2_episodes)

        # Evaluate
        success_rate, metrics = evaluate_agent(agent, env)
        print(f"📊 Performance: Success rate = {success_rate:.3f}")
        print(f"   Robot reward: {metrics['avg_robot_reward']:.2f}")
        print(f"   Human reward: {metrics['avg_human_reward']:.2f}")

        if success_rate > best_performance:
            best_performance = success_rate
            best_agent = agent

        if success_rate >= success_threshold:
            print("✅ Success threshold reached!")
            break
        elif attempt < max_retries - 1:
            print("⚠️  Below threshold, retrying...")

    return best_agent or agent, {
        "success_rate": best_performance,
        "final_performance": metrics,
    }


def save_checkpoint(
    agent, env_name, env_index, performance, checkpoint_dir="simple_checkpoints"
):
    """Save a simple checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"{env_name}_{timestamp}"
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pkl")

    # Save agent
    agent.save_models(checkpoint_path)

    # Save metadata
    metadata = {
        "env_name": env_name,
        "env_index": env_index,
        "timestamp": timestamp,
        "performance": performance,
        "checkpoint_path": checkpoint_path,
    }

    metadata_path = os.path.join(checkpoint_dir, f"{checkpoint_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Simple Curriculum Training")
    parser.add_argument(
        "--start-env", type=int, default=0, help="Starting environment index (0-based)"
    )
    parser.add_argument(
        "--single-env",
        type=int,
        default=None,
        help="Train only on specific environment",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        default=True,
        help="Use neural networks (default: True)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="simple_checkpoints",
        help="Checkpoint directory",
    )

    args = parser.parse_args()

    print("🎓 Simple Curriculum Training System")
    print("=" * 50)

    if args.single_env is not None:
        # Train single environment
        if args.single_env < 0 or args.single_env >= len(CURRICULUM_SEQUENCE):
            print(f"❌ Invalid environment index: {args.single_env}")
            return 1

        env_name, env_config = get_curriculum_env(args.single_env)
        print(f"🎯 Training single environment: {env_name}")

        agent = create_agent_for_env(env_config, network=args.network)
        final_agent, performance = train_single_environment(agent, env_config, env_name)

        save_checkpoint(
            final_agent, env_name, args.single_env, performance, args.checkpoint_dir
        )
        print("✅ Single environment training completed!")

    else:
        # Train curriculum sequence
        print(f"🚀 Starting curriculum from environment {args.start_env}")
        print("📚 Environments to train:")
        for i in range(args.start_env, len(CURRICULUM_SEQUENCE)):
            env_name, _ = get_curriculum_env(i)
            print(f"   {i}: {env_name}")

        agent = None

        for env_idx in range(args.start_env, len(CURRICULUM_SEQUENCE)):
            env_name, env_config = get_curriculum_env(env_idx)

            print(f"\n{'='*60}")
            print(
                f"🎯 Environment {env_idx + 1}/{len(CURRICULUM_SEQUENCE)}: {env_name}"
            )
            print(f"📋 {env_config['description']}")
            print(f"🎚️  Difficulty: {env_config['difficulty']}/5")
            print(f"{'='*60}")

            if agent is None:
                # First environment - create new agent
                agent = create_agent_for_env(env_config, network=args.network)
                print("🆕 Created new agent")
            else:
                # Transfer weights from previous environment
                agent = transfer_agent_weights(agent, env_config)

            # Train on this environment
            agent, performance = train_single_environment(agent, env_config, env_name)

            # Save checkpoint
            save_checkpoint(agent, env_name, env_idx, performance, args.checkpoint_dir)

            print(
                f"✅ Completed {env_name} with success rate: {performance['success_rate']:.3f}"
            )

        print("\n🎉 Curriculum training completed!")

    return 0


if __name__ == "__main__":
    exit(main())
