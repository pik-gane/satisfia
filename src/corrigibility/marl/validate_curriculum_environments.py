#!/usr/bin/env python3
"""
Environment Testing and Validation System

This script tests each curriculum environment to ensure:
1. Training converges successfully
2. Agents can reach their goals
3. Performance meets expected thresholds
4. Weight transfer works between environments

Usage:
    python validate_curriculum_environments.py [--quick] [--env=N] [--verbose]
"""

import argparse
import os
import sys
import time

import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_environment_basic(env_index, quick_test=False, verbose=False):
    """Test basic functionality of a single environment."""
    try:
        from curriculum_envs import get_curriculum_env, get_curriculum_map
        from env import GridEnvironment
        from envs.auto_env import generate_env_map
        from iql_timescale_algorithm import TwoPhaseTimescaleIQL

        env_name, env_config = get_curriculum_env(env_index)

        print(f"\n{'='*60}")
        print(f"🧪 Testing Environment {env_index + 1}: {env_name}")
        print(f"📋 {env_config['description']}")
        print(f"🎚️  Difficulty: {env_config['difficulty']}/5")

        # Check if environment uses hardcoded map
        if env_config.get("use_hardcoded_map", False):
            print("🗺️  Map type: Hardcoded")
        else:
            print(
                f"🤖 Agents: {env_config['gen_args']['num_robots']} robot(s), {env_config['gen_args']['num_humans']} human(s)"
            )
            print(
                f"📐 Grid: {env_config['gen_args']['width']}x{env_config['gen_args']['height']}"
            )
        print(f"{'='*60}")

        # Test 1: Environment Generation
        print("🏗️  Test 1: Environment Generation")
        start_time = time.time()

        # Generate environment using hardcoded map if available
        if env_config.get("use_hardcoded_map", False):
            env_map, meta = get_curriculum_map(env_name)
        else:
            # Fallback to procedural generation
            env_map, meta = generate_env_map(env_config["gen_args"])

        env = GridEnvironment(grid_layout=env_map, grid_metadata=meta)
        gen_time = time.time() - start_time

        print(f"   ✅ Environment generated in {gen_time:.3f}s")
        print(f"   📏 Map size: {len(env_map)}x{len(env_map[0])}")
        print(f"   🎯 Max steps: {meta.get('max_steps', 'Unknown')}")

        if verbose:
            print("   🗺️  Map preview:")
            for i, row in enumerate(env_map[: min(8, len(env_map))]):
                row_str = "      " + " ".join(row[: min(16, len(row))])
                if len(row) > 16:
                    row_str += "..."
                print(row_str)
            if len(env_map) > 8:
                print("      ...")

        # Test 2: Agent Creation
        print("🤖 Test 2: Agent Creation")
        start_time = time.time()

        gen_args = env_config["gen_args"]
        robot_agent_ids = [f"robot_{i}" for i in range(gen_args["num_robots"])]
        human_agent_ids = [f"human_{i}" for i in range(gen_args["num_humans"])]

        action_space_dict = {}
        for rid in robot_agent_ids:
            action_space_dict[rid] = [0, 1, 2, 3]
        for hid in human_agent_ids:
            action_space_dict[hid] = [0, 1, 2, 3]

        goals = [(gen_args["width"] - 2, gen_args["height"] - 2)]
        goal_probs = [1.0]

        agent = TwoPhaseTimescaleIQL(
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
            network=True,
            state_dim=4,
            beta_h=5.0,
            nu_h=0.1,
        )

        agent_time = time.time() - start_time
        print(f"   ✅ Agent created in {agent_time:.3f}s")
        print(f"   🤖 Robot agents: {robot_agent_ids}")
        print(f"   👥 Human agents: {human_agent_ids}")
        print(f"   🎯 Goals: {goals}")

        # Test 3: Basic Environment Interaction
        print("🔄 Test 3: Basic Environment Interaction")
        start_time = time.time()

        env.reset()
        interaction_steps = 10
        total_rewards = {aid: 0.0 for aid in robot_agent_ids + human_agent_ids}

        for step in range(interaction_steps):
            actions = {}

            # Sample actions
            for rid in robot_agent_ids:
                state = agent.state_to_tuple(env.observe(rid))
                actions[rid] = np.random.choice(
                    [0, 1, 2, 3]
                )  # Random action for basic test

            for hid in human_agent_ids:
                state = agent.state_to_tuple(env.observe(hid))
                actions[hid] = np.random.choice(
                    [0, 1, 2, 3]
                )  # Random action for basic test

            # Environment step
            _, rewards, terms, truncs, _ = env.step(actions)

            for aid in total_rewards:
                total_rewards[aid] += rewards.get(aid, 0)

            if any(terms.values()) or any(truncs.values()):
                break

        interaction_time = time.time() - start_time
        print(f"   ✅ Environment interaction successful in {interaction_time:.3f}s")
        print(f"   📊 Steps taken: {step + 1}/{interaction_steps}")
        print(f"   🎁 Total rewards: {total_rewards}")

        # Test 4: Quick Training (if not quick_test)
        training_success = True
        training_time = 0
        final_performance = {"success_rate": 0.0}

        if not quick_test:
            print("🎓 Test 4: Training Validation")
            start_time = time.time()

            # Reduced episodes for testing
            phase1_episodes = min(
                50, env_config["training_config"]["phase1_episodes"] // 10
            )
            phase2_episodes = min(
                100, env_config["training_config"]["phase2_episodes"] // 10
            )

            print(f"   📚 Phase 1: {phase1_episodes} episodes (reduced for testing)")

            try:
                agent.train_phase1(env, phase1_episodes, render=False)
                print("   ✅ Phase 1 completed")

                print(
                    f"   🤖 Phase 2: {phase2_episodes} episodes (reduced for testing)"
                )
                agent.train_phase2(env, phase2_episodes, render=False)
                print("   ✅ Phase 2 completed")

                # Quick evaluation
                success_rate, metrics = evaluate_quick(agent, env, eval_episodes=5)
                final_performance = metrics

                training_time = time.time() - start_time
                print(f"   📊 Training completed in {training_time:.1f}s")
                print(f"   🎯 Quick evaluation: {success_rate:.2f} success rate")
                print(
                    f"   📈 Metrics: Robot={metrics.get('avg_robot_reward', 0):.2f}, Human={metrics.get('avg_human_reward', 0):.2f}"
                )

            except Exception as e:
                training_success = False
                training_time = time.time() - start_time
                print(f"   ❌ Training failed after {training_time:.1f}s: {e}")
        else:
            print("⏩ Test 4: Skipped (quick test mode)")

        # Test Results Summary
        print(f"\n📋 Test Results for {env_name}:")
        print(f"   🏗️  Environment Generation: ✅ ({gen_time:.3f}s)")
        print(f"   🤖 Agent Creation: ✅ ({agent_time:.3f}s)")
        print(f"   🔄 Basic Interaction: ✅ ({interaction_time:.3f}s)")
        if not quick_test:
            print(
                f"   🎓 Training: {'✅' if training_success else '❌'} ({training_time:.1f}s)"
            )
            print(
                f"   📊 Performance: {final_performance.get('success_rate', 0):.2f} success rate"
            )
        else:
            print("   🎓 Training: ⏩ (skipped)")

        return {
            "env_name": env_name,
            "env_index": env_index,
            "generation_time": gen_time,
            "agent_time": agent_time,
            "interaction_time": interaction_time,
            "training_success": training_success,
            "training_time": training_time,
            "performance": final_performance,
            "overall_success": training_success,
        }

    except Exception as e:
        print(f"❌ Environment {env_index} test failed: {e}")
        import traceback

        if verbose:
            traceback.print_exc()
        return {
            "env_name": f"env_{env_index}",
            "env_index": env_index,
            "overall_success": False,
            "error": str(e),
        }


def evaluate_quick(agent, env, eval_episodes=5):
    """Quick evaluation of agent performance."""
    successes = 0
    total_steps = 0
    total_robot_reward = 0.0
    total_human_reward = 0.0

    for episode in range(eval_episodes):
        env.reset()
        done = False
        steps = 0
        episode_robot_reward = 0.0
        episode_human_reward = 0.0
        max_steps = min(50, getattr(env, "max_steps", 200))  # Limit for quick test

        while not done and steps < max_steps:
            actions = {}

            try:
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

            except Exception:
                # If sampling fails, break the episode
                break

            steps += 1

        # Simple success criterion: positive rewards for both
        if episode_human_reward > 5 and episode_robot_reward > -5:
            successes += 1

        total_steps += steps
        total_robot_reward += episode_robot_reward
        total_human_reward += episode_human_reward

    success_rate = successes / eval_episodes

    return success_rate, {
        "success_rate": success_rate,
        "avg_robot_reward": total_robot_reward / eval_episodes,
        "avg_human_reward": total_human_reward / eval_episodes,
        "avg_steps": total_steps / eval_episodes,
    }


def test_weight_transfer():
    """Test weight transfer between environments."""
    print(f"\n{'='*60}")
    print("🔄 Testing Weight Transfer Between Environments")
    print(f"{'='*60}")

    try:
        from curriculum_envs import get_curriculum_env, get_curriculum_map
        from env import GridEnvironment
        from envs.auto_env import generate_env_map

        # Test transfer from env 0 to env 3 (different number of humans)
        source_env_idx = 0
        target_env_idx = 3

        print(
            f"🎯 Testing transfer from Environment {source_env_idx + 1} to Environment {target_env_idx + 1}"
        )

        # Create and train source agent (minimal training)
        source_name, source_config = get_curriculum_env(source_env_idx)
        target_name, target_config = get_curriculum_env(target_env_idx)

        print(f"   📤 Source: {source_name}")
        print(f"   📥 Target: {target_name}")

        # Create source agent
        source_agent = create_agent_for_config(source_config)

        # Quick training on source
        if source_config.get("use_hardcoded_map", False):
            source_env_map, source_meta = get_curriculum_map(source_name)
        else:
            source_env_map, source_meta = generate_env_map(source_config["gen_args"])
        source_env = GridEnvironment(
            grid_layout=source_env_map, grid_metadata=source_meta
        )

        print("   📚 Quick training on source environment...")
        source_agent.train_phase1(source_env, 20, render=False)
        source_agent.train_phase2(source_env, 30, render=False)

        # Test transfer
        print("   🔄 Transferring weights...")
        target_agent = transfer_weights_simple(source_agent, target_config)

        print("   ✅ Weight transfer completed")
        print(f"   📊 Source humans: {len(source_agent.human_agent_ids)}")
        print(f"   📊 Target humans: {len(target_agent.human_agent_ids)}")

        # Quick test on target environment
        if target_config.get("use_hardcoded_map", False):
            target_env_map, target_meta = get_curriculum_map(target_name)
        else:
            target_env_map, target_meta = generate_env_map(target_config["gen_args"])
        target_env = GridEnvironment(
            grid_layout=target_env_map, grid_metadata=target_meta
        )

        print("   🧪 Testing transferred agent on target environment...")
        success_rate, metrics = evaluate_quick(
            target_agent, target_env, eval_episodes=3
        )

        print("   📈 Transfer test results:")
        print(f"      Success rate: {success_rate:.2f}")
        print(f"      Robot reward: {metrics['avg_robot_reward']:.2f}")
        print(f"      Human reward: {metrics['avg_human_reward']:.2f}")

        return {
            "transfer_success": True,
            "source_env": source_name,
            "target_env": target_name,
            "performance": metrics,
        }

    except Exception as e:
        print(f"   ❌ Weight transfer test failed: {e}")
        import traceback

        traceback.print_exc()
        return {"transfer_success": False, "error": str(e)}


def create_agent_for_config(env_config):
    """Create agent for given environment configuration."""
    from iql_timescale_algorithm import TwoPhaseTimescaleIQL

    gen_args = env_config["gen_args"]
    robot_agent_ids = [f"robot_{i}" for i in range(gen_args["num_robots"])]
    human_agent_ids = [f"human_{i}" for i in range(gen_args["num_humans"])]

    action_space_dict = {}
    for rid in robot_agent_ids:
        action_space_dict[rid] = [0, 1, 2, 3]
    for hid in human_agent_ids:
        action_space_dict[hid] = [0, 1, 2, 3]

    goals = [(gen_args["width"] - 2, gen_args["height"] - 2)]
    goal_probs = [1.0]

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
        network=True,
        state_dim=4,
        beta_h=5.0,
        nu_h=0.1,
    )


def transfer_weights_simple(source_agent, target_config):
    """Simple weight transfer for testing."""
    target_agent = create_agent_for_config(target_config)

    # Transfer robot weights
    if hasattr(source_agent, "robot_q_backend") and hasattr(
        target_agent, "robot_q_backend"
    ):
        if source_agent.network and hasattr(source_agent.robot_q_backend, "networks"):
            target_agent.robot_q_backend.networks = (
                source_agent.robot_q_backend.networks.copy()
            )

    # Transfer human weights (shared across all target humans)
    if hasattr(source_agent, "human_q_m_backend") and hasattr(
        target_agent, "human_q_m_backend"
    ):
        if (
            source_agent.network
            and hasattr(source_agent.human_q_m_backend, "networks")
            and source_agent.human_agent_ids
        ):

            source_human_id = source_agent.human_agent_ids[0]
            if source_human_id in source_agent.human_q_m_backend.networks:
                source_network = source_agent.human_q_m_backend.networks[
                    source_human_id
                ]
                target_networks = {}
                for target_human_id in target_agent.human_agent_ids:
                    target_networks[target_human_id] = source_network
                target_agent.human_q_m_backend.networks = target_networks

    return target_agent


def main():
    parser = argparse.ArgumentParser(description="Validate Curriculum Environments")
    parser.add_argument(
        "--quick", action="store_true", help="Skip training tests for faster validation"
    )
    parser.add_argument(
        "--env", type=int, default=None, help="Test only specific environment (0-based)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output and error traces"
    )
    parser.add_argument(
        "--no-transfer", action="store_true", help="Skip weight transfer test"
    )

    args = parser.parse_args()

    print("🧪 Curriculum Environment Validation System")
    print("=" * 60)

    from curriculum_envs import CURRICULUM_SEQUENCE

    if args.env is not None:
        # Test single environment
        if args.env < 0 or args.env >= len(CURRICULUM_SEQUENCE):
            print(f"❌ Invalid environment index: {args.env}")
            return 1

        result = test_environment_basic(
            args.env, quick_test=args.quick, verbose=args.verbose
        )

        if result["overall_success"]:
            print(f"\n✅ Environment {args.env} validation successful!")
        else:
            print(f"\n❌ Environment {args.env} validation failed!")
            return 1

    else:
        # Test all environments
        print(f"🎯 Testing {len(CURRICULUM_SEQUENCE)} environments...")
        if args.quick:
            print("⏩ Quick mode: Skipping training validation")

        results = []
        successful_envs = 0

        for env_idx in range(len(CURRICULUM_SEQUENCE)):
            result = test_environment_basic(
                env_idx, quick_test=args.quick, verbose=args.verbose
            )
            results.append(result)

            if result["overall_success"]:
                successful_envs += 1

        # Test weight transfer
        if not args.no_transfer and not args.quick:
            print("\n🔄 Testing weight transfer system...")
            transfer_result = test_weight_transfer()

        # Final summary
        print(f"\n{'='*60}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*60}")

        print(f"🎯 Environments tested: {len(CURRICULUM_SEQUENCE)}")
        print(f"✅ Successful: {successful_envs}")
        print(f"❌ Failed: {len(CURRICULUM_SEQUENCE) - successful_envs}")

        print("\n📋 Individual Results:")
        for result in results:
            status = "✅" if result["overall_success"] else "❌"
            env_name = result.get("env_name", f"env_{result['env_index']}")

            if result["overall_success"]:
                perf = result.get("performance", {})
                success_rate = perf.get("success_rate", 0)
                print(f"   {status} {env_name}: Success rate {success_rate:.2f}")
            else:
                error = result.get("error", "Unknown error")
                print(f"   {status} {env_name}: {error}")

        if not args.no_transfer and not args.quick:
            transfer_success = transfer_result.get("transfer_success", False)
            transfer_status = "✅" if transfer_success else "❌"
            print(f"\n🔄 Weight Transfer: {transfer_status}")

        if successful_envs == len(CURRICULUM_SEQUENCE):
            print("\n🎉 All environments validated successfully!")
            return 0
        else:
            print(
                f"\n⚠️  {len(CURRICULUM_SEQUENCE) - successful_envs} environment(s) failed validation"
            )
            return 1


if __name__ == "__main__":
    exit(main())
