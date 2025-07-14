"""
Advanced Curriculum Training System

This module provides a comprehensive curriculum training system that:
1. Trains agents through a sequence of progressively difficult environments
2. Transfers both human and robot model weights between environments
3. Supports checkpoint management and resuming from any environment
4. Handles multiple humans with shared weights but different positions
"""

from typing import Any, Dict, List, Optional, Tuple

import wandb

from .checkpoint_manager import CheckpointManager
from .curriculum_envs import (
    CURRICULUM_SEQUENCE,
    get_curriculum_env,
    get_curriculum_map,
)
from .env import CustomEnvironment as GridEnvironment
from .envs.auto_env import generate_env_map
from .iql_timescale_algorithm import TwoPhaseTimescaleIQL


class CurriculumTrainer:
    """
    Advanced curriculum trainer with checkpoint management and weight transfer.
    """

    def __init__(
        self,
        checkpoint_dir: str = "curriculum_checkpoints",
        log_wandb: bool = True,
        project: str = "robot-curriculum-advanced",
    ):
        """
        Initialize curriculum trainer.

        Args:
            checkpoint_dir: Directory for storing checkpoints
            log_wandb: Whether to log to Weights & Biases
            project: WandB project name
        """
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.log_wandb = log_wandb
        self.project = project
        self.training_log = []

        if log_wandb:
            wandb.init(
                project=project,
                config={
                    "curriculum_length": len(CURRICULUM_SEQUENCE),
                    "checkpoint_dir": checkpoint_dir,
                },
            )

    def _load_hardcoded_map(
        self, env_name: str
    ) -> Tuple[List[List[str]], Dict[str, Any]]:
        """Load hardcoded map from environment file."""
        try:
            # Use the centralized map loading function
            return get_curriculum_map(env_name)
        except Exception as e:
            print(f"❌ Failed to load hardcoded map for {env_name}: {e}")
            raise

    def train_full_curriculum(
        self, start_env_index: int = 0, agent_config: Optional[Dict[str, Any]] = None
    ) -> TwoPhaseTimescaleIQL:
        """
        Train through the full curriculum, starting from a specific environment.

        Args:
            start_env_index: Index of environment to start from (0-based)
            agent_config: Configuration for the agent (required if starting from scratch)

        Returns:
            Final trained agent
        """
        print(f"🚀 Starting curriculum training from environment {start_env_index}")
        print(f"📚 Total environments: {len(CURRICULUM_SEQUENCE)}")

        # Initialize or load agent
        if start_env_index == 0:
            # Starting from scratch
            if agent_config is None:
                agent_config = self._get_default_agent_config()
            agent = TwoPhaseTimescaleIQL(**agent_config)
            print("🆕 Created new agent for curriculum training")
        else:
            # Load from previous checkpoint
            agent = self._load_agent_for_continuation(start_env_index)

        # Train through each environment in sequence
        for env_idx in range(start_env_index, len(CURRICULUM_SEQUENCE)):
            env_name, env_config = get_curriculum_env(env_idx)

            print(f"\n{'='*60}")
            print(
                f"🎯 Environment {env_idx + 1}/{len(CURRICULUM_SEQUENCE)}: {env_name}"
            )
            print(f"📋 {env_config['description']}")
            print(f"🎚️  Difficulty: {env_config['difficulty']}/5")
            print(
                f"🎯 Learning objectives: {', '.join(env_config['learning_objectives'])}"
            )
            print(f"{'='*60}")

            # If not starting from scratch, transfer weights to new environment configuration
            if env_idx > start_env_index or (env_idx > 0 and start_env_index == 0):
                agent = self._transfer_weights_for_env(agent, env_config)

            # Train on this environment
            final_agent, performance = self._train_single_environment(
                agent, env_config, env_idx
            )

            # Save checkpoint after successful training
            stage_info = {
                "env_index": env_idx,
                "env_name": env_name,
                "total_envs": len(CURRICULUM_SEQUENCE),
                "training_complete": True,
            }

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                final_agent, env_name, stage_info, performance
            )

            # Log progress to WandB
            if self.log_wandb:
                wandb.log(
                    {
                        "curriculum_stage": env_idx + 1,
                        "env_name": env_name,
                        "success_rate": performance.get("success_rate", 0),
                        "final_human_reward": performance.get("final_human_reward", 0),
                        "final_robot_reward": performance.get("final_robot_reward", 0),
                        "training_episodes": (
                            env_config["training_config"]["phase1_episodes"]
                            + env_config["training_config"]["phase2_episodes"]
                        ),
                    }
                )

            # Update agent for next iteration
            agent = final_agent

            # Add to training log
            self.training_log.append(
                {
                    "env_index": env_idx,
                    "env_name": env_name,
                    "performance": performance,
                    "checkpoint_path": checkpoint_path,
                }
            )

        print("\n🎉 Curriculum training complete!")
        print("📊 Training summary:")
        for entry in self.training_log:
            print(
                f"   {entry['env_name']}: Success rate {entry['performance'].get('success_rate', 0):.2f}"
            )

        if self.log_wandb:
            wandb.finish()

        return agent

    def train_single_env(
        self,
        env_index: int,
        checkpoint_path: Optional[str] = None,
        agent_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TwoPhaseTimescaleIQL, Dict[str, float]]:
        """
        Train on a single environment, optionally loading from checkpoint.

        Args:
            env_index: Index of environment to train on
            checkpoint_path: Optional path to checkpoint to load from
            agent_config: Agent configuration (required if no checkpoint)

        Returns:
            Tuple of (trained_agent, performance_metrics)
        """
        env_name, env_config = get_curriculum_env(env_index)

        # Load or create agent
        if checkpoint_path:
            agent, metadata = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            # Transfer weights if needed for different environment configuration
            agent = self._transfer_weights_for_env(agent, env_config)
        else:
            if agent_config is None:
                agent_config = self._get_default_agent_config()
            agent = TwoPhaseTimescaleIQL(**agent_config)

        print(f"🎯 Training single environment: {env_name}")

        # Train on this environment
        return self._train_single_environment(agent, env_config, env_index)

    def resume_from_checkpoint(
        self, checkpoint_path: str, continue_curriculum: bool = True
    ) -> TwoPhaseTimescaleIQL:
        """
        Resume training from a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint to resume from
            continue_curriculum: Whether to continue with curriculum or just load

        Returns:
            Loaded/trained agent
        """
        agent, metadata = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        if not continue_curriculum:
            return agent

        # Determine where to continue from
        env_name = metadata.get("env_name")
        current_env_idx = None

        for idx, (name, _) in enumerate(CURRICULUM_SEQUENCE):
            if name == env_name:
                current_env_idx = idx
                break

        if current_env_idx is None:
            print(f"⚠️  Warning: Could not determine environment index for {env_name}")
            return agent

        # Continue from the next environment
        next_env_idx = current_env_idx + 1
        if next_env_idx >= len(CURRICULUM_SEQUENCE):
            print("✅ Curriculum already complete!")
            return agent

        print(f"📍 Resuming curriculum from environment {next_env_idx + 1}")
        return self.train_full_curriculum(
            start_env_index=next_env_idx, agent_config=None
        )

    def _train_single_environment(
        self, agent: TwoPhaseTimescaleIQL, env_config: Dict[str, Any], env_idx: int
    ) -> Tuple[TwoPhaseTimescaleIQL, Dict[str, float]]:
        """Train agent on a single environment."""
        # Generate environment using hardcoded map if available
        if env_config.get("use_hardcoded_map", False):
            env_map, meta = self._load_hardcoded_map(env_config["name"])
        else:
            # Fallback to procedural generation
            env_map, meta = generate_env_map(env_config["gen_args"])

        env = GridEnvironment(grid_layout=env_map, grid_metadata=meta)

        # Get training configuration
        training_config = env_config["training_config"]
        phase1_episodes = training_config["phase1_episodes"]
        phase2_episodes = training_config["phase2_episodes"]
        success_threshold = training_config["success_threshold"]
        max_retries = training_config["max_retries"]

        # Training loop with retries
        for attempt in range(max_retries):
            print(f"🔄 Attempt {attempt + 1}/{max_retries}")

            # Phase 1: Human model learning
            print(f"📚 Phase 1: Learning human models ({phase1_episodes} episodes)")
            agent.train_phase1(env, phase1_episodes)

            # Phase 2: Robot policy learning
            print(f"🤖 Phase 2: Learning robot policy ({phase2_episodes} episodes)")
            agent.train_phase2(env, phase2_episodes)

            # Evaluate performance
            success_rate, metrics = self._evaluate_agent(agent, env)

            print(f"📊 Success rate: {success_rate:.3f}")
            print(f"📈 Metrics: {metrics}")

            if success_rate >= success_threshold:
                print(f"✅ Success threshold ({success_threshold}) reached!")
                break
            elif attempt == max_retries - 1:
                print("⚠️  Max retries reached. Proceeding with current performance.")
            else:
                print("🔄 Retrying training...")

        # Final performance metrics
        final_metrics = {
            "success_rate": success_rate,
            "attempts_used": attempt + 1,
            **metrics,
        }

        return agent, final_metrics

    def _evaluate_agent(
        self, agent: TwoPhaseTimescaleIQL, env: GridEnvironment, eval_episodes: int = 20
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate agent performance on environment."""
        successes = 0
        total_robot_reward = 0.0
        total_human_reward = 0.0
        total_steps = 0
        humans_reached_goal_count = 0

        for episode in range(eval_episodes):
            env.reset()
            done = False
            steps = 0
            episode_robot_reward = 0.0
            episode_human_reward = 0.0
            max_steps = getattr(env, "max_steps", 200)

            # Track if humans reached their goals in this episode
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
                    # Use first goal as default (could be made more sophisticated)
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

            # Check success: Did all humans reach their goals?
            if episode_success:
                successes += 1

            total_robot_reward += episode_robot_reward
            total_human_reward += episode_human_reward
            total_steps += steps

        success_rate = successes / eval_episodes
        avg_robot_reward = total_robot_reward / eval_episodes
        avg_human_reward = total_human_reward / eval_episodes
        avg_steps = total_steps / eval_episodes

        metrics = {
            "final_robot_reward": avg_robot_reward,
            "final_human_reward": avg_human_reward,
            "average_steps": avg_steps,
            "humans_reached_goals": humans_reached_goal_count / eval_episodes,
        }

        return success_rate, metrics

    def _transfer_weights_for_env(
        self, source_agent: TwoPhaseTimescaleIQL, env_config: Dict[str, Any]
    ) -> TwoPhaseTimescaleIQL:
        """Transfer weights from a source agent to a new agent for a new environment."""
        print("🧠 Transferring weights for new environment...")

        # Get the configuration for the new agent
        new_agent_config = self._get_default_agent_config()
        new_agent_config.update(env_config.get("agent_params", {}))
        new_agent_config.update(
            {
                "robot_agent_ids": [
                    f"robot_{i}"
                    for i in range(env_config["agent_config"]["num_robots"])
                ],
                "human_agent_ids": [
                    f"human_{i}"
                    for i in range(env_config["agent_config"]["num_humans"])
                ],
                "action_space_dict": env_config["action_space"],
            }
        )

        # Create a new agent with the correct configuration for the new environment
        new_agent = TwoPhaseTimescaleIQL(**new_agent_config)

        # Transfer the weights
        new_agent.human_q_m_backend = source_agent.human_q_m_backend
        new_agent.human_q_e_backend = source_agent.human_q_e_backend
        new_agent.robot_q_backend = source_agent.robot_q_backend

        print("✅ Weight transfer complete.")
        return new_agent

    def _create_agent_config_for_env(
        self, env_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create agent configuration suitable for the given environment."""
        gen_args = env_config["gen_args"]

        # Determine agent IDs based on environment configuration
        robot_agent_ids = [f"robot_{i}" for i in range(gen_args["num_robots"])]
        human_agent_ids = [f"human_{i}" for i in range(gen_args["num_humans"])]

        # Create action space (assume standard 4-direction movement for all agents)
        action_space_dict = {}
        for rid in robot_agent_ids:
            action_space_dict[rid] = [0, 1, 2, 3]  # up, right, down, left
        for hid in human_agent_ids:
            action_space_dict[hid] = [0, 1, 2, 3]

        # Goals (use goal positions from environment if available)
        goals = [
            (gen_args["width"] - 2, gen_args["height"] - 2)
        ]  # Default: near bottom-right
        if gen_args["num_goals"] > 1:
            # Add more goals in different corners
            goals.extend(
                [
                    (1, 1),  # top-left
                    (gen_args["width"] - 2, 1),  # top-right
                    (1, gen_args["height"] - 2),  # bottom-left
                ][: gen_args["num_goals"] - 1]
            )

        goal_probs = [1.0 / len(goals)] * len(goals)

        return {
            "alpha_m": 0.1,
            "alpha_e": 0.2,
            "alpha_r": 0.01,
            "gamma_h": 0.99,
            "gamma_r": 0.99,
            "beta_r_0": 5.0,
            "G": goals,
            "mu_g": goal_probs,
            "p_g": 0.0,
            "action_space_dict": action_space_dict,
            "robot_agent_ids": robot_agent_ids,
            "human_agent_ids": human_agent_ids,
            "network": True,  # Use neural networks for better transfer
            "state_dim": 4,
            "beta_h": 5.0,
            "nu_h": 0.1,
        }

    def _get_default_agent_config(self) -> Dict[str, Any]:
        """Get default agent configuration for starting curriculum."""
        # Start with simplest environment configuration
        env_config = get_curriculum_env(0)[1]  # First environment
        return self._create_agent_config_for_env(env_config)

    def _load_agent_for_continuation(
        self, start_env_index: int
    ) -> TwoPhaseTimescaleIQL:
        """Load agent from checkpoint for curriculum continuation."""
        if start_env_index == 0:
            raise ValueError(
                "Cannot load agent for continuation when starting from index 0"
            )

        # Try to find checkpoint from previous environment
        prev_env_name = CURRICULUM_SEQUENCE[start_env_index - 1][0]
        checkpoint_path = self.checkpoint_manager.get_latest_checkpoint(prev_env_name)

        if checkpoint_path:
            agent, metadata = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            print(f"📁 Loaded checkpoint from {prev_env_name}")
            return agent
        else:
            print(
                f"⚠️  No checkpoint found for {prev_env_name}, starting with default configuration"
            )
            return TwoPhaseTimescaleIQL(**self._get_default_agent_config())

    def list_available_checkpoints(self):
        """Print information about available checkpoints."""
        checkpoints = self.checkpoint_manager.list_checkpoints()

        if not checkpoints:
            print("📭 No checkpoints found")
            return

        print(f"📁 Available checkpoints ({len(checkpoints)}):")
        print("-" * 80)

        for name, metadata in sorted(
            checkpoints.items(), key=lambda x: x[1].get("timestamp", "")
        ):
            env_name = metadata.get("env_name", "Unknown")
            timestamp = metadata.get("timestamp", "Unknown")
            performance = metadata.get("performance_metrics", {})
            success_rate = performance.get("success_rate", 0)

            print(f"🏷️  {name}")
            print(f"   Environment: {env_name}")
            print(f"   Timestamp: {timestamp}")
            print(f"   Success Rate: {success_rate:.3f}")
            print(f"   Performance: {performance}")
            print()


# Convenience functions for easy use


def train_curriculum_from_scratch(
    checkpoint_dir: str = "curriculum_checkpoints",
) -> TwoPhaseTimescaleIQL:
    """Train the full curriculum from scratch."""
    trainer = CurriculumTrainer(checkpoint_dir)
    return trainer.train_full_curriculum(start_env_index=0)


def resume_curriculum_training(checkpoint_path: str) -> TwoPhaseTimescaleIQL:
    """Resume curriculum training from a checkpoint."""
    trainer = CurriculumTrainer()
    return trainer.resume_from_checkpoint(checkpoint_path, continue_curriculum=True)


def train_single_environment(
    env_index: int, checkpoint_path: Optional[str] = None
) -> Tuple[TwoPhaseTimescaleIQL, Dict[str, float]]:
    """Train on a single environment."""
    trainer = CurriculumTrainer()
    return trainer.train_single_env(env_index, checkpoint_path)
