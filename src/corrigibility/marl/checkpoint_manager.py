"""
Checkpoint Manager for Curriculum Training

This module handles saving and loading of agent weights/models between
curriculum environments, allowing for progressive learning where both
human and robot models are carried forward.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .iql_timescale_algorithm import TwoPhaseTimescaleIQL


class CheckpointManager:
    """
    Manages checkpoints for curriculum training, handling both human and robot models.
    Supports weight transfer between environments with proper metadata tracking.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Create subdirectories for organization
        (self.checkpoint_dir / "models").mkdir(exist_ok=True)
        (self.checkpoint_dir / "metadata").mkdir(exist_ok=True)
        (self.checkpoint_dir / "logs").mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        agent,
        env_name: str,
        stage_info: Dict[str, Any],
        performance_metrics: Dict[str, float],
    ) -> str:
        """
        Save a checkpoint including both human and robot models.

        Args:
            agent: The TwoPhaseTimescaleIQL agent to save
            env_name: Name of the environment this checkpoint is from
            stage_info: Information about the training stage
            performance_metrics: Performance metrics achieved

        Returns:
            str: Path to the saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{env_name}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / "models" / f"{checkpoint_name}.pkl"

        # Save the main agent model
        agent.save_models(str(checkpoint_path))

        # Create detailed metadata
        metadata = {
            "checkpoint_name": checkpoint_name,
            "env_name": env_name,
            "timestamp": timestamp,
            "stage_info": stage_info,
            "performance_metrics": performance_metrics,
            "agent_config": {
                "network_mode": getattr(agent, "network", False),
                "robot_agent_ids": getattr(agent, "robot_agent_ids", []),
                "human_agent_ids": getattr(agent, "human_agent_ids", []),
                "state_dim": getattr(agent, "state_dim", 4),
                "action_space_dict": getattr(agent, "action_space_dict", {}),
                "hyperparameters": {
                    "alpha_m": getattr(agent, "alpha_m", None),
                    "alpha_e": getattr(agent, "alpha_e", None),
                    "alpha_r": getattr(agent, "alpha_r", None),
                    "gamma_h": getattr(agent, "gamma_h", None),
                    "gamma_r": getattr(agent, "gamma_r", None),
                    "beta_r_0": getattr(agent, "beta_r_0", None),
                    "beta_h": getattr(agent, "beta_h_final", None),
                    "nu_h": getattr(agent, "nu_h", None),
                },
            },
            "model_files": {
                "main": str(checkpoint_path),
                "human_q_m": str(checkpoint_path).replace(".pkl", "_human_q_m.pkl"),
                "human_q_e": str(checkpoint_path).replace(".pkl", "_human_q_e.pkl"),
                "robot_q": str(checkpoint_path).replace(".pkl", "_robot_q.pkl"),
            },
        }

        # Save metadata
        metadata_path = (
            self.checkpoint_dir / "metadata" / f"{checkpoint_name}_metadata.json"
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Log the checkpoint
        self._log_checkpoint(checkpoint_name, metadata)

        print(f"✅ Checkpoint saved: {checkpoint_name}")
        print(f"   Environment: {env_name}")
        print(f"   Performance: {performance_metrics}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a checkpoint with both models and metadata.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple of (loaded_agent, metadata)
        """

        # Load the agent
        agent = TwoPhaseTimescaleIQL.load_q_values(checkpoint_path)
        if agent is None:
            raise ValueError(f"Failed to load agent from {checkpoint_path}")

        # Load metadata if available
        checkpoint_name = Path(checkpoint_path).stem
        metadata_path = (
            self.checkpoint_dir / "metadata" / f"{checkpoint_name}_metadata.json"
        )

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        print(f"✅ Checkpoint loaded: {checkpoint_name}")
        if metadata:
            print(f"   From environment: {metadata.get('env_name', 'Unknown')}")
            print(f"   Original performance: {metadata.get('performance_metrics', {})}")

        return agent, metadata

    def transfer_weights(
        self, source_agent, target_agent_config: Dict[str, Any]
    ) -> Any:
        """
        Transfer weights from source agent to a new agent with potentially different configuration.
        Handles cases where the number of humans changes but they share the same model weights.

        Args:
            source_agent: Source agent with trained weights
            target_agent_config: Configuration for the target agent

        Returns:
            New agent with transferred weights
        """

        # Create new agent with target configuration
        target_agent = TwoPhaseTimescaleIQL(**target_agent_config)

        # Transfer robot weights (these should be directly compatible)
        if hasattr(source_agent, "robot_q_backend") and hasattr(
            target_agent, "robot_q_backend"
        ):
            if source_agent.network:
                # For neural networks, transfer the actual network weights
                target_agent.robot_q_backend.networks = (
                    source_agent.robot_q_backend.networks.copy()
                )
            else:
                # For tabular, transfer Q-tables
                target_agent.robot_q_backend.q_tables = (
                    source_agent.robot_q_backend.q_tables.copy()
                )

        # Transfer human weights (shared across all humans in target environment)
        if hasattr(source_agent, "human_q_m_backend") and hasattr(
            target_agent, "human_q_m_backend"
        ):
            if source_agent.network:
                # For neural networks, copy and replicate for all target humans
                source_networks = source_agent.human_q_m_backend.networks
                target_networks = {}

                # Use the first human's trained network for all target humans
                if source_agent.human_agent_ids:
                    source_human_id = source_agent.human_agent_ids[0]
                    if source_human_id in source_networks:
                        source_network = source_networks[source_human_id]
                        for target_human_id in target_agent.human_agent_ids:
                            # Create a copy of the network for each target human
                            target_networks[target_human_id] = (
                                source_network  # Shared weights
                            )

                target_agent.human_q_m_backend.networks = target_networks

                # Do the same for Q_e backend
                if hasattr(source_agent, "human_q_e_backend"):
                    source_e_networks = source_agent.human_q_e_backend.networks
                    target_e_networks = {}
                    if (
                        source_agent.human_agent_ids
                        and source_agent.human_agent_ids[0] in source_e_networks
                    ):
                        source_e_network = source_e_networks[
                            source_agent.human_agent_ids[0]
                        ]
                        for target_human_id in target_agent.human_agent_ids:
                            target_e_networks[target_human_id] = (
                                source_e_network  # Shared weights
                            )
                    target_agent.human_q_e_backend.networks = target_e_networks
            else:
                # For tabular, copy and replicate Q-tables
                source_q_tables = source_agent.human_q_m_backend.q_tables
                target_q_tables = {}

                # Use the first human's Q-table for all target humans
                if source_agent.human_agent_ids:
                    source_human_id = source_agent.human_agent_ids[0]
                    if source_human_id in source_q_tables:
                        source_q_table = source_q_tables[source_human_id]
                        for target_human_id in target_agent.human_agent_ids:
                            target_q_tables[target_human_id] = source_q_table.copy()

                target_agent.human_q_m_backend.q_tables = target_q_tables

                # Do the same for Q_e backend
                if hasattr(source_agent, "human_q_e_backend"):
                    source_e_q_tables = source_agent.human_q_e_backend.q_tables
                    target_e_q_tables = {}
                    if (
                        source_agent.human_agent_ids
                        and source_agent.human_agent_ids[0] in source_e_q_tables
                    ):
                        source_e_q_table = source_e_q_tables[
                            source_agent.human_agent_ids[0]
                        ]
                        for target_human_id in target_agent.human_agent_ids:
                            target_e_q_tables[target_human_id] = source_e_q_table.copy()
                    target_agent.human_q_e_backend.q_tables = target_e_q_tables

        # Transfer other learned components
        if hasattr(source_agent, "V_m_h_dict") and hasattr(target_agent, "V_m_h_dict"):
            # Transfer value functions, replicating for new humans
            if source_agent.human_agent_ids:
                source_human_id = source_agent.human_agent_ids[0]
                if source_human_id in source_agent.V_m_h_dict:
                    source_v_dict = source_agent.V_m_h_dict[source_human_id]
                    for target_human_id in target_agent.human_agent_ids:
                        target_agent.V_m_h_dict[target_human_id] = source_v_dict.copy()

        print(
            f"✅ Weights transferred from {len(source_agent.human_agent_ids)} to {len(target_agent.human_agent_ids)} humans"
        )
        print(
            f"   Robot weights: {'✅ Transferred' if hasattr(source_agent, 'robot_q_backend') else '❌ Not found'}"
        )
        print(
            f"   Human weights: {'✅ Shared across all humans' if hasattr(source_agent, 'human_q_m_backend') else '❌ Not found'}"
        )

        return target_agent

    def list_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available checkpoints with their metadata.

        Returns:
            Dict mapping checkpoint names to their metadata
        """
        checkpoints = {}
        metadata_dir = self.checkpoint_dir / "metadata"

        for metadata_file in metadata_dir.glob("*_metadata.json"):
            checkpoint_name = metadata_file.stem.replace("_metadata", "")
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                checkpoints[checkpoint_name] = metadata
            except Exception as e:
                print(f"Warning: Could not load metadata for {checkpoint_name}: {e}")

        return checkpoints

    def get_latest_checkpoint(self, env_name: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the latest checkpoint, optionally filtered by environment.

        Args:
            env_name: Optional environment name to filter by

        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints()

        if env_name:
            checkpoints = {
                k: v for k, v in checkpoints.items() if v.get("env_name") == env_name
            }

        if not checkpoints:
            return None

        # Sort by timestamp and get the latest
        latest = max(
            checkpoints.keys(), key=lambda k: checkpoints[k].get("timestamp", "")
        )

        return str(self.checkpoint_dir / "models" / f"{latest}.pkl")

    def _log_checkpoint(self, checkpoint_name: str, metadata: Dict[str, Any]):
        """Log checkpoint creation to a log file."""
        log_path = self.checkpoint_dir / "logs" / "checkpoint_log.jsonl"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_name": checkpoint_name,
            "env_name": metadata.get("env_name"),
            "performance": metadata.get("performance_metrics", {}),
            "action": "created",
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
