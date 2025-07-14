"""
Pydantic models for MARL training configuration and data validation.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: Literal["robot", "human"] = Field(..., description="Type of agent")
    action_space: List[int] = Field(..., description="Available actions for this agent")

    class Config:
        json_encoders = {np.ndarray: lambda x: x.tolist()}


class EnvironmentConfig(BaseModel):
    """Configuration for the environment."""

    map_name: str = Field(default="simple_map", description="Name of the map to use")
    grid_size: Optional[int] = Field(default=None, description="Size of the grid")
    max_steps: int = Field(default=200, description="Maximum steps per episode")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    debug_level: Literal["minimal", "standard", "verbose"] = Field(
        default="standard", description="Debug level"
    )

    @validator("grid_size")
    def validate_grid_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Grid size must be positive")
        return v

    @validator("max_steps")
    def validate_max_steps(cls, v):
        if v <= 0:
            raise ValueError("Max steps must be positive")
        return v


class NetworkConfig(BaseModel):
    """Configuration for neural network Q-learning."""

    hidden_sizes: List[int] = Field(
        default=[128, 128], description="Hidden layer sizes"
    )
    learning_rate: float = Field(
        default=0.001, description="Learning rate for optimizer"
    )
    device: str = Field(default="cpu", description="Device to run on (cpu/cuda)")
    target_update_freq: int = Field(
        default=100, description="Target network update frequency"
    )

    @validator("hidden_sizes")
    def validate_hidden_sizes(cls, v):
        if not v or any(size <= 0 for size in v):
            raise ValueError("Hidden sizes must be positive integers")
        return v

    @validator("learning_rate")
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v


class IQLConfig(BaseModel):
    """Configuration for IQL algorithm."""

    # Learning rates
    alpha_m: float = Field(
        default=0.1, description="Learning rate for human cautious model"
    )
    alpha_e: float = Field(
        default=0.1, description="Learning rate for human effective model"
    )
    alpha_r: float = Field(default=0.01, description="Learning rate for robot model")
    alpha_p: float = Field(default=0.1, description="Learning rate for power network")

    # Discount factors
    gamma_h: float = Field(default=0.99, description="Discount factor for human")
    gamma_r: float = Field(default=0.99, description="Discount factor for robot")

    # Robot rationality parameter
    beta_r_0: float = Field(
        default=5.0, description="Robot rationality parameter for Phase 2"
    )

    # Human policy parameters
    beta_h: float = Field(default=5.0, description="Human policy temperature")
    policy_update_rate: float = Field(
        default=0.1, description="Policy update smoothing rate"
    )
    nu_h: float = Field(default=0.1, description="Uniform policy mixing parameter")

    # Goal parameters
    p_g: float = Field(default=0.0, description="Goal probability parameter")
    goal_dim: int = Field(default=2, description="Goal dimension")

    # Network configuration
    use_networks: bool = Field(
        default=False, description="Use neural networks instead of tabular"
    )
    network_config: Optional[NetworkConfig] = Field(
        default=None, description="Network configuration"
    )

    @validator("alpha_m", "alpha_e", "alpha_r", "alpha_p")
    def validate_learning_rates(cls, v):
        if v <= 0:
            raise ValueError("Learning rates must be positive")
        return v

    @validator("gamma_h", "gamma_r")
    def validate_discount_factors(cls, v):
        if not (0 < v <= 1):
            raise ValueError("Discount factors must be between 0 and 1")
        return v

    @validator("beta_r_0", "beta_h")
    def validate_temperature_params(cls, v):
        if v <= 0:
            raise ValueError("Temperature parameters must be positive")
        return v

    @validator("network_config", always=True)
    def validate_network_config(cls, v, values):
        if values.get("use_networks") and v is None:
            return NetworkConfig()
        return v


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""

    phase1_episodes: int = Field(
        default=500, description="Number of episodes in Phase 1"
    )
    phase2_episodes: int = Field(
        default=500, description="Number of episodes in Phase 2"
    )
    max_steps_per_episode: int = Field(
        default=200, description="Maximum steps per episode"
    )
    reward_shaping: bool = Field(default=True, description="Enable reward shaping")
    render: bool = Field(default=False, description="Render during training")
    render_delay: int = Field(default=100, description="Delay between renders in ms")

    @validator("phase1_episodes", "phase2_episodes", "max_steps_per_episode")
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Episode counts and max steps must be positive")
        return v

    @validator("render_delay")
    def validate_render_delay(cls, v):
        if v < 0:
            raise ValueError("Render delay must be non-negative")
        return v


class GoalConfig(BaseModel):
    """Configuration for goals."""

    goals: List[Tuple[int, int]] = Field(..., description="List of goal positions")
    goal_distribution: Optional[List[float]] = Field(
        default=None, description="Distribution over goals"
    )

    @validator("goals")
    def validate_goals(cls, v):
        if not v:
            raise ValueError("At least one goal must be specified")
        for goal in v:
            if len(goal) != 2:
                raise ValueError("Goals must be 2D coordinates")
            if any(coord < 0 for coord in goal):
                raise ValueError("Goal coordinates must be non-negative")
        return v

    @validator("goal_distribution")
    def validate_goal_distribution(cls, v, values):
        if v is None:
            return None
        goals = values.get("goals", [])
        if len(v) != len(goals):
            raise ValueError("Goal distribution must have same length as goals")
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError("Goal distribution must sum to 1")
        if any(p < 0 for p in v):
            raise ValueError("Goal distribution probabilities must be non-negative")
        return v


class MARLConfig(BaseModel):
    """Main configuration for MARL system."""

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    iql: IQLConfig = Field(default_factory=IQLConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    agents: List[AgentConfig] = Field(..., description="List of agent configurations")
    goals: GoalConfig = Field(..., description="Goal configuration")

    # File paths
    save_path: str = Field(
        default="q_values.pkl", description="Path to save trained models"
    )
    load_path: Optional[str] = Field(
        default=None, description="Path to load pre-trained models"
    )

    @validator("agents")
    def validate_agents(cls, v):
        if not v:
            raise ValueError("At least one agent must be specified")
        agent_ids = [agent.agent_id for agent in v]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("Agent IDs must be unique")
        return v

    def get_robot_agents(self) -> List[AgentConfig]:
        """Get all robot agents."""
        return [agent for agent in self.agents if agent.agent_type == "robot"]

    def get_human_agents(self) -> List[AgentConfig]:
        """Get all human agents."""
        return [agent for agent in self.agents if agent.agent_type == "human"]

    def get_agent_by_id(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None


class StateObservation(BaseModel):
    """Model for state observations."""

    agent_id: str = Field(..., description="Agent ID")
    state: List[int] = Field(..., description="State observation")
    timestamp: Optional[float] = Field(
        default=None, description="Timestamp of observation"
    )

    @validator("state")
    def validate_state(cls, v):
        if not v:
            raise ValueError("State cannot be empty")
        return v


class ActionSelection(BaseModel):
    """Model for action selections."""

    agent_id: str = Field(..., description="Agent ID")
    action: int = Field(..., description="Selected action")
    q_values: Optional[List[float]] = Field(
        default=None, description="Q-values for all actions"
    )
    policy: Optional[Dict[int, float]] = Field(
        default=None, description="Policy probabilities"
    )

    @validator("action")
    def validate_action(cls, v):
        if v < 0:
            raise ValueError("Action must be non-negative")
        return v


class TrainingStep(BaseModel):
    """Model for a single training step."""

    episode: int = Field(..., description="Episode number")
    step: int = Field(..., description="Step number within episode")
    phase: Literal["phase1", "phase2"] = Field(..., description="Training phase")
    observations: Dict[str, StateObservation] = Field(
        ..., description="Agent observations"
    )
    actions: Dict[str, ActionSelection] = Field(..., description="Agent actions")
    rewards: Dict[str, float] = Field(..., description="Agent rewards")
    done: bool = Field(default=False, description="Whether episode is done")

    @validator("episode", "step")
    def validate_non_negative(cls, v):
        if v < 0:
            raise ValueError("Episode and step must be non-negative")
        return v


class TrainingMetrics(BaseModel):
    """Model for training metrics."""

    total_episodes: int = Field(..., description="Total number of episodes")
    phase1_episodes: int = Field(..., description="Number of Phase 1 episodes")
    phase2_episodes: int = Field(..., description="Number of Phase 2 episodes")
    average_reward: Dict[str, float] = Field(
        ..., description="Average reward per agent"
    )
    success_rate: float = Field(..., description="Success rate (goals reached)")
    convergence_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Convergence metrics"
    )

    @validator("success_rate")
    def validate_success_rate(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Success rate must be between 0 and 1")
        return v


class ModelCheckpoint(BaseModel):
    """Model for saving/loading model checkpoints."""

    config: MARLConfig = Field(..., description="Configuration used for training")
    metrics: TrainingMetrics = Field(..., description="Training metrics")
    model_type: Literal["tabular", "network"] = Field(..., description="Type of model")
    checkpoint_path: str = Field(..., description="Path to checkpoint file")
    timestamp: str = Field(..., description="Timestamp when checkpoint was created")

    class Config:
        json_encoders = {np.ndarray: lambda x: x.tolist()}
