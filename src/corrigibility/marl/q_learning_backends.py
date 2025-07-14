import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .env import Actions
    from .state_encoder import encode_full_observable_state
except ImportError:
    from env import Actions
    from state_encoder import encode_full_observable_state


class QLearningBackend(ABC):
    """Abstract base class for Q-learning backends (tabular vs neural network)."""

    def __init__(
        self,
        agent_ids: List[str],
        action_space_dict: Dict[str, List[int]],
        state_dim: int,
        debug: bool = False,
    ):
        self.agent_ids = agent_ids
        self.action_space_dict = action_space_dict
        self.state_dim = state_dim
        self.debug = debug

    @abstractmethod
    def get_q_values(
        self, agent_id: str, state: Tuple, goal: Optional[Tuple] = None
    ) -> np.ndarray:
        """Get Q-values for a given state (and optionally goal)."""
        pass

    @abstractmethod
    def update_q_values(
        self,
        agent_id: str,
        state: Tuple,
        action: int,
        target: float,
        learning_rate: float,
        goal: Optional[Tuple] = None,
    ):
        """Update Q-values for a given state-action pair."""
        pass

    @abstractmethod
    def get_policy(
        self,
        agent_id: str,
        state: Tuple,
        temperature: float,
        goal: Optional[Tuple] = None,
    ) -> Dict[int, float]:
        """Get softmax policy for a given state."""
        pass

    @abstractmethod
    def save_models(self, filepath: str):
        """Save the models to file."""
        pass

    @abstractmethod
    def load_models(self, filepath: str):
        """Load models from file."""
        pass


class TabularQLearning(QLearningBackend):
    """Tabular Q-learning backend using defaultdicts."""

    def __init__(
        self,
        agent_ids: List[str],
        action_space_dict: Dict[str, List[int]],
        state_dim: int,
        debug: bool = False,
        use_goals: bool = True,
        beta_h: float = 5.0,
        policy_update_rate: float = 0.1,
        nu_h: float = 0.1,
    ):
        super().__init__(agent_ids, action_space_dict, state_dim, debug)
        self.use_goals = use_goals
        self.beta_h = beta_h
        self.policy_update_rate = policy_update_rate
        self.nu_h = nu_h

        # Initialize Q-tables
        if use_goals:
            # For humans: Q(s, g, a)
            self.q_tables = {
                aid: defaultdict(
                    lambda: np.random.uniform(-0.1, 0.1, size=len(Actions))
                )
                for aid in agent_ids
            }
        else:
            # For robots: Q(s, a)
            self.q_tables = {
                aid: defaultdict(
                    lambda: np.random.uniform(
                        -0.1, 0.1, size=len(action_space_dict[aid])
                    )
                )
                for aid in agent_ids
            }

        # Store policies for smooth updates
        self.policies = {aid: {} for aid in agent_ids}

    def _make_key(self, state: Tuple, goal: Optional[Tuple] = None) -> Tuple:
        """Create dictionary key from state and optional goal."""
        if self.use_goals and goal is not None:
            return (state, goal)
        return state

    def get_q_values(
        self, agent_id: str, state: Tuple, goal: Optional[Tuple] = None
    ) -> np.ndarray:
        """Get Q-values for a given state (and optionally goal)."""
        key = self._make_key(state, goal)
        return self.q_tables[agent_id][key].copy()

    def update_q_values(
        self,
        agent_id: str,
        state: Tuple,
        action: int,
        target: float,
        learning_rate: float,
        goal: Optional[Tuple] = None,
    ):
        """Update Q-values for a given state-action pair."""
        key = self._make_key(state, goal)
        current_q = self.q_tables[agent_id][key][action]
        self.q_tables[agent_id][key][action] += learning_rate * (target - current_q)

        # Update policy after Q-value change using smooth updates
        q_values = self.q_tables[agent_id][key]
        self.update_policy_smooth(
            agent_id,
            state,
            q_values,
            self.beta_h,
            self.policy_update_rate,
            self.nu_h,
            goal,
        )

    def get_policy(
        self,
        agent_id: str,
        state: Tuple,
        temperature: float,
        goal: Optional[Tuple] = None,
    ) -> Dict[int, float]:
        """Get softmax policy for a given state."""
        key = self._make_key(state, goal)

        if key in self.policies[agent_id]:
            return self.policies[agent_id][key].copy()

        # Default to uniform policy for unseen states
        allowed_actions = self.action_space_dict[agent_id]
        return {a: 1.0 / len(allowed_actions) for a in allowed_actions}

    def update_policy_smooth(
        self,
        agent_id: str,
        state: Tuple,
        q_values: np.ndarray,
        temperature: float,
        update_rate: float,
        nu_h: float = 0.1,
        goal: Optional[Tuple] = None,
    ):
        """Update policy using smooth policy updates with softmax."""
        key = self._make_key(state, goal)
        allowed_actions = self.action_space_dict[agent_id]

        # Compute softmax policy from Q-values
        q_subset = np.array([q_values[a] for a in allowed_actions])

        # For robot agents, use -log(-Qr) transformation (equation 7)
        # Check if this is a robot agent by looking for 'robot' in agent_id
        if "robot" in agent_id.lower():
            # Transform Q-values: use -log(-Qr) instead of Qr directly
            # Add small epsilon to ensure -Qr is positive for log computation
            epsilon = 1e-8
            neg_q = -q_subset + epsilon  # Ensure positive values for log
            neg_q = np.maximum(neg_q, epsilon)  # Additional safety check
            transformed_q = -np.log(neg_q)
            q_subset = transformed_q

        q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow

        exp_q = np.exp(temperature * q_subset)
        if np.sum(exp_q) == 0 or np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)):
            # Fallback to uniform
            pi_softmax = np.ones(len(allowed_actions)) / len(allowed_actions)
        else:
            pi_softmax = exp_q / np.sum(exp_q)

        # Create softmax policy dict
        pi_softmax_dict = {
            allowed_actions[i]: pi_softmax[i] for i in range(len(allowed_actions))
        }

        # Uniform prior
        pi_norm = {a: 1.0 / len(allowed_actions) for a in allowed_actions}

        # Get old policy
        if key in self.policies[agent_id]:
            pi_old = self.policies[agent_id][key]
        else:
            pi_old = pi_norm.copy()

        # Smooth policy update: π_new = (1-α) * π_old + α * (ν * π_norm + (1-ν) * π_softmax)
        pi_new = {}
        for a in allowed_actions:
            mixed_policy = nu_h * pi_norm[a] + (1 - nu_h) * pi_softmax_dict[a]
            pi_new[a] = (1 - update_rate) * pi_old.get(
                a, pi_norm[a]
            ) + update_rate * mixed_policy

        # Normalize to ensure valid probability distribution
        total_prob = sum(pi_new.values())
        if total_prob > 0:
            pi_new = {a: p / total_prob for a, p in pi_new.items()}
        else:
            pi_new = pi_norm.copy()

        self.policies[agent_id][key] = pi_new

    def update_policy_direct(
        self,
        agent_id: str,
        state: Tuple,
        q_values: np.ndarray,
        temperature: float,
        goal: Optional[Tuple] = None,
    ):
        """Update policy directly from Q-values using softmax."""
        key = self._make_key(state, goal)
        allowed_actions = self.action_space_dict[agent_id]

        # Compute softmax probabilities with temperature
        q_subset = np.array([q_values[a] for a in allowed_actions])

        # For robot agents, use -log(-Qr) transformation (equation 7)
        # Check if this is a robot agent by looking for 'robot' in agent_id
        if "robot" in agent_id.lower():
            # Transform Q-values: use -log(-Qr) instead of Qr directly
            # Add small epsilon to ensure -Qr is positive for log computation
            epsilon = 1e-8
            neg_q = -q_subset + epsilon  # Ensure positive values for log
            neg_q = np.maximum(neg_q, epsilon)  # Additional safety check
            transformed_q = -np.log(neg_q)
            q_subset = transformed_q

        q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow

        exp_q = np.exp(temperature * q_subset)

        # Handle numerical issues
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
            # Fallback to uniform distribution
            probs = np.ones(len(allowed_actions)) / len(allowed_actions)
        else:
            probs = exp_q / np.sum(exp_q)

        # Create policy dictionary
        policy = {allowed_actions[i]: probs[i] for i in range(len(allowed_actions))}
        self.policies[agent_id][key] = policy

    def update_all_policies(self, temperature: float = 1.0):
        """Update all policies from current Q-values."""
        for agent_id in self.agent_ids:
            for key, q_values in self.q_tables[agent_id].items():
                # Extract state and goal from key
                if self.use_goals and isinstance(key, tuple) and len(key) == 2:
                    state, goal = key
                    self.update_policy_direct(
                        agent_id, state, q_values, temperature, goal
                    )
                elif not self.use_goals:
                    state = key
                    self.update_policy_direct(agent_id, state, q_values, temperature)

    def save_models(self, filepath: str):
        """Save the models to file."""
        data = {
            "q_tables": {aid: dict(qtable) for aid, qtable in self.q_tables.items()},
            "policies": self.policies,
            "use_goals": self.use_goals,
            "agent_ids": self.agent_ids,
            "action_space_dict": self.action_space_dict,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load_models(self, filepath: str):
        """Load models from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Reconstruct Q-tables with proper defaultdicts
        for aid, qtable_dict in data["q_tables"].items():
            if self.use_goals:
                q_table = defaultdict(lambda: np.zeros(len(Actions)))
            else:
                q_table = defaultdict(
                    lambda: np.zeros(len(self.action_space_dict[aid]))
                )

            for key_str, values in qtable_dict.items():
                try:
                    # Try to evaluate the key as a tuple
                    key = eval(key_str) if isinstance(key_str, str) else key_str
                    q_table[key] = np.array(values)
                except:
                    # Skip malformed keys
                    continue

            self.q_tables[aid] = q_table

        self.policies = data.get("policies", {aid: {} for aid in self.agent_ids})


class NetworkQLearning(QLearningBackend):
    """Neural network Q-learning backend using PyTorch."""

    def __init__(
        self,
        agent_ids: List[str],
        action_space_dict: Dict[str, List[int]],
        state_dim: int,
        debug: bool = False,
        use_goals: bool = True,
        hidden_sizes: List[int] = [128, 128],
        device: str = "cpu",
        beta_h: float = 5.0,
        policy_update_rate: float = 0.1,
        nu_h: float = 0.1,
    ):
        super().__init__(agent_ids, action_space_dict, state_dim, debug)
        self.use_goals = use_goals
        self.hidden_sizes = hidden_sizes
        self.device = torch.device(device)

        # Calculate input dimension
        if use_goals:
            # State + goal dimensions (assuming goal has same dim as state)
            self.input_dim = state_dim * 2
        else:
            self.input_dim = state_dim

        # Create networks for each agent
        self.networks = {}
        self.optimizers = {}
        self.target_networks = {}

        for aid in agent_ids:
            if use_goals:
                output_dim = len(Actions)  # For humans with goals
            else:
                output_dim = len(action_space_dict[aid])  # For robots

            # Main network
            self.networks[aid] = QNetwork(self.input_dim, output_dim, hidden_sizes).to(
                self.device
            )

            # Target network for stability
            self.target_networks[aid] = QNetwork(
                self.input_dim, output_dim, hidden_sizes
            ).to(self.device)
            self.target_networks[aid].load_state_dict(self.networks[aid].state_dict())

            # Optimizer
            self.optimizers[aid] = torch.optim.Adam(
                self.networks[aid].parameters(), lr=0.001
            )

        # Store policies for consistency
        self.policies = {aid: {} for aid in agent_ids}

        # Update target networks every N steps
        self.target_update_freq = 100
        self.update_counter = 0

    def _state_to_tensor(
        self, state: Tuple, goal: Optional[Tuple] = None, env=None
    ) -> torch.Tensor:
        """Convert state (and optional goal) to tensor. Uses full observable encoding if env is provided."""
        if env is not None:
            state_array = encode_full_observable_state(env, state)
        else:
            state_array = np.array(state, dtype=np.float32)
        if self.use_goals and goal is not None:
            goal_array = np.array(goal, dtype=np.float32)
            input_array = np.concatenate([state_array, goal_array])
        else:
            input_array = state_array
        # Pad or truncate to expected input dimension
        if len(input_array) < self.input_dim:
            input_array = np.pad(input_array, (0, self.input_dim - len(input_array)))
        elif len(input_array) > self.input_dim:
            input_array = input_array[: self.input_dim]
        return torch.FloatTensor(input_array).to(self.device)

    def get_q_values(
        self, agent_id: str, state: Tuple, goal: Optional[Tuple] = None
    ) -> np.ndarray:
        """Get Q-values for a given state (and optionally goal)."""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state, goal).unsqueeze(0)
            q_values = self.networks[agent_id](state_tensor).squeeze(0)
            return q_values.cpu().numpy()

    def update_q_values(
        self,
        agent_id: str,
        state: Tuple,
        action: int,
        target: float,
        learning_rate: float,
        goal: Optional[Tuple] = None,
    ):
        """Update Q-values for a given state-action pair."""
        state_tensor = self._state_to_tensor(state, goal).unsqueeze(0)

        # Forward pass
        q_values = self.networks[agent_id](state_tensor)
        q_value = q_values[0, action]

        # Compute loss
        target_tensor = torch.FloatTensor([target]).to(self.device)
        loss = F.mse_loss(q_value.unsqueeze(0), target_tensor)

        # Backward pass
        self.optimizers[agent_id].zero_grad()
        loss.backward()
        self.optimizers[agent_id].step()

        # Update target networks periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_networks[agent_id].load_state_dict(
                self.networks[agent_id].state_dict()
            )

    def get_policy(
        self,
        agent_id: str,
        state: Tuple,
        temperature: float,
        goal: Optional[Tuple] = None,
    ) -> Dict[int, float]:
        """Get softmax policy for a given state."""
        q_values = self.get_q_values(agent_id, state, goal)
        allowed_actions = self.action_space_dict[agent_id]

        # Compute softmax probabilities with temperature
        q_subset = np.array([q_values[a] for a in allowed_actions])

        # For robot agents, use -log(-Qr) transformation (equation 7)
        # Check if this is a robot agent by looking for 'robot' in agent_id
        if "robot" in agent_id.lower():
            # Transform Q-values: use -log(-Qr) instead of Qr directly
            # Add small epsilon to ensure -Qr is positive for log computation
            epsilon = 1e-8
            neg_q = -q_subset + epsilon  # Ensure positive values for log
            neg_q = np.maximum(neg_q, epsilon)  # Additional safety check
            transformed_q = -np.log(neg_q)
            q_subset = transformed_q

        q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow

        exp_q = np.exp(temperature * q_subset)

        # Handle numerical issues
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
            # Fallback to uniform distribution
            probs = np.ones(len(allowed_actions)) / len(allowed_actions)
        else:
            probs = exp_q / np.sum(exp_q)

        # Create policy dictionary
        return {allowed_actions[i]: probs[i] for i in range(len(allowed_actions))}

    def save_models(self, filepath: str):
        """Save the models to file."""
        checkpoint = {
            "networks": {aid: net.state_dict() for aid, net in self.networks.items()},
            "target_networks": {
                aid: net.state_dict() for aid, net in self.target_networks.items()
            },
            "optimizers": {
                aid: opt.state_dict() for aid, opt in self.optimizers.items()
            },
            "use_goals": self.use_goals,
            "agent_ids": self.agent_ids,
            "action_space_dict": self.action_space_dict,
            "input_dim": self.input_dim,
            "hidden_sizes": self.hidden_sizes,
        }

        torch.save(checkpoint, filepath)

    def load_models(self, filepath: str):
        """Load models from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        for aid in self.agent_ids:
            self.networks[aid].load_state_dict(checkpoint["networks"][aid])
            self.target_networks[aid].load_state_dict(
                checkpoint["target_networks"][aid]
            )
            self.optimizers[aid].load_state_dict(checkpoint["optimizers"][aid])


class QNetwork(nn.Module):
    """Neural network for Q-value approximation."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_sizes: List[int] = [128, 128]
    ):
        super(QNetwork, self).__init__()

        layers = []
        prev_size = input_dim

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def create_q_learning_backend(
    use_networks: bool,
    agent_ids: List[str],
    action_space_dict: Dict[str, List[int]],
    state_dim: int,
    use_goals: bool = True,
    debug: bool = False,
    **kwargs,
) -> QLearningBackend:
    """Factory function to create appropriate Q-learning backend."""
    if use_networks:
        return NetworkQLearning(
            agent_ids, action_space_dict, state_dim, debug, use_goals, **kwargs
        )
    else:
        return TabularQLearning(
            agent_ids, action_space_dict, state_dim, debug, use_goals, **kwargs
        )
