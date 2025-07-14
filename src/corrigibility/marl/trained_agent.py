from __future__ import annotations

import numpy as np

from src.corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL


class TrainedAgent:
    """
    An agent that uses trained Q-values to make decisions.
    This agent can be used for visualization after training is complete.
    """

    def __init__(self, q_values_path="q_values.pkl"):
        """
        Initialize the trained agent by loading pre-trained Q-values.

        Args:
            q_values_path: Path to the saved Q-values file
        """
        self.iql = TwoPhaseTimescaleIQL.load_q_values(q_values_path)
        if self.iql is None:
            raise ValueError(f"Failed to load Q-values from {q_values_path}")

        self.G = self.iql.G  # List of goals from the loaded model
        self.goal_idx = 0  # Default to first goal
        # support multiple humans
        self.human_agent_ids = getattr(self.iql, "human_agent_ids", [])
        self.robot_agent_ids = getattr(self.iql, "robot_agent_ids", [])

        # Detect if this is a network-based or tabular agent
        self.is_network_based = getattr(self.iql, "network", False)

    @property
    def agent_ids(self):
        """
        Returns a list of all valid agent IDs (robots and humans) for this trained agent.
        """
        robot_ids = getattr(self.iql, "robot_agent_ids", [])
        human_ids = getattr(self.iql, "human_agent_ids", [])
        return list(robot_ids) + list(human_ids)

    def choose_action(self, observation, agent_id):
        """
        Choose an action for the given agent based on trained Q-values.

        Args:
            observation: The current observation
            agent_id: The ID of the agent (robot or human)

        Returns:
            The chosen action
        """
        state_tuple = self.iql.state_to_tuple(observation)

        # Check if this is a robot agent
        if (
            hasattr(self.iql, "robot_agent_ids")
            and agent_id in self.iql.robot_agent_ids
        ):
            # For the robot, choose the action with highest Q-value
            if self.is_network_based:
                # Use network backend for Q-value computation
                q_values = self.iql.robot_q_backend.get_q_values(agent_id, state_tuple)
            else:
                # Use tabular Q-table access
                q_table = None
                if hasattr(self.iql, "Q_r_dict") and agent_id in self.iql.Q_r_dict:
                    q_table = self.iql.Q_r_dict[agent_id]
                elif hasattr(self.iql, "Q_r") and agent_id in self.iql.Q_r:
                    q_table = self.iql.Q_r[agent_id]
                elif (
                    hasattr(self.iql, "robot_agent_id")
                    and agent_id == self.iql.robot_agent_id
                ):
                    if hasattr(self.iql, "Q_r"):
                        q_table = self.iql.Q_r.get(agent_id)

                if q_table is None:
                    raise KeyError(f"No Q-table found for robot '{agent_id}'")

                # defaultdict returns default random array for unseen states
                q_values = q_table[state_tuple]

            # Ensure q_values are real numbers
            if np.iscomplexobj(q_values):
                q_values = np.real(q_values)

            return int(np.argmax(q_values))

        elif agent_id in self.human_agent_ids:
            # For the human, sample from the policy distribution
            # Use the current goal for the human (could be modified to choose most likely goal)
            current_goal = self.G[self.goal_idx]
            goal_tuple = self.iql.state_to_tuple(current_goal)

            if self.is_network_based:
                # Use network backend for Q-value computation
                q_values = self.iql.human_q_m_backend.get_q_values(
                    agent_id, state_tuple, goal_tuple
                )
            else:
                # Use tabular Q-table access
                if hasattr(self.iql, "Q_h_dict") and agent_id in self.iql.Q_h_dict:
                    qtable = self.iql.Q_h_dict[agent_id]
                else:
                    qtable = self.iql.Q_h[agent_id]

                q_values = qtable[(state_tuple, goal_tuple)]

            # Ensure q_values are real numbers
            if np.iscomplexobj(q_values):
                q_values = np.real(q_values)

            # Use epsilon-greedy policy with converged epsilon_h_0
            if hasattr(self.iql, "epsilon_h_0"):
                epsilon = self.iql.epsilon_h_0
            else:
                epsilon = 0.1  # Fallback

            # For deterministic visualization, use greedy action (no exploration)
            # If you want some exploration, uncomment the epsilon-greedy code below
            return int(np.argmax(q_values))

            # Epsilon-greedy version (uncomment if you want exploration in visualization):
            # if np.random.random() < epsilon:
            #     if hasattr(self.iql, 'action_space_dict') and agent_id in self.iql.action_space_dict:
            #         allowed_actions = self.iql.action_space_dict[agent_id]
            #         return np.random.choice(allowed_actions)
            #     else:
            #         return np.random.choice(len(q_values))
            # else:
            #     return int(np.argmax(q_values))

        else:
            # Unknown agent
            print(f"Warning: Unknown agent ID: {agent_id}")
            return 0  # Default to a safe action

    def state_to_tuple(self, state):
        return self.iql.state_to_tuple(state)

    def sample_robot_action_phase2(self, robot_id, state):
        return self.iql.sample_robot_action_phase2(robot_id, state)

    def sample_human_action_phase2(self, human_id, state, goal):
        return self.iql.sample_human_action_phase2(human_id, state, goal)
