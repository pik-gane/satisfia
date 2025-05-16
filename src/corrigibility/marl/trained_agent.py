from __future__ import annotations

import numpy as np
from iql_algorithm import TwoTimescaleIQL

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
        self.iql = TwoTimescaleIQL.load_q_values(q_values_path)
        if self.iql is None:
            raise ValueError(f"Failed to load Q-values from {q_values_path}")
            
        self.G = self.iql.G  # List of goals from the loaded model
        self.goal_idx = 0  # Default to first goal
        # support multiple humans
        self.human_agent_ids = getattr(self.iql, 'human_agent_ids', [])
        
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
        
        if agent_id == self.iql.robot_agent_id:
            # For the robot, choose the action with highest Q-value
            # Q_r is now a dict per robot ID
            q_table = self.iql.Q_r.get(agent_id)
            if q_table is None:
                raise KeyError(f"No Q-table found for robot '{agent_id}'")
            # defaultdict returns default random array for unseen states
            q_values = q_table[state_tuple]
            return int(np.argmax(q_values))
        
        elif agent_id in self.human_agent_ids:
            # For the human, sample from the policy distribution
            # Use the current goal for the human (could be modified to choose most likely goal)
            current_goal = self.G[self.goal_idx]
            goal_tuple = self.iql.state_to_tuple(current_goal)
            
            # select Q_h table for this agent
            qtable = self.iql.Q_h[agent_id]
            q_values = qtable[(state_tuple, goal_tuple)]
            exp_q = np.exp(self.iql.beta_h * q_values)
            sum_exp = np.sum(exp_q)
            
            if sum_exp == 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
                # Fallback to uniform distribution
                space = self.iql.action_space_humans.get(agent_id, [])
                probs = np.ones(len(space)) / len(space) if space else []
            else:
                probs = exp_q / sum_exp
            
            # For visualization, we can either sample from the distribution or take the most likely action
            # deterministic choice
            return int(np.argmax(probs))
        
        else:
            # Unknown agent
            print(f"Warning: Unknown agent ID: {agent_id}")
            return 0  # Default to a safe action