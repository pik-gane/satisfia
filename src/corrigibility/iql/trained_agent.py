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
            q_values = self.iql.Q_r[state_tuple]
            return int(np.argmax(q_values))
        
        elif agent_id == self.iql.human_agent_id:
            # For the human, sample from the policy distribution
            # Use the current goal for the human (could be modified to choose most likely goal)
            current_goal = self.G[self.goal_idx]
            goal_tuple = self.iql.state_to_tuple(current_goal)
            
            # Get probability distribution over actions
            q_values = self.iql.Q_h[(state_tuple, goal_tuple)]
            exp_q = np.exp(self.iql.beta_h * q_values)
            sum_exp = np.sum(exp_q)
            
            if sum_exp == 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
                # Fallback to uniform distribution
                probs = np.ones(len(self.iql.action_space_human)) / len(self.iql.action_space_human)
            else:
                probs = exp_q / sum_exp
            
            # For visualization, we can either sample from the distribution or take the most likely action
            # return np.random.choice(self.iql.action_space_human, p=probs)  # Sampling approach
            return int(np.argmax(probs))  # Most likely action approach
        
        else:
            # Unknown agent
            print(f"Warning: Unknown agent ID: {agent_id}")
            return 0  # Default to a safe action