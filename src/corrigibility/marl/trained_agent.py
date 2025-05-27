from __future__ import annotations

import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL

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
        
        # Check if this is a robot agent
        if hasattr(self.iql, 'robot_agent_ids') and agent_id in self.iql.robot_agent_ids:
            # For the robot, choose the action with highest Q-value
            # Try different Q-table access patterns for different algorithm types
            q_table = None
            if hasattr(self.iql, 'Q_r_dict') and agent_id in self.iql.Q_r_dict:
                q_table = self.iql.Q_r_dict[agent_id]
            elif hasattr(self.iql, 'Q_r') and agent_id in self.iql.Q_r:
                q_table = self.iql.Q_r[agent_id]
            elif hasattr(self.iql, 'robot_agent_id') and agent_id == self.iql.robot_agent_id:
                if hasattr(self.iql, 'Q_r'):
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
            
            # select Q_h table for this agent
            if hasattr(self.iql, 'Q_h_dict') and agent_id in self.iql.Q_h_dict:
                qtable = self.iql.Q_h_dict[agent_id]
            else:
                qtable = self.iql.Q_h[agent_id]
            
            q_values = qtable[(state_tuple, goal_tuple)]
            
            # Ensure q_values are real numbers
            if np.iscomplexobj(q_values):
                q_values = np.real(q_values)
            
            exp_q = np.exp(self.iql.beta_h * q_values)
            sum_exp = np.sum(exp_q)
            
            if sum_exp == 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
                # Fallback to uniform distribution
                if hasattr(self.iql, 'action_space_humans') and agent_id in self.iql.action_space_humans:
                    space = self.iql.action_space_humans[agent_id]
                elif hasattr(self.iql, 'action_space_dict') and agent_id in self.iql.action_space_dict:
                    space = self.iql.action_space_dict[agent_id]
                else:
                    space = list(range(len(q_values)))  # Fallback to indices
                probs = np.ones(len(space)) / len(space) if space else []
            else:
                probs = exp_q / sum_exp
            
            # Ensure probabilities are real and positive
            probs = np.real(probs)
            probs = np.maximum(probs, 0)
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)
            else:
                # Fallback to uniform
                probs = np.ones(len(probs)) / len(probs)
            
            # Additional safety check for NaN
            if np.any(np.isnan(probs)):
                probs = np.ones(len(probs)) / len(probs)
            
            # For visualization, we can either sample from the distribution or take the most likely action
            # deterministic choice
            return int(np.argmax(probs))
        
        else:
            # Unknown agent
            print(f"Warning: Unknown agent ID: {agent_id}")
            return 0  # Default to a safe action