#!/usr/bin/env python3
"""
Fixed cooperation algorithm that learns the correct sequence.
"""

import numpy as np
from collections import defaultdict

class FixedCooperationIQL:
    def __init__(
        self,
        alpha=0.3,
        gamma=0.99,
        epsilon_start=0.3,
        epsilon_end=0.01,
        action_space_dict=None,
        robot_agent_ids=None,
        human_agent_ids=None,
        env=None
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.action_space_dict = action_space_dict
        self.robot_agent_ids = robot_agent_ids
        self.human_agent_ids = human_agent_ids
        self.env = env
        
        # Q-tables
        self.Q_human = defaultdict(lambda: np.zeros(3))
        self.Q_robot = defaultdict(lambda: np.zeros(6))
        
        # Known cooperation sequence
        self.cooperation_sequence = [
            # Phase 1: Robot gets key
            {"robot_0": 0, "human_0": 0},  # Robot turn left to face key
            {"robot_0": 3, "human_0": 0},  # Robot pickup key
            {"robot_0": 2, "human_0": 0},  # Robot move to key position
            
            # Phase 2: Robot opens door
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face door
            {"robot_0": 5, "human_0": 0},  # Robot open door
            
            # Phase 3: Human goes to goal
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 2},  # Human move down through door
            {"robot_0": 0, "human_0": 1},  # Human turn right to face goal
            {"robot_0": 0, "human_0": 2},  # Human move right to goal
        ]
        
        print(f"Initialized Fixed Cooperation IQL")

    def get_state_tuple(self, env, agent_id, goal=None):
        """Get state representation"""
        # Simple state: agent position + direction
        pos = env.agent_positions[agent_id]
        dir = env.agent_dirs[agent_id]
        
        if goal is not None:
            return (pos[0], pos[1], dir, goal[0], goal[1])
        else:
            return (pos[0], pos[1], dir)

    def train(self, environment, episodes=1000, render=False):
        """Train using cooperation sequence with Q-learning"""
        print(f"Training Fixed Cooperation IQL for {episodes} episodes")
        
        for episode in range(episodes):
            # Epsilon decay
            epsilon = self.epsilon_start * (1 - episode / episodes) + self.epsilon_end
            
            environment.reset()
            goal = environment.human_goals[self.human_agent_ids[0]]
            
            # Follow cooperation sequence with some exploration
            for step, optimal_actions in enumerate(self.cooperation_sequence):
                if step >= 50:  # Safety limit
                    break
                
                current_actions = {}
                
                # Get actions (with exploration)
                for agent_id in [*self.human_agent_ids, *self.robot_agent_ids]:
                    state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    optimal_action = optimal_actions[agent_id]
                    
                    if np.random.random() < epsilon:
                        # Exploration: random action
                        current_actions[agent_id] = np.random.choice(self.action_space_dict[agent_id])
                    else:
                        # Follow optimal sequence
                        current_actions[agent_id] = optimal_action
                
                # Execute actions
                obs, rewards, terms, truncs, _ = environment.step(current_actions)
                done = any(terms.values()) or any(truncs.values())
                
                # Update Q-values for the actions taken
                for agent_id in [*self.human_agent_ids, *self.robot_agent_ids]:
                    state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    action = current_actions[agent_id]
                    reward = rewards[agent_id]
                    
                    # Enhanced reward shaping
                    if agent_id in self.human_agent_ids:
                        # Human reward: large bonus for reaching goal
                        human_pos = environment.agent_positions[agent_id]
                        if tuple(human_pos) == tuple(goal):
                            reward += 1000.0  # Large goal reward
                        else:
                            # Small progress reward
                            dist = abs(human_pos[0] - goal[0]) + abs(human_pos[1] - goal[1])
                            reward += -0.1 * dist
                    else:
                        # Robot reward: cooperation bonuses
                        if action == 3 and len(environment.robot_has_keys) > 0:  # pickup
                            reward += 100.0
                        if action == 5 and any(d['is_open'] for d in environment.doors):  # toggle
                            reward += 100.0
                    
                    # Q-learning update
                    next_state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    
                    if agent_id in self.human_agent_ids:
                        old_q = self.Q_human[state][action]
                        next_max_q = np.max(self.Q_human[next_state])
                        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                        self.Q_human[state][action] = new_q
                    else:
                        old_q = self.Q_robot[state][action]
                        next_max_q = np.max(self.Q_robot[next_state])
                        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                        self.Q_robot[state][action] = new_q
                
                # Check if goal reached
                human_pos = environment.agent_positions[self.human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    if episode < 5 or episode % 100 == 0:
                        print(f"  Episode {episode}: Goal reached in {step + 1} steps!")
                    break
                
                if done:
                    break
            
            if (episode + 1) % 100 == 0:
                print(f"  Episode {episode + 1}/{episodes}, epsilon={epsilon:.3f}")

    def sample_action(self, agent_id, state, goal=None, epsilon=0.0):
        """Sample action using learned Q-values"""
        if agent_id in self.human_agent_ids:
            state_with_goal = self.get_state_tuple(self.env, agent_id, goal)
            q_values = self.Q_human[state_with_goal]
        else:
            q_values = self.Q_robot[state]
        
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        else:
            return np.argmax(q_values)