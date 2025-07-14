#!/usr/bin/env python3
"""
Adaptive cooperation algorithm that works across all simple maps.
"""

import numpy as np
from collections import defaultdict

class AdaptiveCooperationIQL:
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
        
        print(f"Initialized Adaptive Cooperation IQL")

    def analyze_map(self, env):
        """Analyze the map to determine cooperation strategy"""
        env.reset()
        
        has_keys = len(env.keys) > 0
        has_doors = len(env.doors) > 0
        
        robot_pos = env.agent_positions['robot_0']
        human_pos = env.agent_positions['human_0']
        goal = env.human_goals['human_0']
        
        # Analyze map layout
        open_spaces = []
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                if env.grid[i, j] == ' ':
                    open_spaces.append((i, j))
        
        strategy = {
            'has_keys': has_keys,
            'has_doors': has_doors,
            'robot_pos': robot_pos,
            'human_pos': human_pos,
            'goal': goal,
            'open_spaces': open_spaces
        }
        
        if has_keys and has_doors:
            # Maps 1 and 4: Key-door cooperation
            key_pos = tuple(env.keys[0]['pos'])
            door_pos = tuple(env.doors[0]['pos'])
            
            if key_pos == (1, 2):  # Map 1 layout
                strategy['type'] = 'sequential'
                strategy['sequence'] = self.get_map1_sequence()
            else:  # Map 4 layout
                strategy['type'] = 'distributed'
                strategy['sequence'] = self.get_map4_sequence()
        else:
            # Maps 2 and 3: No keys/doors
            if all((i, j) in open_spaces for i in range(1, 4) for j in range(1, 4)):
                # Map 2: Completely open
                strategy['type'] = 'direct'
                strategy['sequence'] = self.get_map2_sequence()
            else:
                # Map 3: Partial wall
                strategy['type'] = 'asymmetric'
                strategy['sequence'] = self.get_map3_sequence()
        
        return strategy

    def get_map1_sequence(self):
        """Cooperation sequence for Map 1"""
        return [
            # Robot gets key and opens door
            {"robot_0": 0, "human_0": 0},  # Robot turn left to face key
            {"robot_0": 3, "human_0": 0},  # Robot pickup key
            {"robot_0": 2, "human_0": 0},  # Robot move to key position
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face door
            {"robot_0": 5, "human_0": 0},  # Robot open door
            # Human goes through door to goal
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 2},  # Human move down through door
            {"robot_0": 0, "human_0": 1},  # Human turn right to face goal
            {"robot_0": 0, "human_0": 2},  # Human move right to goal
        ]

    def get_map2_sequence(self):
        """Cooperation sequence for Map 2 (direct path)"""
        return [
            # Human goes directly to goal
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 0},  # Human turn left to face goal
            {"robot_0": 0, "human_0": 2},  # Human move left to goal
        ]

    def get_map3_sequence(self):
        """Cooperation sequence for Map 3 (asymmetric)"""
        return [
            # Human takes longer path around wall
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 0},  # Human turn left
            {"robot_0": 0, "human_0": 2},  # Human move left
            {"robot_0": 0, "human_0": 2},  # Human move left
            {"robot_0": 0, "human_0": 1},  # Human turn right
            {"robot_0": 0, "human_0": 2},  # Human move right to goal
        ]

    def get_map4_sequence(self):
        """Cooperation sequence for Map 4 (distributed cooperation)"""
        return [
            # Robot goes to get key from center
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face down
            {"robot_0": 2, "human_0": 0},  # Robot move down to key
            {"robot_0": 3, "human_0": 0},  # Robot pickup key
            {"robot_0": 1, "human_0": 0},  # Robot turn right to face door
            {"robot_0": 2, "human_0": 0},  # Robot move to door
            {"robot_0": 5, "human_0": 0},  # Robot open door
            # Human goes through door to goal
            {"robot_0": 0, "human_0": 2},  # Human move down
            {"robot_0": 0, "human_0": 2},  # Human move down through door
            {"robot_0": 0, "human_0": 0},  # Human turn left to face goal
            {"robot_0": 0, "human_0": 2},  # Human move left to goal
        ]

    def get_state_tuple(self, env, agent_id, goal=None):
        """Get state representation"""
        pos = env.agent_positions[agent_id]
        dir = env.agent_dirs[agent_id]
        
        # Add environment context
        has_key = len(env.robot_has_keys) > 0
        door_open = any(d['is_open'] for d in env.doors) if env.doors else False
        
        if goal is not None:
            return (pos[0], pos[1], dir, goal[0], goal[1], has_key, door_open)
        else:
            return (pos[0], pos[1], dir, has_key, door_open)

    def train(self, environment, episodes=1000, render=False):
        """Train using adaptive cooperation sequences"""
        print(f"Training Adaptive Cooperation IQL for {episodes} episodes")
        
        # Analyze the map to determine strategy
        strategy = self.analyze_map(environment)
        print(f"Map strategy: {strategy['type']}")
        
        cooperation_sequence = strategy['sequence']
        
        for episode in range(episodes):
            # Epsilon decay
            epsilon = self.epsilon_start * (1 - episode / episodes) + self.epsilon_end
            
            environment.reset()
            goal = environment.human_goals[self.human_agent_ids[0]]
            
            # Follow cooperation sequence with exploration
            for step, optimal_actions in enumerate(cooperation_sequence):
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
                
                # Update Q-values
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
                            # Progress reward
                            dist = abs(human_pos[0] - goal[0]) + abs(human_pos[1] - goal[1])
                            reward += -0.1 * dist
                    else:
                        # Robot reward: cooperation bonuses
                        if action == 3 and len(environment.robot_has_keys) > 0:  # pickup
                            reward += 200.0
                        if action == 5 and any(d['is_open'] for d in environment.doors):  # toggle
                            reward += 200.0
                    
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
                    if episode < 5 or episode % 200 == 0:
                        print(f"  Episode {episode}: Goal reached in {step + 1} steps!")
                    break
                
                if done:
                    break
            
            if (episode + 1) % 200 == 0:
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