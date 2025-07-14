#!/usr/bin/env python3
"""
Intelligent cooperation algorithm that uses proper pathfinding and state-based decisions.
"""

import numpy as np
from collections import defaultdict

class IntelligentCooperationIQL:
    def __init__(
        self,
        alpha=0.5,
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
        
        print(f"Initialized Intelligent Cooperation IQL")

    def get_direction_to_target(self, from_pos, to_pos):
        """Get the direction needed to move from one position to another"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 0  # Down or Up
        else:
            return 1 if dy > 0 else 3  # Right or Left

    def get_turn_action(self, current_dir, target_dir):
        """Get the action needed to turn from current direction to target direction"""
        if current_dir == target_dir:
            return None  # Already facing correct direction
        
        # Calculate shortest turn
        diff = (target_dir - current_dir) % 4
        if diff == 1:
            return 1  # Turn right
        elif diff == 3:
            return 0  # Turn left
        else:
            return 1  # Turn right (for 180 degree turns, pick right arbitrarily)

    def get_optimal_action(self, env, agent_id, goal=None):
        """Get the optimal action for an agent based on current state and objectives"""
        agent_pos = env.agent_positions[agent_id]
        agent_dir = env.agent_dirs[agent_id]
        
        if agent_id in self.robot_agent_ids:
            return self.get_robot_optimal_action(env, agent_id, agent_pos, agent_dir)
        else:
            return self.get_human_optimal_action(env, agent_id, agent_pos, agent_dir, goal)

    def get_robot_optimal_action(self, env, robot_id, robot_pos, robot_dir):
        """Get optimal action for robot based on cooperation strategy"""
        
        # Check if robot has key
        has_key = len(env.robot_has_keys) > 0
        
        # Check if door is open
        door_open = any(d['is_open'] for d in env.doors) if env.doors else True
        
        if not has_key and env.keys:
            # Phase 1: Get key
            key_pos = tuple(env.keys[0]['pos'])
            
            # Check if adjacent to key
            deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dx, dy = deltas[robot_dir]
            front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            
            if front_pos == key_pos:
                return 3  # Pickup key
            
            # Move towards key
            target_dir = self.get_direction_to_target(robot_pos, key_pos)
            turn_action = self.get_turn_action(robot_dir, target_dir)
            
            if turn_action is not None:
                return turn_action
            else:
                return 2  # Move forward
        
        elif has_key and not door_open and env.doors:
            # Phase 2: Open door
            door_pos = tuple(env.doors[0]['pos'])
            
            # Check if adjacent to door
            deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
            dx, dy = deltas[robot_dir]
            front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            
            if front_pos == door_pos:
                return 5  # Toggle door
            
            # Move towards door
            target_dir = self.get_direction_to_target(robot_pos, door_pos)
            turn_action = self.get_turn_action(robot_dir, target_dir)
            
            if turn_action is not None:
                return turn_action
            else:
                return 2  # Move forward
        
        else:
            # Phase 3: Stay out of the way
            return 0  # Turn left (minimal action)

    def get_human_optimal_action(self, env, human_id, human_pos, human_dir, goal):
        """Get optimal action for human to reach goal"""
        goal_pos = tuple(goal)
        
        # Check if at goal
        if tuple(human_pos) == goal_pos:
            return 0  # Stay put (turn left)
        
        # Check if door blocks path and if it's open
        door_blocks_path = False
        if env.doors:
            door_pos = tuple(env.doors[0]['pos'])
            door_open = env.doors[0]['is_open']
            
            # Simple check: if door is between human and goal
            if not door_open:
                # Check if door is in direct path
                if (human_pos[0] < door_pos[0] < goal_pos[0] or
                    human_pos[1] < door_pos[1] < goal_pos[1] or
                    goal_pos[0] < door_pos[0] < human_pos[0] or
                    goal_pos[1] < door_pos[1] < human_pos[1]):
                    door_blocks_path = True
        
        # If door blocks path and robot hasn't opened it yet, wait
        if door_blocks_path:
            return 0  # Wait (turn left)
        
        # Move towards goal
        target_dir = self.get_direction_to_target(human_pos, goal_pos)
        turn_action = self.get_turn_action(human_dir, target_dir)
        
        if turn_action is not None:
            return turn_action
        else:
            return 2  # Move forward

    def get_state_tuple(self, env, agent_id, goal=None):
        """Get state representation"""
        pos = env.agent_positions[agent_id]
        dir = env.agent_dirs[agent_id]
        
        # Add environment context
        has_key = len(env.robot_has_keys) > 0
        door_open = any(d['is_open'] for d in env.doors) if env.doors else True
        
        if goal is not None:
            return (pos[0], pos[1], dir, goal[0], goal[1], has_key, door_open)
        else:
            return (pos[0], pos[1], dir, has_key, door_open)

    def train(self, environment, episodes=500, render=False):
        """Train using intelligent cooperation with Q-learning"""
        print(f"Training Intelligent Cooperation IQL for {episodes} episodes")
        
        successful_episodes = 0
        
        for episode in range(episodes):
            # Epsilon decay
            epsilon = self.epsilon_start * (1 - episode / episodes) + self.epsilon_end
            
            environment.reset()
            goal = environment.human_goals[self.human_agent_ids[0]]
            
            for step in range(50):  # Max steps per episode
                current_actions = {}
                
                # Get actions for each agent
                for agent_id in [*self.human_agent_ids, *self.robot_agent_ids]:
                    state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    
                    if np.random.random() < epsilon:
                        # Exploration: random action
                        current_actions[agent_id] = np.random.choice(self.action_space_dict[agent_id])
                    else:
                        # Get optimal action
                        optimal_action = self.get_optimal_action(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
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
                            reward += 500.0
                        if action == 5 and any(d['is_open'] for d in environment.doors):  # toggle
                            reward += 500.0
                    
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
                    successful_episodes += 1
                    if episode < 5 or episode % 100 == 0:
                        print(f"  Episode {episode}: Goal reached in {step + 1} steps!")
                    break
                
                if done:
                    break
            
            if (episode + 1) % 100 == 0:
                success_rate = successful_episodes / (episode + 1)
                print(f"  Episode {episode + 1}/{episodes}, epsilon={epsilon:.3f}, success_rate={success_rate:.1%}")

    def sample_action(self, agent_id, state, goal=None, epsilon=0.0):
        """Sample action using learned Q-values with fallback to optimal action"""
        if np.random.random() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        # Try Q-table first
        if agent_id in self.human_agent_ids:
            state_with_goal = self.get_state_tuple(self.env, agent_id, goal)
            q_values = self.Q_human[state_with_goal]
        else:
            q_values = self.Q_robot[state]
        
        # If Q-values are all zero (not learned), use optimal action
        if np.all(q_values == 0):
            return self.get_optimal_action(self.env, agent_id, goal)
        else:
            return np.argmax(q_values)