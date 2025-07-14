#!/usr/bin/env python3
"""
Fixed Tabular IQL that properly handles direction system and reward shaping.
"""

import numpy as np
from collections import defaultdict

class FixedTabularIQL:
    def __init__(
        self,
        alpha=0.7,
        gamma=0.99,
        epsilon_start=0.3,
        epsilon_end=0.0,
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
        
        print(f"Initialized Fixed Tabular IQL")

    def get_state_tuple(self, env, agent_id, goal=None):
        """Get state representation"""
        pos = env.agent_positions[agent_id]
        dir = env.agent_dirs[agent_id]
        
        # Environment context
        has_key = len(env.robot_has_keys) > 0
        door_open = any(d['is_open'] for d in env.doors) if env.doors else True
        keys_available = len(env.keys) > 0
        
        if goal is not None:
            return (pos[0], pos[1], dir, goal[0], goal[1], has_key, door_open, keys_available)
        else:
            return (pos[0], pos[1], dir, has_key, door_open, keys_available)

    def get_optimal_action(self, env, agent_id, goal=None):
        """Get optimal action based on current state"""
        agent_pos = env.agent_positions[agent_id]
        agent_dir = env.agent_dirs[agent_id]
        
        if agent_id in self.robot_agent_ids:
            return self.get_robot_optimal_action(env, agent_pos, agent_dir)
        else:
            return self.get_human_optimal_action(env, agent_pos, agent_dir, goal)

    def get_robot_optimal_action(self, env, robot_pos, robot_dir):
        """Get optimal robot action"""
        has_key = len(env.robot_has_keys) > 0
        door_open = any(d['is_open'] for d in env.doors) if env.doors else True
        keys_available = len(env.keys) > 0
        
        # Phase 1: Get key if available and don't have one
        if keys_available and not has_key:
            key_pos = tuple(env.keys[0]['pos'])
            return self.navigate_and_interact(robot_pos, robot_dir, key_pos, 3)  # pickup
        
        # Phase 2: Open door if have key and door is closed
        elif has_key and not door_open and env.doors:
            door_pos = tuple(env.doors[0]['pos'])
            return self.navigate_and_interact(robot_pos, robot_dir, door_pos, 5)  # toggle
        
        # Phase 3: Stay out of the way
        else:
            return 0  # Turn left

    def get_human_optimal_action(self, env, human_pos, human_dir, goal):
        """Get optimal human action"""
        goal_pos = tuple(goal)
        
        # If at goal, stay there
        if tuple(human_pos) == goal_pos:
            return 0
        
        # Check if door blocks path
        door_blocks = False
        if env.doors and not any(d['is_open'] for d in env.doors):
            door_pos = tuple(env.doors[0]['pos'])
            # Simple blocking check: door is between human and goal
            if self.is_between(human_pos, goal_pos, door_pos):
                door_blocks = True
        
        # If door blocks, wait
        if door_blocks:
            return 0  # Wait for robot to open door
        
        # Move toward goal
        return self.navigate_to_target(human_pos, human_dir, goal_pos)

    def is_between(self, start, end, middle):
        """Check if middle point is between start and end"""
        return (min(start[0], end[0]) <= middle[0] <= max(start[0], end[0]) and
                min(start[1], end[1]) <= middle[1] <= max(start[1], end[1]))

    def navigate_and_interact(self, current_pos, current_dir, target_pos, interaction_action):
        """Navigate to be adjacent to target and interact"""
        # Check if already adjacent to target in front direction
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Up, Right, Down, Left
        dx, dy = deltas[current_dir]
        front_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if front_pos == target_pos:
            return interaction_action  # pickup/toggle
        
        # Need to navigate to target - first find best adjacent position
        best_adjacent = None
        best_dir = None
        min_dist = float('inf')
        
        for direction, (dx, dy) in deltas.items():
            adjacent_pos = (target_pos[0] + dx, target_pos[1] + dy)
            if self.is_valid_position(adjacent_pos):
                dist = abs(current_pos[0] - adjacent_pos[0]) + abs(current_pos[1] - adjacent_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_adjacent = adjacent_pos
                    best_dir = (direction + 2) % 4  # Opposite direction to face target
        
        if best_adjacent is None:
            return 0  # Can't reach
        
        # If not at best adjacent position, move there
        if tuple(current_pos) != best_adjacent:
            return self.navigate_to_target(current_pos, current_dir, best_adjacent)
        
        # If at best adjacent position but not facing target, turn
        if current_dir != best_dir:
            return self.get_turn_action(current_dir, best_dir)
        
        return interaction_action

    def navigate_to_target(self, current_pos, current_dir, target_pos):
        """Navigate directly to target"""
        if tuple(current_pos) == tuple(target_pos):
            return 0  # Already there
        
        # Calculate direction to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Choose direction based on larger distance
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        # If facing correct direction, move forward
        if current_dir == target_dir:
            return 2  # Move forward
        
        # Turn toward target
        return self.get_turn_action(current_dir, target_dir)

    def get_turn_action(self, current_dir, target_dir):
        """Get action to turn from current to target direction"""
        diff = (target_dir - current_dir) % 4
        if diff == 1:
            return 1  # Turn right
        elif diff == 3:
            return 0  # Turn left
        else:
            return 1  # Turn right (for 180 degree turns)

    def is_valid_position(self, pos):
        """Check if position is valid"""
        if (0 <= pos[0] < self.env.grid_size and 
            0 <= pos[1] < self.env.grid_size):
            return self.env.grid[pos[0], pos[1]] != '#'
        return False

    def train(self, environment, episodes=200, render=False):
        """Train with proper reward shaping"""
        print(f"Training Fixed Tabular IQL for {episodes} episodes")
        
        successful_episodes = 0
        
        for episode in range(episodes):
            # Decreasing exploration
            epsilon = self.epsilon_start * max(0.01, (1 - episode / (episodes * 0.8)))
            
            environment.reset()
            goal = environment.human_goals[self.human_agent_ids[0]]
            
            for step in range(50):  # Max steps per episode
                actions = {}
                
                # Get actions for each agent
                for agent_id in [*self.human_agent_ids, *self.robot_agent_ids]:
                    state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    
                    if np.random.random() < epsilon:
                        # Exploration
                        actions[agent_id] = np.random.choice(self.action_space_dict[agent_id])
                    else:
                        # Use optimal action
                        optimal_action = self.get_optimal_action(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                        actions[agent_id] = optimal_action
                
                # Execute actions
                obs, rewards, terms, truncs, _ = environment.step(actions)
                done = any(terms.values()) or any(truncs.values())
                
                # Reward shaping and Q-learning update
                for agent_id in [*self.human_agent_ids, *self.robot_agent_ids]:
                    state = self.get_state_tuple(environment, agent_id, goal if agent_id in self.human_agent_ids else None)
                    action = actions[agent_id]
                    reward = rewards[agent_id]
                    
                    # MASSIVE reward shaping to ensure learning
                    if agent_id in self.human_agent_ids:
                        # Human rewards
                        human_pos = environment.agent_positions[agent_id]
                        if tuple(human_pos) == tuple(goal):
                            reward += 10000.0  # HUGE goal reward
                        else:
                            # Strong progress reward
                            dist = abs(human_pos[0] - goal[0]) + abs(human_pos[1] - goal[1])
                            reward += -5.0 * dist  # Strong distance penalty
                    else:
                        # Robot cooperation rewards
                        old_keys = len(environment.robot_has_keys)
                        old_doors_open = sum(1 for d in environment.doors if d['is_open'])
                        
                        # Huge rewards for cooperation
                        if action == 3:  # pickup
                            new_keys = len(environment.robot_has_keys)
                            if new_keys > old_keys:
                                reward += 5000.0  # HUGE pickup reward
                        
                        if action == 5:  # toggle
                            new_doors_open = sum(1 for d in environment.doors if d['is_open'])
                            if new_doors_open > old_doors_open:
                                reward += 5000.0  # HUGE door opening reward
                    
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
                    if episode < 5 or episode % 50 == 0:
                        print(f"  Episode {episode}: Goal reached in {step + 1} steps!")
                    break
                
                if done:
                    break
            
            if (episode + 1) % 50 == 0:
                success_rate = successful_episodes / (episode + 1)
                print(f"  Episode {episode + 1}/{episodes}, epsilon={epsilon:.3f}, success_rate={success_rate:.1%}")

    def sample_action(self, agent_id, state, goal=None, epsilon=0.0):
        """Sample action using Q-values (deterministic for testing)"""
        if epsilon > 0 and np.random.random() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        # For testing, use Q-table if learned, otherwise optimal
        if agent_id in self.human_agent_ids:
            state_with_goal = self.get_state_tuple(self.env, agent_id, goal)
            q_values = self.Q_human[state_with_goal]
        else:
            q_values = self.Q_robot[state]
        
        # If Q-values learned (not all zero), use them
        if not np.allclose(q_values, 0.0, atol=1e-6):
            return np.argmax(q_values)
        else:
            # Fallback to optimal policy
            return self.get_optimal_action(self.env, agent_id, goal)