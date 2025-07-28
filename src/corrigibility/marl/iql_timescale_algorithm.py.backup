import numpy as np
from collections import defaultdict
import random
import pickle
import os
import math
import time
import pygame
from env import Actions

class TwoPhaseTimescaleIQL:
    def __init__(self, alpha_m, alpha_e, alpha_r, gamma_h, gamma_r, beta_r_0,
                 G, mu_g, p_g, action_space_dict, robot_agent_ids, human_agent_ids,
                 eta=0.1, epsilon_h_0=0.1, epsilon_r=0.1, decay_epsilon_r_phase1=False, 
                 reward_function='power', concavity_param=1.0, debug=False,
                 zeta=1.0, xi=1.0):
        # Phase 1 parameters
        self.alpha_m = alpha_m  # Phase 1 learning rate for human models
        self.alpha_e = alpha_e  # Phase 2 fast timescale learning rate
        self.alpha_r = alpha_r  # Robot learning rate
        
        # Standard parameters
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        
        # Robot rationality parameters
        self.beta_r_0 = beta_r_0  # Target beta_r for Phase 2
        self.beta_r = 0.1  # Current beta_r (starts low, increases to beta_r_0)
        
        # Human exploration parameters
        self.epsilon_h_0 = epsilon_h_0  # Final epsilon_h for converged policy
        self.epsilon_h = 0.8  # Increased initial epsilon_h (starts high, decreases to epsilon_h_0)
        
        self.G = G
        self.mu_g = mu_g
        self.p_g = p_g
        self.eta = eta
        self.epsilon_r = 0.3  # Increased robot exploration in Phase 1
        self.debug = debug
        
        # NEW: Mathematical formulation parameters
        self.zeta = zeta  # Power parameter in X_h(s) calculation (equation Xh)
        self.xi = xi      # Power parameter in U_r(s) calculation (equation Ur)
        
        # NEW: Exploration bonus parameters
        self.exploration_bonus_initial = 50.0  # Increased initial exploration bonus
        self.exploration_bonus_decay = 0.995   # Slower decay rate
        self.current_exploration_bonus = self.exploration_bonus_initial
        
        # Visit counts for exploration bonus (robot state-action pairs)
        self.robot_visit_counts = defaultdict(int)
        
        # NEW: Human exploration bonus parameters
        self.human_exploration_bonus_initial = 75.0  # Higher human exploration bonus for goal discovery
        self.human_exploration_bonus_decay = 0.998   # Slower decay for longer exploration
        self.current_human_exploration_bonus = self.human_exploration_bonus_initial
        
        # Visit counts for human exploration bonus
        self.human_visit_counts = defaultdict(int)
        
        # Agent configuration
        self.robot_agent_ids = robot_agent_ids if isinstance(robot_agent_ids, list) else [robot_agent_ids]
        self.human_agent_ids = human_agent_ids if isinstance(human_agent_ids, list) else [human_agent_ids]
        self.robot_agent_id = self.robot_agent_ids[0]  # For compatibility
        
        # Action spaces
        self.action_space_dict = action_space_dict
        self.action_space_robot = {rid: action_space_dict[rid] for rid in self.robot_agent_ids}
        self.action_space_humans = {hid: action_space_dict[hid] for hid in self.human_agent_ids}
        
        # Q-tables for Phase 1: Human model learning (Q^m_h)
        action_dim = len(Actions)
        self.Q_m_h_dict = {hid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=action_dim))
                         for hid in self.human_agent_ids}
        
        # Q-tables for Phase 2: Robot policy learning (Q_r)
        self.Q_r_dict = {rid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=len(self.action_space_robot[rid])))
                         for rid in self.robot_agent_ids}
        
        # NEW: Mathematical formulation tables
        # Q^e_h(s,g_h,a_h) - Expected Q-values under robot policy
        self.Q_e_dict = {hid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=action_dim))
                         for hid in self.human_agent_ids}
        
        # V^m_h(s,g_h) - Value function for human model
        self.V_m_h_dict = {hid: defaultdict(float) for hid in self.human_agent_ids}
        
        # V^e_h(s,g_h) - Expected value function under robot policy
        self.V_e_h_dict = {hid: defaultdict(float) for hid in self.human_agent_ids}
        
        # X_h(s) - Aggregated human potential (equation Xh)
        self.X_h_dict = {hid: defaultdict(float) for hid in self.human_agent_ids}
        
        # U_r(s) - Robot utility function (equation Ur)
        self.U_r_dict = {rid: defaultdict(float) for rid in self.robot_agent_ids}
        
        # V_r(s) - Robot value function (equation Vr)
        self.V_r_dict = {rid: defaultdict(float) for rid in self.robot_agent_ids}
        
        # Human policies π_h(s,g_h)
        self.pi_h_dict = {hid: {} for hid in self.human_agent_ids}
        
        # Robot policies π_r(s)
        self.pi_r_dict = {rid: {} for rid in self.robot_agent_ids}
        
        # Wide prior μ_{-h}(s) representing robot's assumption on human's belief about other humans
        # For simplicity, assume uniform distribution over actions
        self.mu_minus_h = lambda state: {hid: np.ones(action_dim) / action_dim for hid in self.human_agent_ids}
        
        # NEW: Robot reward function parameters
        self.reward_function = reward_function  # 'power', 'log', 'bounded', 'generalized_bounded'
        self.concavity_param = concavity_param  # c parameter for generalized bounded function
        
        # Compatibility attributes for trained_agent.py
        self.Q_h_dict = self.Q_m_h_dict  # Backward compatibility
        self.Q_h = self.Q_m_h_dict
        self.Q_r = self.Q_r_dict
        
        # Convergence monitoring
        self.convergence_threshold = 1e-4  # Threshold for Q-value changes
        self.convergence_window = 100      # Episodes to check for convergence
        self.q_value_history = {'robot': [], 'human': []}
        self.last_q_snapshot = {'robot': {}, 'human': {}}
        self.convergence_checks = 0
        
        if self.debug:
            print(f"TwoPhaseTimescaleIQL Initialized:")
            print(f"  Robot IDs: {self.robot_agent_ids}")
            print(f"  Human IDs: {self.human_agent_ids}")
            print(f"  Goals G: {self.G}")
            print(f"  beta_r: {self.beta_r} -> {self.beta_r_0}")
            print(f"  epsilon_h: {self.epsilon_h} -> {self.epsilon_h_0}")
            print(f"  Reward function: {self.reward_function} (c={self.concavity_param})")
            print(f"  Convergence threshold: {self.convergence_threshold}")

    def extract_robot_state(self, env, robot_id: str) -> tuple:
        """
        Extract robot-specific state excluding human positions/directions.
        This includes robot position, direction, and board state (walls, doors, keys, etc.)
        but excludes human agent positions and directions.
        """
        # Get robot's observation
        robot_obs = env.observe(robot_id)
        
        # If observation is a dictionary with structured data
        if hasattr(env, 'agent_positions') and hasattr(env, 'agent_directions'):
            # Get robot position and direction
            robot_pos = env.agent_positions.get(robot_id, (0, 0))
            robot_dir = getattr(env, 'agent_directions', {}).get(robot_id, 0)
            
            # Get board state (static elements like walls, doors, keys)
            board_state = []
            if hasattr(env, 'grid') and hasattr(env.grid, 'grid'):
                # Extract grid information excluding agent positions
                for i in range(env.grid.height):
                    for j in range(env.grid.width):
                        cell = env.grid.get(i, j)
                        if cell is not None:
                            # Include object type and state but not agent positions
                            if hasattr(cell, 'type') and cell.type not in ['agent', 'robot', 'human']:
                                if hasattr(cell, 'is_open'):  # For doors
                                    board_state.append((i, j, cell.type, getattr(cell, 'is_open', False)))
                                else:
                                    board_state.append((i, j, cell.type))
            
            # Combine robot state with board state
            robot_state = (robot_pos, robot_dir, tuple(board_state))
            return robot_state
        else:
            # Fallback: use the original state_to_tuple method
            return self.state_to_tuple(robot_obs)

    def calculate_exploration_bonus(self, env, robot_id: str, action: int) -> float:
        """
        Calculate exploration bonus based on robot state-action visit counts.
        Bonus decreases as visits increase and as episodes progress.
        """
        robot_state = self.extract_robot_state(env, robot_id)
        state_action_key = (robot_state, action)
        visit_count = self.robot_visit_counts[state_action_key]
        
        # Exploration bonus = current_bonus / sqrt(1 + visit_count)
        bonus = self.current_exploration_bonus / np.sqrt(1 + visit_count)
        return bonus

    def update_visit_count(self, env, robot_id: str, action: int):
        """Update visit count for robot state-action pair."""
        robot_state = self.extract_robot_state(env, robot_id)
        state_action_key = (robot_state, action)
        self.robot_visit_counts[state_action_key] += 1

    def update_exploration_bonus(self):
        """Update exploration bonus for next episode (decay over time)."""
        self.current_exploration_bonus *= self.exploration_bonus_decay
        # Ensure it doesn't go below a minimum threshold
        self.current_exploration_bonus = max(self.current_exploration_bonus, 0.01)
        
        # Also update human exploration bonus
        self.current_human_exploration_bonus *= self.human_exploration_bonus_decay
        self.current_human_exploration_bonus = max(self.current_human_exploration_bonus, 0.01)

    def extract_human_state(self, env, human_id: str) -> tuple:
        """
        Extract human-specific state excluding robot position/direction.
        This includes human position, direction, and board state.
        """
        # Get human's observation
        human_obs = env.observe(human_id)
        
        # If observation is a dictionary with structured data
        if hasattr(env, 'agent_positions') and hasattr(env, 'agent_directions'):
            # Get human position and direction
            human_pos = env.agent_positions.get(human_id, (0, 0))
            human_dir = getattr(env, 'agent_directions', {}).get(human_id, 0)
            
            # Get board state (static elements like walls, doors, keys)
            board_state = []
            if hasattr(env, 'grid') and hasattr(env.grid, 'grid'):
                # Extract grid information excluding agent positions
                for i in range(env.grid.height):
                    for j in range(env.grid.width):
                        cell = env.grid.get(i, j)
                        if cell is not None:
                            # Include object type and state but not agent positions
                            if hasattr(cell, 'type') and cell.type not in ['agent', 'robot', 'human']:
                                if hasattr(cell, 'is_open'):  # For doors
                                    board_state.append((i, j, cell.type, getattr(cell, 'is_open', False)))
                                else:
                                    board_state.append((i, j, cell.type))
            
            # Combine human state with board state
            human_state = (human_pos, human_dir, tuple(board_state))
            return human_state
        else:
            # Fallback: use the original state_to_tuple method
            return self.state_to_tuple(human_obs)

    def calculate_human_exploration_bonus(self, env, human_id: str, action: int, goal_tuple: tuple) -> float:
        """
        Calculate exploration bonus for human based on state-action-goal visit counts.
        """
        human_state = self.extract_human_state(env, human_id)
        state_action_goal_key = (human_state, action, goal_tuple)
        visit_count = self.human_visit_counts[state_action_goal_key]
        
        # Exploration bonus = current_bonus / sqrt(1 + visit_count)
        bonus = self.current_human_exploration_bonus / np.sqrt(1 + visit_count)
        return bonus

    def update_human_visit_count(self, env, human_id: str, action: int, goal_tuple: tuple):
        """Update visit count for human state-action-goal triple."""
        human_state = self.extract_human_state(env, human_id)
        state_action_goal_key = (human_state, action, goal_tuple)
        self.human_visit_counts[state_action_goal_key] += 1

    def state_to_tuple(self, state_obs):
        """Convert observation to tuple for Q-table indexing."""
        if isinstance(state_obs, tuple):
            return tuple(int(x) for x in state_obs)
        try:
            arr = np.asarray(state_obs).flatten()
            return tuple(int(x) for x in arr)
        except Exception:
            return tuple(int(x) for x in state_obs)

    def sample_robot_action_phase1(self, robot_id: str, state_tuple: tuple, env=None) -> int:
        """
        Sample robot action using backward induction approach in Phase 1.
        Implements the min_{a_r} part of equation (Qm).
        Now includes exploration bonus.
        """
        allowed = self.action_space_dict[robot_id]
        
        if np.random.random() < self.epsilon_r:
            action = np.random.choice(allowed)
            if env is not None:
                self.update_visit_count(env, robot_id, action)
            return action
        
        # Calculate the action that minimizes expected human future value
        # This implements the "evil" robot behavior for learning human models
        min_expected_values = []
        
        for action in allowed:
            # Calculate expected negative human potential for this action
            neg_value = self.calculate_min_human_value(robot_id, state_tuple, action)
            
            # Add exploration bonus (negative because we're minimizing)
            if env is not None:
                exploration_bonus = self.calculate_exploration_bonus(env, robot_id, action)
                neg_value -= exploration_bonus  # Subtract to encourage exploration
            
            min_expected_values.append(neg_value)
        
        # Choose action that minimizes human expected value (most "evil") plus exploration
        best_action_idx = np.argmin(min_expected_values)
        chosen_action = allowed[best_action_idx]
        
        # Update visit count
        if env is not None:
            self.update_visit_count(env, robot_id, chosen_action)
            
        return chosen_action

    def calculate_min_human_value(self, robot_id: str, state_tuple: tuple, robot_action: int) -> float:
        """
        Calculate min_{a_r} E_{s'} (U_h(s',g_h) + γ_h V^m_h(s',g_h))
        This implements the backward induction for equation (Qm).
        """
        total_expected_value = 0.0
        
        # For each human
        for hid in self.human_agent_ids:
            human_expected_value = 0.0
            
            # For each goal weighted by goal probability
            for i, goal in enumerate(self.G):
                goal_tuple = self.state_to_tuple(goal)
                goal_weight = self.mu_g[i]
                
                # Approximate expected future value under this robot action
                # In a full implementation, this would simulate all possible next states
                key = (state_tuple, goal_tuple)
                if key in self.V_m_h_dict[hid]:
                    future_value = self.V_m_h_dict[hid][key]
                else:
                    # For unseen states, assume pessimistic value
                    future_value = -1.0
                
                # Add immediate utility (reward) estimate
                # This is a simplified version - in full implementation, 
                # we'd simulate the transition and get actual U_h(s',g_h)
                immediate_utility = self.estimate_human_utility(hid, state_tuple, goal_tuple, robot_action)
                
                expected_return = immediate_utility + self.gamma_h * future_value
                human_expected_value += goal_weight * expected_return
            
            total_expected_value += human_expected_value
        
        return total_expected_value

    def calculate_distance(self, state_tuple: tuple, goal_tuple: tuple) -> float:
        """Calculate Euclidean distance between state and goal."""
        if len(state_tuple) >= 2 and len(goal_tuple) >= 2:
            # Assuming first two elements are x, y coordinates
            dx = state_tuple[0] - goal_tuple[0]
            dy = state_tuple[1] - goal_tuple[1]
            return math.sqrt(dx*dx + dy*dy)
        else:
            # Fallback: Manhattan distance for any dimension
            return sum(abs(s - g) for s, g in zip(state_tuple, goal_tuple))

    def calculate_human_utility(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Calculate human utility based on inverse distance to goal.
        Returns 250 at the goal, decreasing with distance.
        """
        distance = self.calculate_distance(state_tuple, goal_tuple)
        
        if distance == 0:
            # At the goal
            return 250.0
        else:
            # Inverse distance utility with scaling
            # Use formula: 250 / (1 + distance) to ensure smooth falloff
            utility = 250.0 / (1.0 + distance)
            return utility

    def estimate_human_utility(self, human_id: str, state_tuple: tuple, goal_tuple: tuple, robot_action: int) -> float:
        """
        Estimate U_h(s',g_h) for a given robot action.
        Uses inverse distance to goal as the utility function.
        """
        # Calculate utility based on current state position relative to goal
        return self.calculate_human_utility(human_id, state_tuple, goal_tuple)

    def sample_robot_action_phase2(self, robot_id: str, state_tuple: tuple, env=None) -> int:
        """Sample robot action using softmax policy in Phase 2 with exploration bonus."""
        allowed = self.action_space_dict[robot_id]
        
        if state_tuple not in self.Q_r_dict[robot_id]:
            # Return random action for unseen states
            action = np.random.choice(allowed)
            if env is not None:
                self.update_visit_count(env, robot_id, action)
            return action
        
        q_values = self.Q_r_dict[robot_id][state_tuple].copy()
        
        # Add exploration bonuses to Q-values
        if env is not None:
            for i, action in enumerate(allowed):
                exploration_bonus = self.calculate_exploration_bonus(env, robot_id, action)
                q_values[action] += exploration_bonus
        
        # Ensure q_values are real numbers
        if np.iscomplexobj(q_values):
            q_values = np.real(q_values)
            
        # Clip extreme values to prevent overflow
        q_values = np.clip(q_values, -500, 500)
        
        # Compute softmax probabilities with current beta_r
        exp_q = np.exp(self.beta_r * q_values)
        
        # Handle numerical issues
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)):
            # Fallback to uniform distribution
            effective_probs = np.ones(len(allowed)) / len(allowed)
        else:
            sum_exp = np.sum(exp_q)
            if sum_exp == 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
                effective_probs = np.ones(len(allowed)) / len(allowed)
            else:
                effective_probs = exp_q / sum_exp
        
        # Ensure probabilities are real and positive
        effective_probs = np.real(effective_probs)
        effective_probs = np.maximum(effective_probs, 0)
        
        # Normalize if sum is not 1
        if np.sum(effective_probs) > 0:
            effective_probs = effective_probs / np.sum(effective_probs)
        else:
            effective_probs = np.ones(len(allowed)) / len(allowed)
        
        # Final safety check
        if np.any(np.isnan(effective_probs)) or np.any(np.isinf(effective_probs)):
            effective_probs = np.ones(len(allowed)) / len(allowed)
        
        # Ensure effective_probs is a real-valued float array
        effective_probs = effective_probs.astype(np.float64)
        
        idx = np.random.choice(len(allowed), p=effective_probs)
        chosen_action = allowed[idx]
        
        # Update visit count
        if env is not None:
            self.update_visit_count(env, robot_id, chosen_action)
            
        return chosen_action

    def sample_human_action_phase1(self, human_id: str, state_tuple: tuple, goal_tuple: tuple, env=None) -> int:
        """Sample human action using epsilon-greedy policy in Phase 1 with exploration bonus."""
        allowed = self.action_space_dict[human_id]
        
        if np.random.random() < self.epsilon_h:
            action = np.random.choice(allowed)
            if env is not None:
                self.update_human_visit_count(env, human_id, action, goal_tuple)
            return action
        
        if (state_tuple, goal_tuple) not in self.Q_h_dict[human_id]:
            action = np.random.choice(allowed)
            if env is not None:
                self.update_human_visit_count(env, human_id, action, goal_tuple)
            return action
        
        q_values = self.Q_h_dict[human_id][(state_tuple, goal_tuple)].copy()
        
        # Add exploration bonuses to Q-values
        if env is not None:
            for action in allowed:
                exploration_bonus = self.calculate_human_exploration_bonus(env, human_id, action, goal_tuple)
                q_values[action] += exploration_bonus
        
        q_subset = [q_values[a] for a in allowed]
        best_action_idx = np.argmax(q_subset)
        chosen_action = allowed[best_action_idx]
        
        # Update visit count
        if env is not None:
            self.update_human_visit_count(env, human_id, chosen_action, goal_tuple)
            
        return chosen_action

    def sample_human_action_phase2(self, human_id: str, state_tuple: tuple, goal_tuple: tuple, env=None) -> int:
        """Sample human action using converged epsilon-greedy policy in Phase 2 with exploration bonus."""
        allowed = self.action_space_dict[human_id]
        
        # Use converged epsilon_h_0 for final policy
        if np.random.random() < self.epsilon_h_0:
            action = np.random.choice(allowed)
            if env is not None:
                self.update_human_visit_count(env, human_id, action, goal_tuple)
            return action
        
        if (state_tuple, goal_tuple) not in self.Q_h_dict[human_id]:
            action = np.random.choice(allowed)
            if env is not None:
                self.update_human_visit_count(env, human_id, action, goal_tuple)
            return action
        
        q_values = self.Q_h_dict[human_id][(state_tuple, goal_tuple)].copy()
        
        # Add exploration bonuses to Q-values (smaller in phase 2)
        if env is not None:
            for action in allowed:
                exploration_bonus = self.calculate_human_exploration_bonus(env, human_id, action, goal_tuple)
                q_values[action] += exploration_bonus * 0.5  # Reduced exploration in phase 2
        
        q_subset = [q_values[a] for a in allowed]
        best_action_idx = np.argmax(q_subset)
        chosen_action = allowed[best_action_idx]
        
        # Update visit count
        if env is not None:
            self.update_human_visit_count(env, human_id, chosen_action, goal_tuple)
            
        return chosen_action

    def train_phase1(self, environment, phase1_episodes, render=False, render_delay=0):
        """Phase 1: Learn cautious human models using conservative Q-learning."""
        print(f"Phase 1: Learning cautious human models for {phase1_episodes} episodes")
        
        # Initialize convergence monitoring
        self.last_q_snapshot = self.take_q_value_snapshot()
        
        for episode in range(phase1_episodes):
            environment.reset()
            
            # Update epsilon_h: decay from 0.5 to epsilon_h_0
            progress = episode / max(phase1_episodes - 1, 1)
            self.epsilon_h = 0.5 * (1 - progress) + self.epsilon_h_0 * progress
            
            # Update epsilon_r: decay to 0
            self.epsilon_r = max(0.1 * (1 - progress), 0.01)
            
            # Sample initial goal for each human
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])
            
            step_count = 0
            episode_human_rewards = {hid: 0.0 for hid in self.human_agent_ids}  # Track individual human rewards
            max_steps = getattr(environment, 'max_steps', 200)
            
            while step_count < max_steps:
                # Get current states
                s_tuples = {aid: self.state_to_tuple(environment.observe(aid)) 
                           for aid in environment.possible_agents}
                
                # Goal dynamics
                for hid in self.human_agent_ids:
                    if np.random.random() < self.p_g:
                        goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                        current_goals[hid] = self.state_to_tuple(self.G[goal_idx])
                
                # Sample actions
                actions = {}
                
                # Human actions (learning)
                for hid in self.human_agent_ids:
                    actions[hid] = self.sample_human_action_phase1(hid, s_tuples[hid], current_goals[hid], environment)
                
                # Robot actions (pessimistic policy)
                for rid in self.robot_agent_ids:
                    actions[rid] = self.sample_robot_action_phase1(rid, s_tuples[rid], environment)
                
                # Environment step
                next_obs, rewards, terms, truncs, infos = environment.step(actions)
                
                if render:
                    environment.render()
                    pygame.time.delay(render_delay)
                
                # Get next states
                next_s_tuples = {aid: self.state_to_tuple(next_obs[aid]) 
                               for aid in environment.possible_agents}
                
                episode_done = any(terms.values()) or any(truncs.values())
                
                # Update human Q-values (conservative)
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    episode_human_rewards[hid] += reward  # Track individual rewards
                    self.update_human_q_phase1(
                        hid, s_tuples[hid], current_goals[hid], actions[hid],
                        reward, next_s_tuples[hid], episode_done
                    )
                
                # NEW: Update Q_e values in Phase 1 as well
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    self.update_q_e(
                        hid, s_tuples[hid], current_goals[hid], actions[hid],
                        reward, next_s_tuples[hid], episode_done
                    )
                
                step_count += 1
                if episode_done:
                    break
            
            # Update exploration bonus at the end of each episode
            self.update_exploration_bonus()
            
            # Print progress and check convergence every 100 episodes
            if (episode + 1) % self.convergence_window == 0 or episode + 1 == phase1_episodes:
                # Calculate average rewards for each human individually
                avg_human_rewards = {hid: episode_human_rewards[hid] / max(step_count, 1) for hid in self.human_agent_ids}
                
                # Format individual human rewards for logging
                human_rewards_str = ", ".join([f"{hid}={avg_human_rewards[hid]:.2f}" for hid in self.human_agent_ids])
                print(f"[PHASE1] Episode {episode + 1}/{phase1_episodes}: humans=({human_rewards_str}), robot=PESSIMISTIC, ε_h={self.epsilon_h:.3f}")
                
                # Q-value change analysis
                if episode >= self.convergence_window - 1:  # Only after sufficient episodes
                    current_snapshot = self.take_q_value_snapshot()
                    changes = self.calculate_q_value_changes(self.last_q_snapshot, current_snapshot)
                    
                    if self.debug:
                        converged = self.log_q_value_changes(episode + 1, "PHASE1", changes)
                        if converged:
                            print(f"🎯 Early convergence detected at episode {episode + 1}!")
                    else:
                        # Brief summary even without debug
                        robot_converged, human_converged = self.check_convergence(changes)
                        if robot_converged and human_converged:
                            print(f"🎯 Q-values converged at episode {episode + 1}")
                    
                    # Update snapshot for next comparison
                    self.last_q_snapshot = current_snapshot

    def train_phase2(self, environment, phase2_episodes, render=False, render_delay=0):
        """Phase 2: Learn robot policy using learned human models."""
        print(f"Phase 2: Learning robot policy for {phase2_episodes} episodes")
        print(f"Robot reward function: {self.reward_function} (concavity_param={self.concavity_param})")
        
        # Reset convergence monitoring for Phase 2
        self.last_q_snapshot = self.take_q_value_snapshot()
        
        for episode in range(phase2_episodes):
            environment.reset()
            
            # Update beta_r: increase from 0.1 to beta_r_0
            progress = episode / max(phase2_episodes - 1, 1)
            self.beta_r = 0.1 + (self.beta_r_0 - 0.1) * progress
            
            # Sample initial goal for each human
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])
            
            step_count = 0
            episode_robot_reward = 0.0
            episode_human_rewards = {hid: 0.0 for hid in self.human_agent_ids}  # Track individual human rewards
            max_steps = getattr(environment, 'max_steps', 200)
            
            while step_count < max_steps:
                # Get current states - consistent observation for all agents
                s_tuples = {aid: self.state_to_tuple(environment.observe(aid)) 
                           for aid in environment.possible_agents}
                
                # Goal dynamics
                for hid in self.human_agent_ids:
                    if np.random.random() < self.p_g:
                        goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                        current_goals[hid] = self.state_to_tuple(self.G[goal_idx])
                
                # Sample actions
                actions = {}
                
                # Robot actions (learning with softmax)
                for rid in self.robot_agent_ids:
                    actions[rid] = self.sample_robot_action_phase2(rid, s_tuples[rid], environment)
                
                # Human actions (using converged policy)
                for hid in self.human_agent_ids:
                    actions[hid] = self.sample_human_action_phase2(hid, s_tuples[hid], current_goals[hid], environment)
                
                # Environment step
                next_obs, rewards, terms, truncs, infos = environment.step(actions)
                
                if render:
                    environment.render()
                    pygame.time.delay(render_delay)
                
                # Get next states - consistent observation for all agents
                next_s_tuples = {aid: self.state_to_tuple(next_obs[aid]) 
                               for aid in environment.possible_agents}
                
                episode_done = any(terms.values()) or any(truncs.values())
                
                # Calculate robot reward using the mathematical formulation
                robot_reward = self.calculate_robot_reward_new(next_s_tuples, current_goals, episode_done)
                episode_robot_reward += robot_reward
                
                # Update robot Q-values
                for rid in self.robot_agent_ids:
                    self.update_robot_q_phase2(
                        rid, s_tuples[rid], actions[rid], robot_reward,
                        next_s_tuples[rid], episode_done
                    )
                
                # Fast timescale human updates AND Q_e updates
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    episode_human_rewards[hid] += reward  # Track individual rewards
                    
                    # Update both Q_h and Q_e
                    self.update_human_q_phase2(
                        hid, s_tuples[hid], current_goals[hid], actions[hid],
                        reward, next_s_tuples[hid], episode_done
                    )
                    self.update_q_e(
                        hid, s_tuples[hid], current_goals[hid], actions[hid],
                        reward, next_s_tuples[hid], episode_done
                    )
                
                step_count += 1
                if episode_done:
                    break
            
            # Update exploration bonus at the end of each episode
            self.update_exploration_bonus()
            
            # Print progress and check convergence every 100 episodes
            if (episode + 1) % self.convergence_window == 0 or episode + 1 == phase2_episodes:
                avg_robot_reward = episode_robot_reward / max(step_count, 1)
                # Calculate average rewards for each human individually
                avg_human_rewards = {hid: episode_human_rewards[hid] / max(step_count, 1) for hid in self.human_agent_ids}
                
                # Format individual human rewards for logging
                human_rewards_str = ", ".join([f"{hid}={avg_human_rewards[hid]:.2f}" for hid in self.human_agent_ids])
                print(f"[PHASE2] Episode {episode + 1}/{phase2_episodes}: humans=({human_rewards_str}), robot={avg_robot_reward:.2f}, β_r={self.beta_r:.3f}")
                
                # Q-value change analysis
                if episode >= self.convergence_window - 1:  # Only after sufficient episodes
                    current_snapshot = self.take_q_value_snapshot()
                    changes = self.calculate_q_value_changes(self.last_q_snapshot, current_snapshot)
                    
                    if self.debug:
                        converged = self.log_q_value_changes(episode + 1, "PHASE2", changes)
                        if converged:
                            print(f"🎯 Early convergence detected at episode {episode + 1}!")
                    else:
                        # Brief summary even without debug
                        robot_converged, human_converged = self.check_convergence(changes)
                        if robot_converged and human_converged:
                            print(f"🎯 Q-values converged at episode {episode + 1}")
                    
                    # Update snapshot for next comparison
                    self.last_q_snapshot = current_snapshot

    def update_human_q_phase1(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done):
        """
        Update Q^m_h using backward induction approach (equation Qm).
        Q^m_h(s,g_h,a_h) ← E_{a_{-h}~μ_{-h}(s)} min_{a_r∈A_r(s)} E_{s'~s,a} (U_h(s',g_h) + γ_h V^m_h(s',g_h))
        """
        key = (state_tuple, goal_tuple)
        current_q = self.Q_m_h_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            # Calculate V^m_h(s',g_h) using equation (Vm)
            next_key = (next_state_tuple, goal_tuple)
            next_v_m_h = self.compute_v_m_h(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_m_h
        
        self.Q_m_h_dict[human_id][key][action] += self.alpha_m * (target - current_q)
        
        # Update π_h(s,g_h) using ε-greedy policy (equation pih)
        self.update_pi_h(human_id, state_tuple, goal_tuple)
        
        # Update V^m_h(s,g_h) using equation (Vm)
        self.update_v_m_h(human_id, state_tuple, goal_tuple)

    def compute_v_m_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Compute V^m_h(s,g_h) = E_{a_h~π_h(s,g_h)} Q^m_h(s,g_h,a_h) (equation Vm)
        """
        key = (state_tuple, goal_tuple)
        if key not in self.Q_m_h_dict[human_id]:
            return 0.0
        
        q_values = self.Q_m_h_dict[human_id][key]
        allowed_actions = self.action_space_dict[human_id]
        
        # Get current policy π_h(s,g_h)
        policy = self.get_pi_h(human_id, state_tuple, goal_tuple)
        
        # Calculate expected value
        expected_value = sum(policy.get(a, 0.0) * q_values[a] for a in allowed_actions)
        return expected_value

    def update_pi_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple):
        """Update π_h(s,g_h) ← ε_h-greedy policy for Q^m_h(s,g_h,·) (equation pih)"""
        key = (state_tuple, goal_tuple)
        if key not in self.Q_m_h_dict[human_id]:
            return
        
        q_values = self.Q_m_h_dict[human_id][key]
        allowed_actions = self.action_space_dict[human_id]
        
        # Find best action
        q_subset = [q_values[a] for a in allowed_actions]
        best_action_idx = np.argmax(q_subset)
        best_action = allowed_actions[best_action_idx]
        
        # Create ε-greedy policy
        policy = {}
        for a in allowed_actions:
            if a == best_action:
                policy[a] = (1.0 - self.epsilon_h) + self.epsilon_h / len(allowed_actions)
            else:
                policy[a] = self.epsilon_h / len(allowed_actions)
        
        self.pi_h_dict[human_id][key] = policy

    def get_pi_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> dict:
        """Get π_h(s,g_h) policy"""
        key = (state_tuple, goal_tuple)
        if key in self.pi_h_dict[human_id]:
            return self.pi_h_dict[human_id][key]
        
        # Default to uniform policy
        allowed_actions = self.action_space_dict[human_id]
        return {a: 1.0 / len(allowed_actions) for a in allowed_actions}

    def update_v_m_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple):
        """Update V^m_h(s,g_h) using equation (Vm)"""
        key = (state_tuple, goal_tuple)
        self.V_m_h_dict[human_id][key] = self.compute_v_m_h(human_id, state_tuple, goal_tuple)

    def update_human_q_phase2(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done):
        """Fast timescale update for human Q-values in Phase 2, keeping Q^m_h updated."""
        key = (state_tuple, goal_tuple)
        current_q = self.Q_h_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            next_key = (next_state_tuple, goal_tuple)
            next_v_m_h = self.compute_v_m_h(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_m_h
        
        # Use fast learning rate for Phase 2 updates
        self.Q_h_dict[human_id][key][action] += self.alpha_e * (target - current_q)
        
        # Also update the mathematical tables
        self.update_pi_h(human_id, state_tuple, goal_tuple)
        self.update_v_m_h(human_id, state_tuple, goal_tuple)

    def update_robot_q_phase2(self, robot_id, state_tuple, action, reward, next_state_tuple, done):
        """
        Update robot Q-values in Phase 2 using equation (Qr):
        Q_r(s,a_r) ← E_g E_{a_H~π_H(s,g)} E_{s'~s,a} γ_r V_r(s')
        """
        current_q = self.Q_r_dict[robot_id][state_tuple][action]
        
        if done:
            target = reward
        else:
            # Calculate V_r(s') using equation (Vr)
            next_v_r = self.compute_v_r(robot_id, next_state_tuple)
            target = reward + self.gamma_r * next_v_r
        
        self.Q_r_dict[robot_id][state_tuple][action] += self.alpha_r * (target - current_q)
        
        # Update π_r(s) using β_r-softmax policy (equation pir)
        self.update_pi_r(robot_id, state_tuple)
        
        # Update V_r(s) using equation (Vr)
        self.update_v_r(robot_id, state_tuple)

    def compute_v_r(self, robot_id: str, state_tuple: tuple) -> float:
        """
        Compute V_r(s) using equation (Vr):
        V_r(s) = U_r(s) + E_{a_r~π_r(s)} Q_r(s,a_r)
        """
        # Get U_r(s) - robot utility function
        U_r = self.U_r_dict[robot_id].get(state_tuple, 0.0)
        
        # Get Q_r values for this state
        if state_tuple not in self.Q_r_dict[robot_id]:
            return U_r
        
        q_values = self.Q_r_dict[robot_id][state_tuple]
        allowed_actions = self.action_space_dict[robot_id]
        
        # Get robot policy π_r(s)
        policy = self.get_pi_r(robot_id, state_tuple)
        
        # Calculate expected Q-value under policy
        expected_q = sum(policy.get(a, 0.0) * q_values[a] for a in allowed_actions)
        
        return U_r + expected_q

    def update_pi_r(self, robot_id: str, state_tuple: tuple):
        """Update π_r(s) ← β_r-softmax policy for Q_r(s,·) (equation pir)"""
        if state_tuple not in self.Q_r_dict[robot_id]:
            return
        
        q_values = self.Q_r_dict[robot_id][state_tuple]
        allowed_actions = self.action_space_dict[robot_id]
        
        # Compute softmax probabilities with current beta_r
        q_subset = np.array([q_values[a] for a in allowed_actions])
        q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow
        
        exp_q = np.exp(self.beta_r * q_subset)
        
        # Handle numerical issues
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
            # Fallback to uniform distribution
            probs = np.ones(len(allowed_actions)) / len(allowed_actions)
        else:
            probs = exp_q / np.sum(exp_q)
        
        # Create policy dictionary
        policy = {allowed_actions[i]: probs[i] for i in range(len(allowed_actions))}
        self.pi_r_dict[robot_id][state_tuple] = policy

    def get_pi_r(self, robot_id: str, state_tuple: tuple) -> dict:
        """Get π_r(s) policy"""
        if state_tuple in self.pi_r_dict[robot_id]:
            return self.pi_r_dict[robot_id][state_tuple]
        
        # Default to uniform policy
        allowed_actions = self.action_space_dict[robot_id]
        return {a: 1.0 / len(allowed_actions) for a in allowed_actions}

    def update_v_r(self, robot_id: str, state_tuple: tuple):
        """Update V_r(s) using equation (Vr)"""
        self.V_r_dict[robot_id][state_tuple] = self.compute_v_r(robot_id, state_tuple)

    def calculate_robot_reward_new(self, next_states, current_goals, done):
        """
        Calculate robot reward using the mathematical formulation:
        U_r(s) = -((∑_h X_h(s)^{-ξ})^η)  (equation Ur)
        where X_h(s) = ∑_{g_h∈G_h} V^e_h(s,g_h)^ζ  (equation Xh)
        """
        if done:
            return 0.0
        
        if self.debug:
            print(f"\n🔧 ROBOT REWARD CALCULATION (Mathematical Formulation):")
            print(f"  Next states: {next_states}")
            print(f"  ζ (zeta): {self.zeta}, ξ (xi): {self.xi}, η (eta): {self.eta}")
        
        # Calculate X_h(s) for each human (equation Xh)
        X_h_values = []
        
        for hid in self.human_agent_ids:
            next_state_hid = next_states.get(hid)
            if next_state_hid is None:
                continue
            
            # Calculate X_h(s) = ∑_{g_h∈G_h} V^e_h(s,g_h)^ζ
            X_h = 0.0
            for i, goal in enumerate(self.G):
                goal_tuple = self.state_to_tuple(goal)
                
                # Compute V^e_h(s,g_h) (equation Ve)
                V_e_h = self.compute_v_e_h(hid, next_state_hid, goal_tuple)
                
                # Apply power ζ and add to sum
                V_e_h_safe = max(V_e_h, 0.0)  # Ensure non-negative for power
                X_h += V_e_h_safe ** self.zeta
                
                if self.debug:
                    print(f"    Human {hid}, Goal {i}: V^e_h = {V_e_h:.3f}, V^e_h^ζ = {V_e_h_safe ** self.zeta:.3f}")
            
            X_h_values.append(X_h)
            self.X_h_dict[hid][next_state_hid] = X_h  # Store for future use
            
            if self.debug:
                print(f"    X_{hid}(s) = {X_h:.3f}")
        
        if len(X_h_values) == 0:
            return 0.0
        
        # Calculate U_r(s) = -((∑_h X_h(s)^(-ξ))^η
        # Add small epsilon to avoid division by zero when X_h is 0
        epsilon = 1e-8
        U_r = 0.0
        for X_h in X_h_values:
            safe_X_h = max(X_h, epsilon)
            U_r += safe_X_h ** (-self.xi)
        
        U_r = -(U_r ** self.eta)
        
        if self.debug:
            print(f"  U_r(s) = {U_r:.4f}")
        
        return U_r

    def compute_v_e_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Compute V^e_h(s,g_h) using the expected Q-values under robot policy:
        V^e_h(s,g_h) = E_{a_r~π_r(s)} Q^e_h(s,g_h,a_h) (equation Ve)
        """
        key = (state_tuple, goal_tuple)
        if key not in self.Q_e_dict[human_id]:
            return 0.0
        
        q_values = self.Q_e_dict[human_id][key]
        allowed_actions = self.action_space_dict[human_id]
        
        # Get robot policy π_r(s) for action selection
        robot_policy = self.get_pi_r(self.robot_agent_id, state_tuple)
        
        # Calculate expected Q-value under robot policy
        expected_q = 0.0
        for a_r, prob in robot_policy.items():
            if a_r in q_values:
                expected_q += prob * q_values[a_r]
        
        return expected_q

    def update_q_e(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done):
        """
        Update Q^e_h(s,g_h,a_h) using the Bellman equation for expected Q-values:
        Q^e_h(s,g_h,a_h) ← E_{s'} (U_h(s',g_h) + γ_h V^e_h(s',g_h))
        """
        key = (state_tuple, goal_tuple)
        current_q_e = self.Q_e_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            # Calculate V^e_h(s',g_h) using the expected value function
            next_key = (next_state_tuple, goal_tuple)
            next_v_e_h = self.compute_v_e_h(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_e_h
        
        self.Q_e_dict[human_id][key][action] += self.alpha_e * (target - current_q_e)

    def take_q_value_snapshot(self):
        """Take a snapshot of Q-values for convergence checking."""
        robot_snapshot = {rid: defaultdict(float) for rid in self.robot_agent_ids}
        human_snapshot = {hid: defaultdict(float) for hid in self.human_agent_ids}
        
        # Average Q-values over all state-action pairs for each agent
        for rid in self.robot_agent_ids:
            for state_action, q_value in self.Q_r_dict[rid].items():
                robot_snapshot[rid][state_action] = np.mean(q_value)
        
        for hid in self.human_agent_ids:
            for state_action, q_value in self.Q_m_h_dict[hid].items():
                human_snapshot[hid][state_action] = np.mean(q_value)
        
        return {'robot': robot_snapshot, 'human': human_snapshot}

    def calculate_q_value_changes(self, snapshot1, snapshot2):
        """Calculate the maximum change in Q-values between two snapshots."""
        max_change_robot = 0.0
        max_change_human = 0.0
        
        # Robot Q-values
        for rid in self.robot_agent_ids:
            for state_action in snapshot1['robot'][rid]:
                old_value = snapshot1['robot'][rid][state_action]
                new_value = snapshot2['robot'][rid].get(state_action, old_value)
                max_change_robot = max(max_change_robot, abs(new_value - old_value))
        
        # Human Q-values
        for hid in self.human_agent_ids:
            for state_action in snapshot1['human'][hid]:
                old_value = snapshot1['human'][hid][state_action]
                new_value = snapshot2['human'][hid].get(state_action, old_value)
                max_change_human = max(max_change_human, abs(new_value - old_value))
        
        return {'robot': max_change_robot, 'human': max_change_human}

    def check_convergence(self, changes):
        """Check if Q-values have converged for robot and human."""
        robot_converged = changes['robot'] < self.convergence_threshold
        human_converged = changes['human'] < self.convergence_threshold
        return robot_converged, human_converged

    def log_q_value_changes(self, episode, phase, changes):
        """Log Q-value changes for debugging and convergence monitoring."""
        if phase == "PHASE1":
            self.q_value_history['human'].append(changes['human'])
            if len(self.q_value_history['human']) > 100:
                self.q_value_history['human'].pop(0)
            
            # Check for convergence in the last 100 episodes
            if len(self.q_value_history['human']) == 100:
                avg_change = np.mean(self.q_value_history['human'])
                if avg_change < self.convergence_threshold:
                    return True
        
        elif phase == "PHASE2":
            self.q_value_history['robot'].append(changes['robot'])
            if len(self.q_value_history['robot']) > 100:
                self.q_value_history['robot'].pop(0)
            
            # Check for convergence in the last 100 episodes
            if len(self.q_value_history['robot']) == 100:
                avg_change = np.mean(self.q_value_history['robot'])
                if avg_change < self.convergence_threshold:
                    return True
        
        return False

    def train(self, environment, phase1_episodes, phase2_episodes, render=False, render_delay=0):
        """
        Complete training method that runs both Phase 1 and Phase 2.
        """
        print(f"Starting Two-Phase Timescale IQL training...")
        print(f"Phase 1: {phase1_episodes} episodes")
        print(f"Phase 2: {phase2_episodes} episodes")
        
        # Phase 1: Learn human models
        self.train_phase1(environment, phase1_episodes, render, render_delay)
        
        # Phase 2: Learn robot policy
        self.train_phase2(environment, phase2_episodes, render, render_delay)
        
        print("Two-Phase Timescale IQL training completed!")

    def save_models(self, filepath: str):
        """Save Q-values and model state to file."""
        import pickle
        
        data = {
            'Q_m_h_dict': dict(self.Q_m_h_dict),
            'Q_r_dict': dict(self.Q_r_dict), 
            'robot_agent_ids': self.robot_agent_ids,
            'human_agent_ids': self.human_agent_ids,
            'action_space_dict': self.action_space_dict,
            'G': self.G,
            'exploration_bonus_initial': self.exploration_bonus_initial,
            'exploration_bonus_decay': self.exploration_bonus_decay,
            'current_exploration_bonus': self.current_exploration_bonus,
            'robot_visit_counts': dict(self.robot_visit_counts),
            'human_exploration_bonus_initial': self.human_exploration_bonus_initial,
            'human_exploration_bonus_decay': self.human_exploration_bonus_decay,
            'current_human_exploration_bonus': self.current_human_exploration_bonus,
            'human_visit_counts': dict(self.human_visit_counts)
        }
        
        # Convert defaultdicts to regular dicts for pickling
        for agent_id in self.Q_m_h_dict:
            data['Q_m_h_dict'][agent_id] = dict(self.Q_m_h_dict[agent_id])
        for agent_id in self.Q_r_dict:
            data['Q_r_dict'][agent_id] = dict(self.Q_r_dict[agent_id])
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Q-values saved to {filepath}")

    @classmethod  
    def load_q_values(cls, filepath: str):
        """Load Q-values and model state from file."""
        import pickle
        import os
        
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist")
            return None
            
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Create instance with basic parameters
            instance = cls(
                alpha_m=0.1, alpha_e=0.1, alpha_r=0.1,
                gamma_h=0.9, gamma_r=0.9, beta_r_0=5.0,
                G=data.get('G', [(3, 2)]), mu_g=[1.0], p_g=[1.0],
                action_space_dict=data.get('action_space_dict', {}),
                robot_agent_ids=data.get('robot_agent_ids', ['robot_0']),
                human_agent_ids=data.get('human_agent_ids', ['human_0'])
            )
            
            # Load the Q-tables
            instance.Q_m_h_dict = data.get('Q_m_h_dict', {})
            instance.Q_r_dict = data.get('Q_r_dict', {})
            instance.current_exploration_bonus = data.get('current_exploration_bonus', 50.0)
            instance.robot_visit_counts = data.get('robot_visit_counts', {})
            instance.current_human_exploration_bonus = data.get('current_human_exploration_bonus', 20.0)
            instance.human_visit_counts = data.get('human_visit_counts', {})
            
            # Backward compatibility
            instance.Q_h_dict = instance.Q_m_h_dict
            instance.Q_h = instance.Q_m_h_dict
            instance.Q_r = instance.Q_r_dict
            
            print(f"Q-values loaded from {filepath}")
            return instance
            
        except Exception as e:
            print(f"Error loading Q-values from {filepath}: {e}")
            return None
