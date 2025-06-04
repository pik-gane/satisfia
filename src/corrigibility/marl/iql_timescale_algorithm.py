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
        self.epsilon_h = 0.5  # Current epsilon_h (starts high, decreases to epsilon_h_0)
        
        self.G = G
        self.mu_g = mu_g
        self.p_g = p_g
        self.eta = eta
        self.epsilon_r = epsilon_r  # Robot exploration in Phase 1
        self.debug = debug
        
        # NEW: Mathematical formulation parameters
        self.zeta = zeta  # Power parameter in X_h(s) calculation (equation Xh)
        self.xi = xi      # Power parameter in U_r(s) calculation (equation Ur)
        
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

    def state_to_tuple(self, state_obs):
        """Convert observation to tuple for Q-table indexing."""
        if isinstance(state_obs, tuple):
            return tuple(int(x) for x in state_obs)
        try:
            arr = np.asarray(state_obs).flatten()
            return tuple(int(x) for x in arr)
        except Exception:
            return tuple(int(x) for x in state_obs)

    def sample_robot_action_phase1(self, robot_id: str, state_tuple: tuple) -> int:
        """
        Sample robot action using backward induction approach in Phase 1.
        Implements the min_{a_r} part of equation (Qm).
        """
        allowed = self.action_space_dict[robot_id]
        
        if np.random.random() < self.epsilon_r:
            return np.random.choice(allowed)
        
        # Calculate the action that minimizes expected human future value
        # This implements the "evil" robot behavior for learning human models
        min_expected_values = []
        
        for action in allowed:
            # Calculate expected negative human potential for this action
            neg_value = self.calculate_min_human_value(robot_id, state_tuple, action)
            min_expected_values.append(neg_value)
        
        # Choose action that minimizes human expected value (most "evil")
        best_action_idx = np.argmin(min_expected_values)
        return allowed[best_action_idx]

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

    def sample_robot_action_phase2(self, robot_id: str, state_tuple: tuple) -> int:
        """Sample robot action using softmax policy in Phase 2."""
        allowed = self.action_space_dict[robot_id]
        
        if state_tuple not in self.Q_r_dict[robot_id]:
            # Return random action for unseen states
            return np.random.choice(allowed)
        
        q_values = self.Q_r_dict[robot_id][state_tuple]
        
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
        return allowed[idx]

    def sample_human_action_phase1(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> int:
        """Sample human action using epsilon-greedy policy in Phase 1."""
        allowed = self.action_space_dict[human_id]
        
        if np.random.random() < self.epsilon_h:
            return np.random.choice(allowed)
        
        if (state_tuple, goal_tuple) not in self.Q_h_dict[human_id]:
            return np.random.choice(allowed)
        
        q_values = self.Q_h_dict[human_id][(state_tuple, goal_tuple)]
        q_subset = [q_values[a] for a in allowed]
        best_action_idx = np.argmax(q_subset)
        return allowed[best_action_idx]

    def sample_human_action_phase2(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> int:
        """Sample human action using converged epsilon-greedy policy in Phase 2."""
        allowed = self.action_space_dict[human_id]
        
        # Use converged epsilon_h_0 for final policy
        if np.random.random() < self.epsilon_h_0:
            return np.random.choice(allowed)
        
        if (state_tuple, goal_tuple) not in self.Q_h_dict[human_id]:
            return np.random.choice(allowed)
        
        q_values = self.Q_h_dict[human_id][(state_tuple, goal_tuple)]
        q_subset = [q_values[a] for a in allowed]
        best_action_idx = np.argmax(q_subset)
        return allowed[best_action_idx]

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
                    actions[hid] = self.sample_human_action_phase1(hid, s_tuples[hid], current_goals[hid])
                
                # Robot actions (pessimistic policy)
                for rid in self.robot_agent_ids:
                    actions[rid] = self.sample_robot_action_phase1(rid, s_tuples[rid])
                
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
                    actions[rid] = self.sample_robot_action_phase2(rid, s_tuples[rid])
                
                # Human actions (using converged policy)
                for hid in self.human_agent_ids:
                    actions[hid] = self.sample_human_action_phase2(hid, s_tuples[hid], current_goals[hid])
                
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
        
        # Calculate U_r(s) = -((∑_h X_h(s)^{-ξ})^η)  (equation Ur)
        sum_X_h_neg_xi = 0.0
        for X_h in X_h_values:
            if X_h > 0:
                sum_X_h_neg_xi += X_h ** (-self.xi)
            else:
                sum_X_h_neg_xi += 1e6  # Large penalty for zero potential
        
        U_r = -((sum_X_h_neg_xi) ** self.eta)
        
        if self.debug:
            print(f"  ∑_h X_h(s)^(-ξ) = {sum_X_h_neg_xi:.3f}")
            print(f"  U_r(s) = {U_r:.3f}")
        
        # Store U_r for all robot agents
        for rid in self.robot_agent_ids:
            next_state_rid = next_states.get(rid)
            if next_state_rid is not None:
                self.U_r_dict[rid][next_state_rid] = U_r
        
        # Ensure finite result
        if not np.isfinite(U_r):
            if self.debug:
                print(f"  ⚠️ Non-finite reward, returning 0.0")
            return 0.0
        
        # Clip to reasonable range
        clipped_reward = np.clip(U_r, -1000, 1000)
        
        if self.debug:
            print(f"  Final clipped reward: {clipped_reward:.3f}")
        
        return clipped_reward

    def compute_v_e_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Compute V^e_h(s,g_h) using equation (Ve):
        V^e_h(s,g_h) = E_{g_{-h}} E_{a_H~π_H(s,g)} E_{a_r~π_r(s)} E_{s'~s,a} (U_h(s',g_h) + γ_h V^e_h(s',g_h))
        """
        key = (state_tuple, goal_tuple)
        if key in self.V_e_h_dict[human_id]:
            return self.V_e_h_dict[human_id][key]
        
        # For now, approximate V^e_h using the human Q-values under current policy
        # This is a simplified implementation - full version would require more computation
        if key not in self.Q_h_dict[human_id]:
            return 0.0
        
        q_values = self.Q_h_dict[human_id][key]
        allowed_actions = self.action_space_dict[human_id]
        
        # Get human policy π_h(s,g_h)
        policy = self.get_pi_h(human_id, state_tuple, goal_tuple)
        
        # Calculate expected value under policy
        expected_value = sum(policy.get(a, 0.0) * q_values[a] for a in allowed_actions)
        
        # Store for future use
        self.V_e_h_dict[human_id][key] = expected_value
        
        return expected_value

    def compute_v_e(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Legacy function for backward compatibility.
        Calls the new compute_v_e_h function.
        """
        return self.compute_v_e_h(human_id, state_tuple, goal_tuple)

    def update_q_e(self, human_id: str, state_tuple: tuple, goal_tuple: tuple, 
                   action: int, reward: float, next_state_tuple: tuple, done: bool):
        """
        Update Q_e^{π_r}(s,g,a_h) assuming current robot policy
        
        Q_e represents the expected Q-value for the human taking action a_h in state s
        for goal g, when the robot follows its current policy π_r.
        """
        key = (state_tuple, goal_tuple)
        current_q_e = self.Q_e_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            # Bootstrap with V_e^{π_r}(s',g) under current robot policy
            next_v_e = self.compute_v_e(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_e
        
        # Update Q_e with learning rate
        self.Q_e_dict[human_id][key][action] += self.alpha_e * (target - current_q_e)

    def concave_function_f(self, z: float) -> float:
        """Implement concave function f(z) for robot reward"""
        if self.reward_function == 'power':
            # Simple power function (not concave, but original formulation)
            return z
        elif self.reward_function == 'log':
            # f(z) = log_2(z) (concave)
            return np.log2(max(z, 1e-8))  # Avoid log(0)
        elif self.reward_function == 'bounded':
            # f(z) = 2 - 2/z (concave, bounded above by 2)
            return 2 - 2 / max(z, 1e-8)
        elif self.reward_function == 'generalized_bounded':
            # f(z) = (z^{-c} - 1) / (2^{-c} - 1) for c > 0
            c = self.concavity_param
            if c <= 0:
                c = 1.0  # Default to c=1
            z_safe = max(z, 1e-8)
            numerator = z_safe**(-c) - 1
            denominator = 2**(-c) - 1
            return numerator / denominator
        else:
            # Default to identity
            return z

    def calculate_robot_q_value_new(self, robot_id: str, state_tuple: tuple, action: int):
        """Calculate Q_r^{π_r}(s,a_r) = E_{g~μ_g} Q_r^{π_r}(s,g,a_r)"""
        # This would require goal-specific robot Q-values, which we approximate
        # by using the current Q_r as an estimate
        if state_tuple not in self.Q_r_dict[robot_id]:
            return 0.0
        
        return self.Q_r_dict[robot_id][state_tuple][action]

    def train(self, environment, phase1_episodes, phase2_episodes, render=False, render_delay=0):
        """Train the two-phase algorithm."""
        self.train_phase1(environment, phase1_episodes, render, render_delay)
        print("Phase 1 complete: Cautious human models learned")
        self.train_phase2(environment, phase2_episodes, render, render_delay)
        print("Phase 2 complete: Robot policy learned")

    def save_models(self, filepath="q_values.pkl"):
        """Save both human and robot models."""
        models = {
            "Q_h_dict": {hid: dict(qtable) for hid, qtable in self.Q_h_dict.items()},
            "Q_r_dict": {rid: dict(qtable) for rid, qtable in self.Q_r_dict.items()},
            "Q_e_dict": {hid: dict(qtable) for hid, qtable in self.Q_e_dict.items()},  # NEW
            "params": {
                "alpha_m": self.alpha_m,
                "alpha_e": self.alpha_e,
                "alpha_r": self.alpha_r,
                "gamma_h": self.gamma_h,
                "gamma_r": self.gamma_r,
                "beta_r_0": self.beta_r_0,
                "G": [tuple(g) for g in self.G],
                "mu_g": self.mu_g.tolist() if isinstance(self.mu_g, np.ndarray) else self.mu_g,
                "action_space_dict": self.action_space_dict,
                "robot_agent_ids": self.robot_agent_ids,
                "human_agent_ids": self.human_agent_ids,
                "eta": self.eta,
                "epsilon_h_0": self.epsilon_h_0,
                "reward_function": self.reward_function,  # NEW
                "concavity_param": self.concavity_param  # NEW
            }
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(models, f)
        
        print(f"Successfully saved models to {filepath}")

    @classmethod
    def load_q_values(cls, filepath):
        """Load saved models."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            params = data["params"]
            
            # Create instance
            instance = cls(
                alpha_m=params.get("alpha_m", 0.1),
                alpha_e=params.get("alpha_e", 0.2),
                alpha_r=params.get("alpha_r", 0.01),
                gamma_h=params.get("gamma_h", 0.99),
                gamma_r=params.get("gamma_r", 0.99),
                beta_r_0=params.get("beta_r_0", 5.0),
                G=params.get("G", [(0, 0)]),
                mu_g=np.array(params.get("mu_g", [1.0])),
                p_g=0.0,
                action_space_dict=params.get("action_space_dict", {}),
                robot_agent_ids=params.get("robot_agent_ids", ["robot_0"]),
                human_agent_ids=params.get("human_agent_ids", ["human_0"]),
                eta=params.get("eta", 0.1),
                epsilon_h_0=params.get("epsilon_h_0", 0.1),
                reward_function=params.get("reward_function", "power"),  # NEW
                concavity_param=params.get("concavity_param", 1.0),  # NEW
                debug=False
            )
            
            # Load Q-tables
            for hid, qtable_dict in data["Q_h_dict"].items():
                q_table = defaultdict(lambda: np.zeros(len(Actions)))
                for key_str, values in qtable_dict.items():
                    try:
                        key = eval(key_str) if isinstance(key_str, str) else key_str
                        q_table[key] = np.array(values)
                    except:
                        pass
                instance.Q_h_dict[hid] = q_table
            
            for rid, qtable_dict in data["Q_r_dict"].items():
                q_table = defaultdict(lambda: np.zeros(len(instance.action_space_dict[rid])))
                for key_str, values in qtable_dict.items():
                    try:
                        key = eval(key_str) if isinstance(key_str, str) else key_str
                        q_table[key] = np.array(values)
                    except:
                        pass
                instance.Q_r_dict[rid] = q_table
            
            # NEW: Load Q_e tables
            if "Q_e_dict" in data:
                for hid, qtable_dict in data["Q_e_dict"].items():
                    q_table = defaultdict(lambda: np.zeros(len(Actions)))
                    for key_str, values in qtable_dict.items():
                        try:
                            key = eval(key_str) if isinstance(key_str, str) else key_str
                            q_table[key] = np.array(values)
                        except:
                            pass
                    instance.Q_e_dict[hid] = q_table
            
            # Update compatibility attributes
            instance.Q_h = instance.Q_h_dict
            instance.Q_r = instance.Q_r_dict
            
            return instance
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

    def take_q_value_snapshot(self):
        """Take a snapshot of current Q-values for convergence monitoring."""
        snapshot = {'robot': {}, 'human': {}}
        
        # Snapshot robot Q-values
        for rid in self.robot_agent_ids:
            snapshot['robot'][rid] = {}
            for state, q_values in self.Q_r_dict[rid].items():
                snapshot['robot'][rid][state] = np.copy(q_values)
        
        # Snapshot human Q-values
        for hid in self.human_agent_ids:
            snapshot['human'][hid] = {}
            for state_goal, q_values in self.Q_h_dict[hid].items():
                snapshot['human'][hid][state_goal] = np.copy(q_values)
        
        return snapshot

    def calculate_q_value_changes(self, old_snapshot, new_snapshot):
        """Calculate the magnitude of Q-value changes between snapshots."""
        changes = {'robot': {}, 'human': {}}
        
        # Calculate robot Q-value changes
        for rid in self.robot_agent_ids:
            total_change = 0.0
            count = 0
            max_change = 0.0
            
            old_robot = old_snapshot['robot'].get(rid, {})
            new_robot = new_snapshot['robot'].get(rid, {})
            
            # Check states present in both snapshots
            common_states = set(old_robot.keys()) & set(new_robot.keys())
            for state in common_states:
                diff = np.abs(new_robot[state] - old_robot[state])
                state_max_change = np.max(diff)
                state_avg_change = np.mean(diff)
                
                total_change += state_avg_change
                max_change = max(max_change, state_max_change)
                count += 1
            
            avg_change = total_change / max(count, 1)
            changes['robot'][rid] = {
                'avg_change': avg_change,
                'max_change': max_change,
                'states_tracked': count
            }
        
        # Calculate human Q-value changes
        for hid in self.human_agent_ids:
            total_change = 0.0
            count = 0
            max_change = 0.0
            
            old_human = old_snapshot['human'].get(hid, {})
            new_human = new_snapshot['human'].get(hid, {})
            
            # Check state-goal pairs present in both snapshots
            common_keys = set(old_human.keys()) & set(new_human.keys())
            for key in common_keys:
                diff = np.abs(new_human[key] - old_human[key])
                key_max_change = np.max(diff)
                key_avg_change = np.mean(diff)
                
                total_change += key_avg_change
                max_change = max(max_change, key_max_change)
                count += 1
            
            avg_change = total_change / max(count, 1)
            changes['human'][hid] = {
                'avg_change': avg_change,
                'max_change': max_change,
                'state_goals_tracked': count
            }
        
        return changes

    def check_convergence(self, changes):
        """Check if Q-values have converged based on recent changes."""
        robot_converged = True
        human_converged = True
        
        # Check robot convergence
        for rid, change_info in changes['robot'].items():
            if change_info['avg_change'] > self.convergence_threshold:
                robot_converged = False
                break
        
        # Check human convergence
        for hid, change_info in changes['human'].items():
            if change_info['avg_change'] > self.convergence_threshold:
                human_converged = False
                break
        
        return robot_converged, human_converged

    def log_q_value_changes(self, episode, phase, changes):
        """Log Q-value changes for debugging."""
        print(f"\n📊 Q-VALUE CHANGE ANALYSIS - {phase} Episode {episode}")
        print("=" * 60)
        
        # Log robot changes
        print("🤖 ROBOT Q-VALUE CHANGES:")
        for rid, change_info in changes['robot'].items():
            print(f"  Agent {rid}:")
            print(f"    Average change: {change_info['avg_change']:.6f}")
            print(f"    Maximum change: {change_info['max_change']:.6f}")
            print(f"    States tracked: {change_info['states_tracked']}")
        
        # Log human changes
        print("👤 HUMAN Q-VALUE CHANGES:")
        for hid, change_info in changes['human'].items():
            print(f"  Agent {hid}:")
            print(f"    Average change: {change_info['avg_change']:.6f}")
            print(f"    Maximum change: {change_info['max_change']:.6f}")
            print(f"    State-goal pairs tracked: {change_info['state_goals_tracked']}")
        
        # Check convergence
        robot_converged, human_converged = self.check_convergence(changes)
        
        print("\n🎯 CONVERGENCE STATUS:")
        print(f"  Robot converged: {'✅ YES' if robot_converged else '❌ NO'}")
        print(f"  Human converged: {'✅ YES' if human_converged else '❌ NO'}")
        print(f"  Threshold: {self.convergence_threshold}")
        
        if robot_converged and human_converged:
            print("🎉 BOTH AGENTS HAVE CONVERGED!")
            return True
        
        print("=" * 60)
        return False
