import numpy as np
from collections import defaultdict
import random
import pickle
import os
import math
import time
import pygame
from env import Actions
from q_learning_backends import create_q_learning_backend, QLearningBackend

class TwoPhaseTimescaleIQL:
    def __init__(self, alpha_m, alpha_e, alpha_r, gamma_h, gamma_r, beta_r_0,
                 G, mu_g, p_g, action_space_dict, robot_agent_ids, human_agent_ids,
                 eta=0.1, epsilon_h_0=0.1, epsilon_r=0.1, decay_epsilon_r_phase1=False, 
                 reward_function='power', concavity_param=1.0, debug=False,
                 zeta=1.0, xi=1.0, beta_h=5.0, nu_h=0.1, network=False, state_dim=4):
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
        
        # Human softmax policy parameters (replacing epsilon-greedy completely)
        self.beta_h_explore = 0.5  # Start with high temperature (more exploration) in Phase 1
        self.beta_h_final = beta_h  # End with low temperature (less exploration)
        self.beta_h = self.beta_h_explore  # Current beta_h (will be updated during training)
        self.nu_h = nu_h  # Weight for prior policy in smooth updates
        self.policy_update_rate = 1.0 / (2.0 * self.beta_h_final)  # Smooth policy update rate
        
        # Robot exploration temperature for Phase 1 (replacing epsilon_r)
        self.phase1_robot_temp_start = 2.0  # Initial exploration
        self.phase1_robot_temp_end = 10.0   # Final exploitation
        self.phase1_robot_temp = self.phase1_robot_temp_start  # Current temperature
        
        # DEPRECATED: Keep for backward compatibility but not used
        self.epsilon_h_0 = epsilon_h_0  # Final epsilon_h for converged policy
        self.epsilon_h = 0.5  # Current epsilon_h (starts high, decreases to epsilon_h_0)
        self.epsilon_r = epsilon_r  # Robot exploration in Phase 1
        
        self.G = G
        self.mu_g = mu_g
        self.p_g = p_g
        self.eta = eta
        self.debug = debug
        
        # NEW: Mathematical formulation parameters
        self.zeta = zeta  # Power parameter in X_h(s) calculation (equation Xh)
        self.xi = xi      # Power parameter in U_r(s) calculation (equation Ur)
        
        # Learning rate decay parameters for provable convergence
        self.alpha_m_initial = alpha_m
        self.alpha_e_initial = alpha_e 
        self.alpha_r_initial = alpha_r
        self.use_learning_rate_decay = True  # Enable provable convergence
        
        # NEW: Network vs Tabular learning
        self.network = network
        self.state_dim = state_dim
        
        # Agent configuration
        self.robot_agent_ids = robot_agent_ids if isinstance(robot_agent_ids, list) else [robot_agent_ids]
        self.human_agent_ids = human_agent_ids if isinstance(human_agent_ids, list) else [human_agent_ids]
        self.robot_agent_id = self.robot_agent_ids[0]  # For compatibility
        
        # Action spaces
        self.action_space_dict = action_space_dict
        self.action_space_robot = {rid: action_space_dict[rid] for rid in self.robot_agent_ids}
        self.action_space_humans = {hid: action_space_dict[hid] for hid in self.human_agent_ids}
        
        # NEW: Modular Q-learning backends
        # Human backends (with goals): Q^m_h(s,g,a) and Q^e_h(s,g,a)
        self.human_q_m_backend = create_q_learning_backend(
            network, self.human_agent_ids, action_space_dict, state_dim, 
            use_goals=True, debug=debug, beta_h=beta_h, policy_update_rate=self.policy_update_rate, nu_h=nu_h
        )
        self.human_q_e_backend = create_q_learning_backend(
            network, self.human_agent_ids, action_space_dict, state_dim, 
            use_goals=True, debug=debug, beta_h=beta_h, policy_update_rate=self.policy_update_rate, nu_h=nu_h
        )
        
        # Robot backend (no goals): Q_r(s,a)
        self.robot_q_backend = create_q_learning_backend(
            network, self.robot_agent_ids, action_space_dict, state_dim, 
            use_goals=False, debug=debug, beta_h=beta_h, policy_update_rate=self.policy_update_rate, nu_h=nu_h
        )
        
        # Backward compatibility: expose Q-tables as properties
        if not network:
            self.Q_m_h_dict = self.human_q_m_backend.q_tables
            self.Q_r_dict = self.robot_q_backend.q_tables
            self.Q_e_dict = self.human_q_e_backend.q_tables
            self.Q_h_dict = self.Q_m_h_dict  # Backward compatibility
            self.Q_h = self.Q_m_h_dict
            self.Q_r = self.Q_r_dict
            # Expose policy dictionaries
            self.pi_h_dict = self.human_q_m_backend.policies
            self.pi_r_dict = self.robot_q_backend.policies
        else:
            # For network mode, create empty dicts for compatibility
            self.Q_m_h_dict = {}
            self.Q_r_dict = {}
            self.Q_e_dict = {}
            self.Q_h_dict = {}
            self.Q_h = {}
            self.Q_r = {}
            self.pi_h_dict = {}
            self.pi_r_dict = {}
        
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
        
        # Habitual policies π^0_h for policy mixing
        self.pi_0_h_dict = {hid: {} for hid in self.human_agent_ids}  # Habitual policies
        
        # Prior beliefs about other humans' behavior μ_{-h}(s)
        self.mu_minus_h_dict = {hid: {} for hid in self.human_agent_ids}
        
        # Wide prior μ_{-h}(s) representing robot's assumption on human's belief about other humans
        # For simplicity, assume uniform distribution over actions
        action_dim = len(Actions)
        self.mu_minus_h = lambda state: {hid: np.ones(action_dim) / action_dim for hid in self.human_agent_ids}
        
        # NEW: Robot reward function parameters
        self.reward_function = reward_function  # 'power', 'log', 'bounded', 'generalized_bounded'
        self.concavity_param = concavity_param  # c parameter for generalized bounded function
        
        # Convergence monitoring
        self.convergence_threshold = 1e-4  # Threshold for Q-value changes
        self.policy_convergence_threshold = 1e-3  # Threshold for policy changes
        self.convergence_window = 100      # Episodes to check for convergence
        self.q_value_history = {'robot': [], 'human': []}
        self.last_q_snapshot = {'robot': {}, 'human': {}}
        self.last_policy_snapshot = {'robot': {}, 'human': {}}
        self.convergence_checks = 0
        
        if self.debug:
            print(f"TwoPhaseTimescaleIQL Initialized:")
            print(f"  Robot IDs: {self.robot_agent_ids}")
            print(f"  Human IDs: {self.human_agent_ids}")
            print(f"  Goals G: {self.G}")
            print(f"  beta_r: {self.beta_r} -> {self.beta_r_0}")
            print(f"  beta_h: {self.beta_h_explore} -> {self.beta_h_final} (softmax inverse temperature)")
            print(f"  nu_h: {self.nu_h} (prior weight)")
            print(f"  policy_update_rate: {self.policy_update_rate:.4f}")
            print(f"  phase1_robot_temp: {self.phase1_robot_temp_start} -> {self.phase1_robot_temp_end}")
            print(f"  Network mode: {self.network}")
            print(f"  State dimension: {self.state_dim}")
            print(f"  Reward function: {self.reward_function} (c={self.concavity_param})")
            print(f"  Convergence threshold: {self.convergence_threshold}")

    def get_decayed_learning_rate(self, initial_rate: float, episode: int, total_episodes: int) -> float:
        """Get learning rate with decay for provable convergence: alpha_t = alpha_0 / (1 + episode * decay_factor)"""
        if not self.use_learning_rate_decay:
            return initial_rate
        
        # Use decay factor that ensures convergence: alpha_t = alpha_0 / (1 + t * 0.001)
        decay_factor = 0.001
        decayed_rate = initial_rate / (1 + episode * decay_factor)
        return max(decayed_rate, initial_rate * 0.1)  # Don't let it go below 10% of original

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
        Sample robot action using backward induction with temperature-based exploration in Phase 1.
        Implements the min_{a_r} part of equation (Qm) with exploration.
        """
        allowed = self.action_space_dict[robot_id]
        
        # Temperature-based exploration for learning
        if self.phase1_robot_temp > 0:
            # Compute values for each action (negative of human value for minimization)
            action_values = []
            for action in allowed:
                neg_value = self.calculate_min_human_value(robot_id, state_tuple, action)
                action_values.append(-neg_value)  # Negate to minimize human value
            
            # Softmax with exploration temperature
            action_values = np.array(action_values)
            action_values = np.clip(action_values, -500, 500)
            exp_values = np.exp(self.phase1_robot_temp * action_values)
            probs = exp_values / np.sum(exp_values)
            
            return np.random.choice(allowed, p=probs)
        else:
            # Pure minimization (no exploration) - original behavior
            min_expected_values = []
            for action in allowed:
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
        
        # Get policy from backend
        policy = self.get_pi_r(robot_id, state_tuple)
        
        # Convert policy to probability array for allowed actions
        probs = np.array([policy.get(a, 1.0 / len(allowed)) for a in allowed])
        
        # Ensure probabilities are normalized and positive
        probs = np.maximum(probs, 1e-8)  # Avoid zero probabilities
        probs = probs / np.sum(probs)
        
        # Sample action according to policy
        idx = np.random.choice(len(allowed), p=probs)
        return allowed[idx]

    def sample_human_action_phase1(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> int:
        """Sample human action using softmax policy in Phase 1."""
        allowed = self.action_space_dict[human_id]
        
        # Get current policy π_h(s,g_h) 
        policy = self.get_pi_h(human_id, state_tuple, goal_tuple)
        
        # Convert policy to probability array for allowed actions
        probs = np.array([policy.get(a, 1.0 / len(allowed)) for a in allowed])
        
        # Ensure probabilities are normalized and positive
        probs = np.maximum(probs, 1e-8)  # Avoid zero probabilities
        probs = probs / np.sum(probs)
        
        # Sample action according to policy
        idx = np.random.choice(len(allowed), p=probs)
        return allowed[idx]

    def sample_human_action_phase2(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> int:
        """Sample human action using converged softmax policy in Phase 2."""
        # Phase 2 uses the same policy as Phase 1, but policies should be converged
        return self.sample_human_action_phase1(human_id, state_tuple, goal_tuple)
        return allowed[best_action_idx]

    def train_phase1(self, environment, phase1_episodes, render=False, render_delay=0):
        """Phase 1: Learn cautious human models using sequential conservative Q-learning."""
        print(f"Phase 1: Learning cautious human models for {phase1_episodes} episodes")
        print(f"Sequential training: Learning individual conservative models for each human")
        
        # Initialize convergence monitoring
        self.last_q_snapshot = self.take_q_value_snapshot()
        
        # Divide episodes among humans for sequential training
        episodes_per_human = max(1, phase1_episodes // len(self.human_agent_ids))
        remaining_episodes = phase1_episodes % len(self.human_agent_ids)
        
        global_episode = 0
        
        # Sequential training: train each human model separately
        for human_idx, target_human_id in enumerate(self.human_agent_ids):
            # Calculate episodes for this human (distribute remainder evenly)
            current_episodes = episodes_per_human + (1 if human_idx < remaining_episodes else 0)
            
            print(f"\n--- Learning conservative model for {target_human_id} ({current_episodes} episodes) ---")
            
            for episode in range(current_episodes):
                environment.reset()
                
                # Update parameters based on global progress
                global_progress = global_episode / max(phase1_episodes - 1, 1)
                
                # Update human beta_h: increase from beta_h_explore to beta_h_final (exploration → exploitation)
                self.beta_h = self.beta_h_explore + (self.beta_h_final - self.beta_h_explore) * global_progress
                
                # Update robot exploration temperature: increase from start to end (some exploration → mostly minimizing)
                self.phase1_robot_temp = (self.phase1_robot_temp_start + 
                                        (self.phase1_robot_temp_end - self.phase1_robot_temp_start) * global_progress)
                
                # Update learning rates with decay for provable convergence
                current_alpha_m = self.get_decayed_learning_rate(self.alpha_m_initial, global_episode, phase1_episodes)
                
                # DEPRECATED: Keep epsilon_r decay for backward compatibility
                self.epsilon_r = max(0.1 * (1 - global_progress), 0.01)
                
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
                    
                    # Human actions (all humans still act to maintain environment realism)
                    for hid in self.human_agent_ids:
                        actions[hid] = self.sample_human_action_phase1(hid, s_tuples[hid], current_goals[hid])
                    
                    # Robot actions (targeted pessimistic policy against specific human)
                    for rid in self.robot_agent_ids:
                        actions[rid] = self.sample_robot_action_phase1_targeted(
                            rid, s_tuples[rid], target_human_id, current_goals[target_human_id]
                        )
                    
                    # Environment step
                    next_obs, rewards, terms, truncs, infos = environment.step(actions)
                    
                    if render:
                        environment.render()
                        pygame.time.delay(render_delay)
                    
                    # Get next states
                    next_s_tuples = {aid: self.state_to_tuple(next_obs[aid]) 
                                   for aid in environment.possible_agents}
                    
                    episode_done = any(terms.values()) or any(truncs.values())
                    
                    # CRITICAL: ONLY update Q-values for the target human
                    # Other human models remain frozen during this sub-phase
                    reward = rewards.get(target_human_id, 0)
                    episode_human_rewards[target_human_id] += reward
                    
                    self.update_human_q_phase1(
                        target_human_id, s_tuples[target_human_id], current_goals[target_human_id], 
                        actions[target_human_id], reward, next_s_tuples[target_human_id], 
                        episode_done, current_alpha_m
                    )
                    
                    # Also update Q_e for the target human
                    self.update_q_e(
                        target_human_id, s_tuples[target_human_id], current_goals[target_human_id], 
                        actions[target_human_id], reward, next_s_tuples[target_human_id], 
                        episode_done, current_alpha_m
                    )
                    
                    step_count += 1
                    if episode_done:
                        break
                
                global_episode += 1
                
                # Print progress for this human's training
                if (episode + 1) % min(50, current_episodes) == 0 or episode + 1 == current_episodes:
                    avg_target_reward = episode_human_rewards[target_human_id] / max(step_count, 1)
                    print(f"[PHASE1-{target_human_id}] Episode {episode + 1}/{current_episodes}: "
                          f"target_human={avg_target_reward:.2f}, β_h={self.beta_h:.3f}, "
                          f"robot_temp={self.phase1_robot_temp:.2f}")
            
            # Check convergence after each human's training
            if global_episode >= self.convergence_window:
                current_snapshot = self.take_q_value_snapshot()
                changes = self.calculate_q_value_changes(self.last_q_snapshot, current_snapshot)
                
                if self.debug:
                    converged = self.log_q_value_changes(global_episode, f"PHASE1-{target_human_id}", changes)
                    if converged:
                        print(f"🎯 Early convergence detected for {target_human_id} at episode {global_episode}!")
                
                # Update snapshot for next comparison
                self.last_q_snapshot = current_snapshot
        
        print(f"\n✅ Phase 1 complete: Individual conservative models learned for all humans")

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
            
            # Update learning rates with decay for provable convergence
            current_alpha_e = self.get_decayed_learning_rate(self.alpha_e_initial, episode, phase2_episodes)
            current_alpha_r = self.get_decayed_learning_rate(self.alpha_r_initial, episode, phase2_episodes)
            
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
                
                # Update robot Q-values with decayed learning rate
                for rid in self.robot_agent_ids:
                    self.update_robot_q_phase2(
                        rid, s_tuples[rid], actions[rid], robot_reward,
                        next_s_tuples[rid], episode_done, current_alpha_r
                    )
                
                # Fast timescale human updates AND Q_e updates
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    episode_human_rewards[hid] += reward  # Track individual rewards
                    
                    # PHASE 2: Do NOT update human Q-values or policies; keep fixed from Phase 1
                    # self.update_human_q_phase2(
                    #     hid, s_tuples[hid], current_goals[hid], actions[hid],
                    #     reward, next_s_tuples[hid], episode_done, current_alpha_e
                    # )
                    # self.update_q_e(
                    #     hid, s_tuples[hid], current_goals[hid], actions[hid],
                    #     reward, next_s_tuples[hid], episode_done, current_alpha_e
                    # )
                
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

    def update_human_q_phase1(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done, alpha_m=None):
        """
        Update Q^m_h using backward induction approach (equation Qm).
        Q^m_h(s,g_h,a_h) ← E_{a_{-h}~μ_{-h}(s)} min_{a_r∈A_r(s)} E_{s'~s,a} (U_h(s',g_h) + γ_h V^m_h(s',g_h))
        """
        if done:
            target = reward
        else:
            # Compute E_{a_{-h}~μ_{-h}(s)} min_{a_r∈A_r(s)} E_{s'~s,a} [...]
            # For now, use simplified approximation - in full implementation would enumerate
            # action combinations of other humans weighted by μ_{-h}
            
            other_humans = [h for h in self.human_agent_ids if h != human_id]
            
            if other_humans:
                # Simplified: assume μ_{-h} leads to expected outcomes similar to current policy
                # TODO: Full implementation would sample/enumerate other human action combinations
                expected_value = 0.0
                
                # Compute min over robot actions (backward induction)
                min_value = float('inf')
                for robot_action in self.action_space_dict[self.robot_agent_ids[0]]:
                    # Compute expected next value given robot_action
                    # This is simplified - full version would use environment model
                    next_v = self.compute_v_m_h(human_id, next_state_tuple, goal_tuple)
                    value = reward + self.gamma_h * next_v
                    min_value = min(min_value, value)
                
                target = min_value
            else:
                # Single human case - use standard calculation
                next_v_m_h = self.compute_v_m_h(human_id, next_state_tuple, goal_tuple)
                target = reward + self.gamma_h * next_v_m_h
        
        # Update Q-values using backend with decayed learning rate
        current_alpha_m = alpha_m if alpha_m is not None else self.alpha_m
        self.human_q_m_backend.update_q_values(human_id, state_tuple, action, target, current_alpha_m, goal_tuple)
        
        # Update π_h(s,g_h) using softmax with habit mixing (equation pih)
        self.update_pi_h(human_id, state_tuple, goal_tuple)
        
        # Update V^m_h(s,g_h) using equation (Vm)
        self.update_v_m_h(human_id, state_tuple, goal_tuple)

    def compute_v_m_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Compute V^m_h(s,g_h) = E_{a_h~π_h(s,g_h)} Q^m_h(s,g_h,a_h) (equation Vm)
        """
        # Get Q-values from backend
        q_values = self.human_q_m_backend.get_q_values(human_id, state_tuple, goal_tuple)
        allowed_actions = self.action_space_dict[human_id]
        
        # Get current policy π_h(s,g_h)
        policy = self.get_pi_h(human_id, state_tuple, goal_tuple)
        
        # Calculate expected value
        expected_value = sum(policy.get(a, 0.0) * q_values[a] for a in allowed_actions)
        return expected_value

    def update_pi_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple):
        """Update π_h(s,g_h) using proper softmax with habit mixing (replacing ε-greedy)"""
        # Get current Q-values from backend
        q_values = self.human_q_m_backend.get_q_values(human_id, state_tuple, goal_tuple)
        allowed_actions = self.action_space_dict[human_id]
        
        # Compute softmax policy
        key = (state_tuple, goal_tuple)
        q_subset = np.array([q_values[a] for a in allowed_actions])
        q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow
        
        exp_q = np.exp(self.beta_h * q_subset)
        softmax_probs = exp_q / np.sum(exp_q)
        
        # Get or initialize habitual policy π^0_h
        habitual_policy = self.pi_0_h_dict[human_id].get(key, 
                         {a: 1.0/len(allowed_actions) for a in allowed_actions})
        
        # Final policy: ν_h * π^0_h + (1-ν_h) * softmax
        policy = {}
        for i, a in enumerate(allowed_actions):
            policy[a] = (self.nu_h * habitual_policy.get(a, 1.0/len(allowed_actions)) + 
                         (1 - self.nu_h) * softmax_probs[i])
        
        if self.network:
            # For networks, we don't store explicit policies
            pass
        else:
            # For tabular, store the mixed policy
            if hasattr(self.human_q_m_backend, 'policies'):
                self.human_q_m_backend.policies[human_id][key] = policy

    def get_pi_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> dict:
        """Get π_h(s,g_h) policy with habit mixing"""
        key = (state_tuple, goal_tuple)
        
        if self.network:
            # For networks, compute policy on-demand from Q-values with habit mixing
            q_values = self.human_q_m_backend.get_q_values(human_id, state_tuple, goal_tuple)
            allowed_actions = self.action_space_dict[human_id]
            
            # Compute softmax policy
            q_subset = np.array([q_values[a] for a in allowed_actions])
            q_subset = np.clip(q_subset, -500, 500)  # Prevent overflow
            
            exp_q = np.exp(self.beta_h * q_subset)
            softmax_probs = exp_q / np.sum(exp_q)
            
            # Get habitual policy
            habitual_policy = self.pi_0_h_dict[human_id].get(key, 
                             {a: 1.0/len(allowed_actions) for a in allowed_actions})
            
            # Mix policies: ν_h * π^0_h + (1-ν_h) * softmax
            policy = {}
            for i, a in enumerate(allowed_actions):
                policy[a] = (self.nu_h * habitual_policy.get(a, 1.0/len(allowed_actions)) + 
                             (1 - self.nu_h) * softmax_probs[i])
            return policy
        else:
            # For tabular, use stored policy or compute if not available
            if hasattr(self.human_q_m_backend, 'policies') and key in self.human_q_m_backend.policies[human_id]:
                return self.human_q_m_backend.policies[human_id][key]
            else:
                # Fallback: compute policy on-demand
                return self.human_q_m_backend.get_policy(human_id, state_tuple, self.beta_h, goal_tuple)

    def update_v_m_h(self, human_id: str, state_tuple: tuple, goal_tuple: tuple):
        """Update V^m_h(s,g_h) using equation (Vm)"""
        key = (state_tuple, goal_tuple)
        self.V_m_h_dict[human_id][key] = self.compute_v_m_h(human_id, state_tuple, goal_tuple)

    def update_human_q_phase2(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done, alpha_e=None):
        """Fast timescale update for human Q-values in Phase 2, keeping Q^m_h updated."""
        if done:
            target = reward
        else:
            next_v_m_h = self.compute_v_m_h(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_m_h
        
        # Use fast learning rate for Phase 2 updates (with decay)
        current_alpha_e = alpha_e if alpha_e is not None else self.alpha_e
        self.human_q_m_backend.update_q_values(human_id, state_tuple, action, target, current_alpha_e, goal_tuple)
        
        # Also update the mathematical tables
        self.update_pi_h(human_id, state_tuple, goal_tuple)
        self.update_v_m_h(human_id, state_tuple, goal_tuple)

    def update_robot_q_phase2(self, robot_id, state_tuple, action, reward, next_state_tuple, done, alpha_r=None):
        """
        Update robot Q-values in Phase 2 using equation (Qr):
        Q_r(s,a_r) ← E_g E_{a_H~π_H(s,g)} E_{s'~s,a} γ_r V_r(s')
        """
        if done:
            target = reward
        else:
            # Calculate V_r(s') using equation (Vr)
            next_v_r = self.compute_v_r(robot_id, next_state_tuple)
            target = reward + self.gamma_r * next_v_r
        
        # Update Q-values using backend with decayed learning rate
        current_alpha_r = alpha_r if alpha_r is not None else self.alpha_r
        self.robot_q_backend.update_q_values(robot_id, state_tuple, action, target, current_alpha_r)
        
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
        q_values = self.robot_q_backend.get_q_values(robot_id, state_tuple)
        allowed_actions = self.action_space_dict[robot_id]
        
        # Get robot policy π_r(s)
        policy = self.get_pi_r(robot_id, state_tuple)
        
        # Calculate expected Q-value under policy
        expected_q = sum(policy.get(a, 0.0) * q_values[a] for a in allowed_actions)
        
        return U_r + expected_q

    def update_pi_r(self, robot_id: str, state_tuple: tuple):
        """Update π_r(s) ← β_r-softmax policy for Q_r(s,·) (equation pir)"""
        # Get Q-values from backend
        q_values = self.robot_q_backend.get_q_values(robot_id, state_tuple)
        
        if self.network:
            # For networks, policy is computed on-demand
            pass
        else:
            # For tabular, update stored policy
            self.robot_q_backend.update_policy_direct(robot_id, state_tuple, q_values, self.beta_r)

    def get_pi_r(self, robot_id: str, state_tuple: tuple) -> dict:
        """Get π_r(s) policy"""
        return self.robot_q_backend.get_policy(robot_id, state_tuple, self.beta_r)

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
        
        TODO: Full implementation requires:
        1. Sampling goals for other humans g_{-h}
        2. Getting actions from all agents' policies π_H(s,g) and π_r(s)
        3. Computing expected next state value under environment model
        Current implementation uses Q_h as approximation for computational tractability.
        """
        key = (state_tuple, goal_tuple)
        if key in self.V_e_h_dict[human_id]:
            return self.V_e_h_dict[human_id][key]
        
        # Simplified approximation: use human Q-values under current policy
        # This approximates the expectation over other agents and environment transitions
        q_values = self.human_q_m_backend.get_q_values(human_id, state_tuple, goal_tuple)
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
                   action: int, reward: float, next_state_tuple: tuple, done: bool, alpha_e=None):
        """
        Update Q_e^{π_r}(s,g,a_h) assuming current robot policy
        
        Q_e represents the expected Q-value for the human taking action a_h in state s
        for goal g, when the robot follows its current policy π_r.
        """
        if done:
            target = reward
        else:
            # Bootstrap with V_e^{π_r}(s',g) under current robot policy
            next_v_e = self.compute_v_e(human_id, next_state_tuple, goal_tuple)
            target = reward + self.gamma_h * next_v_e
        
        # Update Q_e with learning rate using backend (with decay)
        current_alpha_e = alpha_e if alpha_e is not None else self.alpha_e
        self.human_q_e_backend.update_q_values(human_id, state_tuple, action, target, current_alpha_e, goal_tuple)

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
            "params": {
                "alpha_m": self.alpha_m,
                "alpha_e": self.alpha_e,
                "alpha_r": self.alpha_r,
                "gamma_h": self.gamma_h,
                "gamma_r": self.gamma_r,
                "beta_r_0": self.beta_r_0,
                "beta_h": self.beta_h,
                "nu_h": self.nu_h,
                "G": [tuple(g) for g in self.G],
                "mu_g": self.mu_g.tolist() if isinstance(self.mu_g, np.ndarray) else self.mu_g,
                "action_space_dict": self.action_space_dict,
                "robot_agent_ids": self.robot_agent_ids,
                "human_agent_ids": self.human_agent_ids,
                "eta": self.eta,
                "epsilon_h_0": self.epsilon_h_0,
                "reward_function": self.reward_function,
                "concavity_param": self.concavity_param,
                "network": self.network,
                "state_dim": self.state_dim,
                "zeta": self.zeta,
                "xi": self.xi
            }
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save backend models
        if self.network:
            # For neural networks, save to separate files
            base_path = filepath.replace('.pkl', '')
            self.human_q_m_backend.save_models(f"{base_path}_human_q_m.pt")
            self.human_q_e_backend.save_models(f"{base_path}_human_q_e.pt")
            self.robot_q_backend.save_models(f"{base_path}_robot_q.pt")
            models["backend_files"] = {
                "human_q_m": f"{base_path}_human_q_m.pt",
                "human_q_e": f"{base_path}_human_q_e.pt", 
                "robot_q": f"{base_path}_robot_q.pt"
            }
        else:
            # For tabular, include Q-tables in the main file
            self.human_q_m_backend.save_models(filepath.replace('.pkl', '_human_q_m.pkl'))
            self.human_q_e_backend.save_models(filepath.replace('.pkl', '_human_q_e.pkl'))
            self.robot_q_backend.save_models(filepath.replace('.pkl', '_robot_q.pkl'))
            models["backend_files"] = {
                "human_q_m": filepath.replace('.pkl', '_human_q_m.pkl'),
                "human_q_e": filepath.replace('.pkl', '_human_q_e.pkl'),
                "robot_q": filepath.replace('.pkl', '_robot_q.pkl')
            }
        
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
            
            # Create instance with new parameters
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
                reward_function=params.get("reward_function", "power"),
                concavity_param=params.get("concavity_param", 1.0),
                # New parameters for modular backends
                network=params.get("network", False),
                state_dim=params.get("state_dim", 4),
                beta_h=params.get("beta_h", 5.0),
                nu_h=params.get("nu_h", 0.1),
                debug=False
            )
            
            # Load backend models
            if "backend_files" in data:
                # New format with separate backend files
                backend_files = data["backend_files"]
                
                # Load human Q^m backend
                if "human_q_m" in backend_files:
                    instance.human_q_m_backend.load_models(backend_files["human_q_m"])
                
                # Load human Q^e backend
                if "human_q_e" in backend_files:
                    instance.human_q_e_backend.load_models(backend_files["human_q_e"])
                
                # Load robot Q backend
                if "robot_q" in backend_files:
                    instance.robot_q_backend.load_models(backend_files["robot_q"])
                    
            else:
                # Legacy format - convert old Q-tables to new backend format
                print("Loading legacy format, converting to new backend system...")
                
                # Load legacy Q^m tables (Q_h_dict)
                if "Q_h_dict" in data:
                    for hid, qtable_dict in data["Q_h_dict"].items():
                        for key_str, values in qtable_dict.items():
                            try:
                                key = eval(key_str) if isinstance(key_str, str) else key_str
                                instance.human_q_m_backend.q_tables[key] = np.array(values)
                            except:
                                pass
                
                # Load legacy Q^e tables if they exist
                if "Q_e_dict" in data:
                    for hid, qtable_dict in data["Q_e_dict"].items():
                        for key_str, values in qtable_dict.items():
                            try:
                                key = eval(key_str) if isinstance(key_str, str) else key_str
                                instance.human_q_e_backend.q_tables[key] = np.array(values)
                            except:
                                pass
                
                # Load legacy robot Q tables
                if "Q_r_dict" in data:
                    for rid, qtable_dict in data["Q_r_dict"].items():
                        for key_str, values in qtable_dict.items():
                            try:
                                key = eval(key_str) if isinstance(key_str, str) else key_str
                                instance.robot_q_backend.q_tables[key] = np.array(values)
                            except:
                                pass
                
                # Update policies from loaded Q-values for tabular backends
                if not instance.network:
                    instance.human_q_m_backend.update_all_policies()
                    instance.human_q_e_backend.update_all_policies()
                    instance.robot_q_backend.update_all_policies()
            
            # Maintain backward compatibility attributes
            instance.Q_h_dict = instance.human_q_m_backend.q_tables if hasattr(instance.human_q_m_backend, 'q_tables') else {}
            instance.Q_e_dict = instance.human_q_e_backend.q_tables if hasattr(instance.human_q_e_backend, 'q_tables') else {}
            instance.Q_r_dict = instance.robot_q_backend.q_tables if hasattr(instance.robot_q_backend, 'q_tables') else {}
            instance.Q_h = instance.Q_h_dict
            instance.Q_r = instance.Q_r_dict
            
            return instance
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

    def take_q_value_snapshot(self):
        """Take a snapshot of current Q-values for convergence monitoring."""
        snapshot = {'robot': {}, 'human_m': {}, 'human_e': {}}
        
        if self.network:
            # For network backends, we can't easily snapshot all Q-values
            # Instead, snapshot a representative set of states
            snapshot['network_mode'] = True
            # Could implement sampling of representative states if needed
            return snapshot
        else:
            # Tabular mode - snapshot all Q-values
            snapshot['network_mode'] = False
            
            # Snapshot robot Q-values from backend
            snapshot['robot'] = {}
            if hasattr(self.robot_q_backend, 'q_tables'):
                for aid in self.robot_agent_ids:
                    if aid in self.robot_q_backend.q_tables:
                        for state, q_values in self.robot_q_backend.q_tables[aid].items():
                            snapshot['robot'][state] = np.copy(q_values)
            
            # Snapshot human Q^m values from backend
            snapshot['human_m'] = {}
            if hasattr(self.human_q_m_backend, 'q_tables'):
                for aid in self.human_agent_ids:
                    if aid in self.human_q_m_backend.q_tables:
                        for state_goal, q_values in self.human_q_m_backend.q_tables[aid].items():
                            snapshot['human_m'][state_goal] = np.copy(q_values)
            
            # Snapshot human Q^e values from backend
            snapshot['human_e'] = {}
            if hasattr(self.human_q_e_backend, 'q_tables'):
                for aid in self.human_agent_ids:
                    if aid in self.human_q_e_backend.q_tables:
                        for state_goal, q_values in self.human_q_e_backend.q_tables[aid].items():
                            snapshot['human_e'][state_goal] = np.copy(q_values)
        
        return snapshot

    def calculate_q_value_changes(self, old_snapshot, new_snapshot):
        """Calculate the magnitude of Q-value changes between snapshots."""
        changes = {'robot': {}, 'human_m': {}, 'human_e': {}}
        
        # Handle network mode (no detailed change tracking)
        if old_snapshot.get('network_mode', False) or new_snapshot.get('network_mode', False):
            changes['network_mode'] = True
            return changes
        
        changes['network_mode'] = False
        
        # Calculate robot Q-value changes
        total_change = 0.0
        count = 0
        max_change = 0.0
        
        old_robot = old_snapshot.get('robot', {})
        new_robot = new_snapshot.get('robot', {})
        
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
        changes['robot'] = {
            'avg_change': avg_change,
            'max_change': max_change,
            'states_tracked': count
        }
        
        # Calculate human Q^m value changes
        total_change = 0.0
        count = 0
        max_change = 0.0
        
        old_human_m = old_snapshot.get('human_m', {})
        new_human_m = new_snapshot.get('human_m', {})
        
        # Check state-goal pairs present in both snapshots
        common_keys = set(old_human_m.keys()) & set(new_human_m.keys())
        for key in common_keys:
            diff = np.abs(new_human_m[key] - old_human_m[key])
            key_max_change = np.max(diff)
            key_avg_change = np.mean(diff)
            
            total_change += key_avg_change
            max_change = max(max_change, key_max_change)
            count += 1
        
        avg_change = total_change / max(count, 1)
        changes['human_m'] = {
            'avg_change': avg_change,
            'max_change': max_change,
            'state_goal_pairs_tracked': count
        }
        
        # Calculate human Q^e value changes
        total_change = 0.0
        count = 0
        max_change = 0.0
        
        old_human_e = old_snapshot.get('human_e', {})
        new_human_e = new_snapshot.get('human_e', {})
        
        # Check state-goal pairs present in both snapshots
        common_keys = set(old_human_e.keys()) & set(new_human_e.keys())
        for key in common_keys:
            diff = np.abs(new_human_e[key] - old_human_e[key])
            key_max_change = np.max(diff)
            key_avg_change = np.mean(diff)
            
            total_change += key_avg_change
            max_change = max(max_change, key_max_change)
            count += 1
        
        avg_change = total_change / max(count, 1)
        changes['human_e'] = {
            'avg_change': avg_change,
            'max_change': max_change,
            'state_goal_pairs_tracked': count
        }
        
        return changes

    def check_convergence(self, changes, policy_changes=None):
        """Check if Q-values and policies have converged based on recent changes."""
        if changes.get('network_mode', False):
            # For network mode, we can't easily check convergence from Q-values
            # Could implement other convergence criteria if needed
            return False, False
        
        robot_converged = True
        human_converged = True
        
        # Check robot convergence (Q-values)
        if 'robot' in changes and changes['robot']['avg_change'] > self.convergence_threshold:
            robot_converged = False
        
        # Check human convergence (both Q^m and Q^e)
        if ('human_m' in changes and changes['human_m']['avg_change'] > self.convergence_threshold) or \
           ('human_e' in changes and changes['human_e']['avg_change'] > self.convergence_threshold):
            human_converged = False
        
        # Additional policy convergence check if policy_changes provided
        if policy_changes and not policy_changes.get('network_mode', False):
            if 'robot' in policy_changes and policy_changes['robot']['avg_kl_div'] > self.policy_convergence_threshold:
                robot_converged = False
            if 'human' in policy_changes and policy_changes['human']['avg_kl_div'] > self.policy_convergence_threshold:
                human_converged = False
        
        return robot_converged, human_converged

    def log_q_value_changes(self, episode, phase, changes):
        """Log Q-value changes for debugging."""
        print(f"\n📊 Q-VALUE CHANGE ANALYSIS - {phase} Episode {episode}")
        print("=" * 60)
        
        if changes.get('network_mode', False):
            print("🔧 Network mode - detailed change tracking not available")
        else:
            # Log robot changes
            print("🤖 ROBOT Q-VALUE CHANGES:")
            if 'robot' in changes:
                change_info = changes['robot']
                print(f"    Average change: {change_info['avg_change']:.6f}")
                print(f"    Maximum change: {change_info['max_change']:.6f}")
                print(f"    States tracked: {change_info['states_tracked']}")
            
            # Log human Q^m changes
            print("👤 HUMAN Q^m VALUE CHANGES:")
            if 'human_m' in changes:
                change_info = changes['human_m']
                print(f"    Average change: {change_info['avg_change']:.6f}")
                print(f"    Maximum change: {change_info['max_change']:.6f}")
                print(f"    State-goal pairs tracked: {change_info['state_goal_pairs_tracked']}")
            
            # Log human Q^e changes
            print("👤 HUMAN Q^e VALUE CHANGES:")
            if 'human_e' in changes:
                change_info = changes['human_e']
                print(f"    Average change: {change_info['avg_change']:.6f}")
                print(f"    Maximum change: {change_info['max_change']:.6f}")
                print(f"    State-goal pairs tracked: {change_info['state_goal_pairs_tracked']}")
        
        # Check convergence
        robot_converged, human_converged = self.check_convergence(changes)
        
        print("\n🎯 CONVERGENCE STATUS:")
        print(f"  Robot converged: {'✅ YES' if robot_converged else '❌ NO'}")
        print(f"  Human converged: {'✅ YES' if human_converged else '❌ NO'}")
        if not changes.get('network_mode', False):
            print(f"  Threshold: {self.convergence_threshold}")
        
        if robot_converged and human_converged:
            print("🎉 BOTH AGENTS HAVE CONVERGED!")
            return True
        
        print("=" * 60)
        return False

    def take_policy_snapshot(self):
        """Take a snapshot of current policies for convergence monitoring."""
        snapshot = {'robot': {}, 'human': {}}
        
        if self.network:
            # For network mode, policies are computed on-demand
            snapshot['network_mode'] = True
            return snapshot
        else:
            snapshot['network_mode'] = False
            
            # Sample a representative set of states for policy tracking
            sample_states = set()
            
            # Sample robot states
            if hasattr(self.robot_q_backend, 'q_tables'):
                for rid in self.robot_agent_ids:
                    if rid in self.robot_q_backend.q_tables:
                        sample_states.update(list(self.robot_q_backend.q_tables[rid].keys())[:100])  # Sample first 100
            
            # Snapshot robot policies
            snapshot['robot'] = {}
            for state in sample_states:
                try:
                    policy = self.get_pi_r(self.robot_agent_ids[0], state)
                    snapshot['robot'][state] = policy.copy()
                except:
                    pass  # Skip states that cause errors
            
            # Sample human state-goal pairs
            sample_state_goals = set()
            if hasattr(self.human_q_m_backend, 'q_tables'):
                for hid in self.human_agent_ids:
                    if hid in self.human_q_m_backend.q_tables:
                        sample_state_goals.update(list(self.human_q_m_backend.q_tables[hid].keys())[:100])  # Sample first 100
            
            # Snapshot human policies
            snapshot['human'] = {}
            for state_goal in sample_state_goals:
                try:
                    state_tuple, goal_tuple = state_goal
                    policy = self.get_pi_h(self.human_agent_ids[0], state_tuple, goal_tuple)
                    snapshot['human'][state_goal] = policy.copy()
                except:
                    pass  # Skip state-goals that cause errors
                    
        return snapshot
    
    def calculate_policy_changes(self, old_snapshot, new_snapshot):
        """Calculate policy distance between snapshots using KL divergence."""
        changes = {'robot': {}, 'human': {}}
        
        if old_snapshot.get('network_mode', False) or new_snapshot.get('network_mode', False):
            changes['network_mode'] = True
            return changes
            
        changes['network_mode'] = False
        
        # Calculate robot policy changes
        total_kl_div = 0.0
        count = 0
        max_kl_div = 0.0
        
        old_robot = old_snapshot.get('robot', {})
        new_robot = new_snapshot.get('robot', {})
        
        common_states = set(old_robot.keys()) & set(new_robot.keys())
        for state in common_states:
            old_policy = old_robot[state]
            new_policy = new_robot[state]
            
            # Calculate KL divergence between policies
            kl_div = 0.0
            for action in old_policy.keys():
                if action in new_policy:
                    p_old = max(old_policy[action], 1e-8)
                    p_new = max(new_policy[action], 1e-8)
                    kl_div += p_old * np.log(p_old / p_new)
            
            total_kl_div += kl_div
            max_kl_div = max(max_kl_div, kl_div)
            count += 1
        
        avg_kl_div = total_kl_div / max(count, 1)
        changes['robot'] = {
            'avg_kl_div': avg_kl_div,
            'max_kl_div': max_kl_div,
            'states_tracked': count
        }
        
        # Calculate human policy changes
        total_kl_div = 0.0
        count = 0
        max_kl_div = 0.0
        
        old_human = old_snapshot.get('human', {})
        new_human = new_snapshot.get('human', {})
        
        common_state_goals = set(old_human.keys()) & set(new_human.keys())
        for state_goal in common_state_goals:
            old_policy = old_human[state_goal]
            new_policy = new_human[state_goal]
            
            # Calculate KL divergence between policies
            kl_div = 0.0
            for action in old_policy.keys():
                if action in new_policy:
                    p_old = max(old_policy[action], 1e-8)
                    p_new = max(new_policy[action], 1e-8)
                    kl_div += p_old * np.log(p_old / p_new)
            
            total_kl_div += kl_div
            max_kl_div = max(max_kl_div, kl_div)
            count += 1
        
        avg_kl_div = total_kl_div / max(count, 1)
        changes['human'] = {
            'avg_kl_div': avg_kl_div,
            'max_kl_div': max_kl_div,
            'state_goal_pairs_tracked': count
        }
        
        return changes

    def sample_robot_action_phase1_targeted(self, robot_id: str, state_tuple: tuple, 
                                           target_human_id: str, target_goal: tuple) -> int:
        """
        Selects a robot action to minimize the expected future value for ONLY the target human.
        
        This implements the conservative learning for individual human models by having the robot
        act as a specific adversary to one human at a time.
        
        Args:
            robot_id: ID of the robot agent
            state_tuple: Current state
            target_human_id: The specific human whose value we want to minimize
            target_goal: The goal of the target human
            
        Returns:
            Action that minimizes the target human's expected value
        """
        allowed = self.action_space_dict[robot_id]
        
        if self.phase1_robot_temp > 0:
            # Softmax exploration over negative values
            min_expected_values = []
            for action in allowed:
                neg_value = self.calculate_target_human_value(robot_id, state_tuple, action, target_human_id, target_goal)
                min_expected_values.append(neg_value)
            
            # Convert to negative for minimization and apply temperature
            neg_values = np.array([-v for v in min_expected_values])
            neg_values = np.clip(neg_values, -500, 500)  # Prevent overflow
            
            # Apply softmax with temperature
            exp_values = np.exp(neg_values / self.phase1_robot_temp)
            probs = exp_values / np.sum(exp_values)
            
            # Sample action
            idx = np.random.choice(len(allowed), p=probs)
            return allowed[idx]
        else:
            # Pure minimization (no exploration)
            min_expected_values = []
            for action in allowed:
                neg_value = self.calculate_target_human_value(robot_id, state_tuple, action, target_human_id, target_goal)
                min_expected_values.append(neg_value)
            
            # Choose action that minimizes target human's expected value
            best_action_idx = np.argmin(min_expected_values)
            return allowed[best_action_idx]

    def calculate_target_human_value(self, robot_id: str, state_tuple: tuple, robot_action: int, 
                                   target_human_id: str, target_goal: tuple) -> float:
        """
        Calculate expected future value for a specific target human given a robot action.
        This implements focused backward induction for individual human model learning.
        
        Args:
            robot_id: ID of the robot agent
            state_tuple: Current state
            robot_action: Robot action to evaluate
            target_human_id: The specific human whose value to calculate
            target_goal: The goal of the target human
            
        Returns:
            Expected future value for the target human
        """
        # Get current value for the target human at this state-goal pair
        key = (state_tuple, target_goal)
        if key in self.V_m_h_dict[target_human_id]:
            future_value = self.V_m_h_dict[target_human_id][key]
        else:
            # For unseen states, assume pessimistic value for conservative learning
            future_value = -1.0
        
        # Estimate immediate utility for the target human given robot action
        immediate_utility = self.estimate_human_utility(target_human_id, state_tuple, target_goal, robot_action)
        
        # Calculate expected return for this specific human
        expected_return = immediate_utility + self.gamma_h * future_value
        
        return expected_return
