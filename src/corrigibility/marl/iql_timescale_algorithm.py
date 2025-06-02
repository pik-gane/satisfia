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
                 reward_function='power', concavity_param=1.0, debug=False):
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
        
        # Agent configuration
        self.robot_agent_ids = robot_agent_ids if isinstance(robot_agent_ids, list) else [robot_agent_ids]
        self.human_agent_ids = human_agent_ids if isinstance(human_agent_ids, list) else [human_agent_ids]
        self.robot_agent_id = self.robot_agent_ids[0]  # For compatibility
        
        # Action spaces
        self.action_space_dict = action_space_dict
        self.action_space_robot = {rid: action_space_dict[rid] for rid in self.robot_agent_ids}
        self.action_space_humans = {hid: action_space_dict[hid] for hid in self.human_agent_ids}
        
        # Q-tables for Phase 1: Human model learning
        action_dim = len(Actions)
        self.Q_h_dict = {hid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=action_dim))
                         for hid in self.human_agent_ids}
        
        # Q-tables for Phase 2: Robot policy learning
        self.Q_r_dict = {rid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=len(self.action_space_robot[rid])))
                         for rid in self.robot_agent_ids}
        
        # NEW: Q_e tables for expected human Q-values under robot policy
        self.Q_e_dict = {hid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=action_dim))
                         for hid in self.human_agent_ids}
        
        # NEW: Robot reward function parameters
        self.reward_function = reward_function  # 'power', 'log', 'bounded', 'generalized_bounded'
        self.concavity_param = concavity_param  # c parameter for generalized bounded function
        
        # Compatibility attributes for trained_agent.py
        self.Q_h = self.Q_h_dict
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
        """Sample robot action using pessimistic policy in Phase 1."""
        allowed = self.action_space_dict[robot_id]
        
        if np.random.random() < self.epsilon_r:
            return np.random.choice(allowed)
        
        # Calculate pessimistic values for each action
        action_values = []
        for action in allowed:
            # Calculate expected negative human potential for this action
            # This is a simplified pessimistic estimate
            neg_value = self.calculate_pessimistic_value(robot_id, state_tuple, action)
            action_values.append(neg_value)
        
        # Choose action with most negative expected human value (pessimistic)
        best_action_idx = np.argmax(action_values)  # Most negative = worst for human
        return allowed[best_action_idx]

    def calculate_pessimistic_value(self, robot_id: str, state_tuple: tuple, action: int) -> float:
        """Calculate pessimistic estimate of human potential for robot action."""
        # Simple pessimistic heuristic: assume this action leads to worst outcomes for humans
        # In a full implementation, this would simulate the action and calculate
        # the minimum expected human value across all goals
        
        total_pessimistic_value = 0.0
        for hid in self.human_agent_ids:
            human_pessimistic_value = 0.0
            for i, goal in enumerate(self.G):
                goal_tuple = self.state_to_tuple(goal)
                key = (state_tuple, goal_tuple)
                
                if key in self.Q_h_dict[hid]:
                    q_values = self.Q_h_dict[hid][key]
                    # Use minimum Q-value (pessimistic assumption)
                    min_q_value = np.min(q_values)
                    human_pessimistic_value += self.mu_g[i] * min_q_value
                else:
                    # For unseen states, assume negative value
                    human_pessimistic_value += self.mu_g[i] * (-1.0)
            
            total_pessimistic_value += human_pessimistic_value
        
        return total_pessimistic_value

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
            episode_human_reward = 0.0
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
                    episode_human_reward += reward
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
                avg_human_reward = episode_human_reward / max(step_count, 1)
                print(f"[PHASE1] Episode {episode + 1}/{phase1_episodes}: human={avg_human_reward:.2f}, robot=PESSIMISTIC, ε_h={self.epsilon_h:.3f}")
                
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
            episode_human_reward = 0.0
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
                    episode_human_reward += reward
                    
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
                avg_human_reward = episode_human_reward / max(step_count, 1)
                print(f"[PHASE2] Episode {episode + 1}/{phase2_episodes}: human={avg_human_reward:.2f}, robot={avg_robot_reward:.2f}, β_r={self.beta_r:.3f}")
                
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
        """Conservative Q-learning update for human in Phase 1."""
        key = (state_tuple, goal_tuple)
        current_q = self.Q_h_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            next_key = (next_state_tuple, goal_tuple)
            next_q_values = self.Q_h_dict[human_id][next_key]
            # Conservative: use minimum Q-value instead of expected value
            next_value = np.min(next_q_values)
            target = reward + self.gamma_h * next_value
        
        self.Q_h_dict[human_id][key][action] += self.alpha_m * (target - current_q)

    def update_human_q_phase2(self, human_id, state_tuple, goal_tuple, action, reward, next_state_tuple, done):
        """Minimal fast timescale update for human in Phase 2."""
        key = (state_tuple, goal_tuple)
        current_q = self.Q_h_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            next_key = (next_state_tuple, goal_tuple)
            next_q_values = self.Q_h_dict[human_id][next_key]
            # Simple max Q-value for greedy policy
            next_value = np.max(next_q_values)
            target = reward + self.gamma_h * next_value
        
        # Use smaller learning rate for Phase 2 updates
        self.Q_h_dict[human_id][key][action] += self.alpha_e * (target - current_q)

    def update_robot_q_phase2(self, robot_id, state_tuple, action, reward, next_state_tuple, done):
        """Update robot Q-values in Phase 2."""
        current_q = self.Q_r_dict[robot_id][state_tuple][action]
        
        if done:
            target = reward
        else:
            next_q_values = self.Q_r_dict[robot_id][next_state_tuple]
            next_value = np.max(next_q_values)
            target = reward + self.gamma_r * next_value
        
        self.Q_r_dict[robot_id][state_tuple][action] += self.alpha_r * (target - current_q)

    def calculate_robot_reward_new(self, next_states, current_goals, done):
        """
        Calculate robot reward using Q_e and concave function f following the mathematical formulation:
        
        r_r(s', π_r) = f(z(s', π_r))
        where z(s', π_r) = E_{g~μ_g} V_e^{π_r}(s', g)^{1+η}
        and V_e^{π_r}(s', g) = E_{a_h~π_h*(s',g)} Q_e^{π_r}(s', g, a_h)
        """
        if done:
            return 0.0
        
        if self.debug:
            print(f"\n🔧 ROBOT REWARD CALCULATION:")
            print(f"  Next states: {next_states}")
            print(f"  Current goals: {current_goals}")
            print(f"  Reward function: {self.reward_function}")
            print(f"  η (eta): {self.eta}")
        
        # Calculate z(s', π_r) = E_{g~μ_g} V_e^{π_r}(s', g)^{1+η}
        z_total = 0.0
        
        for i, goal in enumerate(self.G):
            goal_tuple = self.state_to_tuple(goal)
            goal_weight = self.mu_g[i]
            
            if self.debug:
                print(f"  Goal {i}: {goal} -> {goal_tuple}, weight: {goal_weight}")
            
            # Calculate V_e^{π_r}(s', g) for each human and average
            v_e_total = 0.0
            for hid in self.human_agent_ids:
                next_state_hid = next_states.get(hid)
                if next_state_hid is None:
                    continue
                    
                v_e = self.compute_v_e(hid, next_state_hid, goal_tuple)
                v_e_total += v_e
                
                if self.debug:
                    print(f"    Human {hid}: V_e({next_state_hid}, {goal_tuple}) = {v_e:.3f}")
            
            # Average across humans
            v_e_avg = v_e_total / max(len(self.human_agent_ids), 1)
            
            # Apply power (1 + η) - ensure non-negative for power operation
            v_e_safe = max(v_e_avg, 0.0)
            v_e_powered = v_e_safe ** (1 + self.eta)
            
            # Weight by goal probability μ_g
            weighted_contribution = goal_weight * v_e_powered
            z_total += weighted_contribution
            
            if self.debug:
                print(f"    Average V_e: {v_e_avg:.3f}")
                print(f"    V_e^(1+η): {v_e_powered:.3f}")
                print(f"    Weighted contribution: {weighted_contribution:.3f}")
        
        if self.debug:
            print(f"  Total z: {z_total:.3f}")
        
        # Apply concave function f(z)
        robot_reward = self.concave_function_f(z_total)
        
        if self.debug:
            print(f"  f(z): {robot_reward:.3f}")
        
        # Ensure finite result
        if not np.isfinite(robot_reward):
            if self.debug:
                print(f"  ⚠️ Non-finite reward, returning 0.0")
            return 0.0
        
        # Clip to reasonable range
        clipped_reward = np.clip(robot_reward, -1000, 1000)
        
        if self.debug:
            print(f"  Final clipped reward: {clipped_reward:.3f}")
        
        return clipped_reward

    def compute_v_e(self, human_id: str, state_tuple: tuple, goal_tuple: tuple) -> float:
        """
        Compute V_e^{π_r}(s,g) = E_{a_h~π_h*(s,g)} Q_e^{π_r}(s,g,a_h)
        
        This represents the expected value for the human following their optimal policy
        in state s for goal g, under the current robot policy π_r.
        """
        key = (state_tuple, goal_tuple)
        if key not in self.Q_e_dict[human_id]:
            # For unseen state-goal pairs, return neutral value
            return 0.0
            
        q_e_values = self.Q_e_dict[human_id][key]
        allowed_actions = self.action_space_dict[human_id]
        
        # Compute human's optimal policy π_h*(s,g) - epsilon-greedy with converged epsilon
        epsilon = self.epsilon_h_0  # Use converged epsilon for policy
        
        # Get Q-values for allowed actions
        q_subset = [q_e_values[a] for a in allowed_actions]
        
        if len(q_subset) == 0:
            return 0.0
        
        # Find best action
        best_action_idx = np.argmax(q_subset)
        
        # Compute action probabilities under epsilon-greedy policy
        action_probs = np.full(len(allowed_actions), epsilon / len(allowed_actions))
        action_probs[best_action_idx] += (1.0 - epsilon)
        
        # Compute expected value: E_{a_h~π_h*(s,g)} Q_e^{π_r}(s,g,a_h)
        expected_value = sum(action_probs[i] * q_e_values[allowed_actions[i]] 
                           for i in range(len(allowed_actions)))
        
        return expected_value

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
