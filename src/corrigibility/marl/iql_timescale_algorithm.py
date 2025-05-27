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
    def __init__(self, alpha_m, alpha_e, alpha_r, gamma_h, gamma_r, beta_h, beta_r,
                 G, mu_g, p_g, action_space_dict, robot_agent_ids, human_agent_ids,
                 eta=0.1, epsilon_h=0.1, debug=False):
        # Phase 1 parameters
        self.alpha_m = alpha_m  # Phase 1 learning rate for human models
        self.alpha_e = alpha_e  # Phase 2 fast timescale learning rate
        self.alpha_r = alpha_r  # Robot learning rate
        
        # Standard parameters
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.beta_h = beta_h
        self.beta_r = beta_r
        self.G = G
        self.mu_g = mu_g
        self.p_g = p_g
        self.eta = eta
        self.epsilon_h = epsilon_h
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
        
        # Compute softmax probabilities
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
        """Sample human action using learned policy in Phase 2."""
        allowed = self.action_space_dict[human_id]
        
        if (state_tuple, goal_tuple) not in self.Q_h_dict[human_id]:
            return np.random.choice(allowed)
        
        q_values = self.Q_h_dict[human_id][(state_tuple, goal_tuple)]
        q_subset = np.array([q_values[a] for a in allowed])
        
        # Ensure q_values are real numbers
        if np.iscomplexobj(q_subset):
            q_subset = np.real(q_subset)
        
        # Clip extreme values to prevent overflow
        q_subset = np.clip(q_subset, -500, 500)
        
        # Use softmax policy
        exp_q = np.exp(self.beta_h * q_subset)
        
        # Handle numerical issues
        if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
            probs = np.ones_like(exp_q) / len(exp_q)
        else:
            sum_exp = np.sum(exp_q)
            if sum_exp <= 0 or not np.isfinite(sum_exp):
                probs = np.ones_like(exp_q) / len(exp_q)
            else:
                probs = exp_q / sum_exp
        
        # Final safety checks for probabilities
        if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
            probs = np.ones_like(probs) / len(probs)
        
        # Ensure probabilities are positive and sum to 1
        probs = np.maximum(probs, 0)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones_like(probs) / len(probs)
        
        # Ensure probs is a real-valued float array
        probs = probs.astype(np.float64)
        
        idx = np.random.choice(len(allowed), p=probs)
        return allowed[idx]

    def train_phase1(self, environment, phase1_episodes, render=False, render_delay=0):
        """Phase 1: Learn cautious human models using conservative Q-learning."""
        print(f"Phase 1: Learning cautious human models for {phase1_episodes} episodes")
        
        # Initialize convergence monitoring
        self.last_q_snapshot = self.take_q_value_snapshot()
        
        for episode in range(phase1_episodes):
            environment.reset()
            
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
                
                # Robot actions (random/fixed for Phase 1)
                for rid in self.robot_agent_ids:
                    actions[rid] = np.random.choice(self.action_space_dict[rid])
                
                # Environment step
                next_obs, rewards, terms, truncs, infos = environment.step(actions)
                
                if render:
                    environment.render()
                    pygame.time.delay(render_delay)
                
                # Get next states
                next_s_tuples = {aid: self.state_to_tuple(next_obs[aid]) 
                               for aid in environment.possible_agents}
                
                episode_done = any(terms.values()) or any(truncs.values())
                
                # Update human Q-values
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    episode_human_reward += reward
                    self.update_human_q_phase1(
                        hid, s_tuples[hid], current_goals[hid], actions[hid],
                        reward, next_s_tuples[hid], episode_done
                    )
                
                step_count += 1
                if episode_done:
                    break
            
            # Print progress and check convergence every 100 episodes
            if (episode + 1) % self.convergence_window == 0 or episode + 1 == phase1_episodes:
                avg_human_reward = episode_human_reward / max(step_count, 1)
                print(f"[PHASE1] Episode {episode + 1}/{phase1_episodes}: human={avg_human_reward:.2f}, robot=0.00")
                
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
        
        # Reset convergence monitoring for Phase 2
        self.last_q_snapshot = self.take_q_value_snapshot()
        
        for episode in range(phase2_episodes):
            environment.reset()
            
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
                
                # Robot actions (learning)
                for rid in self.robot_agent_ids:
                    actions[rid] = self.sample_robot_action_phase2(rid, s_tuples[rid])
                
                # Human actions (using learned policy)
                for hid in self.human_agent_ids:
                    actions[hid] = self.sample_human_action_phase2(hid, s_tuples[hid], current_goals[hid])
                
                # Environment step
                next_obs, rewards, terms, truncs, infos = environment.step(actions)
                
                if render:
                    environment.render()
                    pygame.time.delay(render_delay)
                
                # Get next states
                next_s_tuples = {aid: self.state_to_tuple(next_obs[aid]) 
                               for aid in environment.possible_agents}
                
                episode_done = any(terms.values()) or any(truncs.values())
                
                # Calculate robot reward based on human potential
                robot_reward = self.calculate_robot_reward(next_s_tuples, current_goals, episode_done)
                episode_robot_reward += robot_reward
                
                # Update robot Q-values
                for rid in self.robot_agent_ids:
                    self.update_robot_q_phase2(
                        rid, s_tuples[rid], actions[rid], robot_reward,
                        next_s_tuples[rid], episode_done
                    )
                
                # Fast timescale human updates
                for hid in self.human_agent_ids:
                    reward = rewards.get(hid, 0)
                    episode_human_reward += reward
                    self.update_human_q_phase2(
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
                print(f"[PHASE2] Episode {episode + 1}/{phase2_episodes}: human={avg_human_reward:.2f}, robot={avg_robot_reward:.2f}")
                
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
        """Fast timescale update for human in Phase 2."""
        key = (state_tuple, goal_tuple)
        current_q = self.Q_h_dict[human_id][key][action]
        
        if done:
            target = reward
        else:
            next_key = (next_state_tuple, goal_tuple)
            next_q_values = self.Q_h_dict[human_id][next_key]
            allowed = self.action_space_dict[human_id]
            q_subset = np.array([next_q_values[a] for a in allowed])
            
            # Ensure q_subset is real and clip extreme values
            if np.iscomplexobj(q_subset):
                q_subset = np.real(q_subset)
            q_subset = np.clip(q_subset, -500, 500)
            
            # Expected value under softmax policy with numerical stability
            scaled_q = self.beta_h * q_subset
            # Subtract max for numerical stability
            scaled_q_stable = scaled_q - np.max(scaled_q)
            exp_q = np.exp(scaled_q_stable)
            
            if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
                next_value = np.mean(q_subset)
            else:
                probs = exp_q / np.sum(exp_q)
                if np.any(np.isnan(probs)):
                    next_value = np.mean(q_subset)
                else:
                    next_value = np.sum(probs * q_subset)
            
            target = reward + self.gamma_h * next_value
        
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

    def calculate_robot_reward(self, next_states, current_goals, done):
        """Calculate robot's internal reward based on human potential."""
        if done:
            return 0.0
        
        total_value = 0.0
        for hid in self.human_agent_ids:
            human_value = 0.0
            for i, goal in enumerate(self.G):
                goal_tuple = self.state_to_tuple(goal)
                key = (next_states[hid], goal_tuple)
                
                if key in self.Q_h_dict[hid]:
                    q_values = self.Q_h_dict[hid][key]
                    allowed = self.action_space_dict[hid]
                    q_subset = np.array([q_values[a] for a in allowed])
                    
                    # Ensure q_subset is real and clip extreme values
                    if np.iscomplexobj(q_subset):
                        q_subset = np.real(q_subset)
                    q_subset = np.clip(q_subset, -500, 500)
                    
                    # Expected value under policy with numerical stability
                    scaled_q = self.beta_h * q_subset
                    # Subtract max for numerical stability
                    scaled_q_stable = scaled_q - np.max(scaled_q)
                    exp_q = np.exp(scaled_q_stable)
                    
                    if np.any(np.isnan(exp_q)) or np.any(np.isinf(exp_q)) or np.sum(exp_q) == 0:
                        value = np.mean(q_subset)
                    else:
                        probs = exp_q / np.sum(exp_q)
                        if np.any(np.isnan(probs)):
                            value = np.mean(q_subset)
                        else:
                            value = np.sum(probs * q_subset)
                    
                    # Ensure value is finite and positive for power operation
                    if not np.isfinite(value):
                        value = 0.0
                    
                    # Clip value to prevent overflow in power operation
                    value = np.clip(value, -100, 100)
                    
                    # Safe power operation
                    try:
                        if value >= 0:
                            powered_value = value ** (1 + self.eta)
                        else:
                            # For negative values, use sign preservation
                            powered_value = -((-value) ** (1 + self.eta))
                        
                        if not np.isfinite(powered_value):
                            powered_value = 0.0
                        
                        human_value += self.mu_g[i] * powered_value
                    except (OverflowError, ValueError):
                        # Fallback for numerical issues
                        human_value += self.mu_g[i] * np.sign(value) * min(abs(value), 100)
            
            total_value += human_value
        
        # Ensure final result is finite
        if not np.isfinite(total_value):
            return 0.0
        
        # Clip final result to reasonable range
        return np.clip(total_value, -1000, 1000)

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
            "params": {
                "alpha_m": self.alpha_m,
                "alpha_e": self.alpha_e,
                "alpha_r": self.alpha_r,
                "gamma_h": self.gamma_h,
                "gamma_r": self.gamma_r,
                "beta_h": self.beta_h,
                "beta_r": self.beta_r,
                "G": [tuple(g) for g in self.G],
                "mu_g": self.mu_g.tolist() if isinstance(self.mu_g, np.ndarray) else self.mu_g,
                "action_space_dict": self.action_space_dict,
                "robot_agent_ids": self.robot_agent_ids,
                "human_agent_ids": self.human_agent_ids,
                "eta": self.eta,
                "epsilon_h": self.epsilon_h
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
                beta_h=params.get("beta_h", 5.0),
                beta_r=params.get("beta_r", 5.0),
                G=params.get("G", [(0, 0)]),
                mu_g=np.array(params.get("mu_g", [1.0])),
                p_g=0.0,
                action_space_dict=params.get("action_space_dict", {}),
                robot_agent_ids=params.get("robot_agent_ids", ["robot_0"]),
                human_agent_ids=params.get("human_agent_ids", ["human_0"]),
                eta=params.get("eta", 0.1),
                epsilon_h=params.get("epsilon_h", 0.1),
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
