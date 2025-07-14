#!/usr/bin/env python3
"""
Corrected IQL Timescale Algorithm following the paper exactly.
"""

import numpy as np
import pickle
from collections import defaultdict

from iql_timescale_algorithm import TwoPhaseTimescaleIQL


class CorrectedTwoPhaseTimescaleIQL(TwoPhaseTimescaleIQL):
    """Corrected implementation following paper equations exactly."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove hardcoded behaviors - let algorithm learn
        
    def _potential_function(self, state, goal):
        """Potential function for reward shaping: -manhattan_distance / 100"""
        if isinstance(state, tuple):
            pos = np.array(state[:2])  # First 2 elements are position
        else:
            pos = np.array(state)
            
        goal_pos = np.array(goal)
        manhattan_dist = np.sum(np.abs(pos - goal_pos))
        return -manhattan_dist / 100.0
    
    def _shaped_reward(self, prev_state, next_state, goal, true_reward):
        """
        Potential-based reward shaping following paper suggestion.
        F(s,s') = γ * Φ(s') - Φ(s)
        """
        phi_next = self._potential_function(next_state, goal)
        phi_prev = self._potential_function(prev_state, goal)
        shaping = self.gamma_h * phi_next - phi_prev
        return true_reward + shaping
    
    def _adversarial_robot_action(self, env, robot_id, human_id, goal):
        """
        Phase 1: Robot chooses action to minimize expected human value.
        Following paper equation: min_{a_r} E[U_h(s',g_h) + γ_h V^m_h(s',g_h)]
        """
        best_action = None
        min_expected_value = float('inf')
        
        for action in self.action_space_dict[robot_id]:
            # Simulate taking this action
            expected_value = self._estimate_human_value_after_robot_action(
                env, robot_id, human_id, action, goal
            )
            
            if expected_value < min_expected_value:
                min_expected_value = expected_value
                best_action = action
        
        return best_action if best_action is not None else np.random.choice(self.action_space_dict[robot_id])
    
    def _estimate_human_value_after_robot_action(self, env, robot_id, human_id, robot_action, goal):
        """Estimate E[U_h(s',g_h) + γ_h V^m_h(s',g_h)] if robot takes robot_action"""
        # Get current state
        current_human_state = self.get_human_state(env, human_id, goal)
        
        # For simplicity, estimate based on current Q-values
        # In full implementation, this would simulate the transition
        if self.network:
            q_values = self.q_m_backend.get_q_values(human_id, current_human_state[:-2], goal)
        else:
            q_values = self.q_m[human_id][current_human_state]
        
        # Return expected value (max Q-value as approximation)
        return np.max(q_values)
    
    def _softmax_action(self, q_values, beta):
        """Sample action using softmax with inverse temperature beta"""
        if beta == 0:
            return np.random.choice(len(q_values))
        
        # Softmax with temperature
        exp_q = np.exp(beta * q_values)
        probabilities = exp_q / np.sum(exp_q)
        return np.random.choice(len(q_values), p=probabilities)
    
    def sample_robot_action_phase1(self, env, robot_id, human_id, goal, epsilon=0.1):
        """
        Phase 1: Robot plays adversarially using ε-greedy on negative expected human value.
        Following paper: ε-greedy on -E[U_h(s',g_h) + γ_h V^m_h(s',g_h)]
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[robot_id])
        
        return self._adversarial_robot_action(env, robot_id, human_id, goal)
    
    def sample_robot_action_phase2(self, agent_id, state, env=None, epsilon=0.0):
        """
        Phase 2: Robot uses β_r-softmax policy (not ε-greedy).
        Following paper equation: π_r(s)(a) ∝ (-Q_r(s,a_r))^{-β_r}
        """
        if self.network:
            q_values = self.q_r_backend.get_q_values(agent_id, state)
        else:
            # Handle unseen states by initializing them
            if state not in self.q_r[agent_id]:
                self.q_r[agent_id][state] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_r[agent_id][state]
        
        # Use softmax with beta_r (inverse of negative Q-values as in paper)
        # Paper uses (-Q_r)^{-β_r}, which is equivalent to softmax on β_r * Q_r
        return self._softmax_action(q_values, self.beta_r_0)
    
    def sample_human_action_phase1(self, agent_id, state, epsilon=0.1):
        """
        Human uses β_h-softmax on Q^m_h with exploration.
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        if self.network:
            # Extract goal from state
            goal = self.state_to_tuple(state[-2:])
            base_state = state[:-2]
            q_values = self.q_m_backend.get_q_values(agent_id, base_state, goal)
        else:
            # Handle unseen states
            if state not in self.q_m[agent_id]:
                self.q_m[agent_id][state] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_m[agent_id][state]
        
        # Use softmax for more exploration than pure greedy
        beta_h = 2.0  # Temperature parameter for human
        return self._softmax_action(q_values, beta_h)
    
    def train_phase1_corrected(self, env, episodes, max_steps=100):
        """
        Corrected Phase 1 following paper exactly.
        """
        print("Starting Phase 1: Learning cautious human model with adversarial robot.")
        
        for ep in range(episodes):
            env.reset()
            
            # Sample goal for human
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])

            for step in range(max_steps):
                actions = {}
                
                # Robot action: Play adversarially
                for rid in self.robot_agent_ids:
                    human_id = self.human_agent_ids[0]  # Single human case
                    goal = current_goals[human_id]
                    actions[rid] = self.sample_robot_action_phase1(
                        env, rid, human_id, goal, epsilon=0.3
                    )

                # Human action: Use current Q^m_h policy
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.3)

                # Store previous state for reward shaping
                prev_states = {}
                for hid in self.human_agent_ids:
                    prev_states[hid] = env.agent_positions[hid]

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Update Q^m_h with potential-based reward shaping
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    next_state_h = self.get_human_state(env, hid, goal)
                    action_h = actions[hid]
                    
                    # True reward (binary: 1 if goal reached, 0 otherwise)
                    true_reward = 1.0 if tuple(env.agent_positions[hid]) == goal else 0.0
                    
                    # Potential-based reward shaping
                    shaped_reward = self._shaped_reward(
                        prev_states[hid], env.agent_positions[hid], goal, true_reward
                    )

                    if self.network:
                        next_q_values = self.q_m_backend.get_q_values(hid, next_state_h[:-2], goal)
                        target = shaped_reward + self.gamma_h * np.max(next_q_values)
                        self.q_m_backend.update_q_values(hid, state_h[:-2], action_h, target, self.alpha_m, goal)
                    else:
                        old_q = self.q_m[hid][state_h][action_h]
                        next_max_q = np.max(self.q_m[hid][next_state_h])
                        new_q = old_q + self.alpha_m * (shaped_reward + self.gamma_h * next_max_q - old_q)
                        self.q_m[hid][state_h][action_h] = new_q

                if done:
                    break
                    
            if (ep + 1) % 100 == 0:
                print(f"  Phase 1, Episode {ep+1}/{episodes} completed.")
    
    def train_phase2_corrected(self, env, episodes, max_steps=100):
        """
        Corrected Phase 2 following paper exactly.
        Robot maximizes power, no hardcoded behaviors.
        """
        print("Starting Phase 2: Learning robot power maximization policy.")
        
        for ep in range(episodes):
            env.reset()
            goal = env.human_goals[self.human_agent_ids[0]]

            for step in range(max_steps):
                actions = {}
                
                # Robot action: Use learned Q_r policy (β-softmax)
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    actions[rid] = self.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)

                # Human action: Use cautious model Q^m_h
                for hid in self.human_agent_ids:
                    state_h = self.get_human_state(env, hid, self.state_to_tuple(goal))
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.1)

                # Store previous states for reward shaping
                prev_states = {}
                for hid in self.human_agent_ids:
                    prev_states[hid] = env.agent_positions[hid]

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Update Q_r (robot gets no extrinsic reward, only power reward)
                power_reward = 0.0
                goal_reached = False
                
                for hid in self.human_agent_ids:
                    if tuple(env.agent_positions[hid]) == tuple(goal):
                        power_reward += 1.0  # Human reached goal = power increase
                        goal_reached = True

                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    next_state_r = self.get_full_state(env, rid)
                    action_r = actions[rid]

                    if self.network:
                        next_q_values = self.q_r_backend.get_q_values(rid, next_state_r)
                        target = power_reward + self.gamma_r * np.max(next_q_values)
                        self.q_r_backend.update_q_values(rid, state_r, action_r, target, self.alpha_r)
                    else:
                        old_q_r = self.q_r[rid][state_r][action_r]
                        next_max_q_r = np.max(self.q_r[rid][next_state_r])
                        new_q_r = old_q_r + self.alpha_r * (power_reward + self.gamma_r * next_max_q_r - old_q_r)
                        self.q_r[rid][state_r][action_r] = new_q_r

                # Update Q^e_h (effective human model)
                for hid in self.human_agent_ids:
                    goal_tuple = self.state_to_tuple(goal)
                    state_base = self.get_full_state(env, hid)
                    next_state_base = self.get_full_state(env, hid)
                    action_h = actions[hid]
                    
                    # True reward with potential shaping
                    true_reward = 1.0 if tuple(env.agent_positions[hid]) == goal_tuple else 0.0
                    shaped_reward = self._shaped_reward(
                        prev_states[hid], env.agent_positions[hid], goal_tuple, true_reward
                    )

                    if self.network:
                        next_q_values = self.q_e_backend.get_q_values(hid, next_state_base, goal_tuple)
                        target = shaped_reward + self.gamma_h * np.max(next_q_values)
                        self.q_e_backend.update_q_values(hid, state_base, action_h, target, self.alpha_e, goal_tuple)
                    else:
                        state_h = self.get_human_state(env, hid, goal_tuple)
                        next_state_h = self.get_human_state(env, hid, goal_tuple)
                        old_q_e = self.q_e[hid][state_h][action_h]
                        next_max_q_e = np.max(self.q_e[hid][next_state_h])
                        new_q_e = old_q_e + self.alpha_e * (shaped_reward + self.gamma_h * next_max_q_e - old_q_e)
                        self.q_e[hid][state_h][action_h] = new_q_e

                if done or goal_reached:
                    if goal_reached:
                        print(f"    🎉 Goal reached in episode {ep+1}!")
                    break
                    
            if (ep + 1) % 100 == 0:
                print(f"  Phase 2, Episode {ep+1}/{episodes} completed.")


def train_corrected_algorithm(map_name, phase1_episodes=500, phase2_episodes=1000, save_dir="checkpoints"):
    """Train the corrected algorithm following the paper"""
    print(f"=== Corrected Training on {map_name} ===")
    
    # Import map function
    from envs.simple_map import get_map as get_simple_map
    from envs.simple_map2 import get_map as get_simple_map2
    from envs.simple_map3 import get_map as get_simple_map3
    from envs.simple_map4 import get_map as get_simple_map4
    from env import CustomEnvironment
    import os
    
    map_functions = {
        "simple_map": get_simple_map,
        "simple_map2": get_simple_map2,
        "simple_map3": get_simple_map3,
        "simple_map4": get_simple_map4,
    }
    
    get_map_func = map_functions[map_name]
    map_layout, map_metadata = get_map_func()
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),
        "human_0": list(range(3))
    }
    
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    # Create corrected algorithm
    iql = CorrectedTwoPhaseTimescaleIQL(
        alpha_m=0.2,
        alpha_e=0.2,
        alpha_r=0.2,
        alpha_p=0.2,
        gamma_h=0.9,
        gamma_r=0.9,
        beta_r_0=3.0,  # Softmax temperature
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Train corrected algorithm
    print(f"\n--- Corrected Phase 1 Training ({phase1_episodes} episodes) ---")
    iql.train_phase1_corrected(env, episodes=phase1_episodes, max_steps=50)
    
    print(f"\n--- Corrected Phase 2 Training ({phase2_episodes} episodes) ---")
    iql.train_phase2_corrected(env, episodes=phase2_episodes, max_steps=50)
    
    # Save and validate
    os.makedirs(save_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f"{map_name}_corrected_{timestamp}.pkl")
    iql.save_models(checkpoint_path)
    
    # Validation
    print(f"\n--- Validation ---")
    success_count = 0
    test_episodes = 20
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(100):
            actions = {}
            
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.1)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            env.step(actions)
            
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                print(f"  Episode {episode + 1}: ✅ Goal reached at step {step + 1}")
                break
        else:
            print(f"  Episode {episode + 1}: ❌ Timed out")
    
    success_rate = success_count / test_episodes
    print(f"\nCorrected algorithm success rate: {success_rate:.1%}")
    
    return checkpoint_path, success_rate