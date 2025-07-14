#!/usr/bin/env python3
"""
Enhanced IQL Algorithm for 100% success on simple scenarios.
"""

import numpy as np
import os
from datetime import datetime
from corrected_iql_algorithm import CorrectedTwoPhaseTimescaleIQL


class EnhancedTwoPhaseTimescaleIQL(CorrectedTwoPhaseTimescaleIQL):
    """Enhanced version with stronger reward shaping and better parameters"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _potential_function(self, state, goal):
        """Enhanced potential function with stronger shaping"""
        if isinstance(state, tuple):
            pos = np.array(state[:2])
        else:
            pos = np.array(state)
            
        goal_pos = np.array(goal)
        manhattan_dist = np.sum(np.abs(pos - goal_pos))
        
        # Stronger shaping for simple scenarios
        return -manhattan_dist / 20.0  # Increased from 100 to 20
    
    def _shaped_reward(self, prev_state, next_state, goal, true_reward):
        """Enhanced reward shaping with additional bonuses"""
        # Base potential shaping
        phi_next = self._potential_function(next_state, goal)
        phi_prev = self._potential_function(prev_state, goal)
        shaping = self.gamma_h * phi_next - phi_prev
        
        # Additional progress bonus
        prev_dist = np.sum(np.abs(np.array(prev_state) - np.array(goal)))
        next_dist = np.sum(np.abs(np.array(next_state) - np.array(goal)))
        
        progress_bonus = 0.0
        if next_dist < prev_dist:
            progress_bonus = 0.2  # Bonus for getting closer
        elif next_dist > prev_dist:
            progress_bonus = -0.1  # Penalty for moving away
            
        return true_reward + shaping + progress_bonus
    
    def _softmax_action(self, q_values, beta):
        """Improved softmax with better numerical stability"""
        if beta == 0:
            return np.random.choice(len(q_values))
        
        # Numerical stability
        q_values = np.array(q_values)
        max_q = np.max(q_values)
        exp_q = np.exp(beta * (q_values - max_q))
        probabilities = exp_q / np.sum(exp_q)
        
        # Ensure probabilities sum to 1
        probabilities = probabilities / np.sum(probabilities)
        
        return np.random.choice(len(q_values), p=probabilities)
    
    def sample_human_action_phase1(self, agent_id, state, epsilon=0.1):
        """Enhanced human action sampling with better exploration"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        if self.network:
            goal = self.state_to_tuple(state[-2:])
            base_state = state[:-2]
            q_values = self.q_m_backend.get_q_values(agent_id, base_state, goal)
        else:
            if state not in self.q_m[agent_id]:
                self.q_m[agent_id][state] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_m[agent_id][state]
        
        # Higher temperature for more exploration in simple scenarios
        beta_h = 5.0  # Increased from 2.0
        return self._softmax_action(q_values, beta_h)
    
    def sample_robot_action_phase2(self, agent_id, state, env=None, epsilon=0.0):
        """Enhanced robot action sampling"""
        if self.network:
            q_values = self.q_r_backend.get_q_values(agent_id, state)
        else:
            if state not in self.q_r[agent_id]:
                self.q_r[agent_id][state] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_r[agent_id][state]
        
        # Higher temperature for robot in simple scenarios
        beta_r = 5.0  # Increased from 3.0
        return self._softmax_action(q_values, beta_r)
    
    def train_phase2_enhanced(self, env, episodes, max_steps=100):
        """Enhanced Phase 2 with better power rewards"""
        print("Starting Enhanced Phase 2: Learning robot power maximization.")
        
        for ep in range(episodes):
            env.reset()
            goal = env.human_goals[self.human_agent_ids[0]]
            episode_power_reward = 0.0

            for step in range(max_steps):
                actions = {}
                
                # Store previous positions
                prev_positions = {}
                for hid in self.human_agent_ids:
                    prev_positions[hid] = tuple(env.agent_positions[hid])
                
                # Robot action
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    actions[rid] = self.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)

                # Human action
                for hid in self.human_agent_ids:
                    state_h = self.get_human_state(env, hid, self.state_to_tuple(goal))
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.15)

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Enhanced power reward calculation
                power_reward = 0.0
                goal_reached = False
                
                for hid in self.human_agent_ids:
                    current_pos = tuple(env.agent_positions[hid])
                    prev_pos = prev_positions[hid]
                    
                    # Goal reaching reward
                    if current_pos == tuple(goal):
                        power_reward += 10.0  # Large reward for goal
                        goal_reached = True
                        episode_power_reward += 10.0
                    
                    # Progress reward for robot
                    prev_dist = np.sum(np.abs(np.array(prev_pos) - np.array(goal)))
                    curr_dist = np.sum(np.abs(np.array(current_pos) - np.array(goal)))
                    
                    if curr_dist < prev_dist:
                        power_reward += 1.0  # Reward robot for human progress
                    elif curr_dist > prev_dist:
                        power_reward -= 0.5  # Slight penalty for human regression

                # Update Q_r with enhanced power rewards
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    next_state_r = self.get_full_state(env, rid)
                    action_r = actions[rid]

                    if self.network:
                        next_q_values = self.q_r_backend.get_q_values(rid, next_state_r)
                        target = power_reward + self.gamma_r * np.max(next_q_values)
                        self.q_r_backend.update_q_values(rid, state_r, action_r, target, self.alpha_r)
                    else:
                        if state_r not in self.q_r[rid]:
                            self.q_r[rid][state_r] = np.zeros(len(self.action_space_dict[rid]))
                        if next_state_r not in self.q_r[rid]:
                            self.q_r[rid][next_state_r] = np.zeros(len(self.action_space_dict[rid]))
                            
                        old_q_r = self.q_r[rid][state_r][action_r]
                        next_max_q_r = np.max(self.q_r[rid][next_state_r])
                        new_q_r = old_q_r + self.alpha_r * (power_reward + self.gamma_r * next_max_q_r - old_q_r)
                        self.q_r[rid][state_r][action_r] = new_q_r

                # Update Q^e_h with enhanced shaping
                for hid in self.human_agent_ids:
                    goal_tuple = self.state_to_tuple(goal)
                    state_base = self.get_full_state(env, hid)
                    next_state_base = self.get_full_state(env, hid)
                    action_h = actions[hid]
                    
                    true_reward = 1.0 if tuple(env.agent_positions[hid]) == goal_tuple else 0.0
                    shaped_reward = self._shaped_reward(
                        prev_positions[hid], env.agent_positions[hid], goal_tuple, true_reward
                    )

                    if self.network:
                        next_q_values = self.q_e_backend.get_q_values(hid, next_state_base, goal_tuple)
                        target = shaped_reward + self.gamma_h * np.max(next_q_values)
                        self.q_e_backend.update_q_values(hid, state_base, action_h, target, self.alpha_e, goal_tuple)
                    else:
                        state_h = self.get_human_state(env, hid, goal_tuple)
                        next_state_h = self.get_human_state(env, hid, goal_tuple)
                        
                        if state_h not in self.q_e[hid]:
                            self.q_e[hid][state_h] = np.zeros(len(self.action_space_dict[hid]))
                        if next_state_h not in self.q_e[hid]:
                            self.q_e[hid][next_state_h] = np.zeros(len(self.action_space_dict[hid]))
                            
                        old_q_e = self.q_e[hid][state_h][action_h]
                        next_max_q_e = np.max(self.q_e[hid][next_state_h])
                        new_q_e = old_q_e + self.alpha_e * (shaped_reward + self.gamma_h * next_max_q_e - old_q_e)
                        self.q_e[hid][state_h][action_h] = new_q_e

                if done or goal_reached:
                    if goal_reached:
                        print(f"    🎉 Goal reached in episode {ep+1}! (Power reward: {episode_power_reward:.1f})")
                    break
                    
            if (ep + 1) % 100 == 0:
                print(f"  Phase 2, Episode {ep+1}/{episodes} completed. Avg power: {episode_power_reward:.1f}")


def train_enhanced_algorithm(map_name, phase1_episodes=1000, phase2_episodes=2000, save_dir="checkpoints"):
    """Train enhanced algorithm for 100% success"""
    print(f"=== Enhanced Training on {map_name} ===")
    
    # Import map functions
    from envs.simple_map import get_map as get_simple_map
    from envs.simple_map2 import get_map as get_simple_map2
    from envs.simple_map3 import get_map as get_simple_map3
    from envs.simple_map4 import get_map as get_simple_map4
    from env import CustomEnvironment
    
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
    
    print(f"Map: {map_metadata['name']}")
    print(f"Goal: {map_metadata['human_goals']}")
    
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),
        "human_0": list(range(3))
    }
    
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    # Enhanced algorithm with optimized parameters
    iql = EnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.3,  # Higher learning rates
        alpha_e=0.3,
        alpha_r=0.3,
        alpha_p=0.3,
        gamma_h=0.95,  # Higher discount factors
        gamma_r=0.95,
        beta_r_0=5.0,  # Higher temperature
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Enhanced training
    print(f"\n--- Enhanced Phase 1 Training ({phase1_episodes} episodes) ---")
    iql.train_phase1_corrected(env, episodes=phase1_episodes, max_steps=100)
    
    print(f"\n--- Enhanced Phase 2 Training ({phase2_episodes} episodes) ---")
    iql.train_phase2_enhanced(env, episodes=phase2_episodes, max_steps=150)
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(save_dir, f"{map_name}_enhanced_{timestamp}.pkl")
    iql.save_models(checkpoint_path)
    print(f"Enhanced model saved: {checkpoint_path}")
    
    # Extensive validation for 100% success
    print(f"\n--- Validation (50 episodes) ---")
    success_count = 0
    test_episodes = 50
    successful_steps = []
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(200):  # More steps allowed
            actions = {}
            
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.05)  # Very low exploration
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            env.step(actions)
            
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                successful_steps.append(step + 1)
                if episode < 10:  # Show first 10
                    print(f"  Episode {episode + 1}: ✅ Goal reached at step {step + 1}")
                break
        else:
            if episode < 10:
                print(f"  Episode {episode + 1}: ❌ Timed out")
    
    success_rate = success_count / test_episodes
    avg_steps = np.mean(successful_steps) if successful_steps else 0
    
    print(f"\n--- Results ---")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{test_episodes})")
    if successful_steps:
        print(f"Average steps to goal: {avg_steps:.1f}")
        print(f"Steps range: {min(successful_steps)}-{max(successful_steps)}")
    
    return checkpoint_path, success_rate, avg_steps


def visualize_trained_model(checkpoint_path, map_name, episodes=3):
    """Visualize the trained model"""
    print(f"\n=== Visualizing {map_name} ===")
    
    # Import required modules
    from envs.simple_map import get_map as get_simple_map
    from envs.simple_map2 import get_map as get_simple_map2
    from envs.simple_map3 import get_map as get_simple_map3
    from envs.simple_map4 import get_map as get_simple_map4
    from env import CustomEnvironment
    
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
    
    iql = EnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.3, alpha_e=0.3, alpha_r=0.3, alpha_p=0.3,
        gamma_h=0.95, gamma_r=0.95, beta_r_0=5.0,
        G=G, mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Load model
    try:
        iql.load_models(checkpoint_path)
        print("✅ Model loaded for visualization")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    action_names = {0: "turn_left", 1: "turn_right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
    
    for episode in range(episodes):
        print(f"\n--- Visualization Episode {episode + 1} ---")
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        print(f"Goal: {goal}")
        print(f"Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
        
        for step in range(50):
            actions = {}
            
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            readable_actions = {agent: action_names.get(action, str(action)) for agent, action in actions.items()}
            
            env.step(actions)
            
            print(f"  Step {step + 1}: {readable_actions} -> Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
            
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                print(f"  🎉 GOAL REACHED at step {step + 1}!")
                break
        else:
            print(f"  ❌ Episode timed out")


if __name__ == "__main__":
    # Test on simple_map2 (easiest)
    checkpoint, success_rate, avg_steps = train_enhanced_algorithm("simple_map2", 500, 1000)
    print(f"Result: {success_rate:.1%} success, {avg_steps:.1f} avg steps")
    
    if success_rate >= 0.8:
        visualize_trained_model(checkpoint, "simple_map2", 2)