#!/usr/bin/env python3
"""
Fixed Enhanced IQL Algorithm with proper reward shaping and testing.
"""

import numpy as np
import os
from datetime import datetime
from enhanced_iql_algorithm import EnhancedTwoPhaseTimescaleIQL


class FixedEnhancedTwoPhaseTimescaleIQL(EnhancedTwoPhaseTimescaleIQL):
    """Fixed version with proper reward shaping and human behavior"""
    
    def train_phase1_fixed(self, env, episodes, max_steps=100):
        """Fixed Phase 1 with proper reward shaping for Q^m"""
        print("Starting Fixed Phase 1: Learning cautious model with reward shaping.")
        
        for ep in range(episodes):
            env.reset()
            
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])

            for step in range(max_steps):
                actions = {}
                
                # Store previous positions for reward shaping
                prev_positions = {}
                for hid in self.human_agent_ids:
                    prev_positions[hid] = tuple(env.agent_positions[hid])
                
                # Robot action: Adversarial
                for rid in self.robot_agent_ids:
                    human_id = self.human_agent_ids[0]
                    goal = current_goals[human_id]
                    actions[rid] = self.sample_robot_action_phase1(env, rid, human_id, goal, epsilon=0.3)

                # Human action: Use current Q^m policy
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.3)

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Update Q^m with proper reward shaping
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    next_state_h = self.get_human_state(env, hid, goal)
                    action_h = actions[hid]
                    
                    # True reward with proper shaping
                    true_reward = 1.0 if tuple(env.agent_positions[hid]) == goal else 0.0
                    shaped_reward = self._shaped_reward(
                        prev_positions[hid], env.agent_positions[hid], goal, true_reward
                    )

                    if self.network:
                        next_q_values = self.q_m_backend.get_q_values(hid, next_state_h[:-2], goal)
                        target = shaped_reward + self.gamma_h * np.max(next_q_values)
                        self.q_m_backend.update_q_values(hid, state_h[:-2], action_h, target, self.alpha_m, goal)
                    else:
                        if state_h not in self.q_m[hid]:
                            self.q_m[hid][state_h] = np.zeros(len(self.action_space_dict[hid]))
                        if next_state_h not in self.q_m[hid]:
                            self.q_m[hid][next_state_h] = np.zeros(len(self.action_space_dict[hid]))
                            
                        old_q = self.q_m[hid][state_h][action_h]
                        next_max_q = np.max(self.q_m[hid][next_state_h])
                        new_q = old_q + self.alpha_m * (shaped_reward + self.gamma_h * next_max_q - old_q)
                        self.q_m[hid][state_h][action_h] = new_q

                if done:
                    break
                    
            if (ep + 1) % 100 == 0:
                print(f"  Phase 1, Episode {ep+1}/{episodes} completed.")
    
    def sample_human_action_effective(self, agent_id, state, goal_tuple, epsilon=0.1):
        """Sample human action using effective model Q^e (for testing)"""
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        if self.network:
            base_state = state[:-2] if len(state) > 2 else state
            q_values = self.q_e_backend.get_q_values(agent_id, base_state, goal_tuple)
        else:
            # Construct state properly
            if isinstance(state, tuple):
                state_h = state
            else:
                state_h = tuple(state)
                
            if state_h not in self.q_e[agent_id]:
                self.q_e[agent_id][state_h] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_e[agent_id][state_h]
        
        # Use higher temperature for better action selection
        beta_h = 8.0  # Higher than in training
        return self._softmax_action(q_values, beta_h)
    
    def sample_robot_action_phase2(self, agent_id, state, env=None, epsilon=0.0):
        """FIXED robot action sampling WITH epsilon-greedy exploration"""
        # Use epsilon-greedy exploration as specified
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])
        
        if self.network:
            q_values = self.q_r_backend.get_q_values(agent_id, state)
        else:
            if state not in self.q_r[agent_id]:
                self.q_r[agent_id][state] = np.zeros(len(self.action_space_dict[agent_id]))
            q_values = self.q_r[agent_id][state]
        
        # Use greedy action selection (argmax) for exploitation
        return np.argmax(q_values)
    
    def test_policy(self, env, episodes=20, max_steps=50, verbose=True):
        """Test the learned policy with SAME exploration as training"""
        if verbose:
            print(f"\n--- Testing Policy ({episodes} episodes) ---")
        
        success_count = 0
        successful_steps = []
        
        for episode in range(episodes):
            env.reset()
            goal = env.human_goals[self.human_agent_ids[0]]
            goal_tuple = self.state_to_tuple(goal)
            
            if verbose and episode < 5:
                print(f"\nEpisode {episode + 1}:")
                print(f"  Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
                print(f"  Goal: {goal}")
            
            episode_success = False
            
            for step in range(max_steps):
                actions = {}
                
                # Human uses effective model Q^e WITH SAME EXPLORATION AS TRAINING
                for hid in self.human_agent_ids:
                    state_h = self.get_human_state(env, hid, goal_tuple)
                    actions[hid] = self.sample_human_action_effective(hid, state_h, goal_tuple, epsilon=0.1)
                
                # Robot uses learned policy WITH SAME EXPLORATION AS TRAINING
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    actions[rid] = self.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)
                
                if verbose and episode < 3:
                    action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
                    readable = {k: action_names.get(v, str(v)) for k, v in actions.items()}
                    print(f"    Step {step + 1}: {readable}")
                
                env.step(actions)
                
                if verbose and episode < 3:
                    print(f"      -> Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
                
                # Check for success
                human_pos = env.agent_positions[self.human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    success_count += 1
                    successful_steps.append(step + 1)
                    episode_success = True
                    if verbose and episode < 5:
                        print(f"    🎉 GOAL REACHED at step {step + 1}!")
                    break
            
            if verbose and episode < 5 and not episode_success:
                print(f"    ❌ Episode timed out")
        
        success_rate = success_count / episodes
        avg_steps = np.mean(successful_steps) if successful_steps else 0
        
        if verbose:
            print(f"\nResults: {success_rate:.1%} success ({success_count}/{episodes})")
            if successful_steps:
                print(f"Average steps: {avg_steps:.1f}, Range: {min(successful_steps)}-{max(successful_steps)}")
        
        return success_rate, avg_steps


def train_fixed_algorithm(map_name, phase1_episodes=800, phase2_episodes=1200):
    """Train the fixed algorithm"""
    print(f"=== Fixed Enhanced Training on {map_name} ===")
    
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
    
    # Create fixed algorithm
    iql = FixedEnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.4,  # Higher learning rates
        alpha_e=0.4,
        alpha_r=0.4,
        alpha_p=0.4,
        gamma_h=0.95,
        gamma_r=0.95,
        beta_r_0=5.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Fixed training with proper reward shaping
    print(f"\n--- Fixed Phase 1 Training ({phase1_episodes} episodes) ---")
    iql.train_phase1_fixed(env, episodes=phase1_episodes, max_steps=100)
    
    print(f"\n--- Fixed Phase 2 Training ({phase2_episodes} episodes) ---")
    iql.train_phase2_enhanced(env, episodes=phase2_episodes, max_steps=100)
    
    # Test the policy
    success_rate, avg_steps = iql.test_policy(env, episodes=30, max_steps=100, verbose=True)
    
    # Save if successful
    if success_rate >= 0.5:
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join("checkpoints", f"{map_name}_fixed_enhanced_{timestamp}.pkl")
        iql.save_models(checkpoint_path)
        print(f"Model saved: {checkpoint_path}")
    else:
        checkpoint_path = None
    
    return checkpoint_path, success_rate, avg_steps


if __name__ == "__main__":
    # Test on simple_map2 first
    checkpoint, success_rate, avg_steps = train_fixed_algorithm("simple_map2", 600, 800)
    print(f"\nFINAL RESULT: {success_rate:.1%} success, {avg_steps:.1f} avg steps")
    
    if success_rate >= 0.9:
        print("🎉 SUCCESS! Human is properly moving toward goal!")
    else:
        print("⚠️ Still needs improvement")