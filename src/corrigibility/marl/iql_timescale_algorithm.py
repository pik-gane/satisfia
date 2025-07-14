import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .q_learning_backends import create_q_learning_backend
except ImportError:
    from q_learning_backends import create_q_learning_backend


# Define the Q-network architecture
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TwoPhaseTimescaleIQL:
    def __init__(
        self,
        alpha_m=0.1,
        alpha_e=0.1,
        alpha_r=0.01,
        alpha_p=0.1,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=None,
        mu_g=None,
        p_g=0.0,
        action_space_dict=None,
        robot_agent_ids=None,
        human_agent_ids=None,
        network=False,
        state_dim=None,  # Will be determined by env
        goal_dim=2,
        env=None,  # Pass env to get state dimensions
    ):

        self.action_space_dict = action_space_dict
        self.robot_agent_ids = robot_agent_ids
        self.human_agent_ids = human_agent_ids
        self.num_robots = len(robot_agent_ids)
        self.num_humans = len(human_agent_ids)

        # Learning rates
        self.alpha_m = alpha_m
        self.alpha_e = alpha_e
        self.alpha_r = alpha_r
        self.alpha_p = alpha_p  # Learning rate for the power network

        # Discount factors
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r

        # Robot's rationality parameter for Phase 2
        self.beta_r_0 = beta_r_0

        # Human goal parameters
        self.G = G if G is not None else []
        self.mu_g = mu_g if mu_g is not None else []
        self.p_g = p_g

        # Q-value tables or networks using modern backend
        self.network = network
        if network:
            if env is None:
                raise ValueError(
                    "env must be provided when network=True to determine state_dim"
                )

            # Get actual observation to determine dimensions
            env.reset()
            sample_obs = env.observe(env.possible_agents[0])
            sample_state_tuple = self.state_to_tuple(sample_obs)
            state_dim = len(sample_state_tuple)
            
            # Use large state dimension to handle variable sizes
            max_state_dim = max(state_dim * 3, 64)  # Generous buffer for state variations
            
            print(f"Network backend: base_state_dim={state_dim}, max_state_dim={max_state_dim}")

            # Create modern Q-learning backends for each component
            # Human cautious model (Q_m) - includes goals
            self.q_m_backend = create_q_learning_backend(
                use_networks=True,
                agent_ids=human_agent_ids,
                action_space_dict={hid: action_space_dict[hid] for hid in human_agent_ids},
                state_dim=max_state_dim,
                use_goals=True,
                debug=False,
                hidden_sizes=[128, 64],
                device="cpu"
            )

            # Human effective model (Q_e) - includes goals  
            self.q_e_backend = create_q_learning_backend(
                use_networks=True,
                agent_ids=human_agent_ids,
                action_space_dict={hid: action_space_dict[hid] for hid in human_agent_ids},
                state_dim=max_state_dim,
                use_goals=True,
                debug=False,
                hidden_sizes=[128, 64],
                device="cpu"
            )

            # Robot model (Q_r) - no goals
            self.q_r_backend = create_q_learning_backend(
                use_networks=True,
                agent_ids=robot_agent_ids,
                action_space_dict={rid: action_space_dict[rid] for rid in robot_agent_ids},
                state_dim=max_state_dim,
                use_goals=False,
                debug=False,
                hidden_sizes=[128, 64],
                device="cpu"
            )

            # Power estimation - simplified for now (will be enhanced later)
            self.q_power = {hid: defaultdict(float) for hid in human_agent_ids}
        else:
            self.q_m = {
                hid: defaultdict(lambda: np.zeros(len(action_space_dict[hid])))
                for hid in human_agent_ids
            }
            self.q_e = {
                hid: defaultdict(lambda: np.zeros(len(action_space_dict[hid])))
                for hid in human_agent_ids
            }
            self.q_r = {
                rid: defaultdict(lambda: np.zeros(len(action_space_dict[rid])))
                for rid in robot_agent_ids
            }
            self.q_power = {hid: defaultdict(float) for hid in human_agent_ids}

    def get_full_state(self, env, agent_id):
        """Get the full, flattened observation for an agent."""
        return self.state_to_tuple(env.observe(agent_id))

    def state_to_tuple(self, state_dict):
        """Convert observation to tuple for Q-table indexing."""
        if isinstance(state_dict, tuple):
            return tuple(int(x) for x in state_dict)
        try:
            arr = np.asarray(state_dict).flatten()
            return tuple(int(x) for x in arr)
        except Exception:
            return tuple(int(x) for x in state_dict)

    def get_human_state(self, env, hid, goal):
        # Helper to create the goal-conditioned state for human networks
        base_state = self.get_full_state(env, hid)
        combined_state = base_state + goal
        return combined_state
    
    def _get_movement_action(self, env, agent_id, target_agent_id):
        """Get action to move towards target agent using environment's action scheme."""
        agent_pos = env.agent_positions[agent_id]
        target_pos = env.agent_positions[target_agent_id]
        current_dir = env.agent_dirs[agent_id]
        
        # Calculate desired direction
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        
        # Determine target direction index
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        # Actions: 0=turn_left, 1=turn_right, 2=forward
        if current_dir == target_dir:
            return 2  # Move forward
        else:
            # Calculate turn direction
            left_turn = (current_dir - 1) % 4
            right_turn = (current_dir + 1) % 4
            
            if left_turn == target_dir:
                return 0  # Turn left
            elif right_turn == target_dir:
                return 1  # Turn right
            else:
                # Need 2 turns, pick left arbitrarily
                return 0  # Turn left
    
    def _get_assistive_action(self, env, robot_id):
        """Get action for robot to assist human in reaching goal."""
        robot_pos = env.agent_positions[robot_id]
        current_dir = env.agent_dirs[robot_id]
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        
        # Check if there's a key in front to pick up
        dx, dy = deltas[current_dir]
        front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
        
        for key in env.keys:
            if tuple(key["pos"]) == front_pos:
                return 3  # Actions.pickup
        
        # Check if there's a locked door in front to toggle
        for door in env.doors:
            if (tuple(door["pos"]) == front_pos and 
                door["is_locked"] and 
                door["color"] in env.robot_has_keys):
                return 5  # Actions.toggle
        
        # Look for keys to collect
        for key in env.keys:
            key_pos = tuple(key["pos"])
            # Move towards key
            dx = key_pos[0] - robot_pos[0]
            dy = key_pos[1] - robot_pos[1]
            
            if abs(dx) > abs(dy):
                target_dir = 2 if dx > 0 else 0  # Down or Up
            else:
                target_dir = 1 if dy > 0 else 3  # Right or Left
            
            if current_dir == target_dir:
                return 2  # Move forward
            else:
                left_turn = (current_dir - 1) % 4
                right_turn = (current_dir + 1) % 4
                
                if left_turn == target_dir:
                    return 0  # Turn left
                elif right_turn == target_dir:
                    return 1  # Turn right
                else:
                    return 0  # Turn left arbitrarily
        
        # Look for locked doors to open
        for door in env.doors:
            if door["is_locked"] and door["color"] in env.robot_has_keys:
                door_pos = tuple(door["pos"])
                # Move towards door
                dx = door_pos[0] - robot_pos[0]
                dy = door_pos[1] - robot_pos[1]
                
                if abs(dx) > abs(dy):
                    target_dir = 2 if dx > 0 else 0  # Down or Up
                else:
                    target_dir = 1 if dy > 0 else 3  # Right or Left
                
                if current_dir == target_dir:
                    return 2  # Move forward
                else:
                    left_turn = (current_dir - 1) % 4
                    right_turn = (current_dir + 1) % 4
                    
                    if left_turn == target_dir:
                        return 0  # Turn left
                    elif right_turn == target_dir:
                        return 1  # Turn right
                    else:
                        return 0  # Turn left arbitrarily
        
        # Default: try to move towards human (blocking behavior)
        human_id = self.human_agent_ids[0]
        return self._get_movement_action(env, robot_id, human_id)

    def train(
        self,
        environment,
        phase1_episodes,
        phase2_episodes,
        render=False,
        render_delay=100,
    ):
        self.train_phase1(environment, episodes=phase1_episodes)
        self.train_phase2(environment, episodes=phase2_episodes)

    def train_phase1(self, env, episodes, max_steps=200, reward_shaping=True):
        msg = "Starting Phase 1: Learning cautious human model with robot blocking."
        print(msg)
        for ep in range(episodes):
            env.reset()
            # Sample initial goal for each human
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])

            for step in range(max_steps):
                actions = {}
                # Robot action (Phase 1): Block the human
                human_id = self.human_agent_ids[
                    0
                ]  # Simple assumption for robot targeting
                for rid in self.robot_agent_ids:
                    # Get blocking action towards human
                    actions[rid] = self._get_movement_action(env, rid, human_id)

                # Human action (Phase 1)
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.3)

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Update human's cautious model (Q_m)
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    next_state_h = self.get_human_state(
                        env, hid, goal
                    )  # Goal is constant
                    action_h = actions[hid]
                    reward_h = rewards[hid]

                    # --- Reward Shaping --- #
                    if reward_shaping:
                        # Get current human position
                        current_pos = np.array(env.agent_positions[hid])
                        goal_pos = np.array(goal)
                        
                        # Check if human reached goal
                        if np.array_equal(current_pos, goal_pos):
                            reward_h += 1000.0  # Very large reward for reaching goal
                        # Only give small shaping reward, not large enough to dominate
                        else:
                            # Very small distance-based shaping
                            dist_to_goal = np.sum(np.abs(current_pos - goal_pos))
                            shaping_reward = -dist_to_goal * 0.01  # Small negative for distance
                            reward_h += shaping_reward
                    # --- End Reward Shaping ---

                    if self.network:
                        # Use modern Q-learning backend for Q_m updates
                        next_q_values = self.q_m_backend.get_q_values(hid, next_state_h[:-2], goal)
                        target = reward_h + self.gamma_h * np.max(next_q_values)
                        self.q_m_backend.update_q_values(hid, state_h[:-2], action_h, target, self.alpha_m, goal)
                    else:
                        old_q = self.q_m[hid][state_h][action_h]
                        next_max_q = np.max(self.q_m[hid][next_state_h])
                        new_q = old_q + self.alpha_m * (
                            reward_h + self.gamma_h * next_max_q - old_q
                        )
                        self.q_m[hid][state_h][action_h] = new_q

                if done:
                    break
            if (ep + 1) % 100 == 0:
                print(f"  Phase 1, Episode {ep+1}/{episodes} completed.")

    def train_phase2(self, env, episodes, max_steps=200):
        msg = "Starting Phase 2: Learning assistive robot policy and power estimation."
        print(msg)
        all_possible_goals = env.get_all_possible_human_goals()

        for ep in range(episodes):
            env.reset()
            goal = env.human_goals[self.human_agent_ids[0]]

            for step in range(max_steps):
                actions = {}
                # Robot action (Phase 2): Assist the human
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    actions[rid] = self.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)

                # Human action (Phase 2): Use cautious model (Q_m)
                for hid in self.human_agent_ids:
                    state_h = self.get_human_state(env, hid, goal)
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.3)

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # --- Train Power and Robot Networks ---
                cumulative_power_reward = 0
                for hid in self.human_agent_ids:
                    # Get robot's state for power network (robot observes environment)
                    robot_id = self.robot_agent_ids[
                        0
                    ]  # Use first robot for observations
                    state_r = self.get_full_state(env, robot_id)
                    next_state_r = self.get_full_state(env, robot_id)

                    # 1. Update Power Network
                    if self.network:
                        # Estimate power by sampling possible goals using human's effective policy
                        power_target_val = 0
                        goal_positions = list(all_possible_goals.values())
                        for sample_goal in goal_positions:
                            sample_goal_tuple = self.state_to_tuple(sample_goal)
                            state_base = self.get_full_state(env, hid)
                            # Use modern Q-learning backend for Q_e
                            q_vals = self.q_e_backend.get_q_values(hid, state_base, sample_goal_tuple)
                            power_target_val += np.max(q_vals)
                        power_target_val /= len(goal_positions)
                        
                        # Store power estimate (simplified - could use separate network later)
                        self.q_power[hid][state_r] = power_target_val
                        cumulative_power_reward += power_target_val
                    else:
                        # For tabular case, estimate power from Q_e tables
                        power_target_val = 0
                        goal_positions = list(all_possible_goals.values())
                        for sample_goal in goal_positions:
                            goal_state_h = self.get_human_state(
                                env, hid, self.state_to_tuple(sample_goal)
                            )
                            q_vals = self.q_e[hid][goal_state_h]
                            power_target_val += np.max(q_vals)
                        power_target_val /= len(goal_positions)
                        
                        self.q_power[hid][state_r] = power_target_val
                        cumulative_power_reward += self.q_power[hid][next_state_r]

                # 2. Update Robot's Q_r network
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    next_state_r = self.get_full_state(env, rid)
                    action_r = actions[rid]
                    
                    # Add reward shaping for robot cooperative actions
                    robot_reward = 0
                    prev_keys = len(env.robot_has_keys) - 1 if len(env.robot_has_keys) > 0 else 0
                    prev_doors_open = sum(1 for door in env.doors if door["is_open"]) - 1 if any(door["is_open"] for door in env.doors) else 0
                    
                    # Reward for successfully picking up keys (check if keys increased)
                    if action_r == 3:  # pickup action
                        current_keys = len(env.robot_has_keys)
                        if current_keys > prev_keys:
                            robot_reward += 100.0
                    
                    # Reward for successfully opening doors (check if doors opened)
                    if action_r == 5:  # toggle action
                        current_doors_open = sum(1 for door in env.doors if door["is_open"])
                        if current_doors_open > prev_doors_open:
                            robot_reward += 100.0
                    
                    if self.network:
                        # Use modern Q-learning backend for robot
                        next_q_values = self.q_r_backend.get_q_values(rid, next_state_r)
                        target = robot_reward + cumulative_power_reward + self.gamma_r * np.max(next_q_values)
                        self.q_r_backend.update_q_values(rid, state_r, action_r, target, self.alpha_r)
                    else:
                        old_q_r = self.q_r[rid][state_r][action_r]
                        next_max_q_r = np.max(self.q_r[rid][next_state_r])
                        new_q_r = old_q_r + self.alpha_r * (
                            robot_reward + cumulative_power_reward
                            + self.gamma_r * next_max_q_r
                            - old_q_r
                        )
                        self.q_r[rid][state_r][action_r] = new_q_r

                # 3. Update Human's Effective model (Q_e)
                for hid in self.human_agent_ids:
                    goal_tuple = self.state_to_tuple(goal)
                    state_base = self.get_full_state(env, hid)
                    next_state_base = self.get_full_state(env, hid)
                    action_h = actions[hid]
                    reward_h = rewards[hid]
                    
                    # Add reward shaping for phase 2 as well
                    current_pos = np.array(env.agent_positions[hid])
                    goal_pos = np.array(goal)
                    
                    # Check if human reached goal
                    if np.array_equal(current_pos, goal_pos):
                        reward_h += 1000.0  # Very large reward for reaching goal
                    else:
                        # Very small distance-based shaping
                        dist_to_goal = np.sum(np.abs(current_pos - goal_pos))
                        shaping_reward = -dist_to_goal * 0.01  # Small negative for distance
                        reward_h += shaping_reward
                    if self.network:
                        # Use modern Q-learning backend for Q_e updates
                        next_q_values = self.q_e_backend.get_q_values(hid, next_state_base, goal_tuple)
                        target = reward_h + self.gamma_h * np.max(next_q_values)
                        self.q_e_backend.update_q_values(hid, state_base, action_h, target, self.alpha_e, goal_tuple)
                    else:
                        state_h = self.get_human_state(env, hid, goal)
                        next_state_h = self.get_human_state(env, hid, goal)
                        old_q_e = self.q_e[hid][state_h][action_h]
                        next_max_q_e = np.max(self.q_e[hid][next_state_h])
                        new_q_e = old_q_e + self.alpha_e * (
                            reward_h + self.gamma_h * next_max_q_e - old_q_e
                        )
                        self.q_e[hid][state_h][action_h] = new_q_e

                if done:
                    break
            if (ep + 1) % 100 == 0:
                print(f"  Phase 2, Episode {ep+1}/{episodes} completed.")

    def sample_human_action_phase1(self, agent_id, state, epsilon=0.1):
        """Sample human action using softmax policy in Phase 1."""

        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])

        if self.network:
            # Extract goal from state (last 2 elements) and base state
            goal = state[-2:]
            base_state = state[:-2]
            q_values = self.q_m_backend.get_q_values(agent_id, base_state, goal)
        else:
            q_values = self.q_m[agent_id][state]
        
        # For deterministic testing (epsilon=0.0), use greedy action
        if epsilon == 0.0:
            return np.argmax(q_values)
        
        # Use greedy action for now (can be softmax later if needed)
        return np.argmax(q_values)

    def sample_human_action_phase2(self, agent_id, state, goal, epsilon=0.0):
        # In Phase 2, human uses the cautious policy Q_m, conditioned on the goal
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])

        if self.network:
            goal_tuple = self.state_to_tuple(goal)
            q_values = self.q_m_backend.get_q_values(agent_id, state, goal_tuple)
            return np.argmax(q_values)
        else:
            full_state = self.get_human_state(None, None, goal)  # Reconstruct full state
            full_state = state + goal
            return np.argmax(self.q_m[agent_id][full_state])

    def sample_robot_action_phase2(self, agent_id, state, env=None, epsilon=0.1):
        """Sample robot action using softmax policy in Phase 2."""
        
        # Sometimes use assistive heuristic for exploration
        if env is not None and np.random.rand() < epsilon:
            return self._get_assistive_action(env, agent_id)

        if self.network:
            q_values_np = self.q_r_backend.get_q_values(agent_id, state)
        else:
            q_values_np = self.q_r[agent_id][state]

        # For deterministic testing (epsilon=0.0), use greedy action
        if epsilon == 0.0:
            return np.argmax(q_values_np)

        # Softmax over Q-values with proper scaling
        if np.all(q_values_np == 0):
            # If all Q-values are zero, use uniform random
            return np.random.choice(self.action_space_dict[agent_id])
        
        # Use beta_r_0 for proper softmax scaling
        scaled_q = self.beta_r_0 * (q_values_np - np.max(q_values_np))
        exp_q = np.exp(scaled_q)
        probabilities = exp_q / np.sum(exp_q)

        return np.random.choice(self.action_space_dict[agent_id], p=probabilities)

    def save_models(self, file_prefix):
        if self.network:
            if file_prefix.endswith(".pkl"):
                file_prefix = file_prefix[:-4]
            self.q_m_backend.save_models(f"{file_prefix}_q_m.pt")
            self.q_e_backend.save_models(f"{file_prefix}_q_e.pt")
            self.q_r_backend.save_models(f"{file_prefix}_q_r.pt")
            # Save power estimates as pickle
            q_power_dict = {aid: dict(qtable) for aid, qtable in self.q_power.items()}
            with open(f"{file_prefix}_q_power.pkl", "wb") as f:
                pickle.dump(q_power_dict, f)
        else:
            filename = (
                file_prefix if file_prefix.endswith(".pkl") else f"{file_prefix}.pkl"
            )
            # Convert defaultdicts to regular dicts for pickling
            q_m_dict = {aid: dict(qtable) for aid, qtable in self.q_m.items()}
            q_e_dict = {aid: dict(qtable) for aid, qtable in self.q_e.items()}
            q_r_dict = {aid: dict(qtable) for aid, qtable in self.q_r.items()}
            q_power_dict = {aid: dict(qtable) for aid, qtable in self.q_power.items()}

            with open(filename, "wb") as f:
                pickle.dump(
                    {
                        "q_m": q_m_dict,
                        "q_e": q_e_dict,
                        "q_r": q_r_dict,
                        "q_power": q_power_dict,
                        "action_space_dict": self.action_space_dict,
                        "robot_agent_ids": self.robot_agent_ids,
                        "human_agent_ids": self.human_agent_ids,
                        "G": self.G,
                        "mu_g": self.mu_g,
                    },
                    f,
                )

    def load_models(self, file_prefix):
        if self.network:
            if file_prefix.endswith(".pkl"):
                file_prefix = file_prefix[:-4]
            self.q_m_backend.load_models(f"{file_prefix}_q_m.pt")
            self.q_e_backend.load_models(f"{file_prefix}_q_e.pt")
            self.q_r_backend.load_models(f"{file_prefix}_q_r.pt")
            # Load power estimates
            try:
                with open(f"{file_prefix}_q_power.pkl", "rb") as f:
                    q_power_dict = pickle.load(f)
                    self.q_power = {hid: defaultdict(float, qtable) for hid, qtable in q_power_dict.items()}
            except FileNotFoundError:
                self.q_power = {hid: defaultdict(float) for hid in self.human_agent_ids}
        else:
            filename = (
                file_prefix if file_prefix.endswith(".pkl") else f"{file_prefix}.pkl"
            )
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_m = data["q_m"]
                self.q_e = data["q_e"]
                self.q_r = data["q_r"]
                self.q_power = data.get(
                    "q_power",
                    {hid: defaultdict(float) for hid in self.human_agent_ids},
                )
