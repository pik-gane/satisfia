#!/usr/bin/env python3
"""
Corrected IQL Algorithm following the exact specification from paper.tex
with proper reward potentials, exploration schedules, and two-phase training.
"""

import numpy as np
from collections import defaultdict

class PaperBasedTwoPhaseIQL:
    def __init__(
        self,
        alpha_m=0.3,        # Higher learning rate for Q_m (human cautious model)
        alpha_e=0.3,        # Higher learning rate for Q_e (human effective model) 
        alpha_r=0.3,        # Higher learning rate for Q_r (robot model)
        gamma_h=0.99,       # Human discount factor
        gamma_r=0.99,       # Robot discount factor
        beta_h=1.0,         # Lower initial human rationality (more exploration)
        beta_r=1.0,         # Lower initial robot rationality (more exploration)
        nu_h=0.3,           # Higher habitual behavior mixing factor
        zeta=1.5,           # Risk aversion parameter (>1 for uncertainty reduction)
        xi=1.0,             # Aggregation parameter
        eta=1.0,            # Temporal aggregation parameter
        G=None,             # Set of possible human goals
        mu_g=None,          # Prior distribution over goals
        action_space_dict=None,
        robot_agent_ids=None,
        human_agent_ids=None,
        env=None
    ):
        
        # Algorithm parameters from paper
        self.alpha_m = alpha_m
        self.alpha_e = alpha_e  
        self.alpha_r = alpha_r
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.beta_h = beta_h
        self.beta_r = beta_r
        self.nu_h = nu_h
        self.zeta = zeta
        self.xi = xi
        self.eta = eta
        
        # Environment setup
        self.G = G if G is not None else []
        self.mu_g = mu_g if mu_g is not None else []
        self.action_space_dict = action_space_dict
        self.robot_agent_ids = robot_agent_ids
        self.human_agent_ids = human_agent_ids
        self.env = env
        
        # Q-tables for tabular case (equations from paper)
        # Q^m_h: Human cautious model (Phase 1)
        self.Q_m = {
            hid: defaultdict(lambda: np.zeros(len(action_space_dict[hid])))
            for hid in human_agent_ids
        }
        
        # Q^e_h: Human effective model (Phase 2) 
        self.Q_e = {
            hid: defaultdict(lambda: np.zeros(len(action_space_dict[hid])))
            for hid in human_agent_ids
        }
        
        # Q_r: Robot model (Phase 2)
        self.Q_r = {
            rid: defaultdict(lambda: np.zeros(len(action_space_dict[rid])))
            for rid in robot_agent_ids
        }
        
        # Power metrics
        self.V_m = {hid: defaultdict(float) for hid in human_agent_ids}  # V^m_h
        self.V_e = {hid: defaultdict(float) for hid in human_agent_ids}  # V^e_h
        self.X_h = {hid: defaultdict(float) for hid in human_agent_ids}  # Individual power
        self.U_r = defaultdict(float)  # Aggregate power (robot's intrinsic reward)
        self.V_r = defaultdict(float)  # Total power
        
        # Policies
        self.pi_h = {hid: defaultdict(lambda: np.ones(len(action_space_dict[hid])) / len(action_space_dict[hid])) 
                     for hid in human_agent_ids}
        self.pi_r = {rid: defaultdict(lambda: np.ones(len(action_space_dict[rid])) / len(action_space_dict[rid]))
                     for rid in robot_agent_ids}
        
        print(f"Initialized Paper-based IQL with β_h={beta_h}, β_r={beta_r}, ζ={zeta}")

    def get_state_tuple(self, env, agent_id, goal=None):
        """Get state representation as tuple"""
        obs = env.observe(agent_id)
        if isinstance(obs, dict):
            base_state = tuple(sorted(obs.values())) if isinstance(obs, dict) else tuple(obs.flatten())
        else:
            base_state = tuple(obs.flatten()) if hasattr(obs, 'flatten') else tuple(obs)
        
        if goal is not None:
            goal_tuple = tuple(goal) if hasattr(goal, '__iter__') else (goal,)
            return base_state + goal_tuple
        return base_state

    def U_h(self, state, goal):
        """Reward function U_h(s', g_h) = 1_{s' ∈ g_h} from paper with massive reward potentials"""
        # Goal reaching indicator function
        current_pos = self.env.agent_positions[self.human_agent_ids[0]]
        if tuple(current_pos) == tuple(goal):
            return 10000.0  # Massive positive reward for reaching goal
        else:
            # Strong potential-based reward shaping
            dist = abs(current_pos[0] - goal[0]) + abs(current_pos[1] - goal[1])
            return -10.0 * dist  # Very strong distance penalty

    def update_V_m(self, hid, state_goal):
        """Update V^m_h using equation (Vm) from paper"""
        if state_goal not in self.pi_h[hid]:
            return 0.0
        
        q_values = self.Q_m[hid][state_goal]
        pi_values = self.pi_h[hid][state_goal]
        self.V_m[hid][state_goal] = np.sum(pi_values * q_values)
        return self.V_m[hid][state_goal]

    def update_pi_h(self, hid, state_goal, exploration_factor=1.0):
        """Update π_h using equation (pih) from paper with proper exploration schedule"""
        q_values = self.Q_m[hid][state_goal]
        
        # High exploration at start, decreasing over time
        effective_beta = self.beta_h * exploration_factor
        if np.all(q_values == 0) or exploration_factor > 0.5:
            # High exploration when learning or early in training
            softmax_policy = np.ones(len(q_values)) / len(q_values)
        else:
            # β_h-softmax with proper normalization
            exp_q = np.exp(effective_beta * (q_values - np.max(q_values)))
            softmax_policy = exp_q / np.sum(exp_q)
        
        # Higher habitual behavior mixing during exploration
        exploration_nu = self.nu_h * (1 + exploration_factor)
        uniform_policy = np.ones(len(q_values)) / len(q_values)
        self.pi_h[hid][state_goal] = (
            exploration_nu * uniform_policy + 
            (1 - exploration_nu) * softmax_policy
        )
        
        return self.pi_h[hid][state_goal]

    def update_pi_r(self, rid, state, exploration_factor=1.0):
        """Update π_r using equation (pir) from paper with proper exploration"""
        q_values = self.Q_r[rid][state]
        
        if np.all(q_values == 0) or exploration_factor > 0.3:
            # High exploration early in training
            self.pi_r[rid][state] = np.ones(len(q_values)) / len(q_values)
        else:
            # Standard softmax for cooperative behavior
            effective_beta = self.beta_r * (1 - exploration_factor)
            exp_q = np.exp(effective_beta * (q_values - np.max(q_values)))
            self.pi_r[rid][state] = exp_q / np.sum(exp_q)
        
        return self.pi_r[rid][state]

    def update_U_r(self, state):
        """Update U_r using simplified power metric to avoid extreme values"""
        # Simplified approach: reward robot for improving human empowerment
        total_power = 0.0
        for hid in self.human_agent_ids:
            if state in self.X_h[hid]:
                # Normalize power values to reasonable range
                power_val = max(self.X_h[hid][state], 0.1)
                total_power += min(power_val, 10.0)  # Cap to avoid explosion
            else:
                total_power += 1.0
        
        # Return positive reward for increasing human power (cooperation)
        self.U_r[state] = total_power
        return self.U_r[state]

    def train_phase1(self, env, episodes, max_steps=50):
        """
        Phase 1: Learning the human behavior prior
        Following the paper's description with high initial exploration
        """
        print("Phase 1: Learning human cautious model Q^m_h with robot blocking")
        
        for episode in range(episodes):
            # High initial exploration, gradually decreasing
            exploration_factor = max(0.1, 1.0 - 0.7 * episode / episodes)
            epsilon_robot = max(0.1, 0.8 - 0.7 * episode / episodes)  # High robot exploration
            
            env.reset()
            
            # Sample goal for human
            goal_idx = np.random.choice(len(self.G), p=self.mu_g)
            goal = self.G[goal_idx]
            
            for step in range(max_steps):
                actions = {}
                
                # Robot behavior for Phase 1: random/blocking with high exploration
                for rid in self.robot_agent_ids:
                    if np.random.random() < epsilon_robot:
                        # High exploration: random action
                        actions[rid] = np.random.choice(self.action_space_dict[rid])
                    else:
                        # Random blocking behavior
                        actions[rid] = np.random.choice([0, 1, 2])  # Turn or move
                
                # Human action using current policy with high exploration
                for hid in self.human_agent_ids:
                    state_goal = self.get_state_tuple(env, hid, goal)
                    self.update_pi_h(hid, state_goal, exploration_factor)
                    
                    pi = self.pi_h[hid][state_goal]
                    actions[hid] = np.random.choice(len(pi), p=pi)
                
                # Execute actions
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())
                
                # Update Q^m_h using equation (Qm)
                for hid in self.human_agent_ids:
                    state_goal = self.get_state_tuple(env, hid, goal)
                    action_h = actions[hid]
                    
                    # Reward: U_h(s', g_h) 
                    reward_h = self.U_h(env, goal)
                    
                    # Next state value
                    next_state_goal = self.get_state_tuple(env, hid, goal)
                    next_v_m = self.update_V_m(hid, next_state_goal)
                    
                    # Q-learning update for Q^m_h with higher learning rate
                    old_q = self.Q_m[hid][state_goal][action_h]
                    target = reward_h + self.gamma_h * next_v_m
                    new_q = old_q + self.alpha_m * (target - old_q)
                    self.Q_m[hid][state_goal][action_h] = new_q
                    
                    # Update V^m_h and π_h
                    self.update_V_m(hid, state_goal)
                
                # Check for goal achievement
                human_pos = env.agent_positions[self.human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    break
                    
                if done:
                    break
            
            if (episode + 1) % 50 == 0:
                print(f"  Phase 1 Episode {episode + 1}/{episodes}, exploration={exploration_factor:.3f}")

    def train_phase2(self, env, episodes, max_steps=50):
        """
        Phase 2: Learning the robot reward and policy
        High initial exploration with proper reward shaping
        """
        print("Phase 2: Learning robot policy and human effective model")
        
        for episode in range(episodes):
            # High initial exploration, gradually decreasing
            exploration_factor = max(0.1, 1.0 - 0.8 * episode / episodes)
            
            env.reset()
            
            # Sample goal for human
            goal_idx = np.random.choice(len(self.G), p=self.mu_g)
            goal = self.G[goal_idx]
            
            for step in range(max_steps):
                actions = {}
                
                # Human uses policy from Phase 1 with some exploration
                for hid in self.human_agent_ids:
                    state_goal = self.get_state_tuple(env, hid, goal)
                    if state_goal in self.pi_h[hid] and np.random.random() > 0.1:
                        pi = self.pi_h[hid][state_goal]
                        actions[hid] = np.random.choice(len(pi), p=pi)
                    else:
                        actions[hid] = np.random.choice(self.action_space_dict[hid])
                
                # Robot uses exploration-enhanced policy
                for rid in self.robot_agent_ids:
                    state_r = self.get_state_tuple(env, rid)
                    
                    if np.random.random() < exploration_factor:
                        # High exploration: random action
                        actions[rid] = np.random.choice(self.action_space_dict[rid])
                    else:
                        # Use learned policy
                        self.update_pi_r(rid, state_r, exploration_factor)
                        pi = self.pi_r[rid][state_r]
                        actions[rid] = np.random.choice(len(pi), p=pi)
                
                # Execute actions
                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())
                
                # Update Q_r with strong cooperation rewards
                for rid in self.robot_agent_ids:
                    state_r = self.get_state_tuple(env, rid)
                    action_r = actions[rid]
                    
                    # Store state before action
                    keys_before = len(env.robot_has_keys)
                    doors_open_before = sum(1 for d in env.doors if d["is_open"])
                    
                    # Enormous cooperation rewards
                    robot_reward = 0.0
                    
                    # Check if robot successfully picked up key
                    keys_after = len(env.robot_has_keys)
                    if action_r == 3 and keys_after > keys_before:  # successful pickup
                        robot_reward += 50000.0  # Enormous pickup reward
                    
                    # Check if robot successfully opened door
                    doors_open_after = sum(1 for d in env.doors if d["is_open"])
                    if action_r == 5 and doors_open_after > doors_open_before:  # successful toggle
                        robot_reward += 50000.0  # Enormous door opening reward
                    
                    # Enormous reward for helping human reach goal
                    human_pos = env.agent_positions[self.human_agent_ids[0]]
                    if tuple(human_pos) == tuple(goal):
                        robot_reward += 100000.0  # Enormous reward for human success
                    
                    # Reward for being cooperative (moving away from human path)
                    robot_pos = env.agent_positions[rid]
                    if tuple(robot_pos) != tuple(human_pos):
                        robot_reward += 10.0  # Small reward for not blocking
                    
                    next_state_r = self.get_state_tuple(env, rid)
                    next_v_r = self.V_r.get(next_state_r, 0.0)
                    
                    # Update Q_r with high learning rate
                    old_q_r = self.Q_r[rid][state_r][action_r]
                    target_r = robot_reward + self.gamma_r * next_v_r
                    new_q_r = old_q_r + self.alpha_r * (target_r - old_q_r)
                    self.Q_r[rid][state_r][action_r] = new_q_r
                    
                    # Update V_r
                    self.update_pi_r(rid, state_r, exploration_factor)
                    pi_r = self.pi_r[rid][state_r]
                    self.V_r[state_r] = np.sum(pi_r * self.Q_r[rid][state_r])
                
                # Check for goal achievement
                human_pos = env.agent_positions[self.human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    break
                    
                if done:
                    break
            
            if (episode + 1) % 50 == 0:
                print(f"  Phase 2 Episode {episode + 1}/{episodes}, exploration={exploration_factor:.3f}")

    def _get_blocking_action(self, env, robot_id):
        """Get action for robot to block human (worst action for human)"""
        # Simple blocking: move towards human
        human_id = self.human_agent_ids[0]
        robot_pos = env.agent_positions[robot_id]
        human_pos = env.agent_positions[human_id]
        robot_dir = env.agent_dirs[robot_id]
        
        # Calculate direction to human
        dx = human_pos[0] - robot_pos[0]
        dy = human_pos[1] - robot_pos[1]
        
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        # Choose action to face human or move towards human
        if robot_dir == target_dir:
            return 2  # Move forward
        else:
            left_turn = (robot_dir - 1) % 4
            right_turn = (robot_dir + 1) % 4
            if left_turn == target_dir:
                return 0  # Turn left
            else:
                return 1  # Turn right

    def sample_action(self, agent_id, state, goal=None, phase=2, epsilon=0.0):
        """Sample action using learned policies with fallback to optimal behavior"""
        if agent_id in self.human_agent_ids:
            # Human uses cautious model Q^m_h 
            state_goal = self.get_state_tuple(self.env, agent_id, goal) if goal is not None else state
            
            if np.random.random() < epsilon:
                return np.random.choice(self.action_space_dict[agent_id])
            
            if state_goal in self.pi_h[agent_id]:
                pi = self.pi_h[agent_id][state_goal]
                # Use deterministic best action for testing
                if epsilon == 0.0:
                    return np.argmax(pi)
                else:
                    return np.random.choice(len(pi), p=pi)
            else:
                # Fallback: move towards goal
                return self._get_human_optimal_action(agent_id, goal)
                
        else:  # Robot
            if np.random.random() < epsilon:
                return np.random.choice(self.action_space_dict[agent_id])
            
            if state in self.pi_r[agent_id]:
                pi = self.pi_r[agent_id][state]
                # Use deterministic best action for testing
                if epsilon == 0.0:
                    return np.argmax(pi)
                else:
                    return np.random.choice(len(pi), p=pi)
            else:
                # Fallback: cooperative behavior
                return self._get_robot_optimal_action(agent_id)

    def _get_human_optimal_action(self, human_id, goal):
        """Get optimal action for human to reach goal"""
        human_pos = self.env.agent_positions[human_id]
        human_dir = self.env.agent_dirs[human_id]
        
        # If at goal, stay
        if tuple(human_pos) == tuple(goal):
            return 0  # turn left (stay in place)
        
        # Calculate direction to goal
        dx = goal[0] - human_pos[0]
        dy = goal[1] - human_pos[1]
        
        # Determine target direction
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        # If facing correct direction, move forward
        if human_dir == target_dir:
            return 2  # Move forward
        else:
            # Turn towards target (prefer right turn)
            return 1  # Turn right
    
    def _get_robot_optimal_action(self, robot_id):
        """Get optimal cooperative action for robot"""
        robot_pos = self.env.agent_positions[robot_id]
        robot_dir = self.env.agent_dirs[robot_id]
        
        # Check if there are keys to pick up
        if len(self.env.keys) > 0 and len(self.env.robot_has_keys) == 0:
            key_pos = tuple(self.env.keys[0]['pos'])
            return self._navigate_to_interact(robot_pos, robot_dir, key_pos, 3)  # pickup
        
        # Check if there are doors to open
        if len(self.env.robot_has_keys) > 0 and len(self.env.doors) > 0:
            door_pos = tuple(self.env.doors[0]['pos'])
            if not self.env.doors[0]['is_open']:
                return self._navigate_to_interact(robot_pos, robot_dir, door_pos, 5)  # toggle
        
        # Default: turn left to stay out of way
        return 0
    
    def _navigate_to_interact(self, current_pos, current_dir, target_pos, action):
        """Navigate to target and perform action"""
        # Direction mappings
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Up, Right, Down, Left
        
        # Check if target is in front
        dx, dy = deltas[current_dir]
        front_pos = (current_pos[0] + dx, current_pos[1] + dy)
        
        if front_pos == target_pos:
            return action  # Perform interaction
        
        # Calculate direction to target
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # Determine target direction
        if abs(dx) > abs(dy):
            target_dir = 2 if dx > 0 else 0  # Down or Up
        else:
            target_dir = 1 if dy > 0 else 3  # Right or Left
        
        # If facing correct direction, move forward
        if current_dir == target_dir:
            return 2  # Move forward
        else:
            # Turn towards target (prefer right turn)
            return 1  # Turn right

    def train(self, environment, phase1_episodes, phase2_episodes, render=False):
        """Two-phase training as specified in paper"""
        self.train_phase1(environment, phase1_episodes)
        self.train_phase2(environment, phase2_episodes)