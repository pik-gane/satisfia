import numpy as np
from collections import defaultdict
import random
import pickle
import os
import math
import time
import pygame  # for rendering in training
from env import Actions

class TwoTimescaleIQL:
    def __init__(self, alpha_h, alpha_r, gamma_h, gamma_r, beta_h, epsilon_r,
                 G, mu_g, p_g, E, action_space_dict, robot_agent_id,
                 human_agent_ids, reward_agg_fn=None, debug=False):
        # MODIFIED: human_agent_ids list and reward aggregator
        self.alpha_h = alpha_h
        self.alpha_r = alpha_r
        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.beta_h = beta_h
        self.epsilon_r = epsilon_r
        self.G = G  # List of goals, e.g., [ (x1,y1), (x2,y2) ]
        self.mu_g = mu_g # Prior probability distribution over goals G
        self.p_g = p_g  # Probability of goal change per step
        # E is num_episodes, typically passed to train, but stored if needed
        self.num_episodes_config = E 

        # Support multiple robots similarly to humans
        if isinstance(robot_agent_id, str):
            self.robot_agent_ids = [robot_agent_id]
        else:
            self.robot_agent_ids = list(robot_agent_id)
        # For compatibility, retain single robot_agent_id if only one
        self.robot_agent_id = self.robot_agent_ids[0]
        # ensure list of human IDs
        if isinstance(human_agent_ids, str): human_agent_ids = [human_agent_ids]
        self.human_agent_ids = human_agent_ids
        # default aggregator: simple sum of human values to avoid negative/zero log issues
        self.reward_agg_fn = reward_agg_fn or (lambda rs: sum(rs))

        if not self.G:
            raise ValueError("Goal list G cannot be empty.")
        if not isinstance(self.G[0], tuple) and len(self.G[0]) > 0 : # Basic check for goal format
             print(f"Warning: Goals G are expected to be tuples. Received: {self.G[0]}")


        # action spaces per robot
        self.action_space_robot = {rid: action_space_dict[rid] for rid in self.robot_agent_ids}
        # per-human action spaces
        self.action_space_humans = {hid: action_space_dict[hid] for hid in self.human_agent_ids}

        # Q_r per robot: state->array for each robot's action_space
        self.Q_r = {rid: defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=len(self.action_space_robot[rid])))
                    for rid in self.robot_agent_ids}
        # one Q_h table per human agent: full action dimension for indexing
        from env import Actions
        action_dim = len(Actions)
        self.Q_h = {hid: defaultdict(lambda: np.random.uniform(-0.1,0.1,size=action_dim))
                    for hid in self.human_agent_ids}

        self.debug = bool(debug)
        # Count-based exploration counters for robot and humans
        self.N_r = defaultdict(int)  # counts for robot state-action pairs
        self.N_h = {hid: defaultdict(int) for hid in self.human_agent_ids}
        # Use human beta_h for robot softmax temperature
        self.beta_r = self.beta_h

        self.debug_print_episode_interval = 100 # Print full episode details less frequently
        self.debug_print_step_limit = 3 # Print step details for first few steps of debug episodes
        self.debug_q_update_sample_rate = 0.005 # Print Q-updates very sparsely

        # Prioritized Experience Replay parameters
        self.replay_buffer = []
        self.priorities = []
        self.replay_capacity = 10000
        self.per_alpha = 0.6  # prioritization exponent
        self.per_beta = 0.4   # importance-sampling exponent
        self.batch_size = 32
        self.per_update_interval = 4  # only sample PER every N env steps

        if self.debug:
            print(f"IQL Initialized: Robot IDs='{self.robot_agent_ids}', Human IDs='{self.human_agent_ids}'")
            print(f"  alpha_h={alpha_h}, alpha_r={alpha_r}, gamma_h={gamma_h}, gamma_r={gamma_r}")
            print(f"  beta_h={beta_h}, epsilon_r={epsilon_r}, p_g={p_g}")
            print(f"  Goals G: {G}")
            print(f"  Goal prior mu_g: {mu_g}")
            for rid in self.robot_agent_ids:
                print(f"  Robot ID '{rid}' action space size: {len(self.action_space_robot[rid])}")
            for hid in self.human_agent_ids:
                print(f"  Human ID '{hid}' action space size: {len(self.action_space_humans[hid])}")


    def state_to_tuple(self, state_obs):
        """Converts a numpy array observation to a tuple of Python ints for Q-table keys."""
        # If already a tuple, cast elements to int
        if isinstance(state_obs, tuple):
            return tuple(int(x) for x in state_obs)
        # Numpy array or list-like
        try:
            arr = np.asarray(state_obs).flatten()
            return tuple(int(x) for x in arr)
        except Exception:
            # Fallback
            return tuple(int(x) for x in state_obs)

    def save_q_values(self, filepath="q_values.pkl"):
        """
        Save the trained Q-values to a file.
        
        Args:
            filepath: Path to save the Q-values
        """
        # Convert defaultdict to regular dict before saving
        q_values = {
            "Q_r": {rid: dict(qtable) for rid, qtable in self.Q_r.items()},
            "Q_h": {hid: dict(qtable) for hid, qtable in self.Q_h.items()},
            "params": {
                "alpha_h": self.alpha_h,
                "alpha_r": self.alpha_r,
                "gamma_h": self.gamma_h,
                "gamma_r": self.gamma_r,
                "beta_h": self.beta_h,
                "G": [tuple(g) for g in self.G],
                "mu_g": self.mu_g.tolist() if isinstance(self.mu_g, np.ndarray) else self.mu_g,
                "action_space_robot": {rid: self.action_space_robot[rid] for rid in self.robot_agent_ids},
                "action_space_humans": {hid: self.action_space_humans[hid] for hid in self.human_agent_ids},
                "robot_agent_ids": self.robot_agent_ids,
                "human_agent_ids": self.human_agent_ids
            }
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        print(f"Saving Q-values to {filepath}")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(q_values, f)
            print(f"Successfully saved Q-values with {sum(len(qtable) for qtable in q_values['Q_r'].values())} robot states and {sum(len(qtable) for qtable in q_values['Q_h'].values())} human state-goal pairs")
        except Exception as e:
            print(f"Error saving Q-values: {e}")
            
    @classmethod
    def load_q_values(cls, filepath="q_values.pkl"):
        """
        Load Q-values from a file and create a new TwoTimescaleIQL instance.
        
        Args:
            filepath: Path to the saved Q-values
            
        Returns:
            A new TwoTimescaleIQL instance with loaded Q-values
        """
        print(f"Loading Q-values from {filepath}")
        try:
            with open(filepath, 'rb') as f:
                q_values = pickle.load(f)
                
            params = q_values["params"]
            
            # Convert back to numpy array if needed
            mu_g = np.array(params["mu_g"]) if params.get("mu_g") else np.array([1.0])
            G = params.get("G", [(0, 0)])  # Default goal if not found
            
            # Create action_space_dict from saved params
            action_space_dict = {
                **{rid: params["action_space_robot"][rid] for rid in params.get("robot_agent_ids", ["robot_0"])},
                **{hid: params["action_space_humans"][hid] for hid in params.get("human_agent_ids", ["human_0"])}
            }
            
            # Create a new instance with saved parameters
            instance = cls(
                alpha_h=params.get("alpha_h", 0.1),
                alpha_r=params.get("alpha_r", 0.01),
                gamma_h=params.get("gamma_h", 0.99),
                gamma_r=params.get("gamma_r", 0.99),
                beta_h=params.get("beta_h", 5.0),
                epsilon_r=0.0,  # Set to 0 for deterministic behavior
                G=G,
                mu_g=mu_g,
                p_g=0.0,  # Set to 0 to prevent goal changes
                E=1,  # Not needed for inference
                action_space_dict=action_space_dict,
                robot_agent_id=params.get("robot_agent_ids", ["robot_0"]),
                human_agent_ids=params.get("human_agent_ids", ["human_0"]),
                debug=False
            )
            
            # Replace defaultdict with loaded Q-values
            # Convert the dictionaries back to defaultdicts with numpy arrays
            Q_r_dict = q_values["Q_r"]
            Q_h_dict = q_values["Q_h"]
            
            # Convert string keys back to tuples - with safer handling
            q_r = {}
            for rid, qtable_dict in Q_r_dict.items():
                robot_default_q = np.zeros(len(instance.action_space_robot[rid]))
                q_r_table = defaultdict(lambda: np.copy(robot_default_q))
                for state_str, values in qtable_dict.items():
                    # Handle different key formats safely
                    try:
                        if isinstance(state_str, str):
                            state = eval(state_str)  # Try to convert string representation to tuple
                        else:
                            state = state_str  # Use as-is if not a string (might be a tuple already)
                        q_r_table[state] = np.array(values)
                    except Exception as e:
                        print(f"Warning: Could not convert robot state key {state_str}: {e}")
                q_r[rid] = q_r_table
            
            q_h = {}
            for hid, qtable_dict in Q_h_dict.items():
                human_default_q = np.zeros(len(instance.action_space_humans[hid]))
                q_h_table = defaultdict(lambda: np.copy(human_default_q))
                for state_goal_str, values in qtable_dict.items():
                    # Handle different key formats safely
                    try:
                        if isinstance(state_goal_str, str):
                            state_goal = eval(state_goal_str)  # Try to convert string representation
                        else:
                            state_goal = state_goal_str  # Use as-is if not a string
                        q_h_table[state_goal] = np.array(values)
                    except Exception as e:
                        print(f"Warning: Could not convert human state-goal key {state_goal_str}: {e}")
                q_h[hid] = q_h_table
            
            instance.Q_r = q_r
            instance.Q_h = q_h
            
            print(f"Successfully loaded Q-values with {sum(len(qtable) for qtable in q_r.values())} robot states and {sum(len(qtable) for qtable in q_h.values())} human state-goal pairs")
            return instance
            
        except Exception as e:
            print(f"Error loading Q-values: {e}")
            return None

    def train(self, environment, num_episodes, render=False, render_delay=0):
        max_steps_per_episode = getattr(environment, 'max_steps', 200)
        initial_epsilon_r = self.epsilon_r
        min_epsilon_r = 0.01
        epsilon_decay_rate = (initial_epsilon_r - min_epsilon_r) / (num_episodes * 0.8) # Decay over 80% of episodes
        # Logging interval: 10% of total episodes
        log_interval = max(1, num_episodes // 10)

        print(f"IQL Training Started: {num_episodes} episodes, max_steps/ep={max_steps_per_episode}")
        if self.debug:
            if not all(rid in environment.possible_agents for rid in self.robot_agent_ids) or not all(hid in environment.possible_agents for hid in self.human_agent_ids):
                print(f"ERROR IQL: Agent IDs mismatch with environment possible_agents")
                return

        start_time = time.time()

        # add accumulators for overall stats
        human_reward_total = 0.0
        ai_reward_total = 0.0

        for e in range(num_episodes):
            # Episode start
            ep_start = time.time()
            # per-episode counters
            pickup_success = 0
            toggle_success = 0
            # episode-level sums
            episode_human_sum = 0.0
            episode_ai_sum = 0.0

            if (e) % log_interval == 0:
                print(f"Starting Episode {e+1}/{num_episodes}")

            environment.reset()
            
            current_human_goals = {}
            for hid in self.human_agent_ids:
                current_human_goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_human_goal = self.G[current_human_goal_idx]
                current_human_goals[hid] = self.state_to_tuple(current_human_goal)

            # initial robot states for each robot
            s_r_tuple = {rid: self.state_to_tuple(environment.observe(rid)) for rid in self.robot_agent_ids}
            s_h_tuples = {hid: self.state_to_tuple(environment.observe(hid)) for hid in self.human_agent_ids}

            for step_num in range(max_steps_per_episode):
                # no per-step logging

                # Goal Dynamics
                for hid in self.human_agent_ids:
                    if np.random.rand() < self.p_g:
                        current_human_goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                        current_human_goal = self.G[current_human_goal_idx]
                        current_human_goals[hid] = self.state_to_tuple(current_human_goal)
                
                # Compute potentials before taking step for each human
                phi_s_map = {hid: environment.potential(hid) for hid in self.human_agent_ids}

                # select actions for each robot
                a_r = {rid: self.select_robot_action(rid, s_r_tuple[rid]) for rid in self.robot_agent_ids}
                a_h = {hid: self.select_human_action(hid, s_h_tuples[hid], current_human_goals[hid]) for hid in self.human_agent_ids}

                prev_s_r_tuple = s_r_tuple
                prev_s_h_tuples = s_h_tuples

                # --- Execute actions in parallel ---
                actions = {rid: a_r[rid] for rid in self.robot_agent_ids}
                actions.update(a_h)
                # one parallel step
                obs_dict, reward_dict, term_dict, trunc_dict, info_dict = environment.step(actions)
                # render during training if requested
                if render:
                    environment.render()
                    pygame.time.delay(render_delay)
                # collect human observed rewards
                r_h_obs_env = {hid: reward_dict.get(hid, 0) for hid in self.human_agent_ids}
                # accumulate human rewards
                episode_human_sum += sum(r_h_obs_env.values())

                # Next state observations
                s_r_prime_tuple = {rid: self.state_to_tuple(obs_dict[rid]) for rid in self.robot_agent_ids}
                s_h_prime_tuples = {hid: self.state_to_tuple(obs_dict[hid]) for hid in self.human_agent_ids}

                # Determine overall episode done status for IQL updates
                episode_done_iql = any(term_dict.values()) or any(trunc_dict.values())

                # After step, compute potentials for each human
                phi_s_prime_map = {hid: environment.potential(hid) for hid in self.human_agent_ids}

                # detect interactions
                keys_after = len(environment.keys)
                if keys_after < len(environment.keys) + 1: pickup_success += 1
                doors_open_after = sum(1 for d in environment.doors if d['is_open'])
                if doors_open_after > sum(1 for d in environment.doors if not d['is_open']): toggle_success += 1

                # --- Human Q-Update with potential-based shaping per human ---
                for hid in self.human_agent_ids:
                    r_h = r_h_obs_env.get(hid, 0)
                    # shaped human reward using individual potentials
                    phi_prev = phi_s_map.get(hid, 0.0)
                    phi_post = phi_s_prime_map.get(hid, 0.0)
                    r_h_shaped = r_h + self.gamma_h * phi_post - phi_prev
                    self.update_human_q(hid, prev_s_h_tuples[hid], current_human_goals[hid], a_h[hid], r_h_shaped, s_h_prime_tuples[hid], episode_done_iql, self.debug)
                # --- Robot Q-Updates per robot ---
                for rid in self.robot_agent_ids:
                    base_rr = self.calculate_robot_internal_reward(s_r_prime_tuple[rid], episode_done_iql)
                    # accumulate AI internal rewards
                    episode_ai_sum += base_rr
                    self.update_robot_q(rid, prev_s_r_tuple[rid], a_r[rid], base_rr, s_r_prime_tuple[rid], episode_done_iql, self.debug)
                # Store transition in PER buffer
                for rid in self.robot_agent_ids:
                    q_curr = self.Q_r[rid][prev_s_r_tuple[rid]][a_r[rid]]
                    max_q_next = 0 if episode_done_iql else np.max(self.Q_r[rid][s_r_prime_tuple[rid]])
                    q_target = base_rr + self.gamma_r * max_q_next
                    td_error = abs(q_target - q_curr)
                    # add to buffer or replace
                    transition = (rid, prev_s_r_tuple[rid], a_r[rid], base_rr, s_r_prime_tuple[rid], episode_done_iql)
                    if len(self.replay_buffer) < self.replay_capacity:
                        self.replay_buffer.append(transition)
                        self.priorities.append(td_error + 1e-6)
                    else:
                        idx = random.randrange(self.replay_capacity)
                        self.replay_buffer[idx] = transition
                        self.priorities[idx] = td_error + 1e-6
                 
                s_r_tuple = s_r_prime_tuple
                s_h_tuples = s_h_prime_tuples

                if episode_done_iql:
                    break 
            
            # End of episode: perform batched replay updates
            if len(self.replay_buffer) >= self.batch_size:
                # number of minibatches per episode
                num_batches = max(1, len(self.replay_buffer) // self.batch_size // self.per_update_interval)
                for _ in range(num_batches):
                    # sample from PER
                    pr = np.array(self.priorities) ** self.per_alpha
                    pr_sum = pr.sum()
                    probs = pr / pr_sum if pr_sum > 0 else np.ones_like(pr) / len(pr)
                    idxs = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
                    # IS weights
                    P = probs[idxs]
                    weights = (1.0 / (len(self.replay_buffer) * P)) ** self.per_beta
                    weights = weights / weights.max()
                    for j, idx in enumerate(idxs):
                        rid, s_i, a_i, r_i, s_p_i, done_i = self.replay_buffer[idx]
                        q_current = self.Q_r[rid][s_i][a_i]
                        max_q_next_i = 0 if done_i else np.max(self.Q_r[rid][s_p_i])
                        q_target_i = r_i + self.gamma_r * max_q_next_i
                        td = q_target_i - q_current
                        self.Q_r[rid][s_i][a_i] += self.alpha_r * weights[j] * td
                        self.priorities[idx] = abs(td) + 1e-6

            # compute per-episode averages
            avg_human = episode_human_sum / (step_num+1) if step_num>=0 else 0.0
            avg_ai    = episode_ai_sum    / (step_num+1) if step_num>=0 else 0.0

            # debug output: per-episode human and AI average rewards
            if self.debug:
                print(f"[DEBUG] Episode {e+1}/{num_episodes}: human={avg_human:.2f}, ai={avg_ai:.2f}")

            # Decay epsilon_r per episode
            self.epsilon_r = max(min_epsilon_r, self.epsilon_r - epsilon_decay_rate)
            # optional timing log at intervals
            if not self.debug and ((e+1) % log_interval == 0 or (e+1) == num_episodes):
                elapsed = time.time() - start_time
                avg_time = elapsed / (e+1)
                print(f"Completed {e+1}/{num_episodes} episodes in {elapsed:.2f}s (avg {avg_time:.2f}s/ep)")

        # after all episodes, print overall averages
        print(f"Training complete: avg human reward/episode={human_reward_total/num_episodes:.2f}, avg AI reward/episode={ai_reward_total/num_episodes:.2f}")
        print("IQL Training Finished.")

    def select_robot_action(self, robot_id, s_r_tuple):
        """
        Boltzmann action selection with count-based exploration bonus for robot.
        """
        q_values = self.Q_r[robot_id][s_r_tuple]
        # compute bonus for each action: encourages unvisited actions
        bonuses = []
        for a in self.action_space_robot[robot_id]:
            count = self.N_r[(robot_id, s_r_tuple, a)]
            bonuses.append(1.0 / math.sqrt(count + 1))
        eff_q = np.array(q_values) + np.array(bonuses)
        
        # For robot agents, use -log(-Qr) transformation (equation 7)
        # Transform Q-values: use -log(-Qr) instead of Qr directly
        # Add small epsilon to ensure -Qr is positive for log computation
        epsilon = 1e-8
        neg_q = -eff_q + epsilon  # Ensure positive values for log
        neg_q = np.maximum(neg_q, epsilon)  # Additional safety check
        transformed_q = -np.log(neg_q)
        
        # Boltzmann over transformed Q
        exp_vals = np.exp(self.beta_r * transformed_q)
        # check for invalid or infinite values
        if not np.isfinite(exp_vals).all():
            probs = np.ones_like(exp_vals) / len(exp_vals)
        else:
            sum_exp = exp_vals.sum()
            if sum_exp <= 0 or not np.isfinite(sum_exp):
                probs = np.ones_like(exp_vals) / len(exp_vals)
            else:
                probs = exp_vals / sum_exp
        # sample action according to probabilities
        action = np.random.choice(self.action_space_robot[robot_id], p=probs)
        # update state-action count
        self.N_r[(robot_id, s_r_tuple, action)] += 1
        return action

    def select_human_action(self, hid, s_h_tuple, goal_tuple):
        """
        Boltzmann with count-based exploration for human over allowed actions.
        """
        allowed = self.action_space_humans[hid]
        q_full = self.Q_h[hid][(s_h_tuple, goal_tuple)]
        q_vals = np.array([q_full[a] for a in allowed])
        bonuses = []
        for idx, a in enumerate(allowed):
            count = self.N_h[hid][(s_h_tuple, goal_tuple, a)] + 1
            bonuses.append(1.0 / math.sqrt(count))
        eff_q = q_vals + np.array(bonuses)
        exp_q = np.exp(self.beta_h * eff_q)
        if np.sum(exp_q) == 0 or np.isinf(exp_q).any():
            if self.debug and random.random() < 0.01:
                print(f"    WARN: Softmax issue for human action. s_h={s_h_tuple}, g={goal_tuple}. Using uniform distribution.")
            probs = np.ones_like(exp_q) / len(exp_q)
        else:
            probs = exp_q / np.sum(exp_q)
        idx = np.random.choice(len(allowed), p=probs)
        action = allowed[idx]
        self.N_h[hid][(s_h_tuple, goal_tuple, action)] += 1
        return action

    def update_human_q(self, hid, s_h_tuple, goal_tuple, a_h, r_h_obs, s_h_prime_tuple, done, do_debug_episode):
        q_key = (s_h_tuple, goal_tuple)
        q_current_val = self.Q_h[hid][q_key][a_h]
        v_h_s_prime_g = 0
        if not done:
            q_values_next_state_goal = self.Q_h[hid][(s_h_prime_tuple, goal_tuple)]
            exp_q_next = np.exp(self.beta_h * q_values_next_state_goal)
            sum_exp_q_next = np.sum(exp_q_next)
            if sum_exp_q_next == 0 or np.isinf(sum_exp_q_next) or np.isnan(sum_exp_q_next):
                if self.debug == "verbose" and random.random() < 0.01:
                    print(f"    WARN: Softmax issue in V_h calc for s_h'={s_h_prime_tuple}, goal={goal_tuple}. Q_h'={q_values_next_state_goal}. V_h set to 0.")
                v_h_s_prime_g = 0
            else:
                probs_next = exp_q_next / sum_exp_q_next
                if np.isnan(probs_next).any():
                    if self.debug == "verbose" and random.random() < 0.01:
                        print(f"    WARN: NaN probs in V_h calc for s_h'={s_h_prime_tuple}, goal={goal_tuple}. Q_h'={q_values_next_state_goal}. V_h set to 0.")
                    v_h_s_prime_g = 0
                else:
                    v_h_s_prime_g = np.sum(probs_next * q_values_next_state_goal)
        q_target = r_h_obs + self.gamma_h * v_h_s_prime_g
        self.Q_h[hid][q_key][a_h] += self.alpha_h * (q_target - q_current_val)
        if self.debug == "verbose" and random.random() < self.debug_q_update_sample_rate:
            print(f"    Human Q_h update: hid={hid}, s_h={s_h_tuple}, g={goal_tuple}, a_h={a_h}, r_h={r_h_obs:.2f}, s_h'={s_h_prime_tuple}, done={done}")
            print(f"      Q_h_curr={q_current_val:.3f}, V_h(s',g)={v_h_s_prime_g:.3f}, Q_h_target={q_target:.3f}, New_Q_h={self.Q_h[hid][q_key][a_h]:.3f}")

    def train_phase1(self, environment, phase1_episodes, render=False, render_delay=0):
        """
        Phase 1: Learn conservative human models for each human agent in sequence.
        """
        print(f"Phase 1: Learning cautious human models for {phase1_episodes} episodes")
        num_humans = len(self.human_agent_ids)
        episodes_per_human = phase1_episodes // num_humans
        remaining_episodes = phase1_episodes % num_humans
        for idx, hid in enumerate(self.human_agent_ids):
            current_episodes = episodes_per_human + (1 if idx < remaining_episodes else 0)
            print(f"\n--- Learning conservative model for {hid} ({current_episodes} episodes) ---")
            for ep in range(current_episodes):
                environment.reset()
                # Sample initial goal for this human
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                goal_tuple = self.state_to_tuple(self.G[goal_idx])
                s_h_tuple = self.state_to_tuple(environment.observe(hid))
                step_count = 0
                episode_human_reward = 0.0
                max_steps = getattr(environment, 'max_steps', 200)
                done = False
                while step_count < max_steps and not done:
                    a_h = self.select_human_action(hid, s_h_tuple, goal_tuple)
                    actions = {hid: a_h}
                    obs_dict, reward_dict, term_dict, trunc_dict, info_dict = environment.step(actions)
                    r_h_obs = reward_dict.get(hid, 0)
                    s_h_prime_tuple = self.state_to_tuple(obs_dict[hid])
                    done = term_dict.get(hid, False) or trunc_dict.get(hid, False)
                    self.update_human_q(hid, s_h_tuple, goal_tuple, a_h, r_h_obs, s_h_prime_tuple, done, self.debug)
                    episode_human_reward += r_h_obs
                    s_h_tuple = s_h_prime_tuple
                    step_count += 1
                    if render:
                        environment.render()
                        pygame.time.delay(render_delay)
                if (ep + 1) % max(1, current_episodes // 5) == 0 or (ep + 1) == current_episodes:
                    print(f"[PHASE1-{hid}] Episode {ep+1}/{current_episodes}: avg_reward={episode_human_reward/step_count if step_count else 0:.2f}")
            print(f"[INFO] Conservative model learned for human '{hid}' after {current_episodes} episodes.")
        print(f"\n✅ Phase 1 complete: Individual conservative models learned for all humans")