import numpy as np
from collections import defaultdict
import random
import pickle
import os

class TwoTimescaleIQL:
    def __init__(self, alpha_h, alpha_r, gamma_h, gamma_r, beta_h, epsilon_r,
                 G, mu_g, p_g, E, action_space_dict, robot_agent_id, human_agent_id, debug=False): # MODIFIED: debug default to False
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

        self.robot_agent_id = robot_agent_id
        self.human_agent_id = human_agent_id

        if not self.G:
            raise ValueError("Goal list G cannot be empty.")
        if not isinstance(self.G[0], tuple) and len(self.G[0]) > 0 : # Basic check for goal format
             print(f"Warning: Goals G are expected to be tuples. Received: {self.G[0]}")


        self.action_space_robot = action_space_dict[self.robot_agent_id]
        self.action_space_human = action_space_dict[self.human_agent_id]

        # Q_r[s_r_tuple][action_index]
        self.Q_r = defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=len(self.action_space_robot)))
        # Q_h[(s_h_tuple, goal_tuple)][action_index]
        self.Q_h = defaultdict(lambda: np.random.uniform(-0.1, 0.1, size=len(self.action_space_human)))

        self.debug = debug
        self.debug_print_episode_interval = 100 # Print full episode details less frequently
        self.debug_print_step_limit = 3 # Print step details for first few steps of debug episodes
        self.debug_q_update_sample_rate = 0.005 # Print Q-updates very sparsely

        if self.debug:
            print(f"IQL Initialized: Robot ID='{self.robot_agent_id}', Human ID='{self.human_agent_id}'")
            print(f"  alpha_h={alpha_h}, alpha_r={alpha_r}, gamma_h={gamma_h}, gamma_r={gamma_r}")
            print(f"  beta_h={beta_h}, epsilon_r={epsilon_r}, p_g={p_g}")
            print(f"  Goals G: {G}")
            print(f"  Goal prior mu_g: {mu_g}")
            print(f"  Robot action space size: {len(self.action_space_robot)}")
            print(f"  Human action space size: {len(self.action_space_human)}")


    def state_to_tuple(self, state_obs):
        """Converts a numpy array observation to a hashable tuple for Q-table keys."""
        if isinstance(state_obs, tuple): # Already a tuple
            return state_obs
        try:
            return tuple(state_obs.flatten()) # Flatten in case of multi-dimensional parts
        except AttributeError: # Not a numpy array
            return tuple(state_obs)

    def save_q_values(self, filepath="q_values.pkl"):
        """
        Save the trained Q-values to a file.
        
        Args:
            filepath: Path to save the Q-values
        """
        # Convert defaultdict to regular dict before saving
        q_values = {
            "Q_r": dict(self.Q_r),
            "Q_h": dict(self.Q_h),
            "params": {
                "alpha_h": self.alpha_h,
                "alpha_r": self.alpha_r,
                "gamma_h": self.gamma_h,
                "gamma_r": self.gamma_r,
                "beta_h": self.beta_h,
                "G": [tuple(g) for g in self.G],
                "mu_g": self.mu_g.tolist() if isinstance(self.mu_g, np.ndarray) else self.mu_g,
                "action_space_robot": self.action_space_robot,
                "action_space_human": self.action_space_human,
                "robot_agent_id": self.robot_agent_id,
                "human_agent_id": self.human_agent_id
            }
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        print(f"Saving Q-values to {filepath}")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(q_values, f)
            print(f"Successfully saved Q-values with {len(q_values['Q_r'])} robot states and {len(q_values['Q_h'])} human state-goal pairs")
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
                params.get("robot_agent_id", "robot_0"): params.get("action_space_robot", [0, 1, 2, 3]),
                params.get("human_agent_id", "human_0"): params.get("action_space_human", [0, 1, 2, 3])
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
                robot_agent_id=params.get("robot_agent_id", "robot_0"),
                human_agent_id=params.get("human_agent_id", "human_0"),
                debug=False
            )
            
            # Replace defaultdict with loaded Q-values
            # Convert the dictionaries back to defaultdicts with numpy arrays
            Q_r_dict = q_values["Q_r"]
            Q_h_dict = q_values["Q_h"]
            
            # Convert string keys back to tuples - with safer handling
            robot_default_q = np.zeros(len(instance.action_space_robot))
            q_r = defaultdict(lambda: np.copy(robot_default_q))
            for state_str, values in Q_r_dict.items():
                # Handle different key formats safely
                try:
                    if isinstance(state_str, str):
                        state = eval(state_str)  # Try to convert string representation to tuple
                    else:
                        state = state_str  # Use as-is if not a string (might be a tuple already)
                    q_r[state] = np.array(values)
                except Exception as e:
                    print(f"Warning: Could not convert robot state key {state_str}: {e}")
            
            human_default_q = np.zeros(len(instance.action_space_human))
            q_h = defaultdict(lambda: np.copy(human_default_q))
            for state_goal_str, values in Q_h_dict.items():
                # Handle different key formats safely
                try:
                    if isinstance(state_goal_str, str):
                        state_goal = eval(state_goal_str)  # Try to convert string representation
                    else:
                        state_goal = state_goal_str  # Use as-is if not a string
                    q_h[state_goal] = np.array(values)
                except Exception as e:
                    print(f"Warning: Could not convert human state-goal key {state_goal_str}: {e}")
            
            instance.Q_r = q_r
            instance.Q_h = q_h
            
            print(f"Successfully loaded Q-values with {len(q_r)} robot states and {len(q_h)} human state-goal pairs")
            return instance
            
        except Exception as e:
            print(f"Error loading Q-values: {e}")
            return None

    def train(self, environment, num_episodes):
        max_steps_per_episode = getattr(environment, 'max_steps', 200)
        initial_epsilon_r = self.epsilon_r
        min_epsilon_r = 0.01
        epsilon_decay_rate = (initial_epsilon_r - min_epsilon_r) / (num_episodes * 0.8) # Decay over 80% of episodes


        if self.debug:
            print(f"IQL Training Started: {num_episodes} episodes, max_steps/ep={max_steps_per_episode}")
            if self.robot_agent_id not in environment.possible_agents or self.human_agent_id not in environment.possible_agents:
                print(f"ERROR IQL: robot_id '{self.robot_agent_id}' or human_id '{self.human_agent_id}' not in env.possible_agents {environment.possible_agents}")
                return

        for e in range(num_episodes):
            do_debug_episode = self.debug and (e % self.debug_print_episode_interval == 0 or e == num_episodes - 1)
            if do_debug_episode:
                print(f"--- IQL Episode {e+1}/{num_episodes} (epsilon_r={self.epsilon_r:.3f}) ---")

            environment.reset()
            
            current_human_goal_idx = np.random.choice(len(self.G), p=self.mu_g)
            current_human_goal = self.G[current_human_goal_idx]
            current_human_goal_tuple = self.state_to_tuple(current_human_goal)


            # Initial observations. AECEnv sets agent_selection to the first agent.
            # We need observations for both robot and human based on the *same* underlying env state.
            # env.observe(agent_id) should give the current global state view for that agent.
            s_r_tuple = self.state_to_tuple(environment.observe(self.robot_agent_id))
            s_h_tuple = self.state_to_tuple(environment.observe(self.human_agent_id)) # Same as s_r_tuple if obs is global

            if do_debug_episode and self.debug_print_step_limit > 0:
                print(f"  Initial: s_r={s_r_tuple}, s_h={s_h_tuple}, goal={current_human_goal_tuple}")

            for step_num in range(max_steps_per_episode):
                do_debug_step = do_debug_episode and step_num < self.debug_print_step_limit

                # Goal Dynamics
                if np.random.rand() < self.p_g:
                    current_human_goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                    current_human_goal = self.G[current_human_goal_idx]
                    current_human_goal_tuple = self.state_to_tuple(current_human_goal)
                    if do_debug_step: print(f"    Step {step_num+1}: Goal changed to: {current_human_goal_tuple}")
                
                # Action Selection
                a_r = self.select_robot_action(s_r_tuple)
                a_h = self.select_human_action(s_h_tuple, current_human_goal_tuple)
                if do_debug_step: print(f"    Step {step_num+1}: s_r={s_r_tuple}, s_h={s_h_tuple} -> a_r={a_r}, a_h={a_h}")

                prev_s_r_tuple = s_r_tuple
                prev_s_h_tuple = s_h_tuple

                # --- Execute actions in AECEnv ---
                # Robot's turn
                if environment.agent_selection != self.robot_agent_id:
                    if do_debug_step: print(f"    WARN Step {step_num+1}: Expected robot {self.robot_agent_id}, got {environment.agent_selection}. Forcing turn or checking termination.")
                    # This might happen if an agent terminated out of sync.
                    # If robot is already done, it won't act.
                
                if not environment.terminations[self.robot_agent_id] and not environment.truncations[self.robot_agent_id]:
                    environment.step(a_r) # Robot action
                # else: Robot is done, no action taken. env.step() would handle this if called.

                # Human's turn
                # After robot's step, env.agent_selection should be human_id, unless robot's action ended episode or human is already done.
                if environment.agent_selection != self.human_agent_id:
                    # Check if this is expected (e.g. robot terminated, or human is already done)
                    if not (environment.terminations[self.robot_agent_id] or environment.truncations[self.robot_agent_id] or environment.terminations[self.human_agent_id] or environment.truncations[self.human_agent_id]):
                        if do_debug_step: print(f"    WARN Step {step_num+1}: Expected human {self.human_agent_id} after robot, got {environment.agent_selection}. Robot done: {environment.terminations[self.robot_agent_id] or environment.truncations[self.robot_agent_id]}")
                
                r_h_obs_env = 0 # Default human reward for this conceptual step
                if not environment.terminations[self.human_agent_id] and not environment.truncations[self.human_agent_id]:
                    # Only step human if it's their turn and they are not done.
                    # This check is important if robot's action could terminate the episode for all.
                    if environment.agent_selection == self.human_agent_id :
                        environment.step(a_h) # Human action
                        r_h_obs_env = environment.rewards[self.human_agent_id] 
                # else: Human is done, no action taken. r_h_obs_env remains 0 for this (s,a) pair.

                # Get common next state observation (s_prime)
                s_prime_obs_common = environment.observe(self.robot_agent_id) # Global state view
                s_r_prime_tuple = self.state_to_tuple(s_prime_obs_common)
                s_h_prime_tuple = self.state_to_tuple(s_prime_obs_common)

                # Determine overall episode done status for IQL updates
                # An IQL episode is done if the environment marks any agent as terminated/truncated,
                # or if all agents are removed (env.agents is empty).
                episode_done_iql = False
                if not environment.agents: # All agents removed by PettingZoo
                    episode_done_iql = True
                elif environment.terminations[self.robot_agent_id] or environment.terminations[self.human_agent_id] or environment.truncations[self.robot_agent_id] or environment.truncations[self.human_agent_id]:
                    episode_done_iql = True
                
                if do_debug_step:
                    print(f"    Step {step_num+1}: s_r'={s_r_prime_tuple}, s_h'={s_h_prime_tuple}, r_h_obs={r_h_obs_env:.2f}, done={episode_done_iql}")

                # --- Updates ---
                self.update_human_q(prev_s_h_tuple, current_human_goal_tuple, a_h, r_h_obs_env, s_h_prime_tuple, episode_done_iql, do_debug_episode)
                r_r_calc = self.calculate_robot_internal_reward(s_h_prime_tuple, episode_done_iql, do_debug_episode)
                self.update_robot_q(prev_s_r_tuple, a_r, r_r_calc, s_r_prime_tuple, episode_done_iql, do_debug_episode)

                s_r_tuple = s_r_prime_tuple
                s_h_tuple = s_h_prime_tuple

                if episode_done_iql:
                    if do_debug_episode: print(f"  Episode {e+1} finished at step {step_num+1} due to done={episode_done_iql}.")
                    break 
            
            # Decay epsilon_r per episode
            self.epsilon_r = max(min_epsilon_r, self.epsilon_r - epsilon_decay_rate)

        if self.debug: print("IQL Training Finished.")

    def select_robot_action(self, s_r_tuple):
        if np.random.rand() < self.epsilon_r:
            return np.random.choice(self.action_space_robot)
        else:
            q_values = self.Q_r[s_r_tuple]
            return np.argmax(q_values)

    def select_human_action(self, s_h_tuple, goal_tuple):
        q_values_for_goal = self.Q_h[(s_h_tuple, goal_tuple)]
        exp_q = np.exp(self.beta_h * q_values_for_goal)
        probs = exp_q / np.sum(exp_q)
        if np.isnan(probs).any() or np.isinf(probs).any() or np.sum(exp_q) == 0:
            if self.debug and random.random() < 0.01 : print(f"    WARN: Softmax issue for human action. s_h={s_h_tuple}, goal={goal_tuple}. Q_h={q_values_for_goal}, exp_q={exp_q}. Defaulting to random.")
            return np.random.choice(self.action_space_human)
        return np.random.choice(self.action_space_human, p=probs)

    def update_human_q(self, s_h_tuple, goal_tuple, a_h, r_h_obs, s_h_prime_tuple, done, do_debug_episode):
        q_key = (s_h_tuple, goal_tuple)
        q_current_val = self.Q_h[q_key][a_h]
        
        v_h_s_prime_g = 0
        if not done:
            q_values_next_state_goal = self.Q_h[(s_h_prime_tuple, goal_tuple)]
            exp_q_next = np.exp(self.beta_h * q_values_next_state_goal)
            sum_exp_q_next = np.sum(exp_q_next)
            if sum_exp_q_next == 0 or np.isinf(sum_exp_q_next) or np.isnan(sum_exp_q_next):
                 if self.debug and random.random() < 0.01: print(f"    WARN: Softmax issue in V_h calc for s_h'={s_h_prime_tuple}, goal={goal_tuple}. Q_h'={q_values_next_state_goal}. V_h set to 0.")
                 v_h_s_prime_g = 0
            else:
                probs_next = exp_q_next / sum_exp_q_next
                if np.isnan(probs_next).any():
                    if self.debug and random.random() < 0.01: print(f"    WARN: NaN probs in V_h calc for s_h'={s_h_prime_tuple}, goal={goal_tuple}. Q_h'={q_values_next_state_goal}. V_h set to 0.")
                    v_h_s_prime_g = 0
                else:
                    v_h_s_prime_g = np.sum(probs_next * q_values_next_state_goal)
        
        q_target = r_h_obs + self.gamma_h * v_h_s_prime_g
        self.Q_h[q_key][a_h] += self.alpha_h * (q_target - q_current_val)

        if do_debug_episode and random.random() < self.debug_q_update_sample_rate:
             print(f"    Human Q_h update: s_h={s_h_tuple}, g={goal_tuple}, a_h={a_h}, r_h={r_h_obs:.2f}, s_h'={s_h_prime_tuple}, done={done}")
             print(f"      Q_h_curr={q_current_val:.3f}, V_h(s',g)={v_h_s_prime_g:.3f}, Q_h_target={q_target:.3f}, New_Q_h={self.Q_h[q_key][a_h]:.3f}")

    def calculate_robot_internal_reward(self, s_h_prime_tuple, done, do_debug_episode):
        # Using r_R = E_{g' ~ mu_g} [V_H(s', g')]
        expected_v_h_s_prime = 0
        if not done:
            for i, g_prime_actual in enumerate(self.G): # g_prime_actual is the (x,y) tuple
                g_prime_tuple = self.state_to_tuple(g_prime_actual)
                q_values_s_prime_g_prime = self.Q_h[(s_h_prime_tuple, g_prime_tuple)]
                exp_q_s_prime_g_prime = np.exp(self.beta_h * q_values_s_prime_g_prime)
                sum_exp_q_s_prime_g_prime = np.sum(exp_q_s_prime_g_prime)

                v_h_s_prime_g_prime = 0
                if sum_exp_q_s_prime_g_prime == 0 or np.isinf(sum_exp_q_s_prime_g_prime) or np.isnan(sum_exp_q_s_prime_g_prime):
                    if self.debug and random.random() < 0.01: print(f"    WARN: Softmax issue in robot internal reward V_h calc for s_h'={s_h_prime_tuple}, g'={g_prime_tuple}. Q_h'={q_values_s_prime_g_prime}. V_h for this g' set to 0.")
                else:
                    probs_s_prime_g_prime = exp_q_s_prime_g_prime / sum_exp_q_s_prime_g_prime
                    if np.isnan(probs_s_prime_g_prime).any():
                         if self.debug and random.random() < 0.01: print(f"    WARN: NaN probs in robot internal reward V_h calc for s_h'={s_h_prime_tuple}, g'={g_prime_tuple}. Q_h'={q_values_s_prime_g_prime}. V_h for this g' set to 0.")
                    else:
                        v_h_s_prime_g_prime = np.sum(probs_s_prime_g_prime * q_values_s_prime_g_prime)
                
                expected_v_h_s_prime += self.mu_g[i] * v_h_s_prime_g_prime
        
        r_r_calc = expected_v_h_s_prime
        if do_debug_episode and random.random() < self.debug_q_update_sample_rate :
            print(f"    Robot internal reward calc: s_h'={s_h_prime_tuple}, done={done}, r_r_calc={r_r_calc:.3f}")
        return r_r_calc

    def update_robot_q(self, s_r_tuple, a_r, r_r_calc, s_r_prime_tuple, done, do_debug_episode):
        q_current_val = self.Q_r[s_r_tuple][a_r]
        
        max_q_next = 0
        if not done:
            # Check if s_r_prime_tuple is actually in Q_r, if not, it implies Q values are default (e.g. 0 or small random)
            # This is handled by defaultdict returning default values for new states.
            max_q_next = np.max(self.Q_r[s_r_prime_tuple])
            
        q_target = r_r_calc + self.gamma_r * max_q_next
        self.Q_r[s_r_tuple][a_r] += self.alpha_r * (q_target - q_current_val)

        if do_debug_episode and random.random() < self.debug_q_update_sample_rate:
             print(f"    Robot Q_r update: s_r={s_r_tuple}, a_r={a_r}, r_r_calc={r_r_calc:.3f}, s_r'={s_r_prime_tuple}, done={done}")
             print(f"      Q_r_curr={q_current_val:.3f}, max_Q_r(s',a')={max_q_next:.3f}, Q_r_target={q_target:.3f}, New_Q_r={self.Q_r[s_r_tuple][a_r]:.3f}")