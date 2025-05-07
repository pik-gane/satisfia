import numpy as np
import random
from collections import defaultdict

class IQLPowerAgent:
    """
    Implements Algorithm 1: Two-Timescale Goal-Based IQL.
    Adapted for MiniGrid environment observations.

    State representation for Q-tables: tuple(agent_pos, human_pos)
    """
    def __init__(self,
                 action_space_size,
                 goal_set,          # List of potential goal positions (tuples) for H
                 goal_prior,        # Dict mapping goal index to probability
                 gamma_h=0.99,
                 gamma_r=0.99,
                 alpha_h=0.1,       # Faster learning rate for human model
                 alpha_r=0.01,      # Slower learning rate for robot
                 beta_h=1.0,        # Human rationality (inverse temperature)
                 epsilon_r=1.0,     # Initial robot exploration rate
                 epsilon_r_decay=0.999,
                 epsilon_r_min=0.01,
                 f_func=lambda z: z, # Power transformation function
                 eta=0.0             # Power exponent eta
                 ):

        self.action_space_size = action_space_size
        self.goal_set = goal_set # Store the actual goal coordinates
        self.goal_indices = {goal: i for i, goal in enumerate(goal_set)} # Map goal pos to index
        self.num_goals = len(goal_set)
        self.goal_prior = goal_prior

        if not np.isclose(sum(self.goal_prior.values()), 1.0):
             print(f"Warning: Goal prior probabilities do not sum to 1: {self.goal_prior}")
             # Normalize or raise error
             total_prob = sum(self.goal_prior.values())
             if total_prob > 1e-6:
                 self.goal_prior = {idx: p / total_prob for idx, p in self.goal_prior.items()}
                 print(f"Normalized goal prior: {self.goal_prior}")
             else:
                 print("Error: Cannot normalize zero goal prior. Defaulting to uniform.")
                 self.goal_prior = {i: 1.0/self.num_goals for i in range(self.num_goals)}


        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.alpha_h = alpha_h
        self.alpha_r = alpha_r
        self.beta_h = beta_h
        self.epsilon_r = epsilon_r
        self.epsilon_r_decay = epsilon_r_decay
        self.epsilon_r_min = epsilon_r_min

        self.f_func = f_func
        self.eta = eta

        # Q-tables: Use tuple(agent_pos, human_pos) as state key
        # Q_h[state_tuple][goal_index][action]
        self.Q_h = defaultdict(lambda: np.zeros((self.num_goals, self.action_space_size)))
        # Q_r[state_tuple][action]
        self.Q_r = defaultdict(lambda: np.zeros(self.action_space_size))

    def _get_state_tuple(self, agent_pos, human_pos):
        """Converts positions into a hashable state tuple for Q-table keys."""
        # Make sure positions are tuples
        return (tuple(agent_pos), tuple(human_pos))

    def choose_robot_action(self, agent_pos, human_pos):
        """Chooses robot action using epsilon-greedy policy based on Qr."""
        state_tuple = self._get_state_tuple(agent_pos, human_pos)
        if random.random() < self.epsilon_r:
            return random.randint(0, self.action_space_size - 1) # Explore
        else:
            q_values = self.Q_r[state_tuple]
            # Handle ties by choosing randomly among the best actions
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return random.choice(best_actions) # Exploit

    def _get_human_policy_for_goal(self, state_tuple, goal_index):
        """Calculates the softmax policy for the human for a specific goal."""
        q_values = self.Q_h[state_tuple][goal_index]
        if self.beta_h == 0: # Avoid division by zero, uniform policy
             return np.ones(self.action_space_size) / self.action_space_size
        scaled_q = q_values * self.beta_h
        stable_q = scaled_q - np.max(scaled_q) # Numerical stability
        exp_q = np.exp(stable_q)
        policy_sum = np.sum(exp_q)
        if policy_sum < 1e-8: # Avoid division by zero if all exp_q are tiny
             return np.ones(self.action_space_size) / self.action_space_size
        policy = exp_q / policy_sum
        if np.isnan(policy).any(): # Fallback if something went wrong
            return np.ones(self.action_space_size) / self.action_space_size
        return policy

    def _calculate_V_h(self, state_tuple, goal_index):
        """Calculates the expected value Vh(state, goal_index) under the current human policy."""
        policy = self._get_human_policy_for_goal(state_tuple, goal_index)
        q_values = self.Q_h[state_tuple][goal_index]
        return np.dot(policy, q_values)

    def update(self, agent_pos, human_pos, robot_action, human_action, reward_h_obs, next_agent_pos, next_human_pos, current_goal_idx, done):
        """
        Updates Q_h and Q_r based on the observed transition.
        State is represented by (agent_pos, human_pos).
        """
        state_tuple = self._get_state_tuple(agent_pos, human_pos)
        next_state_tuple = self._get_state_tuple(next_agent_pos, next_human_pos)

        # --- Human Q-Update (Fast Timescale) ---
        if done:
            V_h_next_current_goal = 0.0
        else:
            V_h_next_current_goal = self._calculate_V_h(next_state_tuple, current_goal_idx)

        Q_h_target = reward_h_obs + self.gamma_h * V_h_next_current_goal
        td_error_h = Q_h_target - self.Q_h[state_tuple][current_goal_idx][human_action]
        self.Q_h[state_tuple][current_goal_idx][human_action] += self.alpha_h * td_error_h

        # --- Calculate Robot's Internal Reward (Based on Estimated Human Power Proxy) ---
        z_s_prime = 0.0
        if not done:
            for g_idx in range(self.num_goals):
                 V_h_next_potential_goal = self._calculate_V_h(next_state_tuple, g_idx)
                 goal_prob = self.goal_prior.get(g_idx, 0)
                 base_value = max(0, V_h_next_potential_goal)
                 z_s_prime += goal_prob * (base_value ** (1 + self.eta))

        r_r_calc = self.f_func(z_s_prime)

        # --- Robot Q-Update (Slow Timescale) ---
        if done:
            Q_r_target = r_r_calc
        else:
            max_Q_r_next = np.max(self.Q_r[next_state_tuple])
            Q_r_target = r_r_calc + self.gamma_r * max_Q_r_next

        td_error_r = Q_r_target - self.Q_r[state_tuple][robot_action]
        self.Q_r[state_tuple][robot_action] += self.alpha_r * td_error_r

        # --- Decay Epsilon ---
        self.epsilon_r = max(self.epsilon_r_min, self.epsilon_r * self.epsilon_r_decay)

    def get_human_action_for_simulation(self, agent_pos, human_pos, current_goal_idx):
         """ Samples a human action based on the learned policy for simulation purposes. """
         state_tuple = self._get_state_tuple(agent_pos, human_pos)
         policy = self._get_human_policy_for_goal(state_tuple, current_goal_idx)
         return np.random.choice(self.action_space_size, p=policy)

