#!/usr/bin/env python3
"""
Fixed training script that accounts for environment constraints.
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime

from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment


def get_map_by_name(map_name):
    """Get map function by name"""
    map_functions = {
        "simple_map": get_simple_map,
        "simple_map2": get_simple_map2,
        "simple_map3": get_simple_map3,
        "simple_map4": get_simple_map4,
    }
    
    if map_name not in map_functions:
        raise ValueError(f"Unknown map: {map_name}. Available: {list(map_functions.keys())}")
    
    return map_functions[map_name]


class FixedTwoPhaseTimescaleIQL(TwoPhaseTimescaleIQL):
    """Fixed version of IQL that handles environment constraints better"""
    
    def _get_assistive_action_fixed(self, env, robot_id):
        """Fixed assistive action that works with environment constraints"""
        robot_pos = env.agent_positions[robot_id]
        current_dir = env.agent_dirs[robot_id]
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        
        # Check if there's a key in front to pick up (robot doesn't need to move to key)
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
        
        # Look for keys to face (don't try to move to them)
        for key in env.keys:
            key_pos = tuple(key["pos"])
            
            # Calculate direction needed to face key
            dx = key_pos[0] - robot_pos[0]
            dy = key_pos[1] - robot_pos[1]
            
            if abs(dx) > abs(dy):
                target_dir = 2 if dx > 0 else 0  # Down or Up
            else:
                target_dir = 1 if dy > 0 else 3  # Right or Left
            
            # Turn to face the key
            if current_dir != target_dir:
                left_turn = (current_dir - 1) % 4
                right_turn = (current_dir + 1) % 4
                
                if left_turn == target_dir:
                    return 0  # Turn left
                elif right_turn == target_dir:
                    return 1  # Turn right
                else:
                    return 0  # Turn left arbitrarily
        
        # Look for locked doors to face
        for door in env.doors:
            if door["is_locked"] and door["color"] in env.robot_has_keys:
                door_pos = tuple(door["pos"])
                
                # Calculate direction needed to face door
                dx = door_pos[0] - robot_pos[0]
                dy = door_pos[1] - robot_pos[1]
                
                if abs(dx) > abs(dy):
                    target_dir = 2 if dx > 0 else 0  # Down or Up
                else:
                    target_dir = 1 if dy > 0 else 3  # Right or Left
                
                # Turn to face the door
                if current_dir != target_dir:
                    left_turn = (current_dir - 1) % 4
                    right_turn = (current_dir + 1) % 4
                    
                    if left_turn == target_dir:
                        return 0  # Turn left
                    elif right_turn == target_dir:
                        return 1  # Turn right
                    else:
                        return 0  # Turn left arbitrarily
        
        # If robot has keys but no door to open, try to move to be adjacent to door
        if env.robot_has_keys:
            for door in env.doors:
                if door["is_locked"]:
                    door_pos = tuple(door["pos"])
                    
                    # Try to get adjacent to door
                    adjacent_positions = [
                        (door_pos[0] - 1, door_pos[1]),  # Above
                        (door_pos[0] + 1, door_pos[1]),  # Below
                        (door_pos[0], door_pos[1] - 1),  # Left
                        (door_pos[0], door_pos[1] + 1),  # Right
                    ]
                    
                    for adj_pos in adjacent_positions:
                        if env._is_valid_pos(adj_pos):
                            # Calculate direction to this position
                            dx = adj_pos[0] - robot_pos[0]
                            dy = adj_pos[1] - robot_pos[1]
                            
                            if abs(dx) <= 1 and abs(dy) <= 1 and (dx != 0 or dy != 0):
                                # We can reach this position in one move
                                if abs(dx) > abs(dy):
                                    target_dir = 2 if dx > 0 else 0
                                else:
                                    target_dir = 1 if dy > 0 else 3
                                
                                if current_dir == target_dir:
                                    return 2  # Move forward
                                else:
                                    # Turn to face the target
                                    left_turn = (current_dir - 1) % 4
                                    right_turn = (current_dir + 1) % 4
                                    
                                    if left_turn == target_dir:
                                        return 0
                                    elif right_turn == target_dir:
                                        return 1
                                    else:
                                        return 0
        
        # Default: random action to explore
        return np.random.choice([0, 1, 2])
    
    def sample_robot_action_phase2(self, agent_id, state, env=None, epsilon=0.1):
        """Fixed robot action sampling for Phase 2"""
        
        # Use assistive heuristic more often
        if env is not None and np.random.rand() < 0.7:  # Higher probability
            return self._get_assistive_action_fixed(env, agent_id)

        if self.network:
            q_values_np = self.q_r_backend.get_q_values(agent_id, state)
        else:
            q_values_np = self.q_r[agent_id][state]

        # Add epsilon-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space_dict[agent_id])

        # Softmax over Q-values with higher temperature for more exploration
        exp_q = np.exp(q_values_np / 2.0)  # Higher temperature
        probabilities = exp_q / np.sum(exp_q)

        return np.random.choice(self.action_space_dict[agent_id], p=probabilities)
    
    def train_phase1_fixed(self, env, episodes, max_steps=100):
        """Fixed Phase 1 training with better exploration"""
        print("Starting Phase 1: Learning cautious human model with robot blocking.")
        
        for ep in range(episodes):
            env.reset()
            
            # Sample initial goal for each human
            current_goals = {}
            for hid in self.human_agent_ids:
                goal_idx = np.random.choice(len(self.G), p=self.mu_g)
                current_goals[hid] = self.state_to_tuple(self.G[goal_idx])

            for step in range(max_steps):
                actions = {}
                
                # Robot action (Phase 1): Block the human with more variety
                for rid in self.robot_agent_ids:
                    if np.random.rand() < 0.3:  # Sometimes do random action
                        actions[rid] = np.random.choice([0, 1, 2])
                    else:
                        # Try to block human
                        human_id = self.human_agent_ids[0]
                        actions[rid] = self._get_movement_action(env, rid, human_id)

                # Human action (Phase 1) with more exploration
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.4)  # Higher exploration

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Update human's cautious model (Q_m) with better reward shaping
                for hid in self.human_agent_ids:
                    goal = current_goals[hid]
                    state_h = self.get_human_state(env, hid, goal)
                    next_state_h = self.get_human_state(env, hid, goal)
                    action_h = actions[hid]
                    reward_h = rewards[hid]

                    # Enhanced reward shaping
                    pos = np.array(env.agent_positions[hid])
                    goal_pos = np.array(goal)
                    dist_to_goal = np.sum(np.abs(pos - goal_pos))
                    
                    # Reward for facing the right direction
                    direction_bonus = 0
                    if dist_to_goal > 0:
                        dx = goal_pos[0] - pos[0]
                        dy = goal_pos[1] - pos[1]
                        if abs(dx) > abs(dy):
                            target_dir = 2 if dx > 0 else 0
                        else:
                            target_dir = 1 if dy > 0 else 3
                        
                        if env.agent_dirs[hid] == target_dir:
                            direction_bonus = 0.05
                    
                    shaped_reward = reward_h - 0.05 * dist_to_goal + direction_bonus

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
    
    def train_phase2_fixed(self, env, episodes, max_steps=100):
        """Fixed Phase 2 training with assistive behavior"""
        print("Starting Phase 2: Learning assistive robot policy and power estimation.")
        
        for ep in range(episodes):
            env.reset()
            goal = env.human_goals[self.human_agent_ids[0]]

            for step in range(max_steps):
                actions = {}
                
                # Robot action (Phase 2): Assist the human
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    actions[rid] = self.sample_robot_action_phase2(rid, state_r, env, epsilon=0.4)

                # Human action (Phase 2): Use cautious model with exploration
                for hid in self.human_agent_ids:
                    state_h = self.get_human_state(env, hid, self.state_to_tuple(goal))
                    actions[hid] = self.sample_human_action_phase1(hid, state_h, epsilon=0.3)

                next_obs, rewards, terms, truncs, _ = env.step(actions)
                done = any(terms.values()) or any(truncs.values())

                # Enhanced reward calculation
                door_opened_bonus = 0
                key_collected_bonus = 0
                goal_reached_bonus = 0
                
                # Check if door was opened this step
                if any(not d['is_locked'] for d in env.doors):
                    door_opened_bonus = 1.0
                
                # Check if key was collected
                if len(env.robot_has_keys) > 0:
                    key_collected_bonus = 0.5
                
                # Check if human reached goal
                human_pos = env.agent_positions[self.human_agent_ids[0]]
                if tuple(human_pos) == tuple(goal):
                    goal_reached_bonus = 10.0

                # Update robot Q-values with enhanced rewards
                for rid in self.robot_agent_ids:
                    state_r = self.get_full_state(env, rid)
                    next_state_r = self.get_full_state(env, rid)
                    action_r = actions[rid]
                    
                    enhanced_reward = door_opened_bonus + key_collected_bonus + goal_reached_bonus
                    
                    if self.network:
                        next_q_values = self.q_r_backend.get_q_values(rid, next_state_r)
                        target = enhanced_reward + self.gamma_r * np.max(next_q_values)
                        self.q_r_backend.update_q_values(rid, state_r, action_r, target, self.alpha_r)
                    else:
                        old_q_r = self.q_r[rid][state_r][action_r]
                        next_max_q_r = np.max(self.q_r[rid][next_state_r])
                        new_q_r = old_q_r + self.alpha_r * (enhanced_reward + self.gamma_r * next_max_q_r - old_q_r)
                        self.q_r[rid][state_r][action_r] = new_q_r

                # Update human effective model
                for hid in self.human_agent_ids:
                    goal_tuple = self.state_to_tuple(goal)
                    state_base = self.get_full_state(env, hid)
                    next_state_base = self.get_full_state(env, hid)
                    action_h = actions[hid]
                    
                    # Enhanced reward for human
                    reward_h = rewards[hid] + goal_reached_bonus
                    
                    if self.network:
                        next_q_values = self.q_e_backend.get_q_values(hid, next_state_base, goal_tuple)
                        target = reward_h + self.gamma_h * np.max(next_q_values)
                        self.q_e_backend.update_q_values(hid, state_base, action_h, target, self.alpha_e, goal_tuple)
                    else:
                        state_h = self.get_human_state(env, hid, goal_tuple)
                        next_state_h = self.get_human_state(env, hid, goal_tuple)
                        old_q_e = self.q_e[hid][state_h][action_h]
                        next_max_q_e = np.max(self.q_e[hid][next_state_h])
                        new_q_e = old_q_e + self.alpha_e * (reward_h + self.gamma_h * next_max_q_e - old_q_e)
                        self.q_e[hid][state_h][action_h] = new_q_e

                if done or goal_reached_bonus > 0:
                    if goal_reached_bonus > 0:
                        print(f"    🎉 Goal reached in episode {ep+1}!")
                    break
                    
            if (ep + 1) % 100 == 0:
                print(f"  Phase 2, Episode {ep+1}/{episodes} completed.")


def train_single_map_fixed(map_name, phase1_episodes=300, phase2_episodes=700, save_dir="checkpoints"):
    """Fixed training function"""
    print(f"=== Fixed Training on {map_name} ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    get_map_func = get_map_by_name(map_name)
    map_layout, map_metadata = get_map_func()
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    print(f"Map: {map_metadata['name']}")
    print(f"Size: {map_metadata['size']}")
    print(f"Human goals: {map_metadata['human_goals']}")
    
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),
        "human_0": list(range(3))
    }
    
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    # Create fixed IQL algorithm
    iql = FixedTwoPhaseTimescaleIQL(
        alpha_m=0.3,  # Higher learning rates
        alpha_e=0.3,
        alpha_r=0.3,
        alpha_p=0.3,
        gamma_h=0.9,
        gamma_r=0.9,
        beta_r_0=5.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Fixed training
    print(f"\n--- Fixed Phase 1 Training ({phase1_episodes} episodes) ---")
    iql.train_phase1_fixed(env, episodes=phase1_episodes, max_steps=50)
    
    print(f"\n--- Fixed Phase 2 Training ({phase2_episodes} episodes) ---")
    iql.train_phase2_fixed(env, episodes=phase2_episodes, max_steps=50)
    
    # Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_checkpoint = os.path.join(save_dir, f"{map_name}_fixed_{timestamp}.pkl")
    iql.save_models(final_checkpoint)
    print(f"Fixed checkpoint saved: {final_checkpoint}")
    
    # Validation
    print(f"\n--- Validation ---")
    success_count = 0
    test_episodes = 20
    
    for episode in range(test_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(200):
            actions = {}
            
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
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
    print(f"Validation success rate: {success_rate:.1%}")
    
    return final_checkpoint, success_rate


def main():
    parser = argparse.ArgumentParser(description="Fixed training for IQL")
    parser.add_argument("map_name", choices=["simple_map", "simple_map2", "simple_map3", "simple_map4"],
                       help="Name of the map to train on")
    parser.add_argument("--phase1-episodes", type=int, default=300)
    parser.add_argument("--phase2-episodes", type=int, default=700)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    
    args = parser.parse_args()
    
    try:
        checkpoint_path, success_rate = train_single_map_fixed(
            args.map_name,
            args.phase1_episodes,
            args.phase2_episodes,
            args.save_dir
        )
        
        print(f"\n{'='*60}")
        if success_rate > 0.1:
            print(f"✅ FIXED TRAINING SUCCESSFUL")
            print(f"Success rate: {success_rate:.1%}")
        else:
            print(f"⚠️  FIXED TRAINING PARTIAL SUCCESS")
            print(f"Success rate: {success_rate:.1%}")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"\nTo test: python test_trained_model.py {checkpoint_path} {args.map_name}")
        
    except Exception as e:
        print(f"❌ Fixed training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()