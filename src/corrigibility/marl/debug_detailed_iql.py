#!/usr/bin/env python3
"""
Detailed debugging visualization for IQL tabular case.
"""

import sys
import os


import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def print_grid_with_agents(env):
    """Print the grid with agent positions clearly marked"""
    print("\nCurrent Grid State:")
    for r in range(env.grid_size):
        row = ""
        for c in range(env.grid_size):
            pos = (r, c)
            if pos == env.agent_positions.get('robot_0'):
                row += "R"
            elif pos == env.agent_positions.get('human_0'):
                row += "H"
            else:
                char = env.grid[r, c]
                if char == ' ':
                    row += "."
                else:
                    row += char
        print(f"  {row}")


def debug_detailed():
    """Detailed debugging of the IQL algorithm"""
    print("=== Detailed IQL Debug ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    print(f"Map: {map_metadata['name']}")
    print(f"Goal: {map_metadata['human_goals']}")
    
    # Setup agent IDs and action spaces
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),  # [turn_left, turn_right, forward, pickup, drop, toggle]
        "human_0": list(range(3))   # [turn_left, turn_right, forward]
    }
    
    # Get possible human goals
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    print(f"Human goals: {human_goals}")
    print(f"G: {G}")
    
    # Create IQL algorithm instance
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.3, alpha_e=0.3, alpha_r=0.3, alpha_p=0.3,
        gamma_h=0.9, gamma_r=0.9, beta_r_0=5.0,
        G=G, mu_g=mu_g, action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids, human_agent_ids=human_agent_ids,
        network=False, env=env
    )
    
    # Train with more episodes and debugging
    print("\n" + "="*50)
    print("PHASE 1 TRAINING (Robot blocks human)")
    print("="*50)
    
    # Show some phase 1 training episodes
    for ep in range(3):
        print(f"\nPhase 1 Episode {ep + 1}:")
        env.reset()
        print_grid_with_agents(env)
        
        # Sample initial goal for human
        goal_idx = np.random.choice(len(G), p=mu_g)
        current_goal = iql.state_to_tuple(G[goal_idx])
        print(f"Goal: {current_goal}")
        
        for step in range(10):
            actions = {}
            
            # Robot action (Phase 1): Block the human
            human_id = human_agent_ids[0]
            for rid in robot_agent_ids:
                actions[rid] = iql._get_movement_action(env, rid, human_id)
            
            # Human action (Phase 1)
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, current_goal)
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.2)
            
            print(f"  Step {step + 1}: actions={actions}")
            
            # Execute actions and update Q-values
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            
            # Update human's cautious model (Q_m) - simplified version
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, current_goal)
                action_h = actions[hid]
                reward_h = rewards[hid]
                
                # Add distance-based reward shaping
                pos = np.array(env.agent_positions[hid])
                goal_pos = np.array(current_goal)
                dist_to_goal = np.sum(np.abs(pos - goal_pos))
                shaped_reward = reward_h - 0.1 * dist_to_goal
                
                old_q = iql.q_m[hid][state_h][action_h]
                next_state_h = iql.get_human_state(env, hid, current_goal)
                next_max_q = np.max(iql.q_m[hid][next_state_h])
                new_q = old_q + iql.alpha_m * (shaped_reward + iql.gamma_h * next_max_q - old_q)
                iql.q_m[hid][state_h][action_h] = new_q
                
                print(f"    Human Q-update: state={len(str(state_h))}, action={action_h}, reward={shaped_reward:.2f}, old_q={old_q:.3f}, new_q={new_q:.3f}")
            
            print(f"    Positions: {env.agent_positions}")
            
            done = any(terms.values()) or any(truncs.values())
            if done:
                break
    
    # Run more phase 1 episodes silently
    print(f"\nRunning remaining Phase 1 episodes silently...")
    iql.train_phase1(env, episodes=100, max_steps=50)
    
    print("\n" + "="*50)
    print("PHASE 2 TRAINING (Robot assists human)")
    print("="*50)
    
    # Show some phase 2 training episodes
    for ep in range(3):
        print(f"\nPhase 2 Episode {ep + 1}:")
        env.reset()
        print_grid_with_agents(env)
        
        goal = env.human_goals[human_agent_ids[0]]
        print(f"Goal: {goal}")
        print(f"Keys: {[k['pos'] for k in env.keys]}")
        print(f"Doors: {[(d['pos'], d['is_locked']) for d in env.doors]}")
        
        for step in range(15):
            actions = {}
            
            # Robot action (Phase 2): Assist the human
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                # Use assistive action with high probability during training
                if np.random.rand() < 0.5:
                    actions[rid] = iql._get_assistive_action(env, rid)
                else:
                    actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)
            
            # Human action (Phase 2): Use cautious model
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.1)
            
            action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
            readable_actions = {agent: action_names.get(int(action), str(action)) for agent, action in actions.items()}
            print(f"  Step {step + 1}: {readable_actions}")
            
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            
            print(f"    Positions: {env.agent_positions}")
            print(f"    Robot keys: {env.robot_has_keys}")
            print(f"    Keys left: {[k['pos'] for k in env.keys]}")
            print(f"    Doors: {[(d['pos'], d['is_locked'], d['is_open']) for d in env.doors]}")
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                print(f"    🎉 GOAL REACHED!")
                break
            
            done = any(terms.values()) or any(truncs.values())
            if done:
                break
    
    # Complete phase 2 training
    print(f"\nRunning remaining Phase 2 episodes silently...")
    iql.train_phase2(env, episodes=200, max_steps=50)
    
    print("\n" + "="*50)
    print("FINAL TESTING")
    print("="*50)
    
    # Test final performance
    goal_reached = 0
    test_episodes = 5
    
    for episode in range(test_episodes):
        print(f"\nTest Episode {episode + 1}:")
        env.reset()
        print_grid_with_agents(env)
        
        goal = env.human_goals[human_agent_ids[0]]
        print(f"Goal: {goal}")
        
        for step in range(100):
            actions = {}
            
            # Get trained actions
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.0)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            if step < 10 or step % 10 == 0:  # Print first 10 steps and every 10th step
                action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
                readable_actions = {agent: action_names.get(int(action), str(action)) for agent, action in actions.items()}
                print(f"  Step {step + 1}: {readable_actions}")
                print(f"    Pos: {env.agent_positions}, Keys: {list(env.robot_has_keys)}")
            
            env.step(actions)
            
            # Check if human reached goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                goal_reached += 1
                print(f"  🎉 GOAL REACHED at step {step + 1}!")
                break
        else:
            print(f"  Episode timed out")
    
    success_rate = goal_reached / test_episodes
    print(f"\nFinal Success Rate: {success_rate:.1%}")
    
    return success_rate > 0


if __name__ == "__main__":
    success = debug_detailed()
    print("✅ SUCCESS!" if success else "❌ FAILED!")