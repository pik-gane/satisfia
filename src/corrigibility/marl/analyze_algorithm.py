#!/usr/bin/env python3
"""
Comprehensive analysis of the IQL algorithm to identify issues.
"""

import numpy as np
from collections import defaultdict

from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def analyze_environment():
    """Analyze the environment to understand constraints"""
    print("=== ENVIRONMENT ANALYSIS ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    print(f"Map name: {map_metadata['name']}")
    print(f"Map size: {map_metadata['size']}")
    print(f"Human goals: {map_metadata['human_goals']}")
    
    print(f"\nGrid layout:")
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
    
    print(f"\nAgent starting positions: {env.agent_positions}")
    print(f"Agent starting directions: {env.agent_dirs}")
    print(f"Keys: {[{'pos': k['pos'], 'color': k['color']} for k in env.keys]}")
    print(f"Doors: {[{'pos': d['pos'], 'locked': d['is_locked'], 'color': d['color']} for d in env.doors]}")
    
    # Test manual movement capabilities
    print(f"\n--- Testing Movement Capabilities ---")
    
    # Test robot movement options
    robot_pos = env.agent_positions['robot_0']
    robot_dir = env.agent_dirs['robot_0']
    
    print(f"Robot at {robot_pos}, facing direction {robot_dir}")
    
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # up, right, down, left
    
    # Check what robot can do
    for action in range(6):
        if action <= 2:  # Movement actions
            if action == 2:  # forward
                dx, dy = deltas[robot_dir]
                front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                can_move = env._is_valid_pos(front_pos) if 0 <= front_pos[0] < env.grid_size and 0 <= front_pos[1] < env.grid_size else False
                print(f"  Action {action} (forward): target {front_pos}, valid: {can_move}")
            else:
                print(f"  Action {action} (turn): always valid")
        elif action == 3:  # pickup
            dx, dy = deltas[robot_dir]
            front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            has_key = any(tuple(k['pos']) == front_pos for k in env.keys)
            print(f"  Action {action} (pickup): front {front_pos}, has key: {has_key}")
        elif action == 5:  # toggle
            dx, dy = deltas[robot_dir]
            front_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            has_door = any(tuple(d['pos']) == front_pos for d in env.doors)
            print(f"  Action {action} (toggle): front {front_pos}, has door: {has_door}")
    
    # Test human movement options
    human_pos = env.agent_positions['human_0']
    human_dir = env.agent_dirs['human_0']
    
    print(f"\nHuman at {human_pos}, facing direction {human_dir}")
    
    for action in range(3):
        if action == 2:  # forward
            dx, dy = deltas[human_dir]
            front_pos = (human_pos[0] + dx, human_pos[1] + dy)
            can_move = env._is_valid_pos(front_pos) if 0 <= front_pos[0] < env.grid_size and 0 <= front_pos[1] < env.grid_size else False
            print(f"  Action {action} (forward): target {front_pos}, valid: {can_move}")
        else:
            print(f"  Action {action} (turn): always valid")
    
    return env


def analyze_training_process():
    """Analyze the training process step by step"""
    print(f"\n=== TRAINING PROCESS ANALYSIS ===")
    
    env = analyze_environment()
    
    # Setup algorithm
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),
        "human_0": list(range(3))
    }
    
    human_goals = [(3, 2)]  # From simple_map
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)
    
    iql = TwoPhaseTimescaleIQL(
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
    
    print(f"\n--- Phase 1 Training Sample ---")
    
    # Run a few Phase 1 episodes with detailed logging
    for ep in range(3):
        print(f"\nPhase 1 Episode {ep + 1}:")
        env.reset()
        
        goal_idx = np.random.choice(len(G), p=mu_g)
        current_goal = iql.state_to_tuple(G[goal_idx])
        
        print(f"  Goal: {current_goal}")
        print(f"  Initial: robot={env.agent_positions['robot_0']}, human={env.agent_positions['human_0']}")
        
        q_values_before = {}
        
        for step in range(10):
            actions = {}
            
            # Robot blocks human
            human_id = human_agent_ids[0]
            robot_id = robot_agent_ids[0]
            actions[robot_id] = iql._get_movement_action(env, robot_id, human_id)
            
            # Human action
            state_h = iql.get_human_state(env, human_id, current_goal)
            actions[human_id] = iql.sample_human_action_phase1(human_id, state_h, epsilon=0.3)
            
            # Store Q-values before update
            if step == 0:
                q_values_before[human_id] = iql.q_m[human_id][state_h].copy()
            
            print(f"    Step {step + 1}: robot_action={actions[robot_id]}, human_action={actions[human_id]}")
            
            # Execute and update
            next_obs, rewards, terms, truncs, _ = env.step(actions)
            
            # Manual Q-value update for human (to see what's happening)
            reward_h = rewards[human_id]
            
            # Add distance-based reward shaping
            pos = np.array(env.agent_positions[human_id])
            goal_pos = np.array(current_goal)
            dist_to_goal = np.sum(np.abs(pos - goal_pos))
            shaped_reward = reward_h - 0.1 * dist_to_goal
            
            next_state_h = iql.get_human_state(env, human_id, current_goal)
            old_q = iql.q_m[human_id][state_h][actions[human_id]]
            next_max_q = np.max(iql.q_m[human_id][next_state_h])
            new_q = old_q + iql.alpha_m * (shaped_reward + iql.gamma_h * next_max_q - old_q)
            iql.q_m[human_id][state_h][actions[human_id]] = new_q
            
            print(f"      Positions: {env.agent_positions}")
            print(f"      Reward: {reward_h:.2f}, Shaped: {shaped_reward:.2f}, Dist: {dist_to_goal}")
            print(f"      Q-update: {old_q:.3f} -> {new_q:.3f}")
            
            if any(terms.values()) or any(truncs.values()):
                break
        
        # Show Q-value changes
        if human_id in q_values_before:
            q_after = iql.q_m[human_id][state_h]
            print(f"  Q-values changed: {np.array_equal(q_values_before[human_id], q_after)}")
    
    print(f"\n--- Assistive Action Analysis ---")
    
    # Test assistive actions
    env.reset()
    robot_id = robot_agent_ids[0]
    
    for step in range(10):
        print(f"\nStep {step + 1}:")
        print(f"  Robot pos: {env.agent_positions[robot_id]}, dir: {env.agent_dirs[robot_id]}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}")
        print(f"  Robot has keys: {list(env.robot_has_keys)}")
        print(f"  Doors: {[(d['pos'], d['is_locked']) for d in env.doors]}")
        
        assistive_action = iql._get_assistive_action(env, robot_id)
        action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
        
        print(f"  Assistive action: {assistive_action} ({action_names.get(assistive_action, 'unknown')})")
        
        # Execute assistive action
        actions = {robot_id: assistive_action}
        env.step(actions)
        
        print(f"  After action: pos={env.agent_positions[robot_id]}, keys={list(env.robot_has_keys)}")
        
        # Check if door is unlocked
        door_unlocked = any(not d['is_locked'] for d in env.doors)
        if door_unlocked:
            print(f"  🎉 DOOR UNLOCKED!")
            break
    
    return iql, env


def test_goal_reachability():
    """Test if goal is actually reachable"""
    print(f"\n=== GOAL REACHABILITY TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    # Try manual sequence: robot picks up key, opens door, human walks to goal
    print("Testing manual sequence:")
    print("1. Robot turns to face key")
    print("2. Robot picks up key")
    print("3. Robot turns to face door")
    print("4. Robot opens door")
    print("5. Human walks to goal")
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    # Step 1: Robot turns to face key (key is at (1,2), robot at (1,1))
    key_pos = env.keys[0]['pos']
    robot_pos = env.agent_positions[robot_id]
    
    print(f"\nRobot at {robot_pos}, key at {key_pos}")
    
    # Robot needs to turn right to face key
    env.step({robot_id: 1})  # turn right
    print(f"After turn right: robot_dir={env.agent_dirs[robot_id]}")
    
    # Step 2: Robot picks up key
    env.step({robot_id: 3})  # pickup
    print(f"After pickup: robot_keys={list(env.robot_has_keys)}, keys_left={len(env.keys)}")
    
    # Step 3: Robot turns to face door (door is at (2,2))
    door_pos = env.doors[0]['pos']
    print(f"Door at {door_pos}, robot at {env.agent_positions[robot_id]}")
    
    # Robot needs to turn right again to face down
    env.step({robot_id: 1})  # turn right to face down
    print(f"After turn: robot_dir={env.agent_dirs[robot_id]}")
    
    # Robot moves forward to be adjacent to door
    env.step({robot_id: 2})  # forward
    print(f"After move: robot_pos={env.agent_positions[robot_id]}")
    
    # Step 4: Robot opens door
    env.step({robot_id: 5})  # toggle
    door_state = env.doors[0]
    print(f"After toggle: door_locked={door_state['is_locked']}, door_open={door_state['is_open']}")
    
    # Step 5: Human walks to goal
    goal_pos = (3, 2)
    human_pos = env.agent_positions[human_id]
    
    print(f"\nHuman at {human_pos}, goal at {goal_pos}")
    print("Human needs to: turn down, move forward, move forward")
    
    # Turn human to face down
    env.step({human_id: 1})  # turn right (assuming human starts facing down, this is turn right)
    print(f"Human dir after turn: {env.agent_dirs[human_id]}")
    
    # Move forward twice
    env.step({human_id: 2})  # forward
    print(f"Human pos after move 1: {env.agent_positions[human_id]}")
    
    env.step({human_id: 2})  # forward
    print(f"Human pos after move 2: {env.agent_positions[human_id]}")
    
    # Check if goal reached
    final_human_pos = env.agent_positions[human_id]
    goal_reached = tuple(final_human_pos) == goal_pos
    
    print(f"\nGoal reached: {goal_reached}")
    print(f"Final human position: {final_human_pos}")
    print(f"Goal position: {goal_pos}")
    
    return goal_reached


def main():
    print("IQL ALGORITHM COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # 1. Environment analysis
    env = analyze_environment()
    
    # 2. Training process analysis
    iql, env = analyze_training_process()
    
    # 3. Goal reachability test
    goal_reachable = test_goal_reachability()
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    print(f"✅ Environment loaded successfully")
    print(f"✅ IQL algorithm initialized")
    print(f"✅ Training process executes")
    print(f"{'✅' if goal_reachable else '❌'} Goal is {'reachable' if goal_reachable else 'NOT reachable'} manually")
    
    if not goal_reachable:
        print(f"\n🔍 ISSUE IDENTIFIED:")
        print(f"The goal may not be reachable due to environment constraints.")
        print(f"Check the map layout and agent movement mechanics.")
    else:
        print(f"\n🔍 POSSIBLE ISSUES:")
        print(f"- Learning rates may be too low")
        print(f"- Not enough training episodes")
        print(f"- Exploration (epsilon) too low during training")
        print(f"- Reward shaping insufficient")
        print(f"- State representation issues")


if __name__ == "__main__":
    main()