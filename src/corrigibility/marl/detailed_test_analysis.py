#!/usr/bin/env python3
"""
Detailed analysis of the training vs testing discrepancy
"""

import numpy as np
from fixed_enhanced_algorithm import FixedEnhancedTwoPhaseTimescaleIQL

def analyze_simple_map2():
    """Deep analysis of simple_map2 training vs testing"""
    print("=== Deep Analysis of simple_map2 ===")
    
    from envs.simple_map2 import get_map
    from env import CustomEnvironment
    
    map_layout, map_metadata = get_map()
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
    
    # Create algorithm
    algorithm = FixedEnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.5,
        alpha_e=0.5,
        alpha_r=0.5,
        alpha_p=0.5,
        gamma_h=0.9,
        gamma_r=0.9,
        beta_r_0=3.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Train
    print("\nTraining Phase 1...")
    algorithm.train_phase1_fixed(env, episodes=200, max_steps=50)
    
    print("\nTraining Phase 2...")
    algorithm.train_phase2_enhanced(env, episodes=300, max_steps=50)
    
    # Test with detailed output
    print("\nTesting with detailed output...")
    success_count = 0
    episodes = 10
    
    for episode in range(episodes):
        env.reset()
        goal = env.human_goals[algorithm.human_agent_ids[0]]
        goal_tuple = algorithm.state_to_tuple(goal)
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
        print(f"Goal: {goal}")
        
        episode_success = False
        
        for step in range(50):
            actions = {}
            
            # Human action with same exploration as training
            for hid in algorithm.human_agent_ids:
                state_h = algorithm.get_human_state(env, hid, goal_tuple)
                actions[hid] = algorithm.sample_human_action_effective(hid, state_h, goal_tuple, epsilon=0.1)
            
            # Robot action with same exploration as training
            for rid in algorithm.robot_agent_ids:
                state_r = algorithm.get_full_state(env, rid)
                actions[rid] = algorithm.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)
            
            action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
            readable = {k: action_names.get(v, str(v)) for k, v in actions.items()}
            print(f"  Step {step + 1}: {readable}")
            
            env.step(actions)
            
            print(f"    -> Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
            
            # Check for success
            human_pos = env.agent_positions[algorithm.human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                episode_success = True
                print(f"    🎉 SUCCESS at step {step + 1}!")
                break
        
        if not episode_success:
            print(f"    ❌ Failed")
            
            # Show Q-values for human and robot at final state
            hid = algorithm.human_agent_ids[0]
            rid = algorithm.robot_agent_ids[0]
            
            state_h = algorithm.get_human_state(env, hid, goal_tuple)
            state_r = algorithm.get_full_state(env, rid)
            
            print(f"    Final human state: {state_h}")
            if state_h in algorithm.q_e[hid]:
                print(f"    Human Q^e values: {algorithm.q_e[hid][state_h]}")
            else:
                print(f"    Human Q^e values: [NOT LEARNED]")
            
            print(f"    Final robot state: {state_r}")
            if state_r in algorithm.q_r[rid]:
                print(f"    Robot Q^r values: {algorithm.q_r[rid][state_r]}")
            else:
                print(f"    Robot Q^r values: [NOT LEARNED]")
    
    success_rate = success_count / episodes
    print(f"\nDetailed Test Results: {success_rate:.1%} success ({success_count}/{episodes})")
    
    # Analyze learned policies
    print("\n=== Policy Analysis ===")
    
    # Sample a few states and show learned values
    print("\nSample Q^e values for human:")
    count = 0
    for state, values in algorithm.q_e[algorithm.human_agent_ids[0]].items():
        if count < 5:
            print(f"  State {state}: {values}")
            count += 1
    
    print("\nSample Q^r values for robot:")
    count = 0
    for state, values in algorithm.q_r[algorithm.robot_agent_ids[0]].items():
        if count < 5:
            print(f"  State {state}: {values}")
            count += 1
    
    return success_rate

def analyze_training_vs_testing():
    """Compare training success vs testing success"""
    print("=== Training vs Testing Analysis ===")
    
    # Check if there's a fundamental difference in how we're calling things
    from envs.simple_map2 import get_map
    from env import CustomEnvironment
    
    map_layout, map_metadata = get_map()
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
    
    # Create algorithm
    algorithm = FixedEnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.5,
        alpha_e=0.5,
        alpha_r=0.5,
        alpha_p=0.5,
        gamma_h=0.9,
        gamma_r=0.9,
        beta_r_0=3.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Quick training
    print("Quick training...")
    algorithm.train_phase1_fixed(env, episodes=100, max_steps=50)
    algorithm.train_phase2_enhanced(env, episodes=100, max_steps=50)
    
    # Now let's manually test the exact same way as training
    print("\nManual testing with exact same parameters as training...")
    success_count = 0
    episodes = 20
    
    for episode in range(episodes):
        env.reset()
        goal = env.human_goals[algorithm.human_agent_ids[0]]
        goal_tuple = algorithm.state_to_tuple(goal)
        
        for step in range(50):
            actions = {}
            
            # Use exact same action selection as in training
            for hid in algorithm.human_agent_ids:
                state_h = algorithm.get_human_state(env, hid, goal_tuple)
                actions[hid] = algorithm.sample_human_action_effective(hid, state_h, goal_tuple, epsilon=0.1)
            
            for rid in algorithm.robot_agent_ids:
                state_r = algorithm.get_full_state(env, rid)
                actions[rid] = algorithm.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)
            
            env.step(actions)
            
            # Check for success
            human_pos = env.agent_positions[algorithm.human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                success_count += 1
                print(f"Episode {episode + 1}: SUCCESS at step {step + 1}")
                break
    
    success_rate = success_count / episodes
    print(f"\nManual Test Results: {success_rate:.1%} success ({success_count}/{episodes})")
    
    # Now test the algorithm's built-in test_policy
    print("\nBuilt-in test_policy results:")
    built_in_success, _ = algorithm.test_policy(env, episodes=20, max_steps=50, verbose=False)
    print(f"Built-in Test Results: {built_in_success:.1%} success")
    
    if abs(success_rate - built_in_success) > 0.1:
        print("⚠️ DISCREPANCY FOUND between manual and built-in testing!")
    else:
        print("✅ Manual and built-in testing match")

if __name__ == "__main__":
    analyze_simple_map2()
    analyze_training_vs_testing()