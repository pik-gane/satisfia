#!/usr/bin/env python3
"""
Test with exact same conditions as training to isolate the issue.
"""

import numpy as np
from fixed_enhanced_algorithm import FixedEnhancedTwoPhaseTimescaleIQL

def test_exact_training_conditions():
    """Test using exact same conditions as training"""
    print("=== Testing with Exact Training Conditions ===")
    
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
    
    # Train
    print("Phase 1 training...")
    algorithm.train_phase1_fixed(env, episodes=200, max_steps=50)
    
    print("Phase 2 training...")
    algorithm.train_phase2_enhanced(env, episodes=300, max_steps=50)
    
    # Test with EXACT same parameters as training
    print("\n=== Testing with EXACT training parameters ===")
    
    success_count = 0
    test_episodes = 20
    
    for episode in range(test_episodes):
        env.reset()
        # Use same goal selection as training
        goal_idx = np.random.choice(len(G), p=mu_g)
        goal = G[goal_idx]
        goal_tuple = algorithm.state_to_tuple(goal)
        
        # Set the goal in the environment (like training does)
        env.human_goals[human_agent_ids[0]] = goal
        
        print(f"\nTest Episode {episode + 1}: Goal = {goal}")
        
        episode_success = False
        for step in range(50):  # Same max_steps as training
            actions = {}
            
            # EXACT same action selection as training Phase 2
            for hid in human_agent_ids:
                state_h = algorithm.get_human_state(env, hid, goal_tuple)
                actions[hid] = algorithm.sample_human_action_effective(hid, state_h, goal_tuple, epsilon=0.1)
            
            for rid in robot_agent_ids:
                state_r = algorithm.get_full_state(env, rid)
                actions[rid] = algorithm.sample_robot_action_phase2(rid, state_r, env, epsilon=0.3)
            
            # Print first few steps
            if step < 3:
                action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
                readable = {k: action_names.get(v, str(v)) for k, v in actions.items()}
                print(f"  Step {step + 1}: {readable}")
                print(f"    Before: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
            
            env.step(actions)
            
            if step < 3:
                print(f"    After:  Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
            
            # Check success (exact same as training)
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == goal_tuple:
                success_count += 1
                episode_success = True
                print(f"  🎉 SUCCESS at step {step + 1}!")
                break
        
        if not episode_success:
            print(f"  ❌ Failed")
            # Check final states
            hid = human_agent_ids[0]
            rid = robot_agent_ids[0]
            
            final_human_state = algorithm.get_human_state(env, hid, goal_tuple)
            final_robot_state = algorithm.get_full_state(env, rid)
            
            print(f"    Final positions: Robot={env.agent_positions[rid]}, Human={env.agent_positions[hid]}")
            print(f"    Human state: {final_human_state}")
            print(f"    Robot state: {final_robot_state}")
            
            # Check if states have learned Q-values
            if final_human_state in algorithm.q_e[hid]:
                print(f"    Human Q^e: {algorithm.q_e[hid][final_human_state]}")
            else:
                print(f"    Human Q^e: NOT LEARNED")
            
            if final_robot_state in algorithm.q_r[rid]:
                print(f"    Robot Q^r: {algorithm.q_r[rid][final_robot_state]}")
            else:
                print(f"    Robot Q^r: NOT LEARNED")
    
    success_rate = success_count / test_episodes
    print(f"\n=== RESULTS ===")
    print(f"Success rate: {success_rate:.1%} ({success_count}/{test_episodes})")
    
    if success_rate >= 0.9:
        print("✅ EXCELLENT! Training conditions work in testing!")
    elif success_rate >= 0.5:
        print("⚠️ PARTIAL SUCCESS - need to improve consistency")
    else:
        print("❌ FAILURE - training conditions don't work in testing")
    
    # Analyze learned Q-tables
    print(f"\n=== Q-TABLE ANALYSIS ===")
    print(f"Human Q^e states learned: {len(algorithm.q_e[human_agent_ids[0]])}")
    print(f"Robot Q^r states learned: {len(algorithm.q_r[robot_agent_ids[0]])}")
    
    # Show some high-value Q-entries
    print("\nTop 3 Human Q^e states:")
    human_q_items = list(algorithm.q_e[human_agent_ids[0]].items())
    human_q_items.sort(key=lambda x: np.max(x[1]), reverse=True)
    for i, (state, q_vals) in enumerate(human_q_items[:3]):
        print(f"  {i+1}. State {state}: max Q = {np.max(q_vals):.3f}")
    
    print("\nTop 3 Robot Q^r states:")
    robot_q_items = list(algorithm.q_r[robot_agent_ids[0]].items())
    robot_q_items.sort(key=lambda x: np.max(x[1]), reverse=True)
    for i, (state, q_vals) in enumerate(robot_q_items[:3]):
        print(f"  {i+1}. State {state}: max Q = {np.max(q_vals):.3f}")

if __name__ == "__main__":
    test_exact_training_conditions()