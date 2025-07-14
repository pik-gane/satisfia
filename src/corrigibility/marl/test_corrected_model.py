#!/usr/bin/env python3
"""
Test the corrected algorithm manually.
"""

import numpy as np
from corrected_iql_algorithm import CorrectedTwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def test_corrected_model(checkpoint_path, num_episodes=50):
    """Test the corrected model"""
    print(f"Testing corrected model: {checkpoint_path}")
    
    # Setup environment
    map_layout, map_metadata = get_simple_map()
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
    iql = CorrectedTwoPhaseTimescaleIQL(
        alpha_m=0.2, alpha_e=0.2, alpha_r=0.2, alpha_p=0.2,
        gamma_h=0.9, gamma_r=0.9, beta_r_0=3.0,
        G=G, mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    # Load model
    try:
        iql.load_models(checkpoint_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 0.0
    
    # Test episodes
    success_count = 0
    
    for episode in range(num_episodes):
        env.reset()
        goal = env.human_goals[human_agent_ids[0]]
        
        for step in range(150):  # Longer episodes
            actions = {}
            
            # Use some exploration for better generalization
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.2)
            
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
    
    success_rate = success_count / num_episodes
    print(f"\nSuccess rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    
    return success_rate


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    else:
        checkpoint_path = "checkpoints/simple_map_corrected_20250710_151722.pkl"
        episodes = 50
    
    success_rate = test_corrected_model(checkpoint_path, episodes)
    
    if success_rate >= 0.1:
        print(f"🎉 SUCCESS: {success_rate:.1%} >= 10%")
    else:
        print(f"⚠️ Needs improvement: {success_rate:.1%} < 10%")