#!/usr/bin/env python3
"""
Pygame visualization for trained IQL models.
"""

import numpy as np
import sys
from enhanced_iql_algorithm import EnhancedTwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment


def visualize_with_pygame(checkpoint_path, map_name, episodes=3):
    """Visualize with pygame rendering"""
    print(f"\n=== Pygame Visualization: {map_name} ===")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Map functions
    map_functions = {
        "simple_map": get_simple_map,
        "simple_map2": get_simple_map2, 
        "simple_map3": get_simple_map3,
        "simple_map4": get_simple_map4,
    }
    
    get_map_func = map_functions[map_name]
    map_layout, map_metadata = get_map_func()
    
    # Create environment with pygame rendering
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode="human"  # Enable pygame rendering
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
    
    # Create and load model
    iql = EnhancedTwoPhaseTimescaleIQL(
        alpha_m=0.3, alpha_e=0.3, alpha_r=0.3, alpha_p=0.3,
        gamma_h=0.95, gamma_r=0.95, beta_r_0=5.0,
        G=G, mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,
        env=env
    )
    
    try:
        iql.load_models(checkpoint_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    action_names = {
        0: "turn_left", 1: "turn_right", 2: "forward", 
        3: "pickup", 4: "drop", 5: "toggle"
    }
    
    print(f"\\nStarting visualization with pygame...")
    print(f"Controls: Close window to stop, or wait for episodes to complete")
    
    for episode in range(episodes):
        print(f"\\n--- Episode {episode + 1}/{episodes} ---")
        
        env.reset()
        env.render()  # Show initial state
        
        goal = env.human_goals[human_agent_ids[0]]
        print(f"Goal position: {goal}")
        print(f"Robot starts at: {env.agent_positions['robot_0']}")
        print(f"Human starts at: {env.agent_positions['human_0']}")
        
        episode_success = False
        
        for step in range(100):  # Max 100 steps per episode
            actions = {}
            
            # Get actions from trained model
            for hid in human_agent_ids:
                state_h = iql.get_human_state(env, hid, iql.state_to_tuple(goal))
                actions[hid] = iql.sample_human_action_phase1(hid, state_h, epsilon=0.1)
            
            for rid in robot_agent_ids:
                state_r = iql.get_full_state(env, rid)
                actions[rid] = iql.sample_robot_action_phase2(rid, state_r, env, epsilon=0.0)
            
            # Convert to readable format
            readable_actions = {
                agent: action_names.get(action, str(action)) 
                for agent, action in actions.items()
            }
            
            print(f"  Step {step + 1}: {readable_actions}")
            
            # Execute actions
            env.step(actions)
            env.render()  # Update pygame display
            
            # Check for goal
            human_pos = env.agent_positions[human_agent_ids[0]]
            if tuple(human_pos) == tuple(goal):
                print(f"  🎉 GOAL REACHED at step {step + 1}!")
                episode_success = True
                break
            
            # Small delay to make it watchable
            import time
            time.sleep(0.8)  # 800ms delay between steps
        
        if not episode_success:
            print(f"  ❌ Episode timed out after {step + 1} steps")
        
        print(f"Episode {episode + 1} completed. Continuing to next episode...")
        import time
        time.sleep(2)  # 2 second pause between episodes
    
    print(f"\\nVisualization complete for {map_name}")
    env.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pygame_visualizer.py <checkpoint_path> <map_name>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    map_name = sys.argv[2]
    
    try:
        visualize_with_pygame(checkpoint_path, map_name, episodes=2)
    except KeyboardInterrupt:
        print("\\nVisualization interrupted by user")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()