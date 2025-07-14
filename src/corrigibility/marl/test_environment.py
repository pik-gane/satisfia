#!/usr/bin/env python3
"""
Debug environment state and actions.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def debug_environment():
    """Debug environment state"""
    print("=== Debug Environment ===")
    
    map_layout, map_metadata = get_map()
    print(f"Map layout:")
    for i, row in enumerate(map_layout):
        print(f"  Row {i}: {row}")
    
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"\nInitial state:")
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    print(f"  Human: pos={env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
    
    # Check what actions are available
    print(f"\nAvailable actions:")
    print(f"  Robot action space: {env.action_space('robot_0')}")
    print(f"  Human action space: {env.action_space('human_0')}")
    
    # Test ALL robot actions from initial position
    print(f"\nTesting all robot actions from (1,1):")
    for action in range(6):  # 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle
        env.reset()  # Reset to initial state
        action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle"}
        print(f"  Action {action} ({action_names[action]}):")
        
        initial_pos = env.agent_positions['robot_0']
        initial_dir = env.agent_dirs['robot_0']
        
        actions = {"robot_0": action, "human_0": 0}
        obs, rewards, terms, truncs, info = env.step(actions)
        
        final_pos = env.agent_positions['robot_0']
        final_dir = env.agent_dirs['robot_0']
        
        print(f"    Before: pos={initial_pos}, dir={initial_dir}")
        print(f"    After:  pos={final_pos}, dir={final_dir}")
        print(f"    Rewards: {rewards}")
        print(f"    Done: {terms}, {truncs}")
        
        if action == 2 and final_pos != initial_pos:  # forward movement worked
            print(f"    ✅ Movement successful!")
        elif action in [0, 1] and final_dir != initial_dir:  # turning worked
            print(f"    ✅ Turning successful!")
        elif action in [3, 4, 5]:  # special actions
            print(f"    ⚠️  Special action (may or may not work)")
        else:
            print(f"    ❌ Action had no effect")

if __name__ == "__main__":
    debug_environment()