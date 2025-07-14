#!/usr/bin/env python3
"""
Test movement from specific position (2,1).
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_from_position():
    """Test movement from (2,1)"""
    print("=== Test Movement from (2,1) ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    # Get robot to (2,1)
    print("Step 1: Move robot from (1,1) to (2,1)")
    actions = {"robot_0": 2, "human_0": 0}  # robot forward
    env.step(actions)
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    # Turn robot to face east (dir=1)
    print("Step 2: Turn robot to face east")
    actions = {"robot_0": 0, "human_0": 0}  # robot turn left
    env.step(actions)
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    # Now test all actions from (2,1) facing east
    print(f"\nTesting all actions from (2,1) facing east:")
    initial_pos = env.agent_positions['robot_0']
    initial_dir = env.agent_dirs['robot_0']
    
    for action in range(7):  # Now testing 0-6
        # Save current state
        saved_pos = env.agent_positions['robot_0']
        saved_dir = env.agent_dirs['robot_0']
        
        action_names = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "unknown"}
        print(f"  Action {action} ({action_names.get(action, 'unknown')}):")
        
        actions = {"robot_0": action, "human_0": 0}
        obs, rewards, terms, truncs, info = env.step(actions)
        
        final_pos = env.agent_positions['robot_0']
        final_dir = env.agent_dirs['robot_0']
        
        print(f"    Before: pos={saved_pos}, dir={saved_dir}")
        print(f"    After:  pos={final_pos}, dir={final_dir}")
        
        if action == 2:  # forward
            if final_pos == (2, 2):
                print(f"    🎉 SUCCESS! Robot moved to key position!")
                
                # Test pickup
                print(f"    Testing pickup at key position...")
                actions = {"robot_0": 3, "human_0": 0}  # pickup
                env.step(actions)
                print(f"    After pickup: pos={env.agent_positions['robot_0']}")
                
                # Test moving to door
                print(f"    Testing move to door...")
                actions = {"robot_0": 2, "human_0": 0}  # forward to door
                env.step(actions)
                door_pos = env.agent_positions['robot_0']
                print(f"    After move: pos={door_pos}")
                
                if door_pos == (2, 3):
                    print(f"    🎉 Robot reached door! Testing toggle...")
                    actions = {"robot_0": 5, "human_0": 0}  # toggle
                    env.step(actions)
                    print(f"    After toggle: pos={env.agent_positions['robot_0']}")
                    
                    # Test human movement through door
                    print(f"    Testing human movement through opened door...")
                    human_initial = env.agent_positions['human_0']
                    actions = {"robot_0": 0, "human_0": 2}  # human forward
                    env.step(actions)
                    human_step1 = env.agent_positions['human_0']
                    print(f"    Human step 1: {human_initial} -> {human_step1}")
                    
                    actions = {"robot_0": 0, "human_0": 2}  # human forward
                    env.step(actions)
                    human_final = env.agent_positions['human_0']
                    print(f"    Human step 2: {human_step1} -> {human_final}")
                    
                    if human_final == (3, 3):
                        print(f"    🎉🎉 COMPLETE SUCCESS! Human reached goal!")
                        return True
                
                return  # Exit after testing forward movement
            else:
                print(f"    ❌ Robot couldn't move forward to (2,2)")
        
        # Reset robot to (2,1) facing east for next test
        env.reset()
        actions = {"robot_0": 2, "human_0": 0}  # move to (2,1)
        env.step(actions)
        actions = {"robot_0": 0, "human_0": 0}  # turn to face east
        env.step(actions)
    
    return False

if __name__ == "__main__":
    success = test_from_position()
    print(f"\n{'🎉 COMPLETE SUCCESS!' if success else '❌ FAILED'}")