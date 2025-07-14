#!/usr/bin/env python3
"""
Test complete solution with proper key pickup and door opening.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_complete_solution():
    """Test the complete solution"""
    print("=== Complete Solution Test ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial:")
    print(f"  Robot: {env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    print(f"  Human: {env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
    
    # Step 1: Robot goes to (2,1)
    print(f"\nStep 1: Robot forward to (2,1)")
    env.step({"robot_0": 2, "human_0": 0})
    print(f"  Robot: {env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    # Step 2: Robot picks up key from (2,1) facing south (dir=2)
    print(f"\nStep 2: Robot pickup key (facing south)")
    env.step({"robot_0": 3, "human_0": 0})
    print(f"  Robot: {env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    # Step 3: Robot toggle door
    print(f"\nStep 3: Robot toggle door")
    env.step({"robot_0": 5, "human_0": 0})
    print(f"  Robot: {env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    # Now test human movement carefully
    print(f"\nTesting human movement after door opened:")
    for step in range(10):  # Try up to 10 steps
        human_before = env.agent_positions['human_0']
        human_dir_before = env.agent_dirs['human_0']
        
        print(f"  Step {step+4}: Human forward")
        env.step({"robot_0": 0, "human_0": 2})  # human forward
        
        human_after = env.agent_positions['human_0']
        human_dir_after = env.agent_dirs['human_0']
        
        print(f"    Before: pos={human_before}, dir={human_dir_before}")
        print(f"    After:  pos={human_after}, dir={human_dir_after}")
        
        if human_after == (3, 3):
            print(f"    🎉 SUCCESS! Human reached goal in {step+4} total steps!")
            return True
        
        if human_after == human_before:
            print(f"    Human stopped moving at {human_after}")
            break
    
    print(f"\n❌ Human did not reach goal")
    print(f"Final positions:")
    print(f"  Robot: {env.agent_positions['robot_0']}")
    print(f"  Human: {env.agent_positions['human_0']}")
    
    # Try different human actions
    print(f"\nTrying different human actions:")
    for action in range(4):  # Try different actions
        env.reset()
        
        # Reproduce robot actions
        env.step({"robot_0": 2, "human_0": 0})  # robot to (2,1)
        env.step({"robot_0": 3, "human_0": 0})  # robot pickup
        env.step({"robot_0": 5, "human_0": 0})  # robot toggle
        
        human_initial = env.agent_positions['human_0']
        action_names = {0: "left", 1: "right", 2: "forward", 3: "backward"}
        print(f"  Action {action} ({action_names.get(action, 'unknown')}):")
        
        env.step({"robot_0": 0, "human_0": action})
        human_result = env.agent_positions['human_0']
        
        print(f"    {human_initial} -> {human_result}")
        
        if human_result != human_initial:
            print(f"    ✅ Human can move with action {action}!")
            
            # Continue with this action
            for step in range(5):
                env.step({"robot_0": 0, "human_0": action})
                human_pos = env.agent_positions['human_0']
                print(f"      Step {step+2}: {human_pos}")
                
                if human_pos == (3, 3):
                    print(f"    🎉 SUCCESS with action {action}!")
                    return True
    
    return False

if __name__ == "__main__":
    success = test_complete_solution()
    if success:
        print(f"\n🎉 COMPLETE SOLUTION FOUND!")
    else:
        print(f"\n❌ NO COMPLETE SOLUTION")