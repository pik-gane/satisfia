#!/usr/bin/env python3
"""
Test if robot can toggle door from adjacent positions.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_adjacent_toggle():
    """Test toggle from adjacent positions"""
    print("=== Test Adjacent Toggle ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Door is at (2,3), test toggle from adjacent positions
    adjacent_positions = [
        ((1, 3), "North of door"),
        ((3, 3), "South of door"), 
        ((2, 2), "West of door"),
        ((2, 4), "East of door")
    ]
    
    for target_pos, description in adjacent_positions:
        print(f"\nTesting toggle from {target_pos} ({description}):")
        
        # Check if robot can reach this position
        env.reset()
        reached = False
        
        # Try to reach target_pos
        if target_pos == (1, 3):
            # Try direction 1 then direction 2
            env.step({"robot_0": 1, "human_0": 0})  # turn to dir 1
            env.step({"robot_0": 2, "human_0": 0})  # forward to (1,2)
            if env.agent_positions['robot_0'] == (1, 2):
                env.step({"robot_0": 1, "human_0": 0})  # turn to dir 2  
                env.step({"robot_0": 2, "human_0": 0})  # forward to (1,3)
                if env.agent_positions['robot_0'] == (1, 3):
                    reached = True
        
        elif target_pos == (2, 2):
            # We know robot can't reach (2,2) directly, but let's test anyway
            pass
            
        elif target_pos == (3, 3):
            # Try going (1,1) -> (2,1) -> (3,1) -> (3,2) -> (3,3)
            env.step({"robot_0": 2, "human_0": 0})  # to (2,1)
            env.step({"robot_0": 2, "human_0": 0})  # to (3,1)
            if env.agent_positions['robot_0'] == (3, 1):
                # Turn to face east and try to reach (3,2)
                env.step({"robot_0": 0, "human_0": 0})  # turn left to face east
                env.step({"robot_0": 2, "human_0": 0})  # forward to (3,2)
                if env.agent_positions['robot_0'] == (3, 2):
                    env.step({"robot_0": 2, "human_0": 0})  # forward to (3,3)
                    if env.agent_positions['robot_0'] == (3, 3):
                        reached = True
        
        elif target_pos == (2, 4):
            # Probably can't reach due to walls
            pass
        
        current_pos = env.agent_positions['robot_0']
        print(f"  Robot reached: {current_pos} (target was {target_pos})")
        
        if reached or current_pos == target_pos:
            print(f"  ✅ Robot at {current_pos}, testing toggle...")
            
            # Test toggle action
            env.step({"robot_0": 5, "human_0": 0})  # toggle
            print(f"  Toggle attempted from {current_pos}")
            
            # Test if human can now move
            human_initial = env.agent_positions['human_0']
            print(f"  Human initial: {human_initial}")
            
            env.step({"robot_0": 0, "human_0": 2})  # human forward
            human_after = env.agent_positions['human_0']
            print(f"  Human after step: {human_after}")
            
            if human_after != human_initial:
                print(f"  🎉 SUCCESS! Door opened, human can move!")
                
                # Try to reach goal
                env.step({"robot_0": 0, "human_0": 2})  # human forward again
                human_final = env.agent_positions['human_0']
                print(f"  Human final: {human_final}")
                
                if human_final == (3, 3):
                    print(f"  🎉🎉 COMPLETE SUCCESS! Human reached goal!")
                    return True
            else:
                print(f"  ❌ Door not opened or human still can't move")
        else:
            print(f"  ❌ Robot couldn't reach {target_pos}")
    
    return False

if __name__ == "__main__":
    success = test_adjacent_toggle()
    if success:
        print(f"\n🎉 SOLUTION FOUND!")
    else:
        print(f"\n❌ NO SOLUTION FOUND")