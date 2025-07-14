#!/usr/bin/env python3
"""
Test if robot can pickup key from adjacent positions.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_adjacent_pickup():
    """Test pickup from adjacent positions"""
    print("=== Test Adjacent Pickup ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Key is at (2,2), test pickup from adjacent positions robot can reach
    reachable_adjacent = [
        ((2, 1), "South of key"),
        ((1, 2), "North of key"), 
        ((3, 1), "South-west diagonal"),
    ]
    
    for robot_pos, description in reachable_adjacent:
        print(f"\nTesting pickup from {robot_pos} ({description}):")
        
        env.reset()
        
        # Get robot to target position
        if robot_pos == (2, 1):
            env.step({"robot_0": 2, "human_0": 0})  # forward to (2,1)
        elif robot_pos == (1, 2):
            env.step({"robot_0": 0, "human_0": 0})  # turn left to dir 1
            env.step({"robot_0": 2, "human_0": 0})  # forward to (1,2)
        elif robot_pos == (3, 1):
            env.step({"robot_0": 2, "human_0": 0})  # to (2,1)
            env.step({"robot_0": 2, "human_0": 0})  # to (3,1)
        
        current_pos = env.agent_positions['robot_0']
        print(f"  Robot at: {current_pos}")
        
        if current_pos == robot_pos:
            print(f"  ✅ Robot in position, testing pickup...")
            
            # Test pickup - might need to face the key
            for facing_dir in range(4):
                env.reset()
                
                # Get to position again
                if robot_pos == (2, 1):
                    env.step({"robot_0": 2, "human_0": 0})  # forward to (2,1)
                elif robot_pos == (1, 2):
                    env.step({"robot_0": 0, "human_0": 0})  # turn left to dir 1
                    env.step({"robot_0": 2, "human_0": 0})  # forward to (1,2)
                elif robot_pos == (3, 1):
                    env.step({"robot_0": 2, "human_0": 0})  # to (2,1)
                    env.step({"robot_0": 2, "human_0": 0})  # to (3,1)
                
                # Turn to face direction
                current_dir = env.agent_dirs['robot_0']
                while current_dir != facing_dir:
                    env.step({"robot_0": 1, "human_0": 0})  # turn right
                    current_dir = env.agent_dirs['robot_0']
                
                print(f"    Facing direction {facing_dir}, attempting pickup...")
                env.step({"robot_0": 3, "human_0": 0})  # pickup
                
                # Test if pickup worked by trying toggle
                print(f"    Testing if key was picked up by attempting toggle...")
                env.step({"robot_0": 5, "human_0": 0})  # toggle
                
                # Test if human can move (indicating door opened)
                human_initial = env.agent_positions['human_0']
                env.step({"robot_0": 0, "human_0": 2})  # human forward
                human_after = env.agent_positions['human_0']
                
                if human_after != human_initial:
                    print(f"    🎉 SUCCESS! Key picked up from {robot_pos} facing {facing_dir}!")
                    print(f"    Human moved from {human_initial} to {human_after}")
                    
                    # Try to reach goal
                    env.step({"robot_0": 0, "human_0": 2})  # human forward again
                    human_final = env.agent_positions['human_0']
                    
                    if human_final == (3, 3):
                        print(f"    🎉🎉 COMPLETE SUCCESS! Human reached goal!")
                        return True, robot_pos, facing_dir
                    else:
                        print(f"    Human reached {human_final}, trying one more step...")
                        env.step({"robot_0": 0, "human_0": 2})  # human forward
                        human_final2 = env.agent_positions['human_0']
                        if human_final2 == (3, 3):
                            print(f"    🎉🎉 COMPLETE SUCCESS! Human reached goal!")
                            return True, robot_pos, facing_dir
                else:
                    print(f"    ❌ No effect from direction {facing_dir}")
        else:
            print(f"  ❌ Robot couldn't reach {robot_pos}")
    
    return False, None, None

if __name__ == "__main__":
    success, pos, direction = test_adjacent_pickup()
    if success:
        print(f"\n🎉 SOLUTION FOUND!")
        print(f"Robot picks up key from {pos} facing direction {direction}")
    else:
        print(f"\n❌ NO SOLUTION FOUND")