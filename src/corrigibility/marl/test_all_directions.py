#!/usr/bin/env python3
"""
Test movement in all directions from (2,1).
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_all_directions():
    """Test movement in all directions from (2,1)"""
    print("=== Test All Directions from (2,1) ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Test movement in each direction
    for target_dir in range(4):  # 0=North, 1=East, 2=South, 3=West
        print(f"\nTesting direction {target_dir}:")
        
        env.reset()
        
        # Get robot to (2,1)
        actions = {"robot_0": 2, "human_0": 0}  # robot forward to (2,1)
        env.step(actions)
        
        # Turn robot to face target direction
        current_dir = env.agent_dirs['robot_0']
        while current_dir != target_dir:
            actions = {"robot_0": 1, "human_0": 0}  # turn right
            env.step(actions)
            current_dir = env.agent_dirs['robot_0']
        
        print(f"  Robot at (2,1) facing direction {target_dir}")
        
        # Try to move forward
        initial_pos = env.agent_positions['robot_0']
        actions = {"robot_0": 2, "human_0": 0}  # forward
        env.step(actions)
        final_pos = env.agent_positions['robot_0']
        
        if final_pos != initial_pos:
            print(f"  ✅ SUCCESS: Moved from {initial_pos} to {final_pos}")
            
            # Check if this gets us to the key
            if final_pos == (2, 2):
                print(f"  🔑 REACHED KEY POSITION!")
                
                # Test pickup
                actions = {"robot_0": 3, "human_0": 0}  # pickup
                env.step(actions)
                print(f"  After pickup: pos={env.agent_positions['robot_0']}")
                
                # Try to continue to door
                for door_dir in range(4):
                    env_copy = CustomEnvironment(grid_layout=map_layout, grid_metadata=map_metadata)
                    env_copy.reset()
                    
                    # Recreate state: robot at (2,2) with key
                    actions = {"robot_0": 2, "human_0": 0}  # to (2,1)
                    env_copy.step(actions)
                    
                    # Turn to target_dir and move to (2,2)
                    current_dir = env_copy.agent_dirs['robot_0']
                    while current_dir != target_dir:
                        actions = {"robot_0": 1, "human_0": 0}
                        env_copy.step(actions)
                        current_dir = env_copy.agent_dirs['robot_0']
                    
                    actions = {"robot_0": 2, "human_0": 0}  # to (2,2)
                    env_copy.step(actions)
                    actions = {"robot_0": 3, "human_0": 0}  # pickup key
                    env_copy.step(actions)
                    
                    # Now try direction door_dir from (2,2)
                    current_dir = env_copy.agent_dirs['robot_0']
                    while current_dir != door_dir:
                        actions = {"robot_0": 1, "human_0": 0}
                        env_copy.step(actions)
                        current_dir = env_copy.agent_dirs['robot_0']
                    
                    initial_pos = env_copy.agent_positions['robot_0']
                    actions = {"robot_0": 2, "human_0": 0}  # forward
                    env_copy.step(actions)
                    final_pos = env_copy.agent_positions['robot_0']
                    
                    if final_pos == (2, 3):  # door position
                        print(f"  🚪 REACHED DOOR from (2,2) using direction {door_dir}!")
                        
                        # Test toggle
                        actions = {"robot_0": 5, "human_0": 0}  # toggle
                        env_copy.step(actions)
                        print(f"  Door toggled!")
                        
                        return True, target_dir, door_dir
        else:
            print(f"  ❌ BLOCKED: Cannot move from {initial_pos} in direction {target_dir}")
    
    return False, None, None

if __name__ == "__main__":
    success, key_dir, door_dir = test_all_directions()
    if success:
        print(f"\n🎉 SOLUTION FOUND!")
        print(f"  From (2,1): Use direction {key_dir} to reach key at (2,2)")
        print(f"  From (2,2): Use direction {door_dir} to reach door at (2,3)")
    else:
        print(f"\n❌ NO SOLUTION FOUND")