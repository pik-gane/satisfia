#!/usr/bin/env python3
"""
Fix direction understanding and create working simple_map4 solution.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def understand_direction_system():
    """Properly understand the direction system"""
    print("=== Understanding Direction System ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Robot starts at {env.agent_positions['robot_0']} facing direction {env.agent_dirs['robot_0']}")
    
    # Test all directions by turning
    print(f"\nTesting direction changes:")
    for turn in range(4):
        pos = env.agent_positions['robot_0']
        dir_before = env.agent_dirs['robot_0']
        
        # Turn right
        actions = {"robot_0": 1, "human_0": 0}
        env.step(actions)
        
        dir_after = env.agent_dirs['robot_0']
        print(f"Turn {turn + 1}: dir {dir_before} -> {dir_after}")
    
    # Reset and test movement in each direction
    print(f"\nTesting movement in each direction:")
    env.reset()
    
    for direction in range(4):
        # Turn to specific direction
        while env.agent_dirs['robot_0'] != direction:
            actions = {"robot_0": 1, "human_0": 0}  # turn right
            env.step(actions)
        
        # Test forward movement
        pos_before = env.agent_positions['robot_0']
        actions = {"robot_0": 2, "human_0": 0}  # forward
        env.step(actions)
        pos_after = env.agent_positions['robot_0']
        
        movement = (pos_after[0] - pos_before[0], pos_after[1] - pos_before[1])
        print(f"Direction {direction}: {pos_before} -> {pos_after} (movement: {movement})")
        
        # Reset position
        env.reset()

def create_working_solution():
    """Create working solution based on correct direction understanding"""
    print(f"\n=== Creating Working Solution ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    successes = 0
    total_tests = 10
    
    for test in range(total_tests):
        env.reset()
        print(f"\nTest {test + 1}:")
        print(f"  Start: Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}, Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}")
        
        success = False
        
        # Strategy: Try multiple approaches to reach the goal
        
        # Approach 1: Robot moves to get key, then opens door
        robot_pos = env.agent_positions['robot_0'] 
        robot_dir = env.agent_dirs['robot_0']
        
        # Navigate robot to (2,2) for key
        if robot_pos == (1, 1):
            # Need to get to (2,2) - key position
            
            # First move south to (2,1) if facing south
            if robot_dir == 2:  # facing south
                actions = {"robot_0": 2, "human_0": 0}  # forward
                env.step(actions)
                print(f"    Robot forward: {env.agent_positions['robot_0']}")
                
                # Now turn to face east (need to figure out correct turn)
                # Try turn left first
                actions = {"robot_0": 0, "human_0": 0}  # turn left
                env.step(actions)
                print(f"    Robot turn left: dir={env.agent_dirs['robot_0']}")
                
                # Try forward to reach (2,2)
                actions = {"robot_0": 2, "human_0": 0}  # forward
                env.step(actions)
                print(f"    Robot forward: {env.agent_positions['robot_0']}")
                
                if env.agent_positions['robot_0'] == (2, 2):
                    print(f"    ✅ Robot reached key position!")
                    
                    # Pickup key
                    actions = {"robot_0": 3, "human_0": 0}  # pickup
                    env.step(actions)
                    print(f"    Robot pickup key")
                    
                    # Now navigate to door at (2,3)
                    # Turn to face east/south toward door
                    for turn_attempt in range(4):
                        actions = {"robot_0": 1, "human_0": 0}  # turn right
                        env.step(actions)
                        
                        # Try forward
                        pos_before = env.agent_positions['robot_0']
                        actions = {"robot_0": 2, "human_0": 0}  # forward
                        env.step(actions)
                        pos_after = env.agent_positions['robot_0']
                        
                        if pos_after == (2, 3):
                            print(f"    ✅ Robot reached door position!")
                            
                            # Toggle door
                            actions = {"robot_0": 5, "human_0": 0}  # toggle
                            env.step(actions)
                            print(f"    Robot toggle door")
                            
                            # Now try human movement
                            human_pos = env.agent_positions['human_0']
                            
                            # Human move forward twice to reach (3,3)
                            for human_step in range(3):
                                actions = {"robot_0": 0, "human_0": 2}  # human forward
                                env.step(actions)
                                human_new_pos = env.agent_positions['human_0']
                                print(f"    Human step {human_step + 1}: {human_pos} -> {human_new_pos}")
                                human_pos = human_new_pos
                                
                                if human_pos == (3, 3):
                                    print(f"    🎉 SUCCESS! Human reached goal!")
                                    success = True
                                    successes += 1
                                    break
                            break
                        else:
                            # Reset robot position and try different direction
                            env.reset()
                            # Re-navigate to key first
                            actions = {"robot_0": 2, "human_0": 0}  # forward to (2,1)
                            env.step(actions)
                            actions = {"robot_0": 0, "human_0": 0}  # turn left
                            env.step(actions)
                            actions = {"robot_0": 2, "human_0": 0}  # forward to (2,2)
                            env.step(actions)
                            actions = {"robot_0": 3, "human_0": 0}  # pickup key
                            env.step(actions)
        
        if not success:
            print(f"    ❌ Failed")
    
    success_rate = successes / total_tests
    print(f"\n=== RESULTS ===")
    print(f"Success rate: {success_rate:.1%} ({successes}/{total_tests})")
    
    return success_rate

if __name__ == "__main__":
    understand_direction_system()
    success_rate = create_working_solution()
    
    if success_rate >= 0.5:
        print(f"\n✅ Found working approach with {success_rate:.1%} success!")
    else:
        print(f"\n❌ Need to debug further - only {success_rate:.1%} success")