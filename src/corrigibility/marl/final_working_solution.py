#!/usr/bin/env python3
"""
Final working solution for simple_map4 with proper understanding.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def create_final_solution():
    """Create the final working solution"""
    print("=== Final Working Solution ===")
    
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
        print(f"  Initial: Robot={env.agent_positions['robot_0']} dir={env.agent_dirs['robot_0']}, Human={env.agent_positions['human_0']} dir={env.agent_dirs['human_0']}")
        
        success = False
        
        # Step 1: Robot forward to (2,1)
        actions = {"robot_0": 2, "human_0": 0}
        env.step(actions)
        print(f"  Step 1 - Robot to (2,1): Robot={env.agent_positions['robot_0']}")
        
        # Step 2: Robot pickup key (robot at (2,1) facing south can pickup key at (2,2))
        actions = {"robot_0": 3, "human_0": 0}
        env.step(actions)
        print(f"  Step 2 - Robot pickup key: Robot={env.agent_positions['robot_0']}")
        
        # Step 3: Robot toggle door
        actions = {"robot_0": 5, "human_0": 0}
        env.step(actions)
        print(f"  Step 3 - Robot toggle door: Robot={env.agent_positions['robot_0']}")
        
        # Step 4: Human turn to face south (dir=2) to move toward goal
        human_dir = env.agent_dirs['human_0']
        print(f"  Step 4 - Human current dir: {human_dir}, need dir=2 (south)")
        
        while env.agent_dirs['human_0'] != 2:  # Turn until facing south
            actions = {"robot_0": 0, "human_0": 1}  # human turn right
            env.step(actions)
            print(f"    Human turn: dir={env.agent_dirs['human_0']}")
        
        # Step 5: Human forward to (2,3) 
        actions = {"robot_0": 0, "human_0": 2}
        env.step(actions)
        human_pos = env.agent_positions['human_0']
        print(f"  Step 5 - Human forward: {human_pos}")
        
        # Step 6: Human forward to (3,3) - GOAL!
        actions = {"robot_0": 0, "human_0": 2}
        env.step(actions)
        human_final = env.agent_positions['human_0']
        print(f"  Step 6 - Human forward: {human_final}")
        
        if human_final == (3, 3):
            print(f"  🎉 SUCCESS! Human reached goal!")
            success = True
            successes += 1
        else:
            print(f"  ❌ Failed - Human at {human_final}")
    
    success_rate = successes / total_tests
    print(f"\n=== FINAL RESULTS ===")
    print(f"Success rate: {success_rate:.1%} ({successes}/{total_tests})")
    
    if success_rate >= 0.8:
        print(f"✅ EXCELLENT! Found working solution!")
        print(f"\\nSolution sequence:")
        print(f"1. Robot forward to (2,1)")
        print(f"2. Robot pickup key from (2,1) facing south")
        print(f"3. Robot toggle door")
        print(f"4. Human turn to face south (dir=2)")
        print(f"5. Human forward to (2,3)")
        print(f"6. Human forward to (3,3) - GOAL!")
        return True
    else:
        print(f"❌ Solution not reliable enough")
        return False

if __name__ == "__main__":
    success = create_final_solution()
    
    if success:
        print(f"\\n🎉 READY TO UPDATE SIMPLE_MAP4_SOLVER!")
    else:
        print(f"\\n❌ Need more work")