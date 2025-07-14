#!/usr/bin/env python3
"""
Debug simple_map4 movement step by step.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def debug_movement():
    """Debug movement step by step"""
    print("=== Debug Simple Map 4 Movement ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial state:")
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    print(f"  Human: pos={env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
    
    # Test sequence: Robot goes (1,1) -> (2,1) -> (2,2) 
    print(f"\nStep 1: Robot forward (should go from (1,1) to (2,1))")
    actions = {"robot_0": 2, "human_0": 0}  # robot forward, human turn left
    env.step(actions)
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    print(f"\nStep 2: Robot turn left (should face east)")
    actions = {"robot_0": 0, "human_0": 0}  # robot turn left
    env.step(actions)
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    print(f"\nStep 3: Robot forward (should go from (2,1) to (2,2))")
    actions = {"robot_0": 2, "human_0": 0}  # robot forward
    env.step(actions)
    print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
    
    if env.agent_positions['robot_0'] == (2, 2):
        print(f"✅ SUCCESS! Robot reached key position!")
        
        print(f"\nStep 4: Robot pickup key")
        actions = {"robot_0": 3, "human_0": 0}  # robot pickup
        env.step(actions)
        print(f"  Robot: pos={env.agent_positions['robot_0']}")
        
        print(f"\nStep 5: Robot forward to door (should go from (2,2) to (2,3))")
        actions = {"robot_0": 2, "human_0": 0}  # robot forward
        env.step(actions)
        print(f"  Robot: pos={env.agent_positions['robot_0']}, dir={env.agent_dirs['robot_0']}")
        
        if env.agent_positions['robot_0'] == (2, 3):
            print(f"✅ Robot reached door position!")
            
            print(f"\nStep 6: Robot toggle door")
            actions = {"robot_0": 5, "human_0": 0}  # robot toggle
            env.step(actions)
            print(f"  Robot: pos={env.agent_positions['robot_0']}")
            
            print(f"\nStep 7: Human move forward (should go from (1,3) to (2,3))")
            actions = {"robot_0": 0, "human_0": 2}  # human forward
            env.step(actions)
            print(f"  Human: pos={env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
            
            print(f"\nStep 8: Human move forward (should go from (2,3) to (3,3) - GOAL!)")
            actions = {"robot_0": 0, "human_0": 2}  # human forward
            env.step(actions)
            print(f"  Human: pos={env.agent_positions['human_0']}, dir={env.agent_dirs['human_0']}")
            
            if env.agent_positions['human_0'] == (3, 3):
                print(f"🎉 COMPLETE SUCCESS! Human reached goal!")
                return True
            else:
                print(f"❌ Human failed to reach goal")
        else:
            print(f"❌ Robot failed to reach door")
    else:
        print(f"❌ Robot failed to reach key position")
    
    return False

if __name__ == "__main__":
    success = debug_movement()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}")