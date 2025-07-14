#!/usr/bin/env python3
"""
Understand the exact movement mechanics in simple_map4.
"""

import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def test_basic_movements():
    """Test basic movement actions to understand the action space"""
    print("=== Understanding Movement Mechanics ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
    
    # Test each robot action
    print(f"\nTesting Robot Actions:")
    print(f"Action space: {list(range(6))} = [turn_left, turn_right, forward, pickup, drop, toggle]")
    
    for action in range(6):
        env.reset()
        initial_pos = env.agent_positions['robot_0']
        
        actions = {"robot_0": action, "human_0": 0}  # Human stays still
        env.step(actions)
        
        final_pos = env.agent_positions['robot_0']
        print(f"  Action {action}: {initial_pos} -> {final_pos}")
    
    # Test human actions  
    print(f"\nTesting Human Actions:")
    print(f"Action space: {list(range(3))} = [turn_left, turn_right, forward]")
    
    for action in range(3):
        env.reset()
        initial_pos = env.agent_positions['human_0']
        
        actions = {"robot_0": 0, "human_0": action}  # Robot stays still
        env.step(actions)
        
        final_pos = env.agent_positions['human_0']
        print(f"  Action {action}: {initial_pos} -> {final_pos}")

def test_path_to_key():
    """Test the exact path robot needs to take to reach the key"""
    print(f"\n=== Testing Path to Key at (2,2) ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Robot starts at: {env.agent_positions['robot_0']}")
    print(f"Key is at: (2, 2)")
    print(f"Target path: (1,1) -> (1,2) -> (2,2)")
    
    # Step 1: Robot turn right to face east
    actions = {"robot_0": 1, "human_0": 0}  # turn_right
    env.step(actions)
    print(f"Step 1 - Turn right: {env.agent_positions['robot_0']}")
    
    # Step 2: Robot move forward to (1,2)
    actions = {"robot_0": 2, "human_0": 0}  # forward
    env.step(actions)
    print(f"Step 2 - Forward: {env.agent_positions['robot_0']} (expect (1,2))")
    
    # Step 3: Robot turn right to face south  
    actions = {"robot_0": 1, "human_0": 0}  # turn_right
    env.step(actions)
    print(f"Step 3 - Turn right: {env.agent_positions['robot_0']}")
    
    # Step 4: Robot move forward to (2,2)
    actions = {"robot_0": 2, "human_0": 0}  # forward  
    env.step(actions)
    print(f"Step 4 - Forward: {env.agent_positions['robot_0']} (expect (2,2) - KEY POSITION)")
    
    # Step 5: Robot pickup key
    actions = {"robot_0": 3, "human_0": 0}  # pickup
    env.step(actions)
    print(f"Step 5 - Pickup: {env.agent_positions['robot_0']} (should have key)")

def test_path_to_door():
    """Test robot moving from key to door"""
    print(f"\n=== Testing Path from Key to Door ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Manually set robot at key position (we can't directly, but can navigate there)
    env.reset()
    
    # Quick navigation to key position (2,2)
    actions = {"robot_0": 1, "human_0": 0}  # turn right
    env.step(actions)
    actions = {"robot_0": 2, "human_0": 0}  # forward to (1,2)
    env.step(actions)
    actions = {"robot_0": 1, "human_0": 0}  # turn right to face south
    env.step(actions)
    actions = {"robot_0": 2, "human_0": 0}  # forward to (2,2)
    env.step(actions)
    actions = {"robot_0": 3, "human_0": 0}  # pickup key
    env.step(actions)
    
    print(f"Robot at key position: {env.agent_positions['robot_0']}")
    print(f"Door is at: (2, 3)")
    print(f"Target path: (2,2) -> (2,3)")
    
    # Step 1: Robot turn right to face east
    actions = {"robot_0": 1, "human_0": 0}  # turn_right
    env.step(actions)
    print(f"Turn right: {env.agent_positions['robot_0']}")
    
    # Step 2: Robot move forward to (2,3) - door position
    actions = {"robot_0": 2, "human_0": 0}  # forward
    env.step(actions)
    print(f"Forward to door: {env.agent_positions['robot_0']} (expect (2,3) - DOOR POSITION)")
    
    # Step 3: Robot toggle door
    actions = {"robot_0": 5, "human_0": 0}  # toggle
    env.step(actions)
    print(f"Toggle door: {env.agent_positions['robot_0']} (should have opened door)")

def test_human_path():
    """Test human moving to goal after door is opened"""
    print(f"\n=== Testing Human Path to Goal ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Human starts at: {env.agent_positions['human_0']}")
    print(f"Goal is at: {env.human_goals['human_0']}")
    print(f"Path should be: (1,3) -> (2,3) -> (3,3)")
    
    # First, let's get robot to open the door
    print("First, robot opens door...")
    # Robot to key
    actions = {"robot_0": 1, "human_0": 0}  # turn right
    env.step(actions)
    actions = {"robot_0": 2, "human_0": 0}  # forward to (1,2)
    env.step(actions)
    actions = {"robot_0": 1, "human_0": 0}  # turn right
    env.step(actions)
    actions = {"robot_0": 2, "human_0": 0}  # forward to (2,2)
    env.step(actions)
    actions = {"robot_0": 3, "human_0": 0}  # pickup key
    env.step(actions)
    # Robot to door
    actions = {"robot_0": 1, "human_0": 0}  # turn right
    env.step(actions)
    actions = {"robot_0": 2, "human_0": 0}  # forward to (2,3)
    env.step(actions)
    actions = {"robot_0": 5, "human_0": 0}  # toggle door
    env.step(actions)
    
    print(f"Door opened. Now testing human movement:")
    print(f"Human at: {env.agent_positions['human_0']}")
    
    # Human step 1: forward to (2,3) - through opened door
    actions = {"robot_0": 0, "human_0": 2}  # human forward
    env.step(actions)
    print(f"Human forward: {env.agent_positions['human_0']} (expect (2,3))")
    
    # Human step 2: forward to (3,3) - goal
    actions = {"robot_0": 0, "human_0": 2}  # human forward
    env.step(actions)
    print(f"Human forward: {env.agent_positions['human_0']} (expect (3,3) - GOAL!)")
    
    if tuple(env.agent_positions['human_0']) == tuple(env.human_goals['human_0']):
        print("🎉 SUCCESS! Human reached goal!")
        return True
    else:
        print("❌ Human did not reach goal")
        return False

if __name__ == "__main__":
    test_basic_movements()
    test_path_to_key()
    test_path_to_door()
    success = test_human_path()
    
    if success:
        print("\n✅ Movement mechanics understood! Ready to create successful algorithm.")
    else:
        print("\n❌ Still need to debug movement mechanics.")