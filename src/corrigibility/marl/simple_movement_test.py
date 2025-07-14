#!/usr/bin/env python3
"""
Simple test to check if simple_map4 actually allows any movement at all.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def simple_movement_test():
    """Simple test for movement"""
    print("=== Simple Movement Test ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial: Robot={env.agent_positions['robot_0']}, Human={env.agent_positions['human_0']}")
    
    # Test robot forward (action 2)
    print(f"\nTesting robot forward movement...")
    initial_robot = env.agent_positions['robot_0']
    initial_human = env.agent_positions['human_0']
    
    actions = {"robot_0": 2, "human_0": 0}  # Robot forward, human turn left
    env.step(actions)
    
    final_robot = env.agent_positions['robot_0']
    final_human = env.agent_positions['human_0']
    
    robot_moved = initial_robot != final_robot
    human_moved = initial_human != final_human
    
    print(f"Robot: {initial_robot} -> {final_robot} (moved: {robot_moved})")
    print(f"Human: {initial_human} -> {final_human} (moved: {human_moved})")
    
    # Try more steps to see if any movement happens
    print(f"\nTrying 5 more steps with forward actions...")
    for i in range(5):
        prev_robot = env.agent_positions['robot_0']
        prev_human = env.agent_positions['human_0']
        
        actions = {"robot_0": 2, "human_0": 2}  # Both forward
        env.step(actions)
        
        curr_robot = env.agent_positions['robot_0']
        curr_human = env.agent_positions['human_0']
        
        robot_moved = prev_robot != curr_robot
        human_moved = prev_human != curr_human
        
        print(f"Step {i+1}: Robot {prev_robot}->{curr_robot} ({robot_moved}), Human {prev_human}->{curr_human} ({human_moved})")
        
        if robot_moved or human_moved:
            print("✅ MOVEMENT DETECTED!")
            break
    else:
        print("❌ No movement detected in any step")
        
    # Check if this is maybe a rendering/orientation issue
    print(f"\nChecking if environment has orientation/direction attributes...")
    if hasattr(env, 'agent_states'):
        print(f"Agent states: {env.agent_states}")
    if hasattr(env, 'agent_dirs'):
        print(f"Agent directions: {env.agent_dirs}")
        
    # Test working scenarios - let's check if our working simple_map2 moves
    print(f"\n=== Comparing with working simple_map2 ===")
    from envs.simple_map2 import get_map as get_simple_map2
    
    map_layout2, map_metadata2 = get_simple_map2()
    env2 = CustomEnvironment(
        grid_layout=map_layout2,
        grid_metadata=map_metadata2,
        render_mode=None
    )
    
    env2.reset()
    print(f"simple_map2 Initial: Robot={env2.agent_positions['robot_0']}, Human={env2.agent_positions['human_0']}")
    
    initial_robot2 = env2.agent_positions['robot_0']
    actions = {"robot_0": 2, "human_0": 2}  # Both forward
    env2.step(actions)
    final_robot2 = env2.agent_positions['robot_0']
    
    robot_moved2 = initial_robot2 != final_robot2
    print(f"simple_map2 Robot movement: {initial_robot2} -> {final_robot2} (moved: {robot_moved2})")
    
    if robot_moved2:
        print("✅ simple_map2 has working movement - issue is specific to simple_map4")
    else:
        print("❌ Movement issue affects all maps")

if __name__ == "__main__":
    simple_movement_test()