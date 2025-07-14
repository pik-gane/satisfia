#!/usr/bin/env python3
"""
Analyze how to access the door in the simple map.
"""

import numpy as np

from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment


def analyze_door_accessibility():
    """Analyze how the door can be accessed"""
    print("=== DOOR ACCESSIBILITY ANALYSIS ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    # Print grid with coordinates
    print("Grid with coordinates:")
    print("   0 1 2 3 4")
    for r in range(env.grid_size):
        row = f"{r}: "
        for c in range(env.grid_size):
            pos = (r, c)
            if pos == env.agent_positions.get('robot_0'):
                row += "R "
            elif pos == env.agent_positions.get('human_0'):
                row += "H "
            else:
                char = env.grid[r, c]
                if char == ' ':
                    row += ". "
                else:
                    row += f"{char} "
        print(row)
    
    door_pos = (2, 2)
    print(f"\nDoor at {door_pos}")
    
    # Check all adjacent positions to the door
    adjacent_positions = [
        (door_pos[0] - 1, door_pos[1]),     # Above: (1, 2)
        (door_pos[0] + 1, door_pos[1]),     # Below: (3, 2)
        (door_pos[0], door_pos[1] - 1),     # Left: (2, 1)
        (door_pos[0], door_pos[1] + 1),     # Right: (2, 3)
    ]
    
    print(f"\nPositions adjacent to door:")
    for i, pos in enumerate(adjacent_positions):
        direction = ["above", "below", "left", "right"][i]
        if 0 <= pos[0] < env.grid_size and 0 <= pos[1] < env.grid_size:
            is_valid = env._is_valid_pos(pos)
            char = env.grid[pos]
            print(f"  {direction} {pos}: valid={is_valid}, char='{char}'")
        else:
            print(f"  {direction} {pos}: out of bounds")
    
    # The key insight: position (1, 2) is where the key is!
    # The robot cannot move there, but it can pick up the key while adjacent
    # Position (3, 2) is the goal - valid and accessible
    
    print(f"\nKey insights:")
    print(f"- Position (1, 2) contains the key - robot cannot move there but can pickup from (1, 1)")
    print(f"- Position (3, 2) is the goal - this is where human needs to end up")
    print(f"- Position (2, 1) and (2, 3) are walls - cannot access door from sides")
    print(f"- The door can only be accessed from (3, 2) - the goal position!")
    
    return adjacent_positions


def test_door_opening_from_goal():
    """Test if door can be opened from the goal position"""
    print(f"\n=== DOOR OPENING FROM GOAL TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    # Step 1: Robot gets the key
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})  # turn right
    env.step({robot_id: 3})  # pickup key
    print(f"Robot picked up key: {list(env.robot_has_keys)}")
    
    # Step 2: Give the key to human somehow, or test if human can open door
    # Actually, let's check if the robot can move to position (3,2) to open the door
    
    print(f"\nTesting robot movement to goal position (3,2):")
    goal_pos = (3, 2)
    robot_start = env.agent_positions[robot_id]
    
    print(f"Robot starts at: {robot_start}")
    print(f"Goal position: {goal_pos}")
    print(f"Goal position valid: {env._is_valid_pos(goal_pos)}")
    
    # Robot needs to navigate from (1,1) to (3,2)
    # Path could be: (1,1) -> (1,3) -> (3,3) -> (3,2)
    # But (1,3) is where human is!
    
    # Alternative: Robot moves to (3,1) then (3,2)
    # Path: (1,1) -> face down -> move to (2,1) [WALL] 
    
    # Let's try a different approach: robot moves to (3,1) first
    print(f"\nTrying robot path (1,1) -> (3,1) -> (3,2):")
    
    # First check if (3,1) is accessible
    pos_3_1 = (3, 1)
    print(f"Position (3,1): valid={env._is_valid_pos(pos_3_1)}, char='{env.grid[pos_3_1]}'")
    
    # Robot at (1,1) facing right, needs to go to (3,1)
    # Must go around: (1,1) -> (1,3) -> (3,3) -> (3,1)
    # But human is at (1,3)!
    
    print(f"\nAnalysis: Robot cannot reach (3,2) directly due to:")
    print(f"- Wall at (2,1) blocks downward movement")
    print(f"- Human at (1,3) blocks rightward movement")
    print(f"- Only path would be around, but requires moving human first")


def test_human_opens_door():
    """Test if human can open the door when it reaches goal"""
    print(f"\n=== HUMAN OPENS DOOR TEST ===")
    
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    
    robot_id = "robot_0"
    human_id = "human_0"
    
    # Give robot the key
    while env.agent_dirs[robot_id] != 1:  # Face right
        env.step({robot_id: 1})  # turn right
    env.step({robot_id: 3})  # pickup key
    print(f"Robot picked up key: {list(env.robot_has_keys)}")
    
    # Now manually move human to goal position (3,2)
    print(f"\nManually moving human to goal position...")
    
    # Move human from (1,3) to (3,3) then to (3,2)
    # Human starts facing down, turn left to face right
    env.step({human_id: 0})  # turn left -> now facing right
    env.step({human_id: 0})  # turn left -> now facing up
    env.step({human_id: 0})  # turn left -> now facing left
    env.step({human_id: 0})  # turn left -> now facing down (back to original)
    
    # Actually, let's be more systematic
    # Human at (1,3) facing down, needs to go to (3,3)
    env.step({human_id: 2})  # move forward to (2,3) - but this might be wall!
    
    print(f"Human position after attempting move: {env.agent_positions[human_id]}")
    
    # Check if (2,3) is valid
    pos_2_3 = (2, 3)
    print(f"Position (2,3): valid={env._is_valid_pos(pos_2_3)}, char='{env.grid[pos_2_3]}'")
    
    # Position (2,3) is a wall! Human cannot move there
    # This means human cannot reach (3,2) from (1,3) directly
    
    print(f"\nConclusion: Human cannot reach goal (3,2) because:")
    print(f"- Cannot move down from (1,3) because (2,3) is a wall")
    print(f"- The puzzle appears unsolvable with this layout!")


if __name__ == "__main__":
    analyze_door_accessibility()
    test_door_opening_from_goal()
    test_human_opens_door()