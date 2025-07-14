#!/usr/bin/env python3
"""
Debug the fundamental movement mechanics.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from corrigibility.marl.env import CustomEnvironment, Actions


def debug_directions_and_movement():
    """Debug agent directions and movement mechanics."""
    
    print("=== DIRECTION AND MOVEMENT DEBUG ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    # Check direction mappings
    print("Direction mappings (0=north, 1=east, 2=south, 3=west):")
    
    for agent_id in env.possible_agents:
        direction = env.agent_dirs.get(agent_id, 'unknown')
        pos = env.agent_positions[agent_id]
        print(f"  {agent_id}: position {pos}, facing {direction}")
    
    # Test systematic rotation and movement for robot
    print(f"\n=== ROBOT MOVEMENT TEST ===")
    
    robot_id = 'robot_0'
    
    for test_round in range(4):  # Test 4 different directions
        env.reset()
        print(f"\nTest round {test_round + 1}:")
        
        initial_pos = env.agent_positions[robot_id]
        initial_dir = env.agent_dirs.get(robot_id, 0)
        print(f"  Initial: pos={initial_pos}, dir={initial_dir}")
        
        # First, rotate to face the desired direction
        for rotation in range(test_round):
            actions = {robot_id: 1, 'human_0': 6}  # turn_right
            obs, rewards, terms, truncs, infos = env.step(actions)
        
        current_dir = env.agent_dirs.get(robot_id, 0)
        print(f"  After {test_round} rotations: dir={current_dir}")
        
        # Now try to move forward
        current_pos = env.agent_positions[robot_id]
        actions = {robot_id: 2, 'human_0': 6}  # forward
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        new_pos = env.agent_positions[robot_id]
        moved = new_pos != current_pos
        
        print(f"  Forward movement: {current_pos} -> {new_pos}, moved: {moved}")
        
        if moved:
            # Check what cell was moved to
            target_cell = env.map_layout[new_pos[0]][new_pos[1]]
            print(f"  Moved to cell: '{target_cell}'")


def check_map_structure():
    """Check if the map structure allows movement."""
    
    print(f"\n=== MAP STRUCTURE ANALYSIS ===")
    
    env = CustomEnvironment(map_name="simple_map")
    env.reset()
    
    print("Full map layout:")
    for i, row in enumerate(env.map_layout):
        print(f"  {i}: {row}")
    
    print(f"\nFloor cells (where agents can move):")
    for i, row in enumerate(env.map_layout):
        for j, cell in enumerate(row):
            if cell in ['  ', 'vR', 'vH', 'BK', 'BD', 'GG']:  # Possible walkable cells
                print(f"  ({i},{j}): '{cell}'")
    
    # Check agent spawn positions
    print(f"\nAgent positions relative to map:")
    for agent_id in env.possible_agents:
        pos = env.agent_positions[agent_id]
        if 0 <= pos[0] < len(env.map_layout) and 0 <= pos[1] < len(env.map_layout[0]):
            cell = env.map_layout[pos[0]][pos[1]]
            print(f"  {agent_id} at {pos}: '{cell}'")
        else:
            print(f"  {agent_id} at {pos}: OUT_OF_BOUNDS")


def test_alternative_map():
    """Test with a different map to see if movement works there."""
    
    print(f"\n=== TESTING ALTERNATIVE MAP ===")
    
    try:
        env = CustomEnvironment(map_name="simple_map2")
        env.reset()
        
        print("Alternative map layout:")
        for i, row in enumerate(env.map_layout):
            print(f"  {i}: {row}")
        
        print(f"\nAgent positions:")
        for agent_id in env.possible_agents:
            pos = env.agent_positions[agent_id]
            print(f"  {agent_id}: {pos}")
        
        # Test movement on alternative map
        robot_id = env.robot_agent_ids[0] if env.robot_agent_ids else None
        if robot_id:
            initial_pos = env.agent_positions[robot_id]
            actions = {robot_id: 2}  # forward
            for agent_id in env.possible_agents:
                if agent_id != robot_id:
                    actions[agent_id] = 6  # no-op
            
            obs, rewards, terms, truncs, infos = env.step(actions)
            final_pos = env.agent_positions[robot_id]
            
            print(f"Movement test: {initial_pos} -> {final_pos}, moved: {final_pos != initial_pos}")
    
    except Exception as e:
        print(f"Alternative map test failed: {e}")


if __name__ == "__main__":
    debug_directions_and_movement()
    check_map_structure()
    test_alternative_map()