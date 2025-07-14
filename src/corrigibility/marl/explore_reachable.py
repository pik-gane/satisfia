#!/usr/bin/env python3
"""
Explore all reachable positions for the robot.
"""

from envs.simple_map4 import get_map
from env import CustomEnvironment

def explore_reachable():
    """Explore all positions robot can reach"""
    print("=== Explore Reachable Positions ===")
    
    map_layout, map_metadata = get_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    visited = set()
    positions_to_check = [(1, 1)]  # Start position
    reachable = {}
    
    def try_reach_position(target_pos):
        """Try to reach target position and return path if successful"""
        # Simple BFS to find path
        queue = [((1, 1), [])]  # (position, actions_taken)
        visited_local = {(1, 1)}
        
        while queue:
            current_pos, actions = queue.pop(0)
            
            if current_pos == target_pos:
                return actions
            
            # Try all directions from current position
            for direction in range(4):
                env.reset()
                
                # Execute path to current position
                for action_seq in actions:
                    for action in action_seq:
                        env.step({"robot_0": action, "human_0": 0})
                
                # Turn to target direction
                current_dir = env.agent_dirs['robot_0']
                turn_actions = []
                while current_dir != direction:
                    turn_actions.append(1)  # turn right
                    env.step({"robot_0": 1, "human_0": 0})
                    current_dir = env.agent_dirs['robot_0']
                
                # Try forward
                initial_pos = env.agent_positions['robot_0']
                env.step({"robot_0": 2, "human_0": 0})
                new_pos = tuple(env.agent_positions['robot_0'])
                
                if new_pos != initial_pos and new_pos not in visited_local:
                    visited_local.add(new_pos)
                    new_actions = actions + [turn_actions + [2]]
                    queue.append((new_pos, new_actions))
        
        return None
    
    # Check specific important positions
    important_positions = [
        (1, 1), (2, 1), (3, 1),  # Column 1
        (1, 2), (2, 2), (3, 2),  # Column 2 (key is at 2,2)
        (1, 3), (2, 3), (3, 3),  # Column 3 (door at 2,3, goal at 3,3)
    ]
    
    print("Testing reachability to important positions:")
    for pos in important_positions:
        print(f"\nTesting position {pos}:")
        
        env.reset()
        
        # Simple manual check - try all 4 directions from all reachable positions
        reachable_from_start = []
        
        # Direct movement from start
        for start_dir in range(4):
            env.reset()
            
            # Turn to direction
            current_dir = env.agent_dirs['robot_0']
            while current_dir != start_dir:
                env.step({"robot_0": 1, "human_0": 0})
                current_dir = env.agent_dirs['robot_0']
            
            # Try forward
            initial_pos = env.agent_positions['robot_0']
            env.step({"robot_0": 2, "human_0": 0})
            new_pos = tuple(env.agent_positions['robot_0'])
            
            if new_pos != initial_pos:
                reachable_from_start.append(new_pos)
                print(f"  Can reach {new_pos} directly from start using direction {start_dir}")
                
                # From this new position, try reaching target
                if new_pos == pos:
                    print(f"  ✅ FOUND! Can reach {pos} directly from start")
                    continue
                
                # Try one more step from new position
                for second_dir in range(4):
                    env.reset()
                    
                    # Get to new_pos first
                    current_dir = env.agent_dirs['robot_0']
                    while current_dir != start_dir:
                        env.step({"robot_0": 1, "human_0": 0})
                        current_dir = env.agent_dirs['robot_0']
                    env.step({"robot_0": 2, "human_0": 0})
                    
                    # Turn to second direction
                    current_dir = env.agent_dirs['robot_0']
                    while current_dir != second_dir:
                        env.step({"robot_0": 1, "human_0": 0})
                        current_dir = env.agent_dirs['robot_0']
                    
                    # Try forward
                    second_initial = env.agent_positions['robot_0']
                    env.step({"robot_0": 2, "human_0": 0})
                    final_pos = tuple(env.agent_positions['robot_0'])
                    
                    if final_pos == pos:
                        print(f"  ✅ FOUND! Can reach {pos} via {new_pos} using directions {start_dir} then {second_dir}")
        
        if pos not in reachable_from_start and pos != (1, 1):
            print(f"  ❌ Cannot reach {pos} in 2 steps")

if __name__ == "__main__":
    explore_reachable()