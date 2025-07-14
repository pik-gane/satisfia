#!/usr/bin/env python3
"""
Test individual cooperation sequences on each map to verify they work.
"""

import numpy as np
from envs.simple_map import get_map as get_simple_map
from envs.simple_map2 import get_map as get_simple_map2
from envs.simple_map3 import get_map as get_simple_map3
from envs.simple_map4 import get_map as get_simple_map4
from env import CustomEnvironment

def test_sequence_on_map(map_name, get_map_func, sequence, description):
    """Test a specific sequence on a specific map"""
    print(f"\n{'='*50}")
    print(f"Testing {description} on {map_name}")
    print(f"{'='*50}")
    
    # Create environment
    map_layout, map_metadata = get_map_func()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Goal: {env.human_goals['human_0']}")
    print(f"Keys: {[k['pos'] for k in env.keys]}")
    print(f"Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
    
    # Execute sequence
    for step, actions in enumerate(sequence):
        print(f"\nStep {step + 1}: {actions}")
        print(f"  Before: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"  After: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"  Keys: {[k['pos'] for k in env.keys]}, Robot has: {env.robot_has_keys}")
        print(f"  Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        
        # Check if goal reached
        goal = env.human_goals['human_0']
        if tuple(env.agent_positions['human_0']) == tuple(goal):
            print(f"  🎉 GOAL REACHED!")
            return True
        
        if any(terms.values()) or any(truncs.values()):
            print(f"  Episode ended")
            break
    
    print(f"  ❌ Goal not reached")
    return False

def test_all_sequences():
    """Test all sequences on their intended maps"""
    
    # Map 1 sequence (sequential cooperation)
    map1_sequence = [
        {"robot_0": 0, "human_0": 0},  # Robot turn left to face key
        {"robot_0": 3, "human_0": 0},  # Robot pickup key
        {"robot_0": 2, "human_0": 0},  # Robot move to key position
        {"robot_0": 1, "human_0": 0},  # Robot turn right to face door
        {"robot_0": 5, "human_0": 0},  # Robot open door
        {"robot_0": 0, "human_0": 2},  # Human move down
        {"robot_0": 0, "human_0": 2},  # Human move down through door
        {"robot_0": 0, "human_0": 1},  # Human turn right to face goal
        {"robot_0": 0, "human_0": 2},  # Human move right to goal
    ]
    
    # Map 2 sequence (direct path)
    map2_sequence = [
        {"robot_0": 0, "human_0": 2},  # Human move down
        {"robot_0": 0, "human_0": 2},  # Human move down
        {"robot_0": 0, "human_0": 0},  # Human turn left to face goal
        {"robot_0": 0, "human_0": 2},  # Human move left to goal
    ]
    
    # Map 3 sequence (asymmetric path)
    map3_sequence = [
        {"robot_0": 0, "human_0": 2},  # Human move down
        {"robot_0": 0, "human_0": 0},  # Human turn left
        {"robot_0": 0, "human_0": 2},  # Human move left
        {"robot_0": 0, "human_0": 2},  # Human move left
        {"robot_0": 0, "human_0": 1},  # Human turn right
        {"robot_0": 0, "human_0": 2},  # Human move right to goal
    ]
    
    # Map 4 sequence (distributed cooperation)
    map4_sequence = [
        {"robot_0": 1, "human_0": 0},  # Robot turn right to face down
        {"robot_0": 2, "human_0": 0},  # Robot move down to key
        {"robot_0": 3, "human_0": 0},  # Robot pickup key
        {"robot_0": 1, "human_0": 0},  # Robot turn right to face door
        {"robot_0": 2, "human_0": 0},  # Robot move to door
        {"robot_0": 5, "human_0": 0},  # Robot open door
        {"robot_0": 0, "human_0": 2},  # Human move down
        {"robot_0": 0, "human_0": 2},  # Human move down through door
        {"robot_0": 0, "human_0": 0},  # Human turn left to face goal
        {"robot_0": 0, "human_0": 2},  # Human move left to goal
    ]
    
    # Test each sequence on its intended map
    results = []
    
    results.append(test_sequence_on_map("Map 1", get_simple_map, map1_sequence, "Sequential Cooperation"))
    results.append(test_sequence_on_map("Map 2", get_simple_map2, map2_sequence, "Direct Path"))
    results.append(test_sequence_on_map("Map 3", get_simple_map3, map3_sequence, "Asymmetric Path"))
    results.append(test_sequence_on_map("Map 4", get_simple_map4, map4_sequence, "Distributed Cooperation"))
    
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL SEQUENCE RESULTS")
    print(f"{'='*60}")
    
    map_names = ["Map 1", "Map 2", "Map 3", "Map 4"]
    for i, (name, success) in enumerate(zip(map_names, results)):
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {status}: {name}")
    
    success_count = sum(results)
    print(f"\nOverall: {success_count}/4 sequences work correctly")
    
    return results

if __name__ == "__main__":
    test_all_sequences()