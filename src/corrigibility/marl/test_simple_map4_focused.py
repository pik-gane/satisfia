#!/usr/bin/env python3
"""
Focused test for simple_map4 to achieve 100% success.
"""

from fixed_enhanced_algorithm import train_fixed_algorithm
import numpy as np
from envs.simple_map4 import get_map
from env import CustomEnvironment

def analyze_simple_map4():
    """Analyze simple_map4 structure first"""
    print("=== ANALYZING SIMPLE_MAP4 ===")
    map_layout, map_metadata = get_map()
    
    print("Map Layout:")
    for i, row in enumerate(map_layout):
        print(f"Row {i}: {row}")
    
    print(f"\nMetadata: {map_metadata}")
    
    # Create environment to check positions
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    env.reset()
    
    print(f"\nInitial positions:")
    print(f"Robot: {env.agent_positions['robot_0']}")
    print(f"Human: {env.agent_positions['human_0']}")
    print(f"Goal: {env.human_goals['human_0']}")
    
    # Check objects
    print(f"\nObjects in environment:")
    for pos, obj in env.grid.objects.items():
        print(f"  {pos}: {obj}")
    
    print(f"\nOptimal path analysis:")
    print(f"Human at {env.agent_positions['human_0']} needs to reach {env.human_goals['human_0']}")
    print(f"Distance: {abs(env.agent_positions['human_0'][0] - env.human_goals['human_0'][0]) + abs(env.agent_positions['human_0'][1] - env.human_goals['human_0'][1])}")
    
    # What's blocking the path?
    human_pos = env.agent_positions['human_0']
    goal_pos = env.human_goals['human_0'] 
    print(f"\nPath from {human_pos} to {goal_pos}:")
    
    # Direct path would be: (1,3) -> (2,3) -> (3,3)
    # But (2,3) has a blue door BD
    if (2, 3) in env.grid.objects:
        print(f"  Obstacle at (2,3): {env.grid.objects[(2,3)]}")
    if (2, 2) in env.grid.objects:
        print(f"  Key/tool at (2,2): {env.grid.objects[(2,2)]}")

def test_simple_map4_intensive():
    """Test simple_map4 with intensive training"""
    print(f"\n{'='*60}")
    print("INTENSIVE TRAINING FOR SIMPLE_MAP4")
    print(f"{'='*60}")
    
    checkpoint_path, success_rate, avg_steps = train_fixed_algorithm(
        "simple_map4", 
        phase1_episodes=800,  # More training
        phase2_episodes=1200  # More training
    )
    
    print(f"\nSIMPLE_MAP4 RESULTS:")
    print(f"  🎯 Success Rate: {success_rate:.1%}")
    print(f"  ⏱️  Average Steps: {avg_steps:.1f}")
    print(f"  💾 Checkpoint: {checkpoint_path}")
    
    if success_rate >= 0.95:
        print(f"  ✅ EXCELLENT! simple_map4 achieves ≥95% success")
    elif success_rate >= 0.8:
        print(f"  ⚠️  GOOD: simple_map4 achieves ≥80% success")
    else:
        print(f"  ❌ NEEDS MORE WORK: simple_map4 only {success_rate:.1%} success")
    
    return checkpoint_path, success_rate, avg_steps

if __name__ == "__main__":
    analyze_simple_map4()
    checkpoint_path, success_rate, avg_steps = test_simple_map4_intensive()
    print(f"\nFINAL: simple_map4 achieved {success_rate:.1%} success")