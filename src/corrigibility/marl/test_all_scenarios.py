#!/usr/bin/env python3
"""
Test all simple scenarios with the fixed enhanced algorithm.
"""

from fixed_enhanced_algorithm import train_fixed_algorithm

def test_all_scenarios():
    """Test all simple scenarios"""
    scenarios = ["simple_map", "simple_map2", "simple_map3", "simple_map4"]
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Testing {scenario}")
        print(f"{'='*50}")
        
        checkpoint_path, success_rate, avg_steps = train_fixed_algorithm(
            scenario, 
            phase1_episodes=600, 
            phase2_episodes=800
        )
        
        results[scenario] = {
            "checkpoint": checkpoint_path,
            "success_rate": success_rate,
            "avg_steps": avg_steps
        }
        
        print(f"\n{scenario} Results:")
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        if success_rate >= 0.9:
            print(f"  ✅ EXCELLENT! {scenario} achieves >90% success")
        elif success_rate >= 0.7:
            print(f"  ⚠️  GOOD: {scenario} achieves >70% success")
        else:
            print(f"  ❌ NEEDS WORK: {scenario} only {success_rate:.1%} success")
    
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    
    for scenario, result in results.items():
        status = "✅" if result["success_rate"] >= 0.9 else "⚠️" if result["success_rate"] >= 0.7 else "❌"
        print(f"{status} {scenario}: {result['success_rate']:.1%} success, {result['avg_steps']:.1f} steps")
    
    return results

if __name__ == "__main__":
    results = test_all_scenarios()