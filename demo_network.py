#!/usr/bin/env python3
"""
Quick Demo Script for IQL Network Support

This script demonstrates the key differences between tabular and neural network modes.
"""

import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL
from corrigibility.marl.objects import Actions

def demo_network_support():
    """Demonstrate both tabular and network modes with simple examples."""
    
    print("🎯 IQL Network Support Demo")
    print("=" * 40)
    
    # Common setup
    action_space_dict = {
        'robot_0': list(Actions),
        'human_0': list(Actions)
    }
    
    G = [(0, 0), (2, 2)]
    mu_g = np.array([0.5, 0.5])
    state = (1, 1)
    goal = (2, 2)
    
    print(f"Demo setup:")
    print(f"  State: {state}")
    print(f"  Goal: {goal}")
    print(f"  Available actions: {len(list(Actions))}")
    
    # Tabular Mode Demo
    print(f"\n📊 TABULAR MODE DEMO")
    print("-" * 30)
    
    alg_tabular = TwoPhaseTimescaleIQL(
        alpha_m=0.1, alpha_e=0.1, alpha_r=0.1,
        gamma_h=0.99, gamma_r=0.99, beta_r_0=5.0,
        G=G, mu_g=mu_g, p_g=0.1,
        action_space_dict=action_space_dict,
        robot_agent_ids=['robot_0'],
        human_agent_ids=['human_0'],
        network=False, state_dim=4, debug=False
    )
    
    print("✅ Tabular algorithm initialized")
    
    # Sample some actions
    robot_action_tab = alg_tabular.sample_robot_action_phase1_targeted('robot_0', state, 'human_0', goal)
    human_action_tab = alg_tabular.sample_human_action_phase1('human_0', state, goal)
    
    print(f"🤖 Robot action (targeted): {Actions(robot_action_tab).name}")
    print(f"👤 Human action: {Actions(human_action_tab).name}")
    
    # Get policy
    robot_policy = alg_tabular.get_pi_r('robot_0', state)
    human_policy = alg_tabular.get_pi_h('human_0', state, goal)
    
    print(f"🧠 Robot policy entropy: {-sum(p * np.log(p + 1e-8) for p in robot_policy.values()):.3f}")
    print(f"🧠 Human policy entropy: {-sum(p * np.log(p + 1e-8) for p in human_policy.values()):.3f}")
    
    # Neural Network Mode Demo
    print(f"\n🧠 NEURAL NETWORK MODE DEMO")
    print("-" * 30)
    
    try:
        alg_network = TwoPhaseTimescaleIQL(
            alpha_m=0.1, alpha_e=0.1, alpha_r=0.1,
            gamma_h=0.99, gamma_r=0.99, beta_r_0=5.0,
            G=G, mu_g=mu_g, p_g=0.1,
            action_space_dict=action_space_dict,
            robot_agent_ids=['robot_0'],
            human_agent_ids=['human_0'],
            network=True, state_dim=4, debug=False
        )
        
        print("✅ Neural network algorithm initialized")
        
        # Sample some actions
        robot_action_nn = alg_network.sample_robot_action_phase1_targeted('robot_0', state, 'human_0', goal)
        human_action_nn = alg_network.sample_human_action_phase1('human_0', state, goal)
        
        print(f"🤖 Robot action (targeted): {Actions(robot_action_nn).name}")
        print(f"👤 Human action: {Actions(human_action_nn).name}")
        
        # Get policy
        robot_policy_nn = alg_network.get_pi_r('robot_0', state)
        human_policy_nn = alg_network.get_pi_h('human_0', state, goal)
        
        print(f"🧠 Robot policy entropy: {-sum(p * np.log(p + 1e-8) for p in robot_policy_nn.values()):.3f}")
        print(f"🧠 Human policy entropy: {-sum(p * np.log(p + 1e-8) for p in human_policy_nn.values()):.3f}")
        
        # Show network architecture info
        print(f"🏗️  Network architecture:")
        print(f"   Input dim: {alg_network.human_q_m_backend.input_dim}")
        print(f"   Hidden sizes: {alg_network.human_q_m_backend.hidden_sizes}")
        print(f"   Output dim: {len(list(Actions))}")
        
    except ImportError:
        print("❌ PyTorch not available - neural network mode disabled")
    except Exception as e:
        print(f"❌ Neural network mode failed: {e}")
    
    # Comparison
    print(f"\n🔍 MODE COMPARISON")
    print("-" * 30)
    print(f"Tabular Mode:")
    print(f"  ✅ Fast for small state spaces")
    print(f"  ✅ Exact Q-value storage")
    print(f"  ✅ Complete policy analysis")
    print(f"  ❌ Limited to discrete states")
    
    print(f"\nNeural Network Mode:")
    print(f"  ✅ Scalable to large state spaces")
    print(f"  ✅ Generalization across states")
    print(f"  ✅ Handles continuous features")
    print(f"  ❌ Approximate Q-values")
    
    print(f"\n🎉 Demo completed successfully!")

if __name__ == '__main__':
    demo_network_support()
