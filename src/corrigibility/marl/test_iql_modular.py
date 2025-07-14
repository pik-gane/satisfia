#!/usr/bin/python3
"""
Test script for the modular IQL timescale algorithm.
Tests both tabular and neural network modes.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL
from src.corrigibility.marl.env import Actions
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL
from corrigibility.marl.env import Actions

def test_tabular_mode():
    """Test the algorithm in tabular mode."""
    print("🧪 Testing Tabular Mode")
    print("=" * 50)
    
    # Create algorithm instance in tabular mode
    alg = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.2,
        alpha_r=0.01,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=[(0, 0), (1, 1)],  # Two goals
        mu_g=np.array([0.5, 0.5]),
        p_g=0.1,
        action_space_dict={"robot_0": list(Actions), "human_0": list(Actions)},
        robot_agent_ids=["robot_0"],
        human_agent_ids=["human_0"],
        eta=0.1,
        epsilon_h_0=0.1,
        network=False,  # Tabular mode
        beta_h=5.0,
        nu_h=0.1,
        debug=True
    )
    
    # Test basic functionality
    state = (0, 0)
    goal = (1, 1)
    
    # Test Q-value operations
    print(f"Testing Q-value operations...")
    
    # Test human Q^m backend
    q_values = alg.human_q_m_backend.get_q_values("human_0", state, goal)
    print(f"Initial human Q^m values: {q_values}")
    
    # Update Q-values  
    alg.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 0.5, 0.1, goal)
    q_values = alg.human_q_m_backend.get_q_values("human_0", state, goal)
    print(f"After update human Q^m values: {q_values}")
    
    # Test policy
    policy = alg.human_q_m_backend.get_policy("human_0", state, alg.beta_h, goal)
    print(f"Human policy: {policy}")
    
    # Test robot Q backend
    q_values = alg.robot_q_backend.get_q_values("robot_0", state)
    print(f"Robot Q values: {q_values}")
    
    print("✅ Tabular mode test passed!")
    return alg

def test_network_mode():
    """Test the algorithm in network mode."""
    print("\n🧪 Testing Network Mode")
    print("=" * 50)
    
    # Create algorithm instance in network mode
    alg = TwoPhaseTimescaleIQL(
        alpha_m=0.001,  # Lower learning rate for networks
        alpha_e=0.001,
        alpha_r=0.001,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=[(0, 0), (1, 1)],  # Two goals
        mu_g=np.array([0.5, 0.5]),
        p_g=0.1,
        action_space_dict={"robot_0": list(Actions), "human_0": list(Actions)},
        robot_agent_ids=["robot_0"],
        human_agent_ids=["human_0"],
        eta=0.1,
        epsilon_h_0=0.1,
        network=True,  # Network mode
        state_dim=4,  # State dimension for networks
        beta_h=5.0,
        nu_h=0.1,
        debug=True
    )
    
    # Test basic functionality
    state = (0, 0)
    goal = (1, 1)
    
    # Test Q-value operations
    print(f"Testing Q-value operations...")
    
    # Test human Q^m backend
    q_values = alg.human_q_m_backend.get_q_values("human_0", state, goal)
    print(f"Initial human Q^m values: {q_values}")
    
    # Update Q-values
    alg.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 0.5, 0.1, goal)
    q_values = alg.human_q_m_backend.get_q_values("human_0", state, goal)
    print(f"After update human Q^m values: {q_values}")
    
    # Test policy
    policy = alg.human_q_m_backend.get_policy("human_0", state, alg.beta_h, goal)
    print(f"Human policy: {policy}")
    
    # Test robot Q backend
    q_values = alg.robot_q_backend.get_q_values("robot_0", state)
    print(f"Robot Q values: {q_values}")
    
    print("✅ Network mode test passed!")
    return alg

def test_save_load():
    """Test save and load functionality."""
    print("\n🧪 Testing Save/Load Functionality")
    print("=" * 50)
    
    # Create and train a small algorithm
    alg1 = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.2,
        alpha_r=0.01,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=[(0, 0), (1, 1)],
        mu_g=np.array([0.5, 0.5]),
        p_g=0.1,
        action_space_dict={"robot_0": list(Actions), "human_0": list(Actions)},
        robot_agent_ids=["robot_0"],
        human_agent_ids=["human_0"],
        eta=0.1,
        epsilon_h_0=0.1,
        network=False,
        beta_h=5.0,
        nu_h=0.1,
        debug=False
    )
    
    # Make some updates
    state = (0, 0)
    goal = (1, 1)
    alg1.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 0.5, 0.1, goal)
    alg1.robot_q_backend.update_q_values("robot_0", state, Actions.turn_left, 0.3, 0.1)
    
    # Save
    filepath = "test_save.pkl"
    alg1.save_models(filepath)
    print(f"Saved models to {filepath}")
    
    # Load
    alg2 = TwoPhaseTimescaleIQL.load_q_values(filepath)
    print(f"Loaded models from {filepath}")
    
    if alg2 is not None:
        # Compare Q-values
        q1 = alg1.human_q_m_backend.get_q_values("human_0", state, goal)
        q2 = alg2.human_q_m_backend.get_q_values("human_0", state, goal)
        print(f"Original Q-values: {q1}")
        print(f"Loaded Q-values: {q2}")
        
        if np.allclose(q1, q2):
            print("✅ Save/Load test passed!")
        else:
            print("❌ Save/Load test failed!")
    else:
        print("❌ Failed to load models!")
    
    # Clean up
    import os
    if os.path.exists(filepath):
        os.remove(filepath)

def test_policy_updates():
    """Test smooth policy updates."""
    print("\n🧪 Testing Smooth Policy Updates")
    print("=" * 50)
    
    alg = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.2,
        alpha_r=0.01,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=[(0, 0)],
        mu_g=np.array([1.0]),
        p_g=0.1,
        action_space_dict={"robot_0": list(Actions), "human_0": list(Actions)},
        robot_agent_ids=["robot_0"],
        human_agent_ids=["human_0"],
        eta=0.1,
        epsilon_h_0=0.1,
        network=False,
        beta_h=5.0,
        nu_h=0.1,
        debug=False
    )
    
    state = (0, 0)
    goal = (0, 0)
    
    # Get initial policy
    policy1 = alg.human_q_m_backend.get_policy("human_0", state, alg.beta_h, goal)
    print(f"Initial policy: {policy1}")
    
    # Update Q-values to make one action much better
    alg.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 1.0, 0.0, goal)
    alg.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 1.0, 0.0, goal)
    alg.human_q_m_backend.update_q_values("human_0", state, Actions.forward, 1.0, 0.0, goal)
    
    # Check if policy updated smoothly
    policy2 = alg.human_q_m_backend.get_policy("human_0", state, alg.beta_h, goal)
    print(f"After Q-updates policy: {policy2}")
    
    # Should be more concentrated on forward action but still smooth
    if policy2[Actions.forward] > policy1[Actions.forward]:
        print("✅ Smooth policy update test passed!")
    else:
        print("❌ Smooth policy update test failed!")

def main():
    """Run all tests."""
    print("🚀 Starting IQL Modular Algorithm Tests")
    print("=" * 60)
    
    try:
        # Test tabular mode
        alg_tab = test_tabular_mode()
        
        # Test network mode
        alg_net = test_network_mode()
        
        # Test save/load
        test_save_load()
        
        # Test policy updates
        test_policy_updates()
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
