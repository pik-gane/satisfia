#!/usr/bin/env python3
"""
Simple test to verify all imports work correctly from the MARL directory.
"""

def test_imports():
    """Test that all modules can be imported correctly"""
    print("Testing imports...")
    
    try:
        from iql_timescale_algorithm import TwoPhaseTimescaleIQL
        print("✅ IQL algorithm import successful")
    except ImportError as e:
        print(f"❌ IQL algorithm import failed: {e}")
        return False
    
    try:
        from env import CustomEnvironment
        print("✅ Environment import successful")
    except ImportError as e:
        print(f"❌ Environment import failed: {e}")
        return False
    
    try:
        from envs.simple_map import get_map
        print("✅ Simple map import successful")
    except ImportError as e:
        print(f"❌ Simple map import failed: {e}")
        return False
    
    try:
        from q_learning_backends import create_q_learning_backend
        print("✅ Q-learning backend import successful")
    except ImportError as e:
        print(f"❌ Q-learning backend import failed: {e}")
        return False
    
    print("\n🎉 All imports successful! MARL setup is working correctly.")
    return True


def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from iql_timescale_algorithm import TwoPhaseTimescaleIQL
        from envs.simple_map import get_map
        from env import CustomEnvironment
        
        # Create environment
        map_layout, map_metadata = get_map()
        env = CustomEnvironment(
            grid_layout=map_layout,
            grid_metadata=map_metadata,
            render_mode=None
        )
        
        # Create algorithm
        action_space_dict = {
            "robot_0": list(range(6)),
            "human_0": list(range(3))
        }
        
        human_goals = list(map_metadata["human_goals"].values())
        G = [np.array(goal) for goal in human_goals]
        mu_g = [1.0 / len(G)] * len(G)
        
        iql = TwoPhaseTimescaleIQL(
            alpha_m=0.1, alpha_e=0.1, alpha_r=0.1, alpha_p=0.1,
            gamma_h=0.9, gamma_r=0.9, beta_r_0=5.0,
            G=G, mu_g=mu_g,
            action_space_dict=action_space_dict,
            robot_agent_ids=["robot_0"],
            human_agent_ids=["human_0"],
            network=False,
            env=env
        )
        
        print("✅ Environment and algorithm creation successful")
        
        # Test one episode
        env.reset()
        for _ in range(5):
            actions = {
                "robot_0": 0,  # turn left
                "human_0": 0   # turn left
            }
            env.step(actions)
        
        print("✅ Basic episode execution successful")
        print("\n🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_imports() and test_basic_functionality()
    
    if success:
        print("\n" + "="*50)
        print("✅ MARL SETUP VERIFICATION COMPLETE")
        print("="*50)
        print("You can now run:")
        print("  python test_quick_iql.py")
        print("  python visualize_iql_tabular.py")
        print("  python debug_detailed_iql.py")
    else:
        print("\n" + "="*50)
        print("❌ SETUP VERIFICATION FAILED")
        print("="*50)
        print("Please check the import errors above.")