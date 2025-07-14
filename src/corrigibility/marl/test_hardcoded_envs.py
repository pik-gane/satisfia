#!/usr/bin/env python3
"""
Test script to validate hardcoded curriculum environments.

This script tests that:
1. All environment files can be imported correctly
2. Each environment provides get_map() and get_env_config() functions
3. The maps are valid and contain the expected elements
4. The curriculum system can load all environments

Usage:
    python test_hardcoded_envs.py
"""

import sys
from pathlib import Path

# Add the current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


def test_single_environment(env_name, module_name):
    """Test a single environment module."""
    print(f"🧪 Testing {env_name}...")

    try:
        # Import the module
        module = __import__(f"curriculum_envs.{module_name}", fromlist=[module_name])

        # Test get_env_config
        if not hasattr(module, "get_env_config"):
            raise AttributeError("Missing get_env_config function")

        config = module.get_env_config()
        print(f"   ✅ Configuration loaded: {config['name']}")
        print(f"   📋 Description: {config['description']}")
        print(f"   🎚️  Difficulty: {config['difficulty']}")
        print(f"   🗺️  Uses hardcoded map: {config.get('use_hardcoded_map', False)}")

        # Test get_map
        if not hasattr(module, "get_map"):
            raise AttributeError("Missing get_map function")

        map_layout, metadata = module.get_map()
        print(f"   ✅ Map loaded: {metadata['size']}")
        print(f"   📏 Grid size: {len(map_layout)}x{len(map_layout[0])}")
        print(f"   🎯 Max steps: {metadata['max_steps']}")
        print(f"   👥 Humans: {metadata.get('num_humans', 0)}")
        print(f"   🤖 Robots: {metadata.get('num_robots', 0)}")

        # Validate map structure
        rows = len(map_layout)
        cols = len(map_layout[0]) if rows > 0 else 0

        if metadata["size"] != (rows, cols):
            raise ValueError(
                f"Map size mismatch: expected {metadata['size']}, got ({rows}, {cols})"
            )

        # Count map elements
        elements = {}
        for row in map_layout:
            for cell in row:
                elements[cell] = elements.get(cell, 0) + 1

        print(f"   🔍 Map elements: {dict(sorted(elements.items()))}")

        # Check for required elements
        required_elements = ["vR", "GG"]  # Robot start and goal
        for elem in required_elements:
            if elem not in elements:
                print(f"   ⚠️  Warning: Missing required element '{elem}'")

        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def test_curriculum_imports():
    """Test curriculum system imports."""
    print("\n🔧 Testing curriculum system imports...")

    try:
        from curriculum_envs import (
            get_all_env_names,
            get_curriculum_env,
            get_curriculum_map,
        )

        print("   ✅ Core imports successful")

        env_names = get_all_env_names()
        print(f"   📚 Found {len(env_names)} environments: {env_names}")

        # Test each environment through the curriculum system
        for i, env_name in enumerate(env_names):
            name, config = get_curriculum_env(i)
            if name != env_name:
                raise ValueError(f"Environment name mismatch at index {i}")

            map_layout, metadata = get_curriculum_map(env_name)
            print(f"   ✅ {env_name}: {metadata['size']} grid")

        return True

    except Exception as e:
        print(f"   ❌ Curriculum import error: {e}")
        return False


def test_environment_creation():
    """Test creating GridEnvironment instances from hardcoded maps."""
    print("\n🏗️  Testing environment creation...")

    try:
        from curriculum_envs import get_curriculum_env, get_curriculum_map
        from env import GridEnvironment

        for i in range(5):  # Test first 5 environments
            env_name, config = get_curriculum_env(i)
            map_layout, metadata = get_curriculum_map(env_name)

            # Create environment
            env = GridEnvironment(grid_layout=map_layout, grid_metadata=metadata)

            print(f"   ✅ {env_name}: Environment created successfully")
            print(f"      🎯 Human goals: {env.human_goals}")
            print(f"      🤖 Robot agents: {env.robot_agent_ids}")
            print(f"      👥 Human agents: {env.human_agent_ids}")

        return True

    except Exception as e:
        print(f"   ❌ Environment creation error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Testing Hardcoded Curriculum Environments")
    print("=" * 60)

    # Test individual environment modules
    environments = [
        ("Environment 1: Simple", "env_01_simple"),
        ("Environment 2: Basic Door", "env_02_basic_door"),
        ("Environment 3: Multi-Key", "env_03_multi_key"),
        ("Environment 4: Obstacles", "env_04_obstacles"),
        ("Environment 5: Larger Grid", "env_05_larger_grid"),
    ]

    individual_success = True
    for env_name, module_name in environments:
        success = test_single_environment(env_name, module_name)
        individual_success &= success
        print()

    # Test curriculum system
    curriculum_success = test_curriculum_imports()

    # Test environment creation
    creation_success = test_environment_creation()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Individual environments: {'✅' if individual_success else '❌'}")
    print(f"   Curriculum system: {'✅' if curriculum_success else '❌'}")
    print(f"   Environment creation: {'✅' if creation_success else '❌'}")

    overall_success = individual_success and curriculum_success and creation_success
    print(
        f"   Overall: {'✅ All tests passed!' if overall_success else '❌ Some tests failed'}"
    )
    print("=" * 60)

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
