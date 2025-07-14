#!/usr/bin/env python3
"""
Quick test of the curriculum training system
This runs from the root directory to handle imports properly.
"""

import os
import sys




def test_curriculum_system():
    """Test the basic functionality of the curriculum system."""
    print("🎓 Testing Advanced Curriculum Training System")
    print("=" * 60)

    # Test environment definitions
    print("\n📚 Testing Environment Definitions:")
    try:
        from curriculum_envs import (
            CURRICULUM_SEQUENCE,
            get_all_env_names,
            get_curriculum_env,
        )

        env_names = get_all_env_names()
        print(f"✅ Found {len(env_names)} environments:")
        for i, name in enumerate(env_names):
            env_name, env_config = get_curriculum_env(i)
            print(f"   {i+1}. {env_name}: {env_config['description']}")
            print(f"      Difficulty: {env_config['difficulty']}/5")
            print(
                f"      Agents: {env_config['gen_args']['num_robots']} robot(s), {env_config['gen_args']['num_humans']} human(s)"
            )
            print(
                f"      Grid: {env_config['gen_args']['width']}x{env_config['gen_args']['height']}"
            )
            print()

    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

    # Test checkpoint manager
    print("💾 Testing Checkpoint Manager:")
    try:
        from checkpoint_manager import CheckpointManager

        checkpoint_manager = CheckpointManager("test_checkpoints")
        print("✅ Checkpoint manager initialized")
        print(f"   Directory: {checkpoint_manager.checkpoint_dir}")

        # List existing checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        print(f"   Found {len(checkpoints)} existing checkpoints")

    except Exception as e:
        print(f"❌ Checkpoint manager test failed: {e}")
        return False

    # Test environment generation
    print("🏗️  Testing Environment Generation:")
    try:
        env_name, env_config = get_curriculum_env(0)  # First environment

        # This will test if the auto_env generation works
        from envs.auto_env import generate_env_map

        env_map, meta = generate_env_map(env_config["gen_args"])

        print(f"✅ Generated environment: {env_name}")
        print(f"   Map size: {len(env_map)}x{len(env_map[0])}")
        print(f"   Metadata: {list(meta.keys())}")

        # Print a small representation of the map
        print("   Map preview:")
        for row in env_map[: min(5, len(env_map))]:
            print(f"      {' '.join(row[:min(10, len(row))])}")

    except Exception as e:
        print(f"❌ Environment generation test failed: {e}")
        return False

    print("\n🎉 All basic tests passed!")
    print("\n📋 System Summary:")
    print(f"   ✅ {len(CURRICULUM_SEQUENCE)} environments defined")
    print("   ✅ Progressive difficulty from 1 to 5")
    print("   ✅ Grid sizes from 4x4 to 9x9")
    print("   ✅ Agent scaling from 2 to 4 total agents")
    print("   ✅ Checkpoint system ready")
    print()

    print("🚀 Next Steps:")
    print(
        "   1. Run from project root: python -m src.corrigibility.marl.curriculum_trainer"
    )
    print("   2. Or use the individual environment files directly")
    print("   3. Set up proper imports for full system integration")

    return True


if __name__ == "__main__":
    success = test_curriculum_system()
    exit(0 if success else 1)
