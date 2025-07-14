#!/usr/bin/env python3
"""
Advanced Curriculum Training Demo

This script demonstrates the new curriculum training system with:
- Progressive training through defined environments
- Checkpoint management and weight transfer
- Support for multiple humans with shared models
- Resume capability from any checkpoint

Usage:
    python -m src.corrigibility.marl.advanced_curriculum_demo --mode=full           # Train full curriculum
    python -m src.corrigibility.marl.advanced_curriculum_demo --mode=single --env=2 # Train single environment
    python -m src.corrigibility.marl.advanced_curriculum_demo --mode=resume --checkpoint=path/to/checkpoint.pkl
    python -m src.corrigibility.marl.advanced_curriculum_demo --mode=list          # List available checkpoints
"""

import argparse
from pathlib import Path

from .curriculum_envs import CURRICULUM_SEQUENCE
from .curriculum_trainer import CurriculumTrainer


def main():
    parser = argparse.ArgumentParser(description="Advanced Curriculum Training Demo")
    parser.add_argument(
        "--mode",
        choices=["full", "single", "resume", "list"],
        default="full",
        help="Training mode",
    )
    parser.add_argument(
        "--env", type=int, default=0, help="Environment index for single mode (0-based)"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Checkpoint path for resume mode"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="curriculum_checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--start-env",
        type=int,
        default=0,
        help="Starting environment index for full mode",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = CurriculumTrainer(
        checkpoint_dir=args.checkpoint_dir,
        log_wandb=not args.no_wandb,
        project="robot-curriculum-advanced",
    )

    print("🎓 Advanced Curriculum Training System")
    print("=" * 50)

    if args.mode == "list":
        print("📁 Listing available checkpoints...")
        trainer.list_available_checkpoints()

    elif args.mode == "full":
        print(f"🚀 Starting full curriculum training from environment {args.start_env}")
        print("📚 Available environments:")
        for i, (name, _) in enumerate(CURRICULUM_SEQUENCE):
            status = "🎯" if i >= args.start_env else "✅"
            print(f"   {status} {i}: {name}")

        try:
            final_agent = trainer.train_full_curriculum(start_env_index=args.start_env)
            print("🎉 Curriculum training completed successfully!")

            # Save final agent
            final_checkpoint = trainer.checkpoint_manager.save_checkpoint(
                final_agent,
                "curriculum_complete",
                {"stage": "final", "curriculum_complete": True},
                {"final_training": True},
            )
            print(f"💾 Final agent saved to: {final_checkpoint}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return 1

    elif args.mode == "single":
        if args.env < 0 or args.env >= len(CURRICULUM_SEQUENCE):
            print(
                f"❌ Invalid environment index {args.env}. Must be 0-{len(CURRICULUM_SEQUENCE)-1}"
            )
            return 1

        env_name = CURRICULUM_SEQUENCE[args.env][0]
        print(f"🎯 Training single environment: {env_name} (index {args.env})")

        try:
            agent, performance = trainer.train_single_env(args.env, args.checkpoint)
            print("✅ Training completed!")
            print(f"📊 Final performance: {performance}")

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return 1

    elif args.mode == "resume":
        if not args.checkpoint:
            print("❌ Checkpoint path required for resume mode")
            return 1

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return 1

        print(f"📁 Resuming training from: {checkpoint_path}")

        try:
            final_agent = trainer.resume_from_checkpoint(str(checkpoint_path))
            print("✅ Resume training completed!")

        except Exception as e:
            print(f"❌ Resume failed: {e}")
            return 1

    print("🏁 Demo completed!")
    return 0


if __name__ == "__main__":
    exit(main())
