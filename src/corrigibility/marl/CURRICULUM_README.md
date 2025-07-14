# Advanced Curriculum Training System

This system provides a comprehensive curriculum training framework for multi-agent reinforcement learning with progressive difficulty and model weight transfer.

## Key Features

### 🎓 Structured Curriculum

- **5 Progressive Environments**: From simple 4x4 grids to complex 9x9 environments
- **Difficulty Scaling**: Gradual introduction of obstacles, multiple keys/doors, and more agents
- **Clear Learning Objectives**: Each environment targets specific skills

### 🔄 Weight Transfer System

- **Robot Model Continuity**: Robot weights are carried forward between environments
- **Shared Human Models**: Multiple humans in later environments share the same learned weights
- **Position Independence**: Human models work regardless of starting positions

### 💾 Checkpoint Management

- **Automatic Saving**: Checkpoints saved after each environment completion
- **Resume Capability**: Start training from any environment in the sequence
- **Metadata Tracking**: Performance metrics and training configuration stored
- **Version Control**: Complete training history with timestamps

### 🤖 Multi-Agent Support

- **Scalable Human Models**: Same human model works for 1, 2, or 3+ humans
- **Shared Behavior**: All humans use the same learned policy but act independently
- **Consistent Robot Learning**: Robot learns to interact with varying numbers of humans

## Environment Sequence

### Environment 1: Simple 4x4 Grid

- **Agents**: 1 robot, 1 human
- **Objectives**: Basic movement, goal reaching
- **Training**: 300 + 500 episodes (Phase 1 + Phase 2)
- **Skills Learned**: Navigation, basic coordination

### Environment 2: Basic Door/Key System (5x5)

- **Agents**: 1 robot, 1 human
- **Features**: 1 door, 1 key, sequential tasks
- **Training**: 400 + 600 episodes
- **Skills Learned**: Tool usage, sequential planning

### Environment 3: Multi-Key System (6x6)

- **Agents**: 1 robot, 1 human
- **Features**: 2 doors, 2 keys, multiple goals
- **Training**: 500 + 800 episodes
- **Skills Learned**: Multi-step planning, resource management

### Environment 4: Obstacles and Navigation (7x7)

- **Agents**: 1 robot, 2 humans (shared model)
- **Features**: Obstacles, multiple agents, complex coordination
- **Training**: 600 + 1000 episodes
- **Skills Learned**: Obstacle avoidance, multi-agent coordination

### Environment 5: Large Complex Grid (9x9)

- **Agents**: 1 robot, 3 humans (shared model)
- **Features**: Large scale, complex interactions
- **Training**: 800 + 1200 episodes
- **Skills Learned**: Scalability, advanced planning

## Usage

### Full Curriculum Training

```bash
# Train from scratch through all environments
python advanced_curriculum_demo.py --mode=full

# Start from a specific environment (e.g., Environment 3)
python advanced_curriculum_demo.py --mode=full --start-env=2
```

### Single Environment Training

```bash
# Train only on Environment 2
python advanced_curriculum_demo.py --mode=single --env=1

# Train with a specific checkpoint as starting point
python advanced_curriculum_demo.py --mode=single --env=3 --checkpoint=path/to/checkpoint.pkl
```

### Resume Training

```bash
# Resume from a saved checkpoint
python advanced_curriculum_demo.py --mode=resume --checkpoint=curriculum_checkpoints/models/env_02_basic_door_20250702_143021.pkl
```

### List Checkpoints

```bash
# See all available checkpoints
python advanced_curriculum_demo.py --mode=list
```

## Programmatic Usage

### Basic Training

```python
from curriculum_trainer import CurriculumTrainer

# Train full curriculum
trainer = CurriculumTrainer()
final_agent = trainer.train_full_curriculum()
```

### Advanced Usage

```python
from curriculum_trainer import CurriculumTrainer
from curriculum_envs import get_curriculum_env

# Initialize trainer with custom settings
trainer = CurriculumTrainer(
    checkpoint_dir="my_checkpoints",
    log_wandb=True,
    project="my-experiment"
)

# Train on specific environment
env_name, env_config = get_curriculum_env(2)  # Environment 3
agent, performance = trainer.train_single_env(2)

# Resume from checkpoint
agent = trainer.resume_from_checkpoint("path/to/checkpoint.pkl")
```

## Weight Transfer Mechanism

### How It Works

1. **Robot Weights**: Directly transferred between environments (same policy network/Q-table)
2. **Human Weights**: Single human model copied to all humans in target environment
3. **Shared Learning**: All humans in an environment use the same learned behavior
4. **Position Independence**: Human model adapts to different starting positions

### Benefits

- **Faster Training**: No need to relearn basic behaviors
- **Consistent Behavior**: Humans behave consistently across environments
- **Scalability**: Easy to add more humans without retraining
- **Efficiency**: Reduced training time and computational resources

## File Structure

```
curriculum_envs/
├── __init__.py              # Environment registry
├── env_01_simple.py         # Environment 1 definition
├── env_02_basic_door.py     # Environment 2 definition
├── env_03_multi_key.py      # Environment 3 definition
├── env_04_obstacles.py      # Environment 4 definition
└── env_05_larger_grid.py    # Environment 5 definition

curriculum_trainer.py        # Main training orchestrator
checkpoint_manager.py        # Checkpoint and weight transfer system
advanced_curriculum_demo.py  # Demo script and CLI interface
```

## Checkpoint Structure

```
curriculum_checkpoints/
├── models/                  # Saved model files
│   ├── env_01_simple_timestamp.pkl
│   ├── env_01_simple_timestamp_human_q_m.pkl
│   └── env_01_simple_timestamp_robot_q.pkl
├── metadata/               # Training metadata
│   └── env_01_simple_timestamp_metadata.json
└── logs/                   # Training logs
    └── checkpoint_log.jsonl
```

## Key Concepts

### Two-Phase Training

1. **Phase 1**: Learn conservative human models (cautious behavior)
2. **Phase 2**: Learn robot policy that optimizes for robot objectives

### Shared Human Models

- Multiple humans in later environments use the same learned Q-networks/tables
- Enables consistent behavior across agents
- Reduces training complexity and time

### Progressive Difficulty

- Each environment builds on skills from previous environments
- Gradual introduction of complexity prevents catastrophic forgetting
- Weight transfer ensures knowledge retention

## Monitoring and Logging

### Weights & Biases Integration

- Automatic logging of training progress
- Performance metrics across environments
- Curriculum progression tracking

### Local Logging

- Checkpoint metadata with performance metrics
- Training history and timestamps
- Resume capability from any point

## Customization

### Adding New Environments

1. Create new environment file in `curriculum_envs/`
2. Define `get_env_config()` function with environment parameters
3. Add to `CURRICULUM_SEQUENCE` in `__init__.py`

### Modifying Training Parameters

- Edit environment config files to adjust training episodes
- Modify agent configuration in `curriculum_trainer.py`
- Customize success thresholds and evaluation criteria

This system provides a robust foundation for curriculum learning in multi-agent environments with proper weight management and scalability.
