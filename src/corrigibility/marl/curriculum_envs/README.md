# Curriculum Environments

This document describes the hardcoded environment configurations for the MARL curriculum training system. Each environment represents a specific learning stage with increasing complexity.

## Environment Overview

The curriculum consists of 5 environments with progressive difficulty:

1.  **env_01_simple.py** - Simple 4x4 grid with basic robot-human interaction.
2.  **env_02_basic_door.py** - 5x5 grid introducing doors and keys.
3.  **env_03_multi_key.py** - 7x7 grid with multiple keys and doors.
4.  **env_04_obstacles.py** - 7x7 grid with obstacles.
5.  **env_05_larger_grid.py** - 9x9 grid with large-scale coordination challenges.

## Map Legend

- `##`: Wall
- `  `: Empty space
- `vR0`, `vR1`: Robot starting positions
- `vH0`, `vH1`: Human starting positions
- `BD`, `YD`, `RD`: Blue, Yellow, Red Doors
- `BK`, `YK`, `RK`: Blue, Yellow, Red Keys
- `OL`: Lava/Obstacle
- `GG`: Goal

## Quick Start

### 1. Run Full Curriculum Training

To train the agent through the entire curriculum from scratch, run the main training script:

```bash
# Run the full curriculum training from the project root
python -m src.corrigibility.marl.advanced_curriculum_demo --mode=full
```

This will train agents through all 5 environments sequentially. Progress, including model weights and training metrics, is saved via a checkpointing system.

### 2. List Available Checkpoints

To see a list of all saved checkpoints:

```bash
python -m src.corrigibility.marl.advanced_curriculum_demo --mode=list
```

## Training with Checkpoints

### Checkpoint System

The system automatically saves checkpoints after each environment stage is successfully completed.

- **Directory**: Checkpoints are saved in the `curriculum_checkpoints/models/` directory.
- **Naming**: Checkpoints are named based on the environment and timestamp, e.g., `env_01_simple_20250703_093552.pkl`.

### Checkpoint Structure

Yes, checkpoints store the model weights, along with other data. For each completed environment, the system saves multiple files:

- **`.pkl` file**: This is the main checkpoint file. It's a pickle file containing the agent's overall state, including training parameters, hyperparameters, and performance metrics.
- **`.pt` files**: These are PyTorch files that store the neural network weights for the different models learned by the agent. You will see separate files for the robot's Q-network (`_robot_q.pt`) and the human's Q-networks (`_human_q_m.pt`, `_human_q_e.pt`).

This separation allows for flexible loading and analysis of the agent's learned knowledge.

### Resume Training

You can resume a training run from the last completed checkpoint. This is useful if the training was interrupted.

```bash
# Find the latest checkpoint file in curriculum_checkpoints/models/
CHECKPOINT_PATH="curriculum_checkpoints/models/env_03_multi_key_20250703_094948.pkl"

# Resume the curriculum from the specified checkpoint
python -m src.corrigibility.marl.advanced_curriculum_demo --mode=resume --checkpoint $CHECKPOINT_PATH
```

The script will load the agent from the checkpoint and continue training from the next environment in the sequence.

## Visualization

### Visualize Trained Agents

Use the visualization script to see how a trained agent behaves in its environment.

```bash
# Find the checkpoint you want to visualize
CHECKPOINT_PATH="curriculum_checkpoints/models/env_01_simple_20250703_093552.pkl"

# Visualize environment 0 (env_01_simple) using its checkpoint
python -m src.corrigibility.marl.visualize_curriculum --env 0 --checkpoint $CHECKPOINT_PATH

# Visualize environment 3 (env_04_obstacles) using its checkpoint
CHECKPOINT_PATH_ENV4="curriculum_checkpoints/models/env_04_obstacles_20250703_101533.pkl"
python -m src.corrigibility.marl.visualize_curriculum --env 3 --checkpoint $CHECKPOINT_PATH_ENV4
```

### Visualization Options

- `--env <index>`: The numerical index of the environment to visualize (0-4). (Required)
- `--checkpoint <path>`: Path to the checkpoint file (`.pkl`) for the agent you want to load. (Required)
- `--episodes <N>`: Number of episodes to visualize (default: 3).
- `--delay <ms>`: Delay in milliseconds between visualization steps (default: 500).

## Curriculum Training and Visualization

This project uses a curriculum-based training approach to teach a robot to assist a human in a series of increasingly complex grid-world environments. The system is designed to be modular and extensible, allowing for the easy addition of new environments and training scenarios.

### Checkpoint Structure

The training process generates checkpoints at the end of each curriculum stage. These checkpoints are saved in the root directory and are named according to the stage they correspond to:

- `q_values_stage_1.pkl`: Checkpoint for the first environment.
- `q_values_stage_2.pkl`: Checkpoint for the second environment.
- ...and so on.

Each checkpoint contains the learned Q-values for both the robot and human agents, as well as the agent configuration.

### Correct Usage

To run the curriculum training, execute the following command:

```bash
python -m src.corrigibility.marl.curriculum_train
```

This will train the agent through the entire curriculum, saving a checkpoint at the end of each stage. To visualize the performance of a trained agent in a specific environment, use the `visualize_curriculum.py` script. You must provide the path to the environment configuration and the corresponding checkpoint file.

For example, to visualize the agent's performance in the third environment, run:

```bash
python -m src.corrigibility.marl.visualize_curriculum --env_path src/corrigibility/marl/curriculum_envs/env_03_multi_key.py --checkpoint q_values_stage_3.pkl
```

### Advanced Simulation Scenarios

For more in-depth analysis of agent behavior, you can run simulations under different assumptions about the human's policy. This helps test the robot's robustness and adaptability.

The following scenarios can be implemented by modifying the visualization or evaluation scripts to control the human's action selection logic:

1.  **Human Uses Learned Model (V_m)**:

    - **Scenario**: The human acts according to the policy derived from `V_m`, which is the conservative model the robot learned for the human for a specific goal.
    - **Purpose**: This is the "expected" behavior. It validates that the robot's policy is effective when the human acts as predicted.

2.  **Human Uses Best Response to Robot**:

    - **Scenario**: The human uses a newly learned best-response policy against the robot's final, fixed policy (`pi_r`) for a known goal.
    - **Purpose**: Tests the robot's performance against a more optimal, less conservative human, revealing if the robot's policy is exploitable.

3.  **Human Has Unforeseen Goal**:

    - **Scenario**: The human acts based on a best-response policy for a goal that was _not_ in the robot's set of considered goals during training.
    - **Purpose**: Evaluates the robot's ability to generalize and act safely when its model of the human's intentions is incorrect.

4.  **Human Acts Randomly**:
    - **Scenario**: The human ignores its goals and takes completely random actions.
    - **Purpose**: A stress test to see if the robot's policy remains safe and avoids catastrophic failures when faced with highly unpredictable behavior.

## Environment Details

### Environment 1: Simple Grid (4x4)

- **Agents**: 1 robot, 1 human
- **Objective**: Basic cooperation and goal reaching
- **Complexity**: Low
- **Expected training time**: ~100-200 episodes

### Environment 2: Basic Door (5x5)

- **Agents**: 1 robot, 1 human
- **Features**: Door/key mechanics
- **Objective**: Sequential task coordination
- **Complexity**: Medium-Low
- **Expected training time**: ~200-400 episodes

### Environment 3: Multi Key (7x7)

- **Agents**: 1 robot, 2 humans
- **Features**: Multiple doors and keys
- **Objective**: Complex resource management
- **Complexity**: Medium
- **Expected training time**: ~400-600 episodes

### Environment 4: Obstacles (7x7)

- **Agents**: 1 robot, 2 humans
- **Features**: Lava obstacles, boxes
- **Objective**: Navigation under constraints
- **Complexity**: Medium-High
- **Expected training time**: ~600-800 episodes

### Environment 5: Larger Grid (9x9)

- **Agents**: 1 robot, 3 humans
- **Features**: Large-scale coordination
- **Objective**: Scalable multi-agent planning
- **Complexity**: High
- **Expected training time**: ~800-1200 episodes

## Success Criteria

**Success is defined as all humans reaching their designated goals**, not just maximizing reward. This ensures that the RL system learns meaningful cooperation rather than reward hacking.

### Monitoring Success

- Training logs show success rates per environment
- Checkpoints save success history and convergence metrics
- Visualization clearly shows when humans reach their goals
- Success thresholds are defined per environment (typically 0.6-0.8)

## File Structure

```
curriculum_envs/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── env_01_simple.py            # Environment 1: Simple navigation
├── env_02_basic_door.py        # Environment 2: Basic door mechanics
├── env_03_multi_key.py         # Environment 3: Multiple keys/doors
├── env_04_obstacles.py         # Environment 4: Obstacles and boxes
└── env_05_larger_grid.py       # Environment 5: Large complex grid
```

## Environment Configuration Structure

Each environment file contains:

1. **SIMPLE_MAP**: Hardcoded 2D grid layout
2. **MAP_METADATA**: Environment properties and constraints
3. **get_map()**: Returns map layout and metadata
4. **get_env_config()**: Returns training configuration
5. **get_expected_performance()**: Expected performance metrics

### Example Environment Configuration

```python
# Map layout
SIMPLE_MAP = [
    ["##", "##", "##", "##", "##"],
    ["##", "vR0", "  ", "BK", "##"],
    ["##", "  ", "BD", "  ", "##"],
    ["##", "vH0", "  ", "GG", "##"],
    ["##", "##", "##", "##", "##"],
]

# Metadata
MAP_METADATA = {
    "name": "Basic Door Environment",
    "description": "Simple door and key mechanics",
    "size": (5, 5),
    "max_steps": 50,
    "human_goals": {"human_0": (3, 3)},
    "num_robots": 1,
    "num_humans": 1,
    "difficulty": 2,
}
```

## Training Configuration

Each environment defines training parameters:

- **phase1_episodes**: Initial training episodes
- **phase2_episodes**: Extended training episodes
- **success_threshold**: Minimum success rate to advance
- **max_retries**: Maximum retry attempts per environment

## Success Criteria

Training success is measured by:

1. **Human Goal Achievement**: Humans must reach their designated goals
2. **Success Rate**: Percentage of episodes where goals are achieved
3. **Convergence**: Stable performance over evaluation episodes
4. **Coordination Metrics**: Quality of robot-human cooperation

## Validation and Testing

### Validate Environments

```bash
# Validate all environments
python validate_curriculum_environments.py

# Test specific environment
python test_hardcoded_envs.py --env env_02_basic_door
```

### Performance Monitoring

The system tracks:

- Episode success rates
- Convergence metrics
- Human goal achievement rates
- Robot helping behavior scores
- Training time and efficiency

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the MARL directory
2. **Checkpoint Loading**: Check checkpoint file paths and permissions
3. **Visualization Issues**: Verify matplotlib and rendering dependencies
4. **Memory Issues**: Reduce batch sizes for large environments

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed output
trainer = CurriculumTrainer(debug=True)
```

## Advanced Features

### Custom Environment Creation

To create a new curriculum environment:

1. Copy an existing environment file
2. Modify the `SIMPLE_MAP` layout
3. Update `MAP_METADATA` with new parameters
4. Adjust training configuration in `get_env_config()`
5. Add to curriculum sequence in trainer

### Performance Tuning

Adjust hyperparameters:

```python
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "target_update_freq": 1000,
    "exploration_decay": 0.995,
    "memory_size": 10000,
}
```

## Dependencies

Required packages:

- numpy
- torch
- matplotlib (for visualization)
- wandb (for logging, optional)

Install with:

```bash
pip install numpy torch matplotlib wandb
```

## Contributing

When adding new environments:

1. Follow the naming convention: `env_XX_description.py`
2. Include comprehensive docstrings
3. Test with validation scripts
4. Update difficulty progression logically
5. Document learning objectives clearly

## License

This code is part of the Satisfia project. See the main project LICENSE for details.
