# Multi-Agent Reinforcement Learning (MARL)

This directory contains the implementation of the **IQL Timescale Algorithm** for multi-agent reinforcement learning in cooperative environments with human-robot interaction.

## Overview

The IQL (Independent Q-Learning) Timescale Algorithm is designed for scenarios where:
- **Humans** have goals they want to achieve
- **Robots** can either block or assist humans
- The environment contains puzzles (e.g., key-door mechanics) requiring cooperation

### Key Features

- **Two-Phase Training**:
  - **Phase 1**: Robot learns to block human, human learns cautious policies
  - **Phase 2**: Robot learns to assist human, enabling goal achievement
- **Goal-Conditioned Learning**: Humans learn policies conditioned on their goals
- **Tabular and Neural Network backends**: Support for both Q-tables and deep Q-networks
- **Assistive Behaviors**: Robots learn to collect keys, open doors, and help humans navigate

## Quick Start

### 1. Basic Test (No Visualization)
```bash
cd src/corrigibility/marl
python test_quick_iql.py
```

### 2. Full Test Suite (All Simple Maps)
```bash
cd src/corrigibility/marl
python test_iql_tabular.py
```

### 3. Visual Demo (with Pygame)
```bash
cd src/corrigibility/marl
python visualize_iql_tabular.py
```

### 4. Detailed Debugging
```bash
cd src/corrigibility/marl
python debug_detailed_iql.py
```

## File Structure

```
src/corrigibility/marl/
├── README_MARL.md                      # This file
├── __init__.py                         # Package initialization
│
# Core Algorithm
├── iql_timescale_algorithm.py          # Main IQL implementation
├── q_learning_backends.py              # Q-learning backends (tabular/neural)
│
# Environment
├── env.py                              # Main environment (CustomEnvironment)
├── objects.py                          # Game objects (walls, keys, doors, etc.)
├── rendering_utils.py                  # Pygame rendering utilities
├── state_encoder.py                    # State encoding for neural networks
│
# Maps and Environments
├── envs/
│   ├── __init__.py
│   ├── map_loader.py                   # Map loading utilities
│   ├── simple_map.py                   # Simple locking door map
│   ├── simple_map2.py                  # Variant 2
│   ├── simple_map3.py                  # Variant 3
│   ├── simple_map4.py                  # Variant 4
│   └── auto_env.py                     # Auto-generated environments
│
# Training and Utilities
├── main.py                             # Main training script
├── trained_agent.py                    # Trained agent utilities
├── deterministic_algorithm.py          # Deterministic baseline
│
# Testing and Visualization
├── test_iql_tabular.py                 # Comprehensive test suite
├── test_quick_iql.py                   # Quick functionality test
├── visualize_iql_tabular.py            # Visual demonstration
├── debug_detailed_iql.py               # Detailed debugging
├── debug_movement_new.py               # Movement debugging
│
# Curriculum Learning (Advanced)
├── curriculum_envs/                    # Curriculum learning environments
├── curriculum_train.py                 # Curriculum training
├── curriculum_trainer.py               # Curriculum trainer class
├── checkpoint_manager.py               # Model checkpointing
├── advanced_curriculum_demo.py         # Advanced curriculum demo
├── simple_curriculum_runner.py         # Simple curriculum runner
├── test_curriculum_system.py           # Curriculum system tests
└── validate_curriculum_environments.py # Environment validation
```

## Algorithm Details

### IQL Timescale Algorithm

The algorithm maintains several Q-functions:

1. **Q_m (Human Cautious Model)**: Learned during Phase 1 when robot blocks
2. **Q_e (Human Effective Model)**: Learned during Phase 2 when robot assists  
3. **Q_r (Robot Model)**: Robot's policy for maximizing human power/goal achievement
4. **Power Estimation**: Estimates human's ability to achieve goals

### Training Process

#### Phase 1: Learning Cautious Policies
```python
# Robot blocks human movement
for episode in phase1_episodes:
    robot_action = block_human()
    human_action = sample_from_Q_m(state, goal)
    # Update Q_m based on experience
```

#### Phase 2: Learning Assistive Policies  
```python
# Robot assists human
for episode in phase2_episodes:
    robot_action = sample_from_Q_r(state)  # or assistive_heuristic()
    human_action = sample_from_Q_m(state, goal)  # Use cautious policy
    # Update Q_r and Q_e based on experience
```

## Environment Details

### Simple Maps
The environment features **locking door puzzles**:

```
#####
#RKH#  # R=Robot, K=Key, H=Human  
##D##  # D=Door (locked)
# G #  # G=Goal
#####
```

**Objective**: Robot must collect the key and open the door so the human can reach the goal.

### Actions
- **Human Actions**: `[turn_left, turn_right, forward]`
- **Robot Actions**: `[turn_left, turn_right, forward, pickup, drop, toggle]`

### Mechanics
- **Movement**: Agents must turn to face direction, then move forward
- **Key Collection**: Only robots can pick up keys
- **Door Opening**: Only robots can toggle doors (requires matching key)
- **Goal**: Human reaches the goal position

## Configuration Options

### Algorithm Parameters
```python
TwoPhaseTimescaleIQL(
    alpha_m=0.1,      # Learning rate for Q_m (cautious model)
    alpha_e=0.1,      # Learning rate for Q_e (effective model)  
    alpha_r=0.1,      # Learning rate for Q_r (robot model)
    alpha_p=0.1,      # Learning rate for power estimation
    gamma_h=0.99,     # Discount factor for human
    gamma_r=0.99,     # Discount factor for robot
    beta_r_0=5.0,     # Robot rationality parameter
    network=False,    # Use tabular (False) or neural network (True)
    # ... other parameters
)
```

### Environment Parameters
```python
CustomEnvironment(
    grid_layout=map_layout,        # Map layout from simple_map.py
    grid_metadata=map_metadata,    # Map metadata (goals, size, etc.)
    render_mode="human",           # "human" for visual, None for headless
    debug_mode=True,              # Enable debug logging
)
```

## Testing and Validation

### Test Scripts

1. **`test_quick_iql.py`**: Fast test with minimal episodes
   - Trains for 50+100 episodes (Phase 1+2)
   - Tests on 10 episodes
   - Good for quick validation

2. **`test_iql_tabular.py`**: Comprehensive test suite
   - Tests all 4 simple maps
   - Multiple trials per map
   - Statistical success rate analysis

3. **`visualize_iql_tabular.py`**: Visual demonstration
   - Shows training process
   - Step-by-step episode visualization
   - Real-time rendering with Pygame

4. **`debug_detailed_iql.py`**: Detailed debugging
   - Shows Q-value updates
   - Prints agent actions and reasoning
   - Grid state visualization

### Success Metrics
- **Goal Achievement**: Human reaches goal position
- **Success Rate**: Percentage of episodes where goal is reached
- **Training Convergence**: Q-values stabilize over episodes

## Advanced Features

### Curriculum Learning
For complex multi-environment training:

```bash
cd src/corrigibility/marl
python curriculum_train.py
```

### Model Checkpointing
Save and load trained models:

```python
# Save models
iql.save_models("checkpoints/my_model")

# Load models  
iql.load_models("checkpoints/my_model")
```

### Neural Network Backend
Enable deep Q-learning:

```python
iql = TwoPhaseTimescaleIQL(
    network=True,           # Enable neural networks
    state_dim=64,          # State dimension
    hidden_sizes=[128, 64], # Network architecture
    # ... other parameters
)
```

## Troubleshooting

### Common Issues

1. **Agents not moving**: Check action space and environment setup
2. **Low success rates**: Increase training episodes or adjust learning rates
3. **Pygame display issues**: Use `--no-render` flag for headless testing
4. **Memory issues**: Use tabular backend (`network=False`) for debugging

### Debug Commands

```bash
# Test movement mechanics
python debug_movement_new.py

# Detailed algorithm debugging  
python debug_detailed_iql.py

# Quick functionality check
python test_quick_iql.py --no-render
```

### Expected Results

✅ **Successful Training Should Show**:
- Robot collects keys and opens doors
- Human learns goal-directed movement
- Success rate > 0% after training
- Q-values converge during training

❌ **Common Failure Modes**:
- Agents stuck in starting positions (environment issue)
- Random actions (insufficient training)
- 0% success rate (algorithm or environment misconfiguration)

## Research Applications

This implementation supports research in:
- **Cooperative AI**: Human-robot interaction and assistance
- **Corrigibility**: Robot behavior alignment with human preferences  
- **Multi-agent RL**: Independent learning in shared environments
- **Goal-conditioned RL**: Learning policies for multiple objectives

## Citation

If you use this implementation in your research, please cite:
```bibtex
@software{iql_timescale_marl,
  title={IQL Timescale Algorithm for Multi-Agent Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/satisfia}
}
```

## Contributing

1. **Bug Reports**: File issues with reproducible examples
2. **Feature Requests**: Propose new environments or algorithm variants
3. **Pull Requests**: Follow the existing code style and include tests

## License

This project is licensed under the MIT License - see the LICENSE file for details.