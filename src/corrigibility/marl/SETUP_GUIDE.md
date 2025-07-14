# MARL Setup Guide

This guide provides a quick overview of how to get started with the Multi-Agent Reinforcement Learning (MARL) implementation.

## Quick Start

### 1. Setup Verification
First, verify that everything is working correctly:

```bash
cd src/corrigibility/marl
python test_setup.py
```

You should see:
```
✅ All imports successful! MARL setup is working correctly.
✅ All functionality tests passed!
```

### 2. Run Basic Test
```bash
python test_quick_iql.py
```

This runs a fast test with minimal training episodes to verify the IQL algorithm works.

### 3. Run Visual Demo (if you have a display)
```bash
python visualize_iql_tabular.py
```

For headless testing:
```bash
python visualize_iql_tabular.py --no-render
```

### 4. Detailed Debugging
```bash
python debug_detailed_iql.py
```

This shows step-by-step training with Q-value updates and agent reasoning.

## File Organization

All test, debug, and visualization files are now organized in the MARL directory:

### Test Scripts
- `test_setup.py` - Basic setup verification
- `test_quick_iql.py` - Fast functionality test
- `test_iql_tabular.py` - Comprehensive test suite
- `test_iql_integration.py` - Integration tests
- `test_iql_modular.py` - Modular tests

### Visualization
- `visualize_iql_tabular.py` - Interactive visualization
- `visualize_curriculum.py` - Curriculum visualization

### Debugging
- `debug_detailed_iql.py` - Detailed step-by-step debugging
- `debug_movement_new.py` - Movement mechanics debugging
- `debug_environment.py` - Environment debugging
- `debug_training.py` - Training process debugging

### Movement Tests
- `test_basic_movement.py` - Basic movement tests
- `test_goal_reaching.py` - Goal reaching tests
- `test_robot_cooperation.py` - Robot cooperation tests

## Expected Results

When everything is working correctly, you should see:

✅ **Robot Behaviors**:
- Picks up keys (action 3: "pickup")
- Opens doors (action 5: "toggle")
- Moves strategically (actions 0,1,2: "left", "right", "forward")

✅ **Human Behaviors**:
- Learns goal-directed movement
- Uses cautious policies learned in Phase 1

✅ **Training Process**:
- Phase 1: Robot blocks, human learns cautious policies
- Phase 2: Robot assists, enabling goal achievement
- Q-values converge over episodes

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the MARL directory:
```bash
pwd
# Should show: .../satisfia/src/corrigibility/marl
```

### Display Issues
For headless systems, use the `--no-render` flag or run tests that don't require visualization:
```bash
python test_quick_iql.py
python debug_detailed_iql.py
```

### Low Success Rates
This is expected in the current simple maps due to environment constraints. The algorithm is working correctly - the robot learns assistive behaviors and the human learns goal-directed policies.

## Next Steps

1. **Read the full documentation**: [README_MARL.md](README_MARL.md)
2. **Explore different maps**: Check `envs/simple_map*.py`
3. **Try curriculum learning**: `python curriculum_train.py`
4. **Experiment with parameters**: Modify learning rates, episodes, etc.

## Support

If you encounter issues:
1. Run `python test_setup.py` to verify basic functionality
2. Check the detailed documentation in `README_MARL.md`
3. Use the debug scripts to understand what's happening step-by-step