# satisfia

Intelligent agents that satisfy aspirations. This is the main repository of code produced in the [SatisfIA project](https://pik-gane.github.io/satisfia/).

![image](https://github.com/pik-gane/satisfia/assets/22815964/d2c2e297-eb51-4cef-8989-875e8bf20719)

## Requirements

tested with:

- python 3.10.0
- gymnasium 0.29.0
- pygame 2.5.0
- mujoco-py 2.1.2.14

## Getting started

### Multi-Agent Reinforcement Learning (MARL)

For the IQL Timescale Algorithm and cooperative human-robot environments:

```bash
cd src/corrigibility/marl
python test_quick_iql.py        # Quick functionality test
python visualize_iql_tabular.py # Visual demonstration
```

See the [MARL README](src/corrigibility/marl/README_MARL.md) for comprehensive documentation.

### Running around a very simple gridworld and a larger random gridworld:

```
python scripts/test_simple_gridworld.py
```

### A GUI for exploring parameters:

```
python scripts/explore_parameters.py
```

### Adding more gridworlds:

Open `src/environments/very_simple_gridworlds.py` and add another `elif` section.

### Running the IQL Timescale Algorithm

You can train and visualize agents using the `run_iql.py` script.

#### Training

**Tabular (default):**
To train a tabular agent, run:

```bash
python3 run_iql.py --mode train --save tabular_q_values.pkl --phase1-episodes 100 --phase2-episodes 100
```

**Neural Network:**
To train a neural network agent, add the `--network` flag:

```bash
python3 run_iql.py --mode train --save network_q_values --network --phase1-episodes 100 --phase2-episodes 100
```

_Note: For network mode, the model files will be saved with `.pt` extensions, so you only need to provide the base name._

#### Visualization

**Tabular:**
To visualize a trained tabular agent:

```bash
python3 run_iql.py --mode visualize --load tabular_q_values.pkl
```

**Neural Network:**
To visualize a trained network agent:

```bash
python3 run_iql.py --mode visualize --load network_q_values --network
```
