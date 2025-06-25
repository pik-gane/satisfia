### Command Examples

**Training with Tabular Q-learning (Default):**

```bash
python main.py --mode train --phase1-episodes 1000 --phase2-episodes 1000 --save saved/q_values.pkl --map simple_map2

python main.py --mode train --phase1-episodes 1000 --phase2-episodes 1000 --save saved/q_values.pkl --map team_map
```

**Training with Neural Network Q-learning:**

```bash
python src/corrigibility/marl/main.py --mode train --algorithm timescale --map team_map --save network_agent.pkl --network

python src/corrigibility/marl/main.py --mode visualize --load network_agent.pkl --map team_map --network


python src/corrigibility/marl/main.py --mode train --algorithm timescale --map simple_map2 --save network_agent.pkl --network

python src/corrigibility/marl/main.py --mode visualize --load network_agent.pkl --map simple_map2 --network
```

**Visualization:**

```bash
python main.py --mode visualize --load saved/q_values.pkl --map simple_map --delay 200

python main.py --mode visualize --load saved/q_values_simple_map4.pkl --map simple_map4 --delay 200

# Visualize neural network trained model
python main.py --mode visualize --load saved/q_values_nn.pkl --map simple_map2 --delay 200 --network --state-dim 4
```

# Two-Timescale Goal-Based IQL in a Custom Grid Environment

## Overview

This project implements a Two-Timescale Goal-Based Independent Q-Learning (IQL) algorithm in a custom grid environment. The environment is designed using the `PettingZoo` library and rendered with `pygame`. The agents in the environment are a **robot** and a **human**, and the robot's goal is to maximize the human's potential to achieve various goals.

The algorithm follows a two-phase approach:

- **Phase 1**: Learn cautious human behavior models while the robot acts pessimistically
- **Phase 2**: Learn an optimal robot policy using the learned human models

## Environment Description

The environment is a grid-based world with the following elements:

- **Walls (`#`)**: Impassable obstacles.
- **Goal (`G`)**: The target location for the human.
- **Key (`K`)**: An object that may be required to unlock doors.
- **Door (`D`)**: A barrier that can be opened with a key.
- **Lava (`L`)**: Hazardous tiles that terminate the episode if stepped on.
- **Robot (`R`)**: The learning agent controlled by the IQL algorithm.
- **Human (`H`)**: A simulated agent with predefined behavior.

## Actions

The environment supports the following actions, similar to MiniGrid:

| Action | Name   | Description               |
| ------ | ------ | ------------------------- |
| 0      | Left   | Move left                 |
| 1      | Right  | Move right                |
| 2      | Up     | Move up                   |
| 3      | Down   | Move down                 |
| 4      | Pickup | Pick up an object         |
| 5      | Drop   | Drop an object            |
| 6      | Toggle | Toggle/activate an object |
| 7      | No-op  | Do nothing                |

### Objects

The environment includes the following objects, rendered similarly to MiniGrid:

- **Walls**: Impassable obstacles.
- **Goal**: The target location for the agent.
- **Key**: An object required to unlock doors.
- **Door**: A barrier that can be opened with a key.
- **Lava**: Hazardous tiles that terminate the episode if stepped on.
- **Ball**: A movable object.
- **Box**: A container that may hold other objects.

## Possible Actions

The environment supports the following actions:

- move_up
- move_down
- move_left
- move_right

### Rendering

The environment is rendered using `pygame`. The grid elements are displayed with distinct colors:

- **Walls**: Black
- **Goal**: Green
- **Key**: Yellow
- **Door**: Brown
- **Lava**: Red
- **Robot**: Blue
- **Human**: Magenta

## Algorithm

The Two-Timescale Goal-Based Independent Q-Learning (IQL) algorithm is implemented with the following key features:

Add multiple Envs as well

### Learning Backends

The algorithm supports two learning backends:

1. **Tabular Q-Learning (Default)**: Uses lookup tables for Q-values and policies

   - Suitable for small discrete state spaces
   - Fast convergence for simple environments
   - Complete policy tracking and analysis

2. **Neural Network Q-Learning**: Uses PyTorch neural networks for Q-value approximation
   - Suitable for large or continuous state spaces
   - Scalable to complex environments
   - Automatic feature learning

### Phase 1: Learning Human Behavior Models

- **Human Policy**: ε-greedy with ε_h decreasing from 0.5 to ε_h_0
- **Robot Policy**: Pessimistic policy that assumes worst-case human outcomes
- **Goal**: Learn conservative estimates of human Q-values Q_h^m
- **Updates**: Conservative Q-learning for humans, pessimistic action selection for robot

### Phase 2: Learning Robot Policy

- **Human Policy**: Fixed ε_h_0-greedy policy using learned Q_h^m
- **Robot Policy**: β_r-softmax with β_r increasing from 0.1 to β_r_0
- **Goal**: Learn optimal robot policy Q_r to maximize human potential
- **Updates**: Standard Q-learning for robot, minimal updates for humans

**Key Parameters:**

- Learning rates: `alpha_m` (Phase 1 human), `alpha_e` (Phase 2 human), `alpha_r` (robot)
- Discount factors: `gamma_h`, `gamma_r`
- Human exploration: `epsilon_h` (current, starts at 0.5), `epsilon_h_0` (target, final epsilon for converged policy)
- Robot rationality: `beta_r` (current, starts at 0.1), `beta_r_0` (target, final beta for Phase 2)
- Robot exploration: `epsilon_r` (Phase 1 only)
- Set of possible human goals: `G`
- Prior probability distribution over goals: `mu_g`
- Probability of goal change per step: `p_g`
- Power function exponent: `eta`
- **Neural Network Parameters:**
  - `network`: Boolean flag to enable neural network mode (default: False)
  - `state_dim`: Dimension of state vector for neural networks (default: 4)
  - `beta_h`: Softmax inverse temperature for human policies (default: 5.0)
  - `nu_h`: Prior weight in smooth policy updates (default: 0.1)

## Gridworld Application of IQL

This section details the specific application of the IQL algorithm within the custom gridworld environment.

- **State ($s$):** A tuple representing the full observable state of the gridworld, e.g., (robot_position, human_position, status_of_items, door_open, etc.). It's assumed to be fully observable by both the robot and human in the tabular learning setup.
- **Actions ($a_r, a_h$):** Discrete actions available to the robot and human. Examples include:
  - Movement: North, South, East, West
  - Wait
  - Pickup (item)
  - Drop (item)
  - Interact (e.g., toggle a switch, open a door)
    The sets of actions are denoted by $\\mathcal{A}_r$ and $\\mathcal{A}_h$.
- **Goals ($\\mathcal{G}$):** The set of potential goals for the human.
  - Example: A predefined subset of grid cells that the human might want to reach, e.g., $\\mathcal{G} = \\{(x_1, y_1), (x_2, y_2), \\dots\\}$.
  - In the "simplest case" formulation, each goal could be to reach one of the terminal states of the environment ($\\mathcal{G} = \\mathcal{S}^\\top$).
- **Goal Prior ($\\mu_g$):** The assumed probability distribution over the set of goals $\\mathcal{G}$.
  - Example: A uniform distribution, $\\mu_g(g) = 1/|\\mathcal{G}|$ for all $g \\in \\mathcal{G}$.
- **Base Human Reward ($r_h^{obs} = r_h(s, a_r, a_h, s', g)$):** This reward is provided by the environment simulator and signals the human's success in achieving their current goal $g$.
  - Example: If the human's goal $g$ is to reach cell $(x_g, y_g)$, then $r_h^{obs} = +1$ if the human's position in the next state $s'$ is $(x_g, y_g)$, and $0$ otherwise. A small negative penalty for each step taken (e.g., -0.01) can also be included.
  - For the "simplest case", $r_h(s, a_r, a_h, s', g) = 1_{s'=g}$ (reward of 1 for reaching state $g$).
- **Robot's Reward ($r_r$):** This reward is calculated internally by the robot, not provided by the environment. It is based on the human's expected future value, averaged over potential goals, possibly transformed by a function $f$ and an exponent $\\eta$.
  - It is typically formulated as $r_r = f(\\mathbb{E}_{g' \\sim \\mu_g} [\\hat{V}_h(s', g')^{1+\\eta}])$, where $\\hat{V}_h(s', g')$ is the learned value function for the human achieving goal $g'$ from state $s'$. The expectation $\\mathbb{E}_{g' \\sim \\mu_g} [\\cdot]$ is the $r_r^{calc}$ term from step 2.f.ii in the algorithm description.
- **Q-Tables:** The learning process involves maintaining two main Q-tables:
  - Robot's Q-table: $Q_r[s][a_r]$ of size $|\\mathcal{S}| \\times |\\mathcal{A}_r|$.
  - Human's Q-table: $Q_h[s][g][a_h]$ of size $|\\mathcal{S}| \\times |\\mathcal{G}| \\times |\\mathcal{A}_h|$.
- **Typical Parameters for Gridworld:**
  - Learning rates: `alpha_m ≈ 0.1`, `alpha_e ≈ 0.2`, `alpha_r ≈ 0.01`
  - Discount factors: `gamma_h = 0.99`, `gamma_r = 0.99`
  - Human exploration: `epsilon_h` starts at 0.5, decays to `epsilon_h_0 ∈ [0.05, 0.2]` (final exploration level)
  - Robot rationality: `beta_r` starts at 0.1, increases to `beta_r_0 ∈ [5, 20]` (higher = more deterministic)
  - Robot exploration: `epsilon_r` starts at 0.1, decays to 0.01 in Phase 1
  - Goal change probability: `p_g ≈ 0.01` for infrequent goal changes
  - Power exponent: `eta = 0` (linear) or `eta = 1` (quadratic emphasis)

## Code Structure

- **`env.py`**: Defines the custom grid environment.
- **`iql_algorithm.py`**: Implements the Two-Timescale Goal-Based IQL algorithm.
- **`main.py`**: Runs the training loop and renders the environment.
- **`trained_agent.py`**: Implements an agent that uses saved Q-values for deterministic visualization.
- **`envs/`**: Contains map definitions for different grid layouts:
  - **`map_loader.py`**: Utility functions for loading map layouts.
  - **`simple_map.py`**: A simple map with a door separating the robot from the goal.
  - **`complex_map.py`**: A more complex map with multiple rooms and hazards.

## Maps and Environment Layout

The environment supports customizable maps that define the layout of walls, doors, keys, goals, and agent starting positions. Maps are defined in individual Python files in the `envs/` directory.

### Map Format

Maps are defined as a list of strings, where each character represents a specific element in the grid:

- **`#`**: Wall
- **` `** (space): Empty space
- **`R`**: Robot starting position
- **`H`**: Human starting position
- **`D`**: Door
- **`K`**: Key
- **`G`**: Goal
- **`L`**: Lava (hazardous tile)

### Available Maps

- **`simple_map`**: A small, simple layout with a door separating the robot and human from the goal.
- **`complex_map`**: A larger, more complex layout with multiple rooms and hazards.

### Creating Your Own Maps

You can create your own map by adding a new Python file in the `envs/` directory. The file should define:

1. A list of strings representing the grid layout
2. Metadata including name, description, size, and maximum steps
3. A `get_map()` function that returns the layout and metadata

Example:

```python
# Define the map layout
MY_MAP = [
    "#########",
    "#R      #",
    "#   D   #",
    "#       #",
    "#K     G#",
    "#########"
]

# Define metadata
MAP_METADATA = {
    "name": "My Custom Map",
    "description": "A simple custom map",
    "size": (5, 9),
    "max_steps": 100,
}

def get_map():
    return MY_MAP, MAP_METADATA
```

## Running the Code

### Choosing the Right Mode

**Tabular Mode (Default):**

- Use for small discrete state spaces (typically < 10,000 states)
- Faster training for simple environments
- Complete policy analysis and visualization
- Example: Simple gridworlds with few objects

**Neural Network Mode:**

- Use for large state spaces or when state representation is complex
- Better generalization across similar states
- Scalable to continuous or high-dimensional state spaces
- Example: Large gridworlds, complex object interactions, or continuous control

### Installation and Setup

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   cd src/corrigibility/marl
   ```

2. **For Neural Network Support (Optional):**
   ```bash
   pip install torch  # PyTorch for neural networks
   ```

### Running the Code

**Note:** Use `run_iql.py` from the root directory instead of `main.py` for easier imports and full feature support.

### Complete Examples

**Example 1: Quick Tabular Training (Small Map)**

```bash
# Train with tabular Q-learning on a simple map
python run_iql.py --mode train --phase1-episodes 500 --phase2-episodes 500 --save saved/simple_tabular.pkl --map simple_map

# Visualize the results
python run_iql.py --mode visualize --load saved/simple_tabular.pkl --map simple_map --delay 200
```

**Example 2: Neural Network Training (Complex Map)**

```bash
# Train with neural networks for scalability
python run_iql.py --mode train --phase1-episodes 1000 --phase2-episodes 1000 --save saved/complex_network.pkl --map complex_map --network --state-dim 6

# Visualize with network mode
python run_iql.py --mode visualize --load saved/complex_network.pkl --map complex_map --network --state-dim 6 --delay 150
```

**Example 3: Training with Rendering and Debug**

```bash
# Watch the training process with debug output
python run_iql.py --mode train --phase1-episodes 200 --phase2-episodes 200 --save saved/debug_run.pkl --map simple_map2 --render --debug_prints --debug_level verbose
```

**Example 4: Comparing Reward Functions**

```bash
# Train with different robot reward functions
python run_iql.py --mode train --phase1-episodes 500 --phase2-episodes 500 --save saved/power_reward.pkl --map team_map --reward-function power

python run_iql.py --mode train --phase1-episodes 500 --phase2-episodes 500 --save saved/bounded_reward.pkl --map team_map --reward-function bounded --concavity-param 2.0
```

### Visualization Examples

5. **Visualize Tabular Trained Agent:**

   ```bash
   python run_iql.py --mode visualize --load saved/q_values.pkl --delay 100 --map simple_map
   ```

6. **Visualize Neural Network Trained Agent:**

   ```bash
   python run_iql.py --mode visualize --load saved/q_values_nn.pkl --delay 100 --map complex_map --network --state-dim 6
   ```

7. **Run Baseline Deterministic Test:**
   ```bash
   python run_iql.py --mode test --delay 100 --map simple_map
   ```

### Command-line Arguments

The following command-line arguments are available:

- `--mode`: Choose between `train` (train the model), `visualize` (run trained model), or `test` (run deterministic test). Default: `train`.
- `--algorithm`: Choose between `timescale` (two-phase algorithm) or `standard` (original IQL). Default: `timescale`.
- `--save`: Path to save trained Q-values. Default: `saved/q_values.pkl`.
- `--load`: Path to load trained Q-values for visualization. Default: `saved/q_values.pkl`.
- `--episodes`: Number of episodes for training (standard algorithm). Default: `1000`.
- `--phase1-episodes`: Number of episodes for Phase 1 (timescale algorithm). Default: `500`.
- `--phase2-episodes`: Number of episodes for Phase 2 (timescale algorithm). Default: `500`.
- `--delay`: Delay in milliseconds between steps during visualization. Default: `100`.
- `--map`: The map to use (e.g., `simple_map`, `complex_map`). Default: `simple_map`.
- `--grid-size`: The size of the grid (optional, default is derived from map).
- `--render`: Enable rendering during training.
- `--debug_prints`: Enable detailed debug prints during training.
- `--debug_level`: Level of debug output (`minimal`, `standard`, `verbose`). Default: `standard`.
- `--reward-function`: Robot reward function (`power`, `log`, `bounded`, `generalized_bounded`). Default: `power`.
- `--concavity-param`: Concavity parameter for generalized_bounded function. Default: `1.0`.
- **Neural Network Arguments:**
  - `--network`: Use neural network Q-learning instead of tabular. Default: `False`.
  - `--state-dim`: State vector dimension for neural networks. Default: `4`.

### Training and Visualization Flow

1. **Training**: The agent is trained in deterministic=false mode for a specified number of episodes.
2. **Saving**: After training, Q-values are saved to the specified file.
3. **Visualization**: The trained Q-values can be loaded and used to visualize the agent's behavior in deterministic mode.

## Future Work

- Add more complex human behaviors.
- Introduce additional grid elements and interactions.
- Experiment with different reward functions and learning rates.
- Implement convergence metrics to automatically determine when training should stop.
- Create more diverse map layouts for different training scenarios.
