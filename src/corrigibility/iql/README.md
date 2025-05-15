# Two-Timescale Goal-Based IQL in a Custom Grid Environment

## Overview

This project implements a Two-Timescale Goal-Based Independent Q-Learning (IQL) algorithm in a custom grid environment. The environment is designed using the `PettingZoo` library and rendered with `pygame`. The agents in the environment are a **robot** and a **human**, and the robot's goal is to maximize the human's potential to achieve various goals.

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

The Two-Timescale Goal-Based Independent Q-Learning (IQL) algorithm is implemented as described below. It involves learning a human model ($Q_h$) and a robot policy ($Q_r$) on different timescales.

**Inputs:**

- Learning rates: $\\alpha_h, \\alpha_r$ (with $\\alpha_h \\gg \\alpha_r$)
- Discount factors: $\\gamma_h, \\gamma_r$
- Human rationality parameter: $\\beta_h$
- Robot exploration parameter: $\\epsilon_r$
- Set of possible human goals: $G$
- Prior probability distribution over goals: $\\mu_g$
- Probability of goal change per step: $p_g$
- Total number of training episodes: $E$
- An environment that provides agent observations, human rewards, and state transitions.

**Initialization:**

1.  Initialize robot's Q-function $Q_r(s_r, a_r)$ with small random values (e.g., $U[-0.1, 0.1]$) for all robot states $s_r$ and actions $a_r$.
2.  Initialize human's Q-function $Q_h(s_h, g, a_h)$ with small random values (e.g., $U[-0.1, 0.1]$) for all human states $s_h$, goals $g \\in G$, and human actions $a_h$.
3.  Initialize the robot's exploration parameter $\\epsilon_r$.

**Training Loop (for each episode $e = 1, \\dots, E$):**

1.  Start in an initial state $s$. Sample an initial goal $g$ for the human from the prior $\\mu_g$.
2.  **While the current state $s$ is not terminal:**
    a. **Goal Dynamics:** With probability $p_g$, resample the human's goal $g$ from $\\mu_g$.
    b. **Action Selection:**
    i. The robot chooses action $a_r$ using an $\\epsilon_r$-greedy policy based on $Q_r(s_r, \\cdot)$.
    ii. The human chooses action $a_h$ using a softmax policy: $\\pi_h^*(s_h, g, \\cdot) \\propto \\exp(\\beta_h Q_h(s_h, g, \\cdot))$.
    c. **Environment Interaction:** Execute actions $(a_r, a_h)$. Observe the next state $s'$ and the human's reward $r_h^{obs}$.
    d. **Calculate Human's Expected Next State Value ($V_h(s', g)$):**
    i. If $s'$ is a terminal state, $V_h(s', g) = 0$.
    ii. Otherwise, calculate the human's policy $\\pi_h^*(s_h', g, \\cdot)$ at the next state $s_h'$ and compute $V_h(s', g) = \\sum_{a_h'} \\pi_h^*(s_h', g, a_h') Q_h(s_h', g, a_h')$.
    e. **Human Q-Update (Fast Timescale):**
    i. Calculate the TD target: $Q_h^{target} = r_h^{obs} + \\gamma_h V_h(s', g)$.
    ii. Update $Q_h(s_h, g, a_h) \\leftarrow Q_h(s_h, g, a_h) + \\alpha_h [Q_h^{target} - Q_h(s_h, g, a_h)]$.
    f. **Calculate Robot's Internal Reward ($r_r^{calc}$):** This reward is based on the human's potential to achieve various goals from the next state $s'$.
    i. Initialize $r_r^{calc} = 0$.
    ii. For each potential goal $g' \\in G$: 1. Calculate $V_h(s', g')$ (the expected value for the human to achieve goal $g'$ from state $s'$), similar to step 2.d. 2. Add to the robot's reward: $r_r^{calc} \\leftarrow r_r^{calc} + \\mu_g(g') V_h(s', g')$.
    _(The actual reward used for the robot's Q-update might involve applying a concave function $f$ and an exponent $\\eta$ to this expectation, e.g., $r_r = f(\\mathbb{E}_{g' \\sim \\mu*g} [V_h(s', g')^{1+\\eta}])$ as detailed in the gridworld application).*
    g. **Robot Q-Update (Slow Timescale):**
    i. If $s'$ is a terminal state, the robot's TD target is $Q_r^{target} = r_r^{calc}$.
    ii. Otherwise, $Q_r^{target} = r_r^{calc} + \\gamma_r \\max_{a_r'} Q_r(s_r', a_r')$.
    iii. Update $Q_r(s_r, a_r) \\leftarrow Q_r(s_r, a_r) + \\alpha_r [Q_r^{target} - Q_r(s_r, a_r)]$.
    h. **Update State:** $s \\leftarrow s'$.
    i. Optionally, decay the robot's exploration parameter $\\epsilon_r$.

**Output:** The learned Q-functions $Q_r$ and $Q_h$.

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
  - Learning rates: $\\alpha_h \\approx 0.1$, $\\alpha_r \\approx 0.01$ (ensure $\\alpha_h \\gg \\alpha_r$).
  - Discount factors: $\\gamma_h = 0.99$ (or $\\gamma_h=1$ for the simplest case), $\\gamma_r = 0.99$.
  - Human rationality: $\\beta_h \\in [1, 10]$ (e.g., start with $\\beta_h=5$). Higher values mean the human is more deterministic in following their optimal policy for a given goal.
  - Robot exploration ($\\epsilon_r$): Start at $1.0` and decay linearly to a small value (e.g., $0.1`) over the course of training. (For the "simplest case" with $\\beta_r=\\infty$, the robot is a full optimizer, implying $\\epsilon_r=0$ after learning or during evaluation).
  - Goal change probability ($p_g$): A small value, e.g., $0.01`, for infrequent changes in the human's goal during an episode.
  - Power function/parameters for robot's reward:
    - Function $f(z)$: e.g., $f(z) = 2 - 2/z` (corresponds to $c=1` in some power formulations) or $f(z) = \\log_2(z)` .
    - Exponent $\\eta`: e.g., $\\eta = 0` (simplifies to $f(\\mathbb{E}[V_h])`) or $\\eta = 1` (as in the "simplest case"). Start with $f(z)=2-2/z` and $\\eta=0`.
  - Number of episodes ($E`): Highly dependent on the size and complexity of the gridworld (e.g., 10,000 to 1,000,000+).

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

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   cd src/corrigibility/iql
   ```

2. Train the IQL algorithm:

   ```bash
   python main.py --mode train --episodes 1000 --save saved/q_values.pkl --map simple_map
   ```

3. Visualize the trained agent:

   ```bash
   python main.py --mode visualize --load saved/q_values.pkl --delay 100 --map simple_map
   ```

4. Run the baseline deterministic test:
   ```bash
   python main.py --mode test --delay 100 --map complex_map
   ```

### Command-line Arguments

The following command-line arguments are available:

- `--mode`: Choose between `train` (train the model), `visualize` (run trained model), or `test` (run deterministic test). Default: `train`.
- `--save`: Path to save trained Q-values. Default: `saved/q_values.pkl`.
- `--load`: Path to load trained Q-values for visualization. Default: `saved/q_values.pkl`.
- `--episodes`: Number of episodes for training. Default: `1000`.
- `--delay`: Delay in milliseconds between steps during visualization. Default: `100`.
- `--map`: The map to use (e.g., `simple_map`, `complex_map`). Default: `simple_map`.
- `--grid-size`: The size of the grid (optional, default is derived from map).

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
