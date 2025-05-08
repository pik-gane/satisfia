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

| Action | Name     | Description            |
|--------|----------|------------------------|
| 0      | Left     | Turn left              |
| 1      | Right    | Turn right             |
| 2      | Forward  | Move forward           |
| 3      | Pickup   | Pick up an object      |
| 4      | Drop     | Drop an object         |
| 5      | Toggle   | Toggle/activate an object |
| 6      | Done     | Signal done            |

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

The Two-Timescale Goal-Based IQL algorithm is implemented as follows:

1. **Human Model (`Q_h`)**: The robot learns a model of the human's behavior for each potential goal.
2. **Robot Policy (`Q_r`)**: The robot learns its own policy to maximize a reward signal based on the human's potential.
3. **Two Timescales**: The human model is updated at a faster rate than the robot's policy.
4. **Power Proxy**: The robot calculates an internal reward based on the human's potential to achieve various goals.

## Code Structure

- **`minigrid_power_env.py`**: Defines the custom grid environment.
- **`iql_agent.py`**: Implements the Two-Timescale Goal-Based IQL algorithm.
- **`main.py`**: Runs the training loop and renders the environment.

## Running the Code

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python src/corrigibility/iql/main.py
   ```

3. Observe the environment being rendered in a `pygame` window and the training progress in the terminal.

## Future Work

- Add more complex human behaviors.
- Introduce additional grid elements and interactions.
- Experiment with different reward functions and learning rates.

