import numpy as np

def encode_full_observable_state(env, state):
    """
    Encode the full grid, all agent positions, and all object states into a flat vector.
    - env: the environment instance (must have map_layout, agent_positions, etc.)
    - state: the environment state (tuple or dict)
    Returns a 1D numpy array suitable for neural network input.
    """
    # 1. Encode the grid as integers (row-major)
    grid = env.map_layout
    grid_shape = (len(grid), len(grid[0]))
    grid_flat = []
    for row in grid:
        for cell in row:
            # Map each cell type to an integer (customize as needed)
            if cell == '#':
                grid_flat.append(1)
            elif cell == 'G':
                grid_flat.append(2)
            elif cell == 'L':
                grid_flat.append(3)
            elif cell == ',':
                grid_flat.append(4)
            elif cell == 'K':
                grid_flat.append(5)
            elif cell == 'D':
                grid_flat.append(6)
            elif cell == 'A':
                grid_flat.append(7)
            elif cell == 'H':
                grid_flat.append(8)
            else:
                grid_flat.append(0)  # empty or unknown
    # 2. Encode agent positions (robot and humans)
    agent_vec = []
    for aid in env.robot_agent_ids + env.human_agent_ids:
        pos = env.agent_positions.get(aid, (-1, -1))
        agent_vec.extend([pos[0], pos[1]])
    # 3. Encode object states (keys, doors, boxes, etc.)
    object_vec = []
    for key in env.keys:
        object_vec.extend([key['pos'][0], key['pos'][1], int(key.get('picked', False))])
    for door in env.doors:
        object_vec.extend([door['pos'][0], door['pos'][1], int(door.get('is_open', False)), int(door.get('is_locked', False))])
    for box in env.boxes:
        object_vec.extend([box['pos'][0], box['pos'][1]])
    # 4. Concatenate all parts
    state_vec = np.array(grid_flat + agent_vec + object_vec, dtype=np.float32)
    return state_vec
