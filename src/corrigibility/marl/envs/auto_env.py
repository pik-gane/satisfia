"""
Auto-generated grid environment for multi-agent experiments.

This module provides a function to generate a gridworld environment based on a configuration dictionary.
The config can specify the number of humans, robots, doors, keys, boxes, goals, and other elements.
The environment is validated to ensure it is solvable (e.g., all goals are reachable, walls enclose the map, etc.).
A random seed ensures reproducibility.

Example config:
config = {
    'width': 8,
    'height': 8,
    'num_humans': 2,
    'num_robots': 1,
    'num_doors': 2,
    'num_keys': 2,
    'num_boxes': 1,
    'num_goals': 2,
    'lava_prob': 0.05,  # probability of lava per cell
    'seed': 42
}

Call generate_env_map(config) to get (map_grid, metadata).
"""

import random
from typing import Dict, List, Tuple

import numpy as np


def generate_env_map(config: Dict) -> Tuple[List[List[str]], Dict]:
    width = config.get("width", 8)
    height = config.get("height", 8)
    num_humans = config.get("num_humans", 1)
    num_robots = config.get("num_robots", 1)
    num_doors = config.get("num_doors", 1)
    num_keys = config.get("num_keys", 1)
    num_boxes = config.get("num_boxes", 0)
    num_goals = config.get("num_goals", num_humans)
    lava_prob = config.get("lava_prob", 0.0)
    seed = config.get("seed", None)
    max_steps = config.get("max_steps", 100)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize empty map with walls
    grid = [["##" for _ in range(width)] for _ in range(height)]
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            grid[r][c] = "  "

    # Place robots
    robot_positions = []
    for i in range(num_robots):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = f"vR{i}" if num_robots > 1 else "vR"
                robot_positions.append((r, c))
                break

    # Place humans
    human_positions = []
    for i in range(num_humans):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = f"vH{i}" if num_humans > 1 else "vH0"
                human_positions.append((r, c))
                break

    # Place goals
    goal_positions = []
    for i in range(num_goals):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = "GG"
                goal_positions.append((r, c))
                break

    # Place doors
    for i in range(num_doors):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = "YD"
                break

    # Place keys
    for i in range(num_keys):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = "YK"
                break

    # Place boxes
    for i in range(num_boxes):
        while True:
            r, c = random.randint(1, height - 2), random.randint(1, width - 2)
            if grid[r][c] == "  ":
                grid[r][c] = "BX"
                break

    # Place lava
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if grid[r][c] == "  " and random.random() < lava_prob:
                grid[r][c] = "OL"

    # Validate map: ensure all goals are reachable from at least one agent
    def is_reachable(start, targets):
        from collections import deque

        visited = set()
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            if pos in targets:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pos[0] + dr, pos[1] + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and grid[nr][nc] not in ("##", "YD")
                    and (nr, nc) not in visited
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # Check each human can reach at least one goal
    for hpos in human_positions:
        if not is_reachable(hpos, set(goal_positions)):
            raise ValueError(
                f"Human at {hpos} cannot reach any goal. Try a different seed or config."
            )

    # Check each robot can reach at least one goal
    for rpos in robot_positions:
        if not is_reachable(rpos, set(goal_positions)):
            raise ValueError(
                f"Robot at {rpos} cannot reach any goal. Try a different seed or config."
            )

    # Metadata
    metadata = {
        "name": "Auto-generated Gridworld",
        "description": f"Gridworld with {num_robots} robot(s), {num_humans} human(s), {num_doors} door(s), {num_keys} key(s), {num_boxes} box(es), {num_goals} goal(s)",
        "size": (height, width),
        "max_steps": max_steps,
        "human_goals": {
            f"human_{i}": goal_positions[i]
            for i in range(min(num_humans, len(goal_positions)))
        },
        "robot_starts": robot_positions,
        "human_starts": human_positions,
        "goal_positions": goal_positions,
        "seed": seed,
    }
    return grid, metadata
