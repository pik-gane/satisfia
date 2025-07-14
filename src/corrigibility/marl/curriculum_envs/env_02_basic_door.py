"""
Environment 2: Basic Door/Key System
Description: 5x5 grid with 1 robot, 1 human, 1 door, 1 key, 1 goal
Difficulty: Easy
Learning objectives: Key collection, door unlocking, sequential tasks

Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'BD': Blue Door
- 'BK': Blue Key
- 'GG': Goal
"""

# Define the hardcoded map layout for Environment 2
SIMPLE_MAP = [
    ["##", "##", "##", "##", "##"],
    ["##", "vR0", "BK", "vH0", "##"],
    ["##", "##", "BD", "##", "##"],
    ["##", "  ", "GG", "  ", "##"],
    ["##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Basic Door/Key System",
    "description": "5x5 grid with basic door/key mechanics",
    "size": (5, 5),  # (rows, cols)
    "max_steps": 75,
    "human_goals": {"human_0": (3, 2)},  # Human's goal is the main goal
    "num_robots": 1,
    "num_humans": 1,
    "num_doors": 1,
    "num_keys": 1,
    "difficulty": 2,
}


def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA


def get_env_config():
    """
    Returns the configuration for Environment 2: Basic Door/Key System

    Returns:
        dict: Environment configuration dictionary with hardcoded map
    """
    return {
        "name": "env_02_basic_door",
        "description": "5x5 grid with basic door/key mechanics",
        "difficulty": 2,
        "learning_objectives": [
            "Key collection",
            "Door unlocking",
            "Sequential planning",
        ],
        "use_hardcoded_map": True,  # Flag to indicate hardcoded map usage
        "gen_args": {
            "num_robots": 1,
            "num_humans": 1,
            "num_goals": 1,
            "width": 5,
            "height": 5,
        },
        "training_config": {
            "phase1_episodes": 400,
            "phase2_episodes": 600,
            "success_threshold": 0.75,
            "max_retries": 3,
        },
        "expected_skills": [
            "Tool usage (key collection)",
            "Sequential task execution",
            "Environmental interaction",
        ],
    }


def get_expected_performance():
    """Expected performance metrics for this environment."""
    return {
        "target_success_rate": 0.75,
        "expected_phase1_convergence": 300,
        "expected_phase2_convergence": 500,
        "key_metrics": [
            "success_rate",
            "key_collection_time",
            "door_unlock_efficiency",
        ],
    }
