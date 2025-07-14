"""
Environment 1: Simple 4x4 Grid
Description: Minimal environment with 1 robot, 1 human, 1 goal
Difficulty: Beginner
Learning objectives: Basic movement, goal reaching

Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'GG': Goal
"""

# Define the hardcoded map layout for Environment 1
SIMPLE_MAP = [
    ["##", "##", "##", "##"],
    ["##", "vR0", "vH0", "##"],
    ["##", "  ", "GG", "##"],
    ["##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Simple 4x4 Grid",
    "description": "Minimal environment with 1 robot, 1 human, 1 goal",
    "size": (4, 4),  # (rows, cols)
    "max_steps": 50,
    "human_goals": {"human_0": (2, 2)},  # Human's goal is the main goal
    "num_robots": 1,
    "num_humans": 1,
    "num_doors": 0,
    "num_keys": 0,
    "difficulty": 1,
}


def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA


def get_env_config():
    """
    Returns the configuration for Environment 1: Simple 4x4 Grid

    Returns:
        dict: Environment configuration dictionary with hardcoded map
    """
    return {
        "name": "env_01_simple",
        "description": "Simple 4x4 grid with 1 robot, 1 human, 1 goal",
        "difficulty": 1,
        "learning_objectives": [
            "Basic movement",
            "Goal reaching",
            "Agent coordination",
        ],
        "use_hardcoded_map": True,  # Flag to indicate hardcoded map usage
        "gen_args": {
            "num_robots": 1,
            "num_humans": 1,
            "num_goals": 1,
            "width": 4,
            "height": 4,
        },
        "training_config": {
            "phase1_episodes": 300,
            "phase2_episodes": 500,
            "success_threshold": 0.8,
            "max_retries": 2,
        },
        "expected_skills": ["Basic navigation", "Simple goal pursuit"],
    }


def get_expected_performance():
    """Expected performance metrics for this environment."""
    return {
        "target_success_rate": 0.8,
        "expected_phase1_convergence": 200,
        "expected_phase2_convergence": 400,
        "key_metrics": ["success_rate", "steps_to_goal", "agent_cooperation"],
    }
