"""
Environment 3: Multi-Key System
Description: 6x6 grid with 1 robot, 1 human, 2 doors, 2 keys, multiple goals
Difficulty: Medium
Learning objectives: Multi-step planning, resource management, coordination

Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'BD': Blue Door
- 'BK': Blue Key
- 'YD': Yellow Door
- 'YK': Yellow Key
- 'GG': Goal
"""

# Define the hardcoded map layout for Environment 3
SIMPLE_MAP = [
    ["##", "##", "##", "##", "##", "##"],
    ["##", "vR0", "BK", "  ", "YK", "##"],
    ["##", "##", "BD", "##", "YD", "##"],
    ["##", "vH0", "  ", "GG", "  ", "##"],
    ["##", "  ", "  ", "  ", "GG", "##"],
    ["##", "##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Multi-Key System",
    "description": "6x6 grid with multiple keys and doors",
    "size": (6, 6),  # (rows, cols)
    "max_steps": 100,
    "human_goals": {"human_0": (3, 3)},  # Human's goal is one of the goals
    "num_robots": 1,
    "num_humans": 1,
    "num_doors": 2,
    "num_keys": 2,
    "difficulty": 3,
}


def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA


def get_env_config():
    """
    Returns the configuration for Environment 3: Multi-Key System

    Returns:
        dict: Environment configuration dictionary with hardcoded map
    """
    return {
        "name": "env_03_multi_key",
        "description": "6x6 grid with multiple keys and doors",
        "difficulty": 3,
        "learning_objectives": [
            "Multi-step planning",
            "Resource management",
            "Complex coordination",
        ],
        "use_hardcoded_map": True,  # Flag to indicate hardcoded map usage
        "gen_args": {
            "num_robots": 1,
            "num_humans": 1,
            "num_goals": 2,
            "width": 6,
            "height": 6,
        },
        "training_config": {
            "phase1_episodes": 500,
            "phase2_episodes": 800,
            "success_threshold": 0.7,
            "max_retries": 3,
        },
        "expected_skills": [
            "Multi-object interaction",
            "Strategic planning",
            "Resource allocation",
            "Complex goal prioritization",
        ],
    }


def get_expected_performance():
    """Expected performance metrics for this environment."""
    return {
        "target_success_rate": 0.7,
        "expected_phase1_convergence": 400,
        "expected_phase2_convergence": 650,
        "key_metrics": [
            "success_rate",
            "planning_efficiency",
            "resource_utilization",
            "coordination_score",
        ],
    }
