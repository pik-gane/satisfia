"""
Environment 5: Large Complex Grid
Description: 9x9 grid with multiple robots, humans, complex interactions
Difficulty: Hard
Learning objectives: Large-scale coordination, complex planning, scalability

Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'BD': Blue Door
- 'BK': Blue Key
- 'YD': Yellow Door
- 'YK': Yellow Key
- 'RD': Red Door
- 'RK': Red Key
- 'OL': Lava/Obstacle
- 'BB': Box
- 'GG': Goal
"""

# Define the hardcoded map layout for Environment 5
SIMPLE_MAP = [
    ["##", "##", "##", "##", "##", "##", "##", "##", "##"],
    ["##", "vR0", "BK", "  ", "OL", "  ", "YK", "  ", "##"],
    ["##", "  ", "##", "BD", "##", "YD", "##", "RK", "##"],
    ["##", "vH0", "  ", "  ", "BB", "  ", "  ", "  ", "##"],
    ["##", "OL", "##", "  ", "##", "  ", "##", "OL", "##"],
    ["##", "  ", "  ", "GG", "  ", "GG", "  ", "vH1", "##"],
    ["##", "##", "RD", "##", "##", "##", "BB", "##", "##"],
    ["##", "vH2", "  ", "  ", "OL", "  ", "  ", "GG", "##"],
    ["##", "##", "##", "##", "##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Large Complex Grid",
    "description": "9x9 grid with complex multi-agent interactions",
    "size": (9, 9),  # (rows, cols)
    "max_steps": 200,
    "human_goals": {
        "human_0": (5, 3),
        "human_1": (5, 5),
        "human_2": (7, 7),
    },  # Different goals for different humans
    "num_robots": 1,
    "num_humans": 3,
    "num_doors": 3,
    "num_keys": 3,
    "difficulty": 5,
}


def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA


def get_env_config():
    """
    Returns the configuration for Environment 5: Large Complex Grid

    Returns:
        dict: Environment configuration dictionary with hardcoded map
    """
    return {
        "name": "env_05_larger_grid",
        "description": "9x9 grid with complex multi-agent interactions",
        "difficulty": 5,
        "learning_objectives": [
            "Large-scale coordination",
            "Complex planning",
            "Model scalability",
        ],
        "use_hardcoded_map": True,  # Flag to indicate hardcoded map usage
        "training_config": {
            "phase1_episodes": 800,
            "phase2_episodes": 1200,
            "success_threshold": 0.6,
            "max_retries": 5,
        },
        "expected_skills": [
            "Large-scale navigation",
            "Complex multi-agent coordination",
            "Advanced planning under constraints",
            "Robust model generalization",
        ],
        "gen_args": {},
        "training_params": {
            "num_training_episodes": 50000,
            "evaluation_interval": 1000,
        },
    }


def get_expected_performance():
    """Expected performance metrics for this environment."""
    return {
        "target_success_rate": 0.6,
        "expected_phase1_convergence": 650,
        "expected_phase2_convergence": 1000,
        "key_metrics": [
            "success_rate",
            "scalability_score",
            "coordination_efficiency",
            "task_completion_time",
        ],
    }
