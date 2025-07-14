"""
Environment 4: Obstacles and Navigation
Description: 7x7 grid with obstacles, 1 robot, 2 humans, multiple keys/doors
Difficulty: Medium-Hard
Learning objectives: Navigation with obstacles, multi-agent coordination, spatial reasoning

Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'BD': Blue Door
- 'BK': Blue Key
- 'YD': Yellow Door
- 'YK': Yellow Key
- 'OL': Lava/Obstacle
- 'BB': Box
- 'GG': Goal
"""

# Define the hardcoded map layout for Environment 4
SIMPLE_MAP = [
    ["##", "##", "##", "##", "##", "##", "##"],
    ["##", "vR0", "BK", "OL", "  ", "YK", "##"],
    ["##", "  ", "##", "OL", "##", "  ", "##"],
    ["##", "vH0", "BD", "  ", "YD", "vH1", "##"],
    ["##", "  ", "##", "BB", "##", "  ", "##"],
    ["##", "  ", "OL", "GG", "OL", "GG", "##"],
    ["##", "##", "##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Obstacles and Navigation",
    "description": "7x7 grid with obstacles and multiple agents",
    "size": (7, 7),  # (rows, cols)
    "max_steps": 150,
    "human_goals": {
        "human_0": (5, 3),
        "human_1": (5, 5),
    },  # Different goals for different humans
    "num_robots": 1,
    "num_humans": 2,
    "num_doors": 2,
    "num_keys": 2,
    "difficulty": 4,
}


def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA


def get_env_config():
    """
    Returns the configuration for Environment 4: Obstacles and Navigation

    Returns:
        dict: Environment configuration dictionary with hardcoded map
    """
    return {
        "name": "env_04_obstacles",
        "description": "7x7 grid with obstacles and multiple agents",
        "difficulty": 4,
        "learning_objectives": [
            "Obstacle navigation",
            "Multi-agent coordination",
            "Spatial reasoning",
        ],
        "use_hardcoded_map": True,  # Flag to indicate hardcoded map usage
        "training_config": {
            "phase1_episodes": 600,
            "phase2_episodes": 1000,
            "success_threshold": 0.65,
            "max_retries": 4,
        },
        "expected_skills": [
            "Obstacle avoidance",
            "Multi-agent path planning",
            "Shared model coordination",
            "Complex environment navigation",
        ],
        "gen_args": {},
        "training_params": {
            "num_training_episodes": 50000,
            "evaluation_interval": 1000,
            "success_reward": 10,
            "failure_penalty": -10,
            "max_steps": 150,
            "checkpoint_interval": 5000,
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "exploration_rate": 1.0,
            "exploration_decay": 0.995,
            "min_exploration_rate": 0.01,
            "batch_size": 32,
            "buffer_size": 100000,
            "num_hidden_units": 128,
            "num_layers": 3,
            "activation_function": "relu",
            "optimizer": "adam",
            "loss_function": "mean_squared_error",
            "metrics": ["mae", "mse"],
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
            "lr_scheduler": "ReduceLROnPlateau",
            "clip_norm": 1.0,
            "use_gae": True,
            "gae_lambda": 0.95,
            "use_automatic_entropy_tuning": True,
            "target_entropy": "auto",
            "ent_coef": "auto",
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "rms_prop_eps": 1e-5,
            "use_sde": True,
            "sde_sample_freq": 4,
            "train_freq": (1, "step"),
            "target_update_interval": 100,
            "num_steps": 5,
            "gae_lambda": 0.95,
            "normalize_advantages": True,
            "use_torch": True,
            "torch_deterministic": True,
            "torch_benchmark": True,
            "num_envs": 1,
            "env_name": "env_04_obstacles",
            "log_dir": "./logs",
            "save_model": True,
            "load_model": False,
            "model_dir": "./models",
            "model_name": "obstacle_navigation_model",
            "render": False,
            "seed": 42,
            "capture_video": False,
            "video_folder": "./videos",
            "disable_env_checker": True,
            "wrapper": "Default",
            "custom_metrics": {},
            "custom_callbacks": {},
            "debug": False,
            "verbose": 1,
        },
    }


def get_expected_performance():
    """Expected performance metrics for this environment."""
    return {
        "target_success_rate": 0.65,
        "expected_phase1_convergence": 500,
        "expected_phase2_convergence": 800,
        "key_metrics": [
            "success_rate",
            "path_efficiency",
            "collision_avoidance",
            "multi_agent_coordination",
        ],
    }
