# Curriculum Environments Package
"""
This package contains a sequence of environments for curriculum training.
Each environment is defined in its own module with specific configurations
and difficulty progressions.
"""

from .env_01_simple import get_env_config as env_01_config
from .env_01_simple import get_map as env_01_map
from .env_02_basic_door import get_env_config as env_02_config
from .env_02_basic_door import get_map as env_02_map
from .env_03_multi_key import get_env_config as env_03_config
from .env_03_multi_key import get_map as env_03_map
from .env_04_obstacles import get_env_config as env_04_config
from .env_04_obstacles import get_map as env_04_map
from .env_05_larger_grid import get_env_config as env_05_config
from .env_05_larger_grid import get_map as env_05_map

# List of all environment configs in training order
CURRICULUM_SEQUENCE = [
    ("env_01_simple", env_01_config),
    ("env_02_basic_door", env_02_config),
    ("env_03_multi_key", env_03_config),
    ("env_04_obstacles", env_04_config),
    ("env_05_larger_grid", env_05_config),
]

# Map environment names to their get_map functions
ENV_MAP_FUNCTIONS = {
    "env_01_simple": env_01_map,
    "env_02_basic_door": env_02_map,
    "env_03_multi_key": env_03_map,
    "env_04_obstacles": env_04_map,
    "env_05_larger_grid": env_05_map,
}


def get_curriculum_env(env_index):
    """Get environment config by index."""
    if env_index < 0 or env_index >= len(CURRICULUM_SEQUENCE):
        raise ValueError(
            f"Environment index {env_index} out of range [0, {len(CURRICULUM_SEQUENCE)-1}]"
        )

    env_name, env_config_func = CURRICULUM_SEQUENCE[env_index]
    return env_name, env_config_func()


def get_curriculum_map(env_name):
    """Get hardcoded map for environment by name."""
    if env_name in ENV_MAP_FUNCTIONS:
        return ENV_MAP_FUNCTIONS[env_name]()
    else:
        raise ValueError(f"Environment {env_name} not found in curriculum")


def get_all_env_names():
    """Get list of all environment names."""
    return [name for name, _ in CURRICULUM_SEQUENCE]


__all__ = [
    "CURRICULUM_SEQUENCE",
    "ENV_MAP_FUNCTIONS",
    "get_curriculum_env",
    "get_curriculum_map",
    "get_all_env_names",
    "env_01_config",
    "env_01_map",
    "env_02_config",
    "env_02_map",
    "env_03_config",
    "env_03_map",
    "env_04_config",
    "env_04_map",
    "env_05_config",
    "env_05_map",
]
