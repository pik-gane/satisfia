"""
Map loader module for loading different environment maps.
"""

import importlib
import sys
from pathlib import Path

# Default map if none specified
DEFAULT_MAP = "simple_map"


def list_available_maps():
    """List all available map modules in the 'envs' directory."""
    maps = []
    envs_dir = Path(__file__).parent

    for file_path in envs_dir.glob("*.py"):
        if file_path.stem != "map_loader" and not file_path.stem.startswith("__"):
            maps.append(file_path.stem)

    return sorted(maps)


def load_map(map_name=DEFAULT_MAP):
    """
    Load a map by name.
    Args:
        map_name: Name of the map module without the .py extension

    Returns:
        Tuple of (map_layout, map_metadata)

    Raises:
        ImportError: If the map cannot be loaded
    """
    try:
        # First try relative import (when running from root with proper package structure)
        map_module = importlib.import_module(
            f".{map_name}", package="corrigibility.marl.envs"
        )
        return map_module.get_map()
    except (ImportError, ModuleNotFoundError):
        try:
            # If relative import fails, try direct import from current directory
            # This handles the case when running from the marl folder
            envs_dir = Path(__file__).parent
            map_file = envs_dir / f"{map_name}.py"

            if not map_file.exists():
                available_maps = list_available_maps()
                raise ImportError(
                    f"Map file '{map_name}.py' not found. Available maps: {', '.join(available_maps)}"
                )

            # Add the envs directory to sys.path temporarily if not already there
            envs_dir_str = str(envs_dir)
            if envs_dir_str not in sys.path:
                sys.path.insert(0, envs_dir_str)

            # Import the module directly
            map_module = importlib.import_module(map_name)
            return map_module.get_map()

        except (ImportError, AttributeError) as e:
            available_maps = list_available_maps()
            raise ImportError(
                f"Failed to load map '{map_name}': {e}. "
                f"Available maps: {', '.join(available_maps)}"
            )
