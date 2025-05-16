"""
Map loader module for loading different environment maps.
"""
import importlib
import os
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
        # Import the map module dynamically
        map_module = importlib.import_module(f".{map_name}", package="envs")
        # Call get_map() function to retrieve map data
        return map_module.get_map()
    except (ImportError, AttributeError) as e:
        available_maps = list_available_maps()
        raise ImportError(
            f"Failed to load map '{map_name}': {e}. "
            f"Available maps: {', '.join(available_maps)}"
        )