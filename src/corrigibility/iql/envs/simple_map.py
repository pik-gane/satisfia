"""
Simple map definition for the locking door environment.
Map legend:
- '#': Wall
- ' ': Empty space
- 'R': Robot starting position
- 'H': Human starting position
- 'D': Door
- 'K': Key
- 'G': Goal
- 'L': Lava
"""

# Define the map layout as a list of lists (grid)
# Each inner list represents a row, and each character in the row represents a cell
SIMPLE_MAP = [
    "##########",
    "#R      K#",
    "#        #",
    "#        #",
    "####D#####",
    "#        #",
    "#        #",
    "#        #",
    "#G       #",
    "##########"
]

# Metadata for this map
MAP_METADATA = {
    "name": "Simple Locking Door",
    "description": "A simple map with a locking door separating the robot from the goal",
    "size": (10, 10),  # (rows, cols)
    "max_steps": 200,
}

def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA