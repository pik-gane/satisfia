"""
Simple map definition for the locking door environment.
Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'YD': Door
- 'YK': Key
- 'GG': Goal
- 'OL': Lava
"""

# Define the map layout as a list of lists (grid)
# Each inner list represents a row, and each character in the row represents a cell
SIMPLE_MAP = [
    # Row of walls
    ["##", "##", "##", "##", "##"],
    # Robot facing down (vR), Yellow key (YK), Human facing down (vH)
    ["##", "vR", "BK", "vH", "##"],
    # Walls around, Yellow door (YD)
    ["##", "##", "BD", "##", "##"],
    # Floor, Green goal (GG), Floor
    ["##", "  ", "GG", "  ", "##"],
    # Bottom walls
    ["##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Simple Locking Door",
    "description": "A simple map with a locking door separating the robot from the goal",
    "size": (5, 5),  # (rows, cols)
    "max_steps": 200,
    # Map human agent IDs to their goal positions (row, col) in map layout
    "human_goals": {"human_0": (3, 2)},
}

def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA