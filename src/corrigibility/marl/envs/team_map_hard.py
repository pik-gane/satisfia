"""
Team map definition for multi-human environment.
Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH0': Human 0 starting position
- 'vH1': Human 1 starting position
- 'YD': Door
- 'YK': Key
- 'GG': Green Goal (for human_0)
- 'RG': Red Goal (for human_1)
- 'OL': Lava
"""

# Define the map layout as a list of lists (grid)
# Each inner list represents a row, and each character in the row represents a cell
SIMPLE_MAP = [
    # Top walls
    ["##", "##", "##", "##", "##", "##", "##", "##"],
    # Starting positions: Robot in center, humans on sides
    ["##", "vH0", "  ", "vR", "  ", "  ", "vH1", "##"],
    # Empty middle area
    ["##", "##", "##", "  ", "  ", "  ", "##", "##"],
    # Goals at bottom: Green goal (left) and Red goal (right)
    ["##", "GG", "BD", "BK", "  ", "RK", "RG", "##"],
    # Bottom walls
    ["##", "##", "##", "##", "##", "##", "##", "##"],
]

# Metadata for this map
MAP_METADATA = {
    "name": "Team Multi-Goal Map",
    "description": "A map with multiple humans who have different goals to reach",
    "size": (5, 7),  # (rows, cols)
    "max_steps": 200,
    # Map human agent IDs to their goal positions (row, col) in map layout
    "human_goals": {"human_0": (3, 1), "human_1": (3, 5)},  # GG and RG positions
}

def get_map():
    """Return the map layout and metadata."""
    return SIMPLE_MAP, MAP_METADATA