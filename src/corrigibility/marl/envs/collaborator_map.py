"""
Collaborator map definition for the environment.
Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position
- 'vH': Human starting position
- 'BX': Box
"""

# Define the map layout as a list of lists (grid)
# Each inner list represents a row, and each character in the row represents a cell
COLLABORATOR_MAP = [
    # Top walls
    ["##"] * 7,
    # H0 has many options
    ["##", "##", "##", "vH0", "##", "##", "##"],
    # First box branch
    ["##", "  ", "  ", "BX", "  ", "  ", "##"],
    # Middle wall
    ["##", "  ", "  ", "##", "  ", "  ", "##"],
    # Second box branch
    ["##", "  ", "  ", "BX", "  ", "  ", "##"],
    # Robot and second human H1
    ["##", "vR0", "##", "vH1", "##", "##", "##"],
    # Third box branch
    ["##", "  ", "  ", "BX", "  ", "  ", "##"],
    # Free corridor
    ["##", "##", "##", "  ", "##", "##", "##"],
    # One k-row free row
    ["##", "  ", "  ", "  ", "  ", "  ", "##"],
    # Bottom walls
    ["##"] * 7,
]

# Metadata for this map
MAP_METADATA = {
    "name": "Collaborator Map",
    "description": "Two humans and a robot must choose box pushes; robot has only 8 steps",
    "size": (10, 7),  # (rows, cols)
    "max_steps": 8,  # robot power limit
    # Map human agent IDs to their goal positions (raw map indices before centering)
    "human_goals": {"human_0": (1, 3), "human_1": (5, 3)},
}

def get_map():
    """Return the map layout and metadata."""
    return COLLABORATOR_MAP, MAP_METADATA