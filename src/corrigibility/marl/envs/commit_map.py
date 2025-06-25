"""
Commit map definition for a minimal but slightly more complex human-robot environment.
Map legend:
- '##': Wall
- '  ': Empty space
- 'vR': Robot starting position (facing down)
- 'vH0': Human 0 starting position (facing down)
- 'BX': Box
- 'GG': Goal
- '↑', '↓', '←', '→': Arrow buttons (key tiles for human)
"""

# Define the map layout as a list of lists (grid)
COMMIT_MAP = [
    ["##", "##", "##", "##", "##", "##"],
    ["##", "vR", "BX", "  ", "##", "##"],
    ["##", "  ", "  ", "  ", "##", "##"],
    ["##", "##", "##", "  ", "##", "##"],
    ["##", "  ", "↑", "vH0", "↓", "##"],
    ["##", "  ", "←", "GG", "→", "##"],
    ["##", "##", "##", "##", "##", "##"],
]

MAP_METADATA = {
    "name": "Commit Map with Human Buttons and Robot Room",
    "description": "A map with a robot in a box room and a human (H0) surrounded by arrow buttons (key tiles) and a goal.",
    "size": (7, 6),  # (rows, cols)
    "max_steps": 60,
    # Map human agent IDs to their goal positions (row, col) in map layout
    "human_goals": {"human_0": (5, 3)},
}

def get_map():
    """Return the map layout and metadata."""
    return COMMIT_MAP, MAP_METADATA