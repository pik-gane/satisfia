"""
Complex map definition with multiple rooms and challenges.
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

# Define a more complex map layout
COMPLEX_MAP = [
    "###############",
    "#R           K#",
    "# ########### #",
    "# #         # #",
    "# # ####### # #",
    "# # #     # # #",
    "# # # ### # # #",
    "# # # # # # # #",
    "# # # # # # # #",
    "# ### # # ### #",
    "#     # #     #",
    "####### #######",
    "#L     D     L#",
    "#             #",
    "###  #####  ###",
    "#  H         G#",
    "###############"
]

# Metadata for this map
MAP_METADATA = {
    "name": "Complex Maze",
    "description": "A more complex map with multiple paths and hazards",
    "size": (17, 15),  # (rows, cols)
    "max_steps": 300,  # More steps for a more complex map
}

def get_map():
    """Return the map layout and metadata."""
    return COMPLEX_MAP, MAP_METADATA