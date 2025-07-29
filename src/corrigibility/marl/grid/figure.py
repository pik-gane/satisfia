
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# --- Constants ---
GRID_SIZE = 3
CELL_SIZE = 1
GRID_COLOR = 'lightgray'

# Note: Emoji rendering depends on the font support of the underlying OS and Matplotlib backend. 
# If emojis do not appear correctly, you may need to install a font that supports them.
GOAL_EMOJI = '🏁'
LOCK_EMOJI = '🔒'
WALL_EMOJI = '🧱'

# Colors
GOAL_COLOR = 'lightgreen'
ROBOT_PATH_COLOR = 'blue'
HUMAN_PATH_COLOR = 'darkgreen'
ROBOT_LABEL_BG = 'lightblue'
HUMAN_LABEL_BG = 'lightgreen'

# Coordinates (y, x) from plan, but plotting uses (x, y)
ROBOT_START = (0, 2)
HUMAN_START = (2, 2)
KEY_POS = (1, 1)
LOCK_POS = (2, 1)
GOAL_POS = (2, 0)
WALL_POS = (1, 0)

ROBOT_PATH = [(0, 2), (0, 1), (1, 1)]
HUMAN_PATH = [(2, 2), (2, 1), (2, 0)]

# Image paths
ROBOT_IMG_PATH = 'assets/robot.png'
HUMAN_IMG_PATH = 'assets/human.png'
KEY_IMG_PATH = 'assets/key.png'
GOAL_IMG_PATH = 'assets/goal.png'
LOCK_IMG_PATH = 'assets/lock.png'
WALL_IMG_PATH = 'assets/wall.png'

def draw_grid_and_background(ax):
    """Draws the grid lines and sets the background color."""
    ax.set_facecolor('white')
    for x in range(GRID_SIZE + 1):
        ax.axvline(x, color=GRID_COLOR, linewidth=2)
    for y in range(GRID_SIZE + 1):
        ax.axhline(y, color=GRID_COLOR, linewidth=2)
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])

def draw_environment(ax):
    """Draws the environment objects (goal, lock, wall)."""
    draw_image(ax, GOAL_POS, GOAL_IMG_PATH, zoom=0.2, zorder=1)
    draw_image(ax, LOCK_POS, LOCK_IMG_PATH, zoom=0.2, zorder=1)
    draw_image(ax, WALL_POS, WALL_IMG_PATH, zoom=0.2, zorder=1)


def draw_paths(ax):
    """Draws the dashed-line paths for the robot and human."""
    robot_path_x, robot_path_y = zip(*[(x + 0.5, y + 0.5) for x, y in ROBOT_PATH])
    human_path_x, human_path_y = zip(*[(x + 0.5, y + 0.5) for x, y in HUMAN_PATH])
    ax.plot(robot_path_x, robot_path_y, color=ROBOT_PATH_COLOR, linestyle='--', linewidth=2, zorder=2)
    ax.plot(human_path_x, human_path_y, color=HUMAN_PATH_COLOR, linestyle='--', linewidth=2, zorder=2)

def draw_image(ax, pos, img_path, zoom=0.8, zorder=3, alpha=1.0):
    """Helper function to draw an image on the grid."""
    img = plt.imread(img_path)
    imagebox = OffsetImage(img, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(imagebox, (pos[0] + 0.5, pos[1] + 0.5), frameon=False, pad=0)
    ax.add_artist(ab)
    ab.set_zorder(zorder)

def draw_static_elements(ax):
    """Draws the starting agent icons and the key icon."""
    draw_image(ax, ROBOT_START, ROBOT_IMG_PATH, zoom=0.2, zorder=3)
    draw_image(ax, HUMAN_START, HUMAN_IMG_PATH, zoom=0.2, zorder=3)
    # Offset the key slightly to the right to make it clear the robot picks it up
    draw_image(ax, (KEY_POS[0] + 0.2, KEY_POS[1]), KEY_IMG_PATH, zoom=0.15, zorder=3)


def draw_mirage_effect(ax):
    """Draws the semi-transparent 'mirage' robot."""
    draw_image(ax, KEY_POS, ROBOT_IMG_PATH, zoom=0.2, zorder=4, alpha=0.5)

def main():
    """Main function to create the figure."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Layer 0: Grid and Background
    draw_grid_and_background(ax)
    
    # Layer 1: Environment
    draw_environment(ax)
    
    # Layer 2: Paths
    draw_paths(ax)
    
    # Layer 3: Static Elements
    draw_static_elements(ax)
    
    # Layer 4: Mirage Effect
    draw_mirage_effect(ax)
    
    plt.tight_layout(pad=1.5)
    plt.savefig("figure.png", dpi=300)
    plt.close(fig)
    print("Figure saved to figure.png")

if __name__ == "__main__":
    main()
