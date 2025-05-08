from __future__ import annotations

import numpy as np
from rendering_utils import fill_coords, point_in_rect, point_in_circle, point_in_line

# Simplified color palette similar to MiniGrid
COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "black": np.array([0,0,0]),
    "brown": np.array([139, 69, 19]), # For Door
    "orange": np.array([255, 165, 0]) # For Lava
}

TILE_PIXELS = 32 # Default tile size, can be adjusted

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type: str, color: str):
        self.type = type
        self.color_name = color # Store color name
        self.color = COLORS[color]
        self.contains = None
        self.init_pos: tuple[int, int] | None = None
        self.cur_pos: tuple[int, int] | None = None

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return False
    
    def see_behind(self) -> bool:
        return True

    def toggle(self, env, pos: tuple[int, int]) -> bool:
        return False

    def encode(self) -> tuple[int, int, int]:
        # Placeholder, adapt if encoding is needed like in MiniGrid
        raise NotImplementedError

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        # Placeholder, adapt if decoding is needed
        raise NotImplementedError

    def render(self, img: np.ndarray):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Goal(WorldObj):
    def __init__(self, color: str = "green"):
        super().__init__("goal", color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)


class Floor(WorldObj):
    def __init__(self, color: str = "grey"): # Changed default color for visibility
        super().__init__("floor", color)
        # Floor color is often paler or a background, so apply a modification
        self.render_color = self.color * 0.5 


    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), self.render_color)


class Lava(WorldObj):
    def __init__(self, color: str = "orange"): # Using orange for lava
        super().__init__("lava", color)

    def can_overlap(self):
        return True # Agent "overlaps" then terminates

    def render(self, img):
        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)
        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), COLORS["black"])
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), COLORS["black"])
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), COLORS["black"])
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), COLORS["black"])


class Wall(WorldObj):
    def __init__(self, color: str = "grey"):
        super().__init__("wall", color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), self.color)


class Door(WorldObj):
    def __init__(self, color: str = "brown", is_open: bool = False, is_locked: bool = False, is_permanently_locked: bool = False):
        super().__init__("door", color)
        self.is_open = is_open
        self.is_locked = is_locked
        self.is_permanently_locked = is_permanently_locked

    def can_overlap(self):
        return self.is_open
    
    def see_behind(self) -> bool:
        return self.is_open

    def toggle(self, env, pos): # env is the MiniGridEnv instance
        if self.is_locked:
            # In our custom env, robot_has_key is a property of the env/agent
            if hasattr(env, 'robot_has_key') and env.robot_has_key: # Check if env has this attribute
                self.is_locked = False
                self.is_open = True
                # env.robot_has_key = False # Key is consumed, optional
                return True
            return False

        self.is_open = not self.is_open
        return True

    def render(self, img):
        # Determine color based on state
        if self.is_permanently_locked:
            color = COLORS['red']  # Red for permanently locked
        elif not self.is_locked: # Equivalent to is_open
            color = COLORS['green'] # Green for open
        else: # Closed/normally locked but not permanently
            color = COLORS['yellow'] # Yellow for normally locked/closed

        fill_coords(img, point_in_rect(0.0, 1.0, 0.0, 1.0), color) # Fill entire tile

        # Draw door frame
        fill_coords(img, point_in_rect(0, 0.12, 0, 1), COLORS['black']) # Left frame
        fill_coords(img, point_in_rect(0.88, 1, 0, 1), COLORS['black']) # Right frame
        fill_coords(img, point_in_rect(0, 1, 0, 0.12), COLORS['black']) # Top frame
        fill_coords(img, point_in_rect(0, 1, 0.88, 1), COLORS['black']) # Bottom frame

        # Draw doorknob or handle based on open/locked state if not permalocked
        if not self.is_permanently_locked:
            if not self.is_locked: # Open
                # Handle for open door
                fill_coords(img, point_in_rect(0.75, 0.85, 0.45, 0.55), COLORS['grey'])
            else: # Closed/locked
                # Knob for closed/locked door
                fill_coords(img, point_in_circle(0.78, 0.5, 0.1), COLORS['grey'])


class Key(WorldObj):
    def __init__(self, color: str = "yellow"):
        super().__init__("key", color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = self.color
        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)
        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)
        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), COLORS["black"])

# Placeholder for Ball and Box if needed later, similar to MiniGrid
class Ball(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("ball", color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), self.color)

class Box(WorldObj):
    def __init__(self, color, contains: WorldObj | None = None):
        super().__init__("box", color)
        self.contains = contains # Should be a WorldObj instance

    def can_pickup(self):
        return True # Or False, depending on game mechanics

    def render(self, img):
        c = self.color
        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), COLORS["black"])
        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents if it has any
        if self.contains:
            # This logic assumes env.grid can store WorldObj instances
            # or env has a method to place objects
            # env.grid.set(pos[0], pos[1], self.contains)
            pass # Needs integration with how env handles grid objects
        return True

# Mapping from characters in grid to WorldObj classes for rendering
CHAR_TO_OBJ_CLASS = {
    'G': Goal,
    'K': Key,
    '#': Wall,
    'D': Door,
    'L': Lava,
    ' ': Floor, # Represent empty space as a floor tile
    # 'R' and 'H' will be rendered separately as agents on top of tiles
}