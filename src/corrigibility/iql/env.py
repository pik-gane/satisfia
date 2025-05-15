from __future__ import annotations

import functools
import random
from copy import copy
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete
from enum import IntEnum
# ADDED: PettingZoo imports
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

from objects import WorldObj, Goal, Key, Wall, Door, Lava, Floor, CHAR_TO_OBJ_CLASS, COLORS, TILE_PIXELS
from rendering_utils import fill_coords, point_in_circle
from envs.map_loader import load_map, DEFAULT_MAP

# Define Actions enum
class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    up = 2
    down = 3
    # Pick up an object
    pickup = 4
    # Drop an object
    drop = 5
    # Toggle/activate an object
    toggle = 6
    # ADDED: No-op action for when an agent might not act in a parallel step
    no_op = 7


class CustomEnvironment(AECEnv): # MODIFIED: Inherit from AECEnv
    """Custom grid environment, now inheriting from PettingZoo AECEnv."""

    metadata = {
        "name": "custom_environment_v0",
        "render_modes": ["human"],
        "is_parallelizable": False, # AECEnv is not parallel by default
        "render_fps": 10,
    }

    # Action constants (mirrors Actions enum for direct use)
    ACTION_LEFT = Actions.left
    ACTION_RIGHT = Actions.right
    ACTION_UP = Actions.up
    ACTION_DOWN = Actions.down
    ACTION_PICKUP = Actions.pickup
    ACTION_DROP = Actions.drop
    ACTION_TOGGLE = Actions.toggle
    ACTION_NO_OP = Actions.no_op

    def __init__(self, map_name=DEFAULT_MAP, grid_size=None):
        super().__init__() # ADDED: Call to AECEnv superclass
        
        # Load map layout and metadata
        self.map_layout, self.map_metadata = load_map(map_name)
        
        # Use grid size from map metadata if not explicitly provided
        if grid_size is None and "size" in self.map_metadata:
            rows, cols = self.map_metadata["size"]
            self.grid_size = max(rows, cols)  # Use the larger dimension for square grid
        else:
            self.grid_size = grid_size or 30  # Default grid size
        
        # Override max_steps from map metadata if provided
        if "max_steps" in self.map_metadata:
            self.max_steps = self.map_metadata["max_steps"]
        else:
            self.max_steps = 200  # Default max steps
        
        self.cell_size = TILE_PIXELS
        self.grid_viz_size = self.grid_size * self.cell_size
        self.text_panel_height = 120
        self.window_height = self.grid_viz_size + self.text_panel_height
        self.window_width = self.grid_viz_size
        self.screen = None
        self.clock = None
        self.timestep = 0
        
        # MODIFIED: Agent IDs and management for AECEnv
        self.robot_id_str = "robot_0" # Using PettingZoo typical naming
        self.human_id_str = "human_0"
        self.possible_agents = [self.robot_id_str, self.human_id_str]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector.agent_selector(self.possible_agents) # MODIFIED: Call class within module
        self.agent_selection = None # Will be set in reset
        
        # Initialize environment state variables
        self.door_is_open = False
        self.door_is_key_locked = True
        self.key_pos = None
        self.door_pos = None
        self.goal_pos = None
        self.agent_pos = None  # Robot position
        self.human_pos = None
        self.robot_has_key = False # Specific to robot agent
        self.lava_positions = []
        
        obs_shape_array = np.array(
            [self.grid_size, self.grid_size] * 5 +
            [2] * 3 
        )
        # MODIFIED: Observation and Action spaces as per AECEnv requirements
        self._observation_spaces = {
            agent: MultiDiscrete(obs_shape_array)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: Discrete(len(Actions))
            for agent in self.possible_agents
        }
        
        # MODIFIED: AECEnv requires these dictionaries
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        
        # Initialize grid
        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='U1')

    def _is_cardinally_adjacent(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
        """Check if pos1 is cardinally adjacent to pos2."""
        r1, c1 = pos1
        r2, c2 = pos2
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

    def _parse_map_layout(self):
        """Parse the map layout to set up the environment."""
        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='U1')
        self.lava_positions = []
        
        # Get dimensions of the map
        map_height = len(self.map_layout)
        map_width = max(len(row) for row in self.map_layout)
        
        # Calculate offsets to center the map in the grid
        row_offset = (self.grid_size - map_height) // 2
        col_offset = (self.grid_size - map_width) // 2
        
        # Parse the map layout
        for r, row in enumerate(self.map_layout):
            for c, cell in enumerate(row):
                grid_r = r + row_offset
                grid_c = c + col_offset
                
                if grid_r < 0 or grid_r >= self.grid_size or grid_c < 0 or grid_c >= self.grid_size:
                    continue  # Skip if out of bounds
                
                # Set grid cell based on map character
                if cell == '#':
                    # Wall
                    self.grid[grid_r, grid_c] = '#'
                elif cell == 'D':
                    # Door
                    self.door_pos = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = 'D'
                elif cell == 'K':
                    # Key
                    self.key_pos = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = 'K'
                elif cell == 'G':
                    # Goal
                    self.goal_pos = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = 'G'
                elif cell == 'L':
                    # Lava
                    lava_pos = (grid_r, grid_c)
                    self.lava_positions.append(lava_pos)
                    self.grid[grid_r, grid_c] = 'L'
                elif cell == 'R':
                    # Robot starting position
                    self.agent_pos = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = 'R'
                elif cell == 'H':
                    # Human starting position
                    self.human_pos = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = 'H'
        
        # If robot and human start at the same position, adjust the grid character
        if self.human_pos == self.agent_pos:
            self.grid[self.human_pos] = 'R'  # Both agents at same spot, show robot for now
        
        # Ensure essential elements are defined
        if self.agent_pos is None:
            self.agent_pos = (1, 1)  # Default robot position
            self.grid[self.agent_pos] = 'R'
            
        if self.human_pos is None:
            self.human_pos = self.agent_pos  # Default to same as robot
            
        if self.door_pos is None:
            # Try to find a suitable position for door if not defined in map
            center_row = self.grid_size // 2
            for c in range(1, self.grid_size - 1):
                if self.grid[center_row, c] == ' ':
                    self.door_pos = (center_row, c)
                    self.grid[self.door_pos] = 'D'
                    break
        
        if self.goal_pos is None:
            # Try to find a suitable position for goal
            for r in range(self.grid_size - 2, 0, -1):
                if self.grid[r, self.grid_size // 2] == ' ':
                    self.goal_pos = (r, self.grid_size // 2)
                    self.grid[self.goal_pos] = 'G'
                    break
        
        if self.key_pos is None:
            # Try to find a suitable position for key
            for r in range(1, self.grid_size - 1):
                for c in range(1, self.grid_size - 1):
                    if self.grid[r, c] == ' ':
                        self.key_pos = (r, c)
                        self.grid[self.key_pos] = 'K'
                        return
    
    def reset(self, seed=None, options=None): # MODIFIED: AECEnv reset signature
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        # MODIFIED: AECEnv agent and state reset
        self.agents = self.possible_agents[:] # Active agents
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next() # Set current agent
        self.timestep = 0
        self.robot_has_key = False
        self.door_is_open = False
        self.door_is_key_locked = True
        
        # Reset rewards, terminations, etc.
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Parse map layout to set up environment
        self._parse_map_layout()
        
        # AECEnv reset doesn't return observations directly.
        # Observations are fetched via observe() or last().

    def _get_obs(self, agent_id): # agent_id is for context, obs is global
        obs_door_pos = self.door_pos if self.door_pos is not None else (-1,-1)
        obs_goal_pos = self.goal_pos if self.goal_pos is not None else (-1,-1)
        obs_agent_pos = self.agent_pos if self.agent_pos is not None else (-1,-1) # This is robot's pos
        obs_human_pos = self.human_pos if self.human_pos is not None else (-1,-1)
        obs_key_pos = self.key_pos if self.key_pos is not None else (-1,-1)
        return np.array([
            obs_agent_pos[0], obs_agent_pos[1],
            obs_human_pos[0], obs_human_pos[1],
            obs_door_pos[0], obs_door_pos[1],
            obs_goal_pos[0], obs_goal_pos[1],
            obs_key_pos[0], obs_key_pos[1],
            int(self.robot_has_key), # Robot-specific part of state
            int(self.door_is_open),
            int(self.door_is_key_locked)
        ])

    def observe(self, agent_id): # MODIFIED: Standard AECEnv observe method
        # Note: agent_id here is one of self.possible_agents (e.g., "robot_0")
        # The observation is global but returned for the specified agent.
        obs_door_pos = self.door_pos if self.door_pos is not None else (-1,-1)
        obs_goal_pos = self.goal_pos if self.goal_pos is not None else (-1,-1)
        obs_agent_pos = self.agent_pos if self.agent_pos is not None else (-1,-1) # This is robot's pos
        obs_human_pos = self.human_pos if self.human_pos is not None else (-1,-1)
        obs_key_pos = self.key_pos if self.key_pos is not None else (-1,-1)
        return np.array([
            obs_agent_pos[0], obs_agent_pos[1],
            obs_human_pos[0], obs_human_pos[1],
            obs_door_pos[0], obs_door_pos[1],
            obs_goal_pos[0], obs_goal_pos[1],
            obs_key_pos[0], obs_key_pos[1],
            int(self.robot_has_key), # Robot-specific part of state
            int(self.door_is_open),
            int(self.door_is_key_locked)
        ])

    def _move_agent_vanilla(self, current_pos, action):
        x, y = current_pos
        if action == self.ACTION_LEFT: new_pos = (x, y - 1)
        elif action == self.ACTION_RIGHT: new_pos = (x, y + 1)
        elif action == self.ACTION_UP: new_pos = (x - 1, y)
        elif action == self.ACTION_DOWN: new_pos = (x + 1, y)
        else: new_pos = current_pos
        return new_pos

    def _is_valid_pos(self, pos):
        x, y = pos
        return (0 <= x < self.grid_size and
                0 <= y < self.grid_size and
                self.grid[pos] != "#")

    def _handle_movement(self, agent_to_move_id, new_pos): # agent_to_move_id is "robot_0" or "human_0"
        old_char_on_grid = " " 
        current_agent_char = "R" if agent_to_move_id == self.robot_id_str else "H"
        current_pos = self.agent_pos if agent_to_move_id == self.robot_id_str else self.human_pos
        if self.grid[current_pos] != current_agent_char: 
            old_char_on_grid = self.grid[current_pos]
        elif current_pos == self.key_pos and not (agent_to_move_id == self.robot_id_str and self.robot_has_key):
            old_char_on_grid = "K"
        elif current_pos == self.door_pos: old_char_on_grid = "D"
        elif current_pos == self.goal_pos: old_char_on_grid = "G"
        self.grid[current_pos] = old_char_on_grid
        
        if agent_to_move_id == self.robot_id_str:
            self.agent_pos = new_pos
        elif agent_to_move_id == self.human_id_str:
            self.human_pos = new_pos
        
        if self.grid[new_pos] == " ":
            self.grid[new_pos] = current_agent_char

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self.agent_selection = self._agent_selector.next()
            return

        current_agent_id = self.agent_selection
        action_enum = Actions(action) # Convert int to enum

        # Reset reward for current agent for this step
        self.rewards[current_agent_id] = 0

        # --- AGENT ACTION (Robot or Human) ---
        if current_agent_id == self.robot_id_str:
            if action_enum != self.ACTION_NO_OP:
                current_pos = self.agent_pos
                if self.ACTION_LEFT <= action_enum <= self.ACTION_DOWN: # Movement
                    new_pos = self._move_agent_vanilla(current_pos, action_enum)
                    can_move = True
                    if not self._is_valid_pos(new_pos): can_move = False
                    elif new_pos == self.door_pos and not self.door_is_open: can_move = False
                    if can_move: self._handle_movement(self.robot_id_str, new_pos)
                elif action_enum == self.ACTION_PICKUP:
                    if self.agent_pos == self.key_pos and self.key_pos is not None and not self.robot_has_key:
                        self.robot_has_key = True; self.key_pos = None
                elif action_enum == self.ACTION_DROP:
                    if self.robot_has_key and self.key_pos is None and (self.grid[self.agent_pos] == 'R' or self.grid[self.agent_pos] == ' '):
                        self.robot_has_key = False; self.key_pos = self.agent_pos; self.grid[self.key_pos] = "K"
                elif action_enum == self.ACTION_TOGGLE:
                    if self._is_cardinally_adjacent(self.agent_pos, self.door_pos):
                        if self.door_is_key_locked:
                            if self.robot_has_key: self.door_is_key_locked = False; self.door_is_open = True
                        else: self.door_is_open = not self.door_is_open
        
        elif current_agent_id == self.human_id_str:
            if action_enum != self.ACTION_NO_OP:
                current_pos = self.human_pos
                if self.ACTION_LEFT <= action_enum <= self.ACTION_DOWN: # Movement
                    new_pos = self._move_agent_vanilla(current_pos, action_enum)
                    can_move_human = True
                    if not self._is_valid_pos(new_pos): can_move_human = False
                    elif new_pos == self.door_pos and not self.door_is_open: can_move_human = False
                    if can_move_human: self._handle_movement(self.human_id_str, new_pos)
            # Human specific rewards are based on reaching goal, checked globally below.

        # --- GLOBAL TERMINAL CONDITIONS & REWARDS (after action) ---
        # These conditions affect all agents' done status or provide rewards.
        human_reached_goal = (self.human_pos == self.goal_pos)
        robot_in_lava = (self.agent_pos in self.lava_positions)
        human_in_lava = (self.human_pos in self.lava_positions)
        
        current_agent_is_human = (current_agent_id == self.human_id_str)
        current_agent_is_robot = (current_agent_id == self.robot_id_str)

        if human_reached_goal:
            if current_agent_is_human: self.rewards[current_agent_id] += 1
            if current_agent_is_robot: self.rewards[current_agent_id] += 1 # Robot also gets reward
            self.terminations = {agent: True for agent in self.agents} # Episode ends for all

        if robot_in_lava:
            if current_agent_is_robot: self.rewards[current_agent_id] -= 1
            self.terminations = {agent: True for agent in self.agents}

        if human_in_lava:
            if current_agent_is_human: self.rewards[current_agent_id] -= 1
            self.terminations = {agent: True for agent in self.agents}
        
        # Update cumulative rewards
        self._cumulative_rewards[current_agent_id] += self.rewards[current_agent_id]

        if self._agent_selector.is_last():
            self.timestep += 1 # One full round of agent steps

        if self.timestep >= self.max_steps:
            self.truncations = {agent: True for agent in self.agents}

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def render(self): 
        if self.render_mode is None: 
            return

        if self.screen is None and self.render_mode == "human": 
            pygame.init()
            pygame.font.init() 
            pygame.display.set_caption("Locking Door Environment")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.agent_char_font = pygame.font.SysFont("Arial", int(self.cell_size * 0.75)) 
            self.info_font = pygame.font.SysFont("Arial", 20) 

        pygame.event.pump()
        self.screen.fill(COLORS["grey"] * 0.8) 

        grid_surface = pygame.Surface((self.grid_viz_size, self.grid_viz_size))
        grid_surface.fill(COLORS["grey"] * 0.5)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                char_on_grid = self.grid[r, c] 
                tile_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                tile_img_np = np.zeros((self.cell_size, self.cell_size, 3), dtype=np.uint8)
                
                floor_obj = Floor()
                floor_obj.render(tile_img_np)

                obj_to_render = None
                current_tile_coords = (r, c)

                if current_tile_coords == self.door_pos:
                    obj_to_render = Door(
                        is_locked=self.door_is_key_locked,
                        is_open=self.door_is_open
                    )
                elif current_tile_coords == self.key_pos: 
                    obj_to_render = Key()
                elif current_tile_coords == self.goal_pos:
                    obj_to_render = Goal()
                elif char_on_grid == '#': 
                    obj_to_render = Wall()
                elif char_on_grid == 'L': 
                    obj_to_render = Lava()
                
                if obj_to_render:
                    obj_to_render.render(tile_img_np)

                agent_char_to_render = None
                char_color = COLORS["black"] 

                current_tile_tuple = (r,c)
                if current_tile_tuple == self.agent_pos:
                    agent_char_to_render = "R"
                    char_color = COLORS["blue"]
                elif current_tile_tuple == self.human_pos:
                    agent_char_to_render = "H"
                    char_color = COLORS["purple"]

                if agent_char_to_render:
                    temp_tile_surface = pygame.surfarray.make_surface(np.transpose(tile_img_np, (1, 0, 2)))
                    
                    text_surface = self.agent_char_font.render(agent_char_to_render, True, char_color)
                    text_rect = text_surface.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
                    temp_tile_surface.blit(text_surface, text_rect)
                    
                    tile_img_np = pygame.surfarray.array3d(temp_tile_surface) 
                    tile_img_np = np.transpose(tile_img_np, (1,0,2)) 

                tile_surface = pygame.surfarray.make_surface(np.transpose(tile_img_np, (1,0,2)))
                grid_surface.blit(tile_surface, tile_rect.topleft)
                
                pygame.draw.rect(grid_surface, COLORS["black"], tile_rect, 1)

        self.screen.blit(grid_surface, (0,0))
        
        text_y_start = self.grid_viz_size + 5 
        line_height = 22 

        text_panel_rect = pygame.Rect(0, self.grid_viz_size, self.window_width, self.text_panel_height)
        self.screen.fill(COLORS["grey"] * 0.8, text_panel_rect) 

        map_name = self.map_metadata.get("name", "Unknown Map")
        
        texts_to_render = [
            f"Map: {map_name} | Step: {self.timestep}/{self.max_steps}", 
            f"Robot@: {self.agent_pos} | Human@: {self.human_pos}",
            f"Key@: {self.key_pos if self.key_pos else 'Picked Up'} | Robot Has Key: {self.robot_has_key}",
            f"Door@: {self.door_pos} | Open: {self.door_is_open} | KeyLock: {self.door_is_key_locked}",
            f"Goal@: {self.goal_pos}"
        ]

        for i, text_content in enumerate(texts_to_render):
            text_surface = self.info_font.render(text_content, True, COLORS["black"])
            self.screen.blit(text_surface, (5, text_y_start + i * line_height))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self): 
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id): 
        return self._observation_spaces[agent_id]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id): 
        return self._action_spaces[agent_id]

class LockingDoorEnvironment(CustomEnvironment):
    """Environment with a locking door. Inherits AECEnv behavior from CustomEnvironment."""
    def __init__(self, map_name=DEFAULT_MAP, grid_size=None):
        super().__init__(map_name=map_name, grid_size=grid_size)
        # Any LockingDoorEnvironment specific configuration can be done here