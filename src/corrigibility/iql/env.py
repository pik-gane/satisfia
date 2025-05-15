from __future__ import annotations

import functools
import random
from copy import copy
from typing import Tuple

import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete
from enum import IntEnum
# ADDED: PettingZoo imports
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import agent_selector

from objects import WorldObj, Goal, Key, Wall, Door, Lava, Floor, CHAR_TO_OBJ_CLASS, COLORS, TILE_PIXELS
from rendering_utils import fill_coords, point_in_circle

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

    def __init__(self, grid_size=30):
        super().__init__() # ADDED: Call to AECEnv superclass
        self.grid_size = grid_size
        self.cell_size = TILE_PIXELS
        self.grid_viz_size = self.grid_size * self.cell_size
        self.text_panel_height = 120
        self.window_height = self.grid_viz_size + self.text_panel_height
        self.window_width = self.grid_viz_size

        self.screen = None
        self.clock = None

        self.timestep = 0
        self.max_steps = 200 # Max steps per episode

        # MODIFIED: Agent IDs and management for AECEnv
        self.robot_id_str = "robot_0" # Using PettingZoo typical naming
        self.human_id_str = "human_0"
        self.possible_agents = [self.robot_id_str, self.human_id_str]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector.agent_selector(self.possible_agents) # MODIFIED: Call class within module
        self.agent_selection = None # Will be set in reset

        self.door_is_open = False
        self.door_is_key_locked = True
        self.key_pos = None
        self.robot_has_key = False # Specific to robot agent

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

        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='U1')
        
        # Note: AECEnv typically calls reset() externally after __init__.
        # However, to ensure internal state like grid is ready for first render or obs,
        # we can call parts of reset logic or a full reset here.
        # For now, we'll let the first external reset() call fully initialize.
        # self.reset() # If called here, ensure it doesn't conflict with external calls.


    def _is_cardinally_adjacent(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> bool:
        """Check if pos1 is cardinally adjacent to pos2."""
        r1, c1 = pos1
        r2, c2 = pos2
        return (abs(r1 - r2) == 1 and c1 == c2) or (abs(c1 - c2) == 1 and r1 == r2)

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

        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='U1')
        
        self.grid[0, :] = "#"
        self.grid[-1, :] = "#"
        self.grid[:, 0] = "#"
        self.grid[:, -1] = "#"

        # Adjust room, door, key, goal, agent, human positions for larger grid
        room_wall_row = self.grid_size // 2 
        for c in range(1, self.grid_size - 1):
            self.grid[room_wall_row, c] = "#"

        self.door_pos = (room_wall_row, self.grid_size // 2)
        self.grid[self.door_pos] = "D"

        # Key position: top middle tile
        self.key_pos = (1, self.grid_size // 2)
        self.grid[self.key_pos] = "K"

        # Goal position: should be it bottom center tile
        self.goal_pos = (self.grid_size - 2, self.grid_size // 2)
        
        self.grid[self.goal_pos] = "G"
        
        self.lava_positions = [] # No lava for this scenario

        # Agent and Human start at top-left (e.g., (1,1) inside border walls)
        self.agent_pos = (1, 1)
        self.grid[self.agent_pos] = "R" 

        self.human_pos = (1, 1) # Human also starts at (1,1)
        # If human and robot start at the same spot, rendering will show one on top.
        # The logic in _handle_movement and rendering should correctly show them if they move apart.
        # Ensure the grid character for human is placed if they are not at the same spot as robot initially,
        # or rely on rendering to draw both if at same spot.
        if self.human_pos != self.agent_pos: # This condition will be false if they start at same spot
            self.grid[self.human_pos] = "H"
        
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

    # REMOVED: Parallel step(self, action_robot, action_human)
    # def step(self, action_robot: int, action_human: int): ...

    # ADDED: AECEnv step(self, action)
    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Handle dead agent: select next agent and return
            # PettingZoo AECEnv typically expects step(None) for dead agents,
            # but IQL might not call it this way. If it does, this handles it.
            # If IQL skips stepping dead agents, this won't be an issue.
            self._was_dead_step = True # Internal flag if needed
            # No reward update for dead agent on this step
            # self.rewards[self.agent_selection] = 0 # Already done or not applicable
            if self._agent_selector.is_last():
                # If it was the last agent in the cycle, and it's dead,
                # this might indicate a need to check global done conditions.
                # However, individual done flags are primary.
                pass
            else:
                # Select next agent if current one is done
                # This logic might be complex if all agents become done simultaneously.
                # The IQL loop will likely manage its own agent cycling.
                pass # Agent selector will be advanced by IQL or main loop
            # For AECEnv, after a dead agent's "turn", we still need to select the next agent.
            # However, the IQL will manage its own calls.
            # The key is that self.rewards, self.terminations, etc. for agent_selection are set.
            # The IQL will need to fetch these after its conceptual joint step.
            # This step() is for a single agent.
            # If IQL calls step for a done agent, we just ensure its state is consistent.
            # The agent_selector is advanced *after* this block.
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
            # Both agents get reward if human reaches goal, but only set for current agent if it's their turn
            # The IQL will need to sum rewards or assign appropriately.
            # For AECEnv, reward is for the current agent.
            if current_agent_is_human: self.rewards[current_agent_id] += 1
            if current_agent_is_robot: self.rewards[current_agent_id] += 1 # Robot also gets reward
            # print(f"Human reached goal at: {self.human_pos}") # ADDED: Print statement
            self.terminations = {agent: True for agent in self.agents} # Episode ends for all

        if robot_in_lava:
            if current_agent_is_robot: self.rewards[current_agent_id] -= 1
            self.terminations = {agent: True for agent in self.agents}

        if human_in_lava:
            if current_agent_is_human: self.rewards[current_agent_id] -= 1
            self.terminations = {agent: True for agent in self.agents}
        
        # Update cumulative rewards
        self._cumulative_rewards[current_agent_id] += self.rewards[current_agent_id]

        # Advance timestep only if it's the last agent in a "round"
        # Or, more simply for IQL, IQL can manage its own step counting for episodes.
        # For AECEnv, timestep usually means full cycles.
        if self._agent_selector.is_last():
            self.timestep += 1 # One full round of agent steps

        # Check for truncation (max_steps)
        if self.timestep >= self.max_steps:
            self.truncations = {agent: True for agent in self.agents}
            # No specific reward for truncation itself unless designed.

        # If any agent is terminated or truncated, they should be removed from self.agents
        # This is standard AECEnv practice.
        # However, IQL might want to see the "done" state before removal.
        # For now, let's assume IQL checks terminations/truncations from the dicts.
        # If an agent is done, it should not act again.
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()

        # Render if in human mode (AECEnv standard practice)
        if self.render_mode == "human":
            self.render()


    # REMOVED: is_terminal(self, s_dict=None) - AECEnv uses terminations/truncations dicts

    def render(self): # Render method remains largely the same, uses self.agent_pos, self.human_pos etc.
        if self.render_mode is None: # From AECEnv, allow render_mode to be None
            # gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.screen is None and self.render_mode == "human": # Check render_mode
            pygame.init()
            pygame.font.init() 
            pygame.display.set_caption("Locking Door Environment")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.agent_char_font = pygame.font.SysFont("Arial", int(self.cell_size * 0.75)) 
            self.info_font = pygame.font.SysFont("Arial", 20) # Smaller font for more text

        pygame.event.pump()
        self.screen.fill(COLORS["grey"] * 0.8) # Background for text panel area

        grid_surface = pygame.Surface((self.grid_viz_size, self.grid_viz_size))
        grid_surface.fill(COLORS["grey"] * 0.5)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                char_on_grid = self.grid[r, c] # Keep for static elements if needed
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
                elif current_tile_coords == self.key_pos: # Render key only if its at its position (key_pos is None if held)
                    obj_to_render = Key()
                elif current_tile_coords == self.goal_pos:
                    obj_to_render = Goal()
                elif char_on_grid == '#': # Wall character from grid
                    obj_to_render = Wall()
                elif char_on_grid == 'L': # Lava character from grid (assuming 'L' is used for Lava)
                    obj_to_render = Lava()
                # Add other static objects based on char_on_grid if necessary
                
                if obj_to_render:
                    obj_to_render.render(tile_img_np)

                agent_char_to_render = None
                char_color = COLORS["black"] # Default character color

                current_tile_tuple = (r,c)
                if current_tile_tuple == self.agent_pos:
                    agent_char_to_render = "R"
                    char_color = COLORS["blue"]
                elif current_tile_tuple == self.human_pos:
                    agent_char_to_render = "H"
                    char_color = COLORS["purple"]

                if agent_char_to_render:
                    # Create a Pygame Surface from the current tile_img_np to draw text on it
                    temp_tile_surface = pygame.surfarray.make_surface(np.transpose(tile_img_np, (1, 0, 2)))
                    
                    text_surface = self.agent_char_font.render(agent_char_to_render, True, char_color)
                    text_rect = text_surface.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
                    temp_tile_surface.blit(text_surface, text_rect)
                    
                    # Convert back to numpy array for the main rendering path
                    tile_img_np = pygame.surfarray.array3d(temp_tile_surface) # Removed 'out' argument
                    tile_img_np = np.transpose(tile_img_np, (1,0,2)) # Transpose back because make_surface did

                tile_surface = pygame.surfarray.make_surface(np.transpose(tile_img_np, (1,0,2)))
                grid_surface.blit(tile_surface, tile_rect.topleft)
                
                pygame.draw.rect(grid_surface, COLORS["black"], tile_rect, 1)


        self.screen.blit(grid_surface, (0,0))
        
        text_y_start = self.grid_viz_size + 5 # Start text panel just below grid
        line_height = 22 # Adjusted line height for smaller font

        text_panel_rect = pygame.Rect(0, self.grid_viz_size, self.window_width, self.text_panel_height)
        self.screen.fill(COLORS["grey"] * 0.8, text_panel_rect) 

        texts_to_render = [
            f"Step: {self.timestep}/{self.max_steps}", # MODIFIED: Show max_steps
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

    def close(self): # Standard AECEnv close
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id): # MODIFIED: agent_id is "robot_0" or "human_0"
        return self._observation_spaces[agent_id]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id): # MODIFIED: agent_id is "robot_0" or "human_0"
        return self._action_spaces[agent_id]

class LockingDoorEnvironment(CustomEnvironment):
    """Environment with a locking door. Inherits AECEnv behavior from CustomEnvironment."""
    def __init__(self, grid_size=5): 
        super().__init__(grid_size=grid_size)
        # Specifics for LockingDoorEnvironment if any, e.g., different max_steps
        # self.max_steps = 150 
        pass