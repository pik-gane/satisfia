from __future__ import annotations

import functools
import random
from copy import copy
from typing import Tuple, List, Dict, Any, Optional, Literal

import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete
from enum import IntEnum
# ADDED: PettingZoo imports
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import agent_selector

from objects import WorldObj, Goal, Key, Wall, Door, Lava, Floor, Box, CHAR_TO_OBJ_CLASS, COLORS, TILE_PIXELS
from rendering_utils import fill_coords, point_in_circle
from envs.map_loader import load_map, DEFAULT_MAP

# Mapping from code letter to color names
COLOR_CODE_MAP = {'B': 'blue', 'G': 'green', 'Y': 'yellow', 'P': 'purple', 'O': 'orange', 'R': 'red', 'W': 'white'}
# Direction code to orientation index
DIR_CODE_MAP = {'^': 0, '>': 1, 'v': 2, '<': 3}

# Define Actions enum
class Actions(IntEnum):
    # Turn left, turn right, move forward
    turn_left = 0
    turn_right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5
    # No-op action for when an agent might not act in a parallel step
    done = 6


class CustomEnvironment(ParallelEnv):
    """Custom grid environment using Parallel API."""

    metadata = {
        "name": "custom_environment_v0",
        "render_modes": ["human"],
        "is_parallelizable": True,  # parallel api enabled
        "render_fps": 10,
    }

    # Action constants (mirrors Actions enum for direct use)
    ACTION_LEFT = Actions.turn_left
    ACTION_RIGHT = Actions.turn_right
    ACTION_UP = Actions.forward
    ACTION_DOWN = Actions.forward
    ACTION_PICKUP = Actions.pickup
    ACTION_DROP = Actions.drop
    ACTION_TOGGLE = Actions.toggle
    ACTION_NO_OP = Actions.done

    def __init__(self, map_name=DEFAULT_MAP, grid_size=None, debug_mode=False, debug_level='standard'): # Added debug_level
        super().__init__()
        self.debug_mode = debug_mode # Store debug_mode
        self.debug_level = debug_level # Store debug_level
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
        
        # Determine agent IDs (support multiple humans/robots) from map codes
        robot_ids = set()
        human_ids = set()
        for row in self.map_layout:
            for code in row:
                if isinstance(code, str) and len(code) >= 3:
                    if code[1] == 'R':
                        robot_ids.add(code[2])
                    if code[1] == 'H':
                        human_ids.add(code[2])
        # Default to '0' if none found
        if not robot_ids:
            robot_ids.add('0')
        if not human_ids:
            human_ids.add('0')
        # Build agent ID strings
        self.robot_agent_ids = [f"robot_{rid}" for rid in sorted(robot_ids)]
        self.human_agent_ids = [f"human_{hid}" for hid in sorted(human_ids)]
        self.possible_agents = self.robot_agent_ids + self.human_agent_ids
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._agent_selector = agent_selector.agent_selector(self.possible_agents)
        self.agent_selection = None # Will be set in reset
        # Initialize orientation mapping for newly identified agent IDs
        self.agent_dirs = {aid: 2 for aid in self.possible_agents}
        
        # Initialize environment state variables for multiple keys, doors, and boxes
        self.keys = []
        self.doors = []
        self.boxes = []  # list of {'pos':(r,c), 'color':str}
        self.initial_num_keys = 0
        self.robot_has_keys = set()
        # map human agent IDs to goal positions (to be filled in reset)
        self.human_goals: dict[str, tuple[int,int]] = {}
        # track positions per agent
        self.agent_positions: dict[str, tuple[int,int]] = {}
        # lava positions
        self.lava_positions = []
        
        # Reduced observation space: robot_pos(2) + human_pos(2) + robot_dir(1) + human_dir(1)
        obs_shape_array = np.array([
            self.grid_size, self.grid_size,  # robot position
            self.grid_size, self.grid_size,  # human position  
            4,  # robot direction (0-3)
            4   # human direction (0-3)
        ])
        # MODIFIED: Observation and Action spaces as per ParallelEnv requirements
        self._observation_spaces = {
            agent: MultiDiscrete(obs_shape_array)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: Discrete(len(Actions))
            for agent in self.possible_agents
        }
        # Initialize orientations: 0=up,1=right,2=down,3=left; default down
        self.agent_dirs = {agent: 2 for agent in self.possible_agents}
        self.directions = ['up', 'right', 'down', 'left']
        
        # MODIFIED: ParallelEnv requires these dictionaries
        self.rewards = {agent: 0 for agent in self.possible_agents}
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
        map_width = len(self.map_layout[0])
        
        # Calculate offsets to center the map in the grid
        self.row_offset = (self.grid_size - map_height) // 2
        self.col_offset = (self.grid_size - map_width) // 2
        
        # Parse the map layout
        for r, row in enumerate(self.map_layout):
            for c, code in enumerate(row):
                grid_r = r + self.row_offset
                grid_c = c + self.col_offset
                
                if grid_r < 0 or grid_r >= self.grid_size or grid_c < 0 or grid_c >= self.grid_size:
                    continue
                # expect code as 2- or 3-char string (direction+entity+id)
                if not isinstance(code, str) or len(code) < 2:
                    continue
                # for 3-char codes, ignore ID suffix
                color_code = code[0]
                cell = code[1]
                entity_id = code[2] if len(code) >= 3 else None
                color = COLOR_CODE_MAP.get(color_code, 'grey')
                # Set grid cell based on map character
                if cell == '#':  # wall
                    self.grid[grid_r, grid_c] = '#'
                elif cell == 'D':  # door
                    self.doors.append({'pos': (grid_r, grid_c), 'color': color, 'is_open': False, 'is_locked': True})
                    self.grid[grid_r, grid_c] = 'D'
                elif cell == 'K':  # key
                    self.keys.append({'pos': (grid_r, grid_c), 'color': color})
                    self.grid[grid_r, grid_c] = 'K'
                elif cell == 'X':  # box
                    self.boxes.append({'pos': (grid_r, grid_c), 'color': color})
                    self.grid[grid_r, grid_c] = 'X'
                elif cell == 'G':  # goal
                    # Mark goal tile for rendering
                    self.grid[grid_r, grid_c] = 'G'
                elif cell == 'L':  # lava
                    self.lava_positions.append((grid_r, grid_c))
                    self.grid[grid_r, grid_c] = 'L'
                elif cell == 'R':  # robot with ID
                    # entity_id holds the robot suffix
                    rid = entity_id or self.robot_agent_ids[0].split('_')[1]
                    agent_str = f"robot_{rid}"
                    self.agent_positions[agent_str] = (grid_r, grid_c)
                    # clear grid cell for agent overlay
                    self.grid[grid_r, grid_c] = ' '
                    # set orientation
                    self.agent_dirs[agent_str] = DIR_CODE_MAP.get(color_code, 2)
                elif cell == 'H':  # human with ID
                    hid = entity_id or self.human_agent_ids[0].split('_')[1]
                    agent_str = f"human_{hid}"
                    self.agent_positions[agent_str] = (grid_r, grid_c)
                    self.grid[grid_r, grid_c] = ' '
                    self.agent_dirs[agent_str] = DIR_CODE_MAP.get(color_code, 2)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        # reset active agents
        self.agents = self.possible_agents[:]
        # clear items and agent positions so _parse_map_layout can reinitialize
        self.keys = []
        self.doors = []
        self.boxes = []
        self.agent_positions = {}
        self.timestep = 0
        # Reset orientations to default down
        self.agent_dirs = {agent: 2 for agent in self.agents}
        # Reset keys and doors
        self.robot_has_keys = set()
        self.door_is_open = False
        self.door_is_key_locked = True

        # reset step-level data
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # re-initialize map
        self._parse_map_layout()
        # record initial key count after parsing
        self.initial_num_keys = len(self.keys)
        # map human IDs to goal positions from metadata (apply offsets)
        for agent_id, raw in self.map_metadata.get('human_goals', {}).items():
            if agent_id in self.possible_agents:
                raw_r, raw_c = raw
                grid_r = raw_r + self.row_offset
                grid_c = raw_c + self.col_offset
                self.human_goals[agent_id] = (grid_r, grid_c)
        
        # update single-agent positions for backwards compatibility
        # robot position
        rid = self.robot_agent_ids[0]
        if rid in self.agent_positions:
            self.agent_pos = self.agent_positions[rid]
        # human_0 position
        hid0 = self.human_agent_ids[0]
        if hid0 in self.agent_positions:
            self.human_pos = self.agent_positions[hid0]
        # return initial observations for all agents
        return {agent: self.observe(agent) for agent in self.agents}

    def _get_obs(self, agent_id): # agent_id is for context, obs is global
        # Reduced state space: only agent positions and directions
        obs_agent_pos = self.agent_pos if self.agent_pos is not None else (-1,-1) # Robot position
        obs_human_pos = self.human_pos if self.human_pos is not None else (-1,-1) # Human position
        obs_agent_dir = self.agent_dirs.get(self.robot_agent_ids[0], 0) # Robot direction
        obs_human_dir = self.agent_dirs.get(self.human_agent_ids[0], 0) # Human direction
        
        return np.array([
            obs_agent_pos[0], obs_agent_pos[1],
            obs_human_pos[0], obs_human_pos[1],
            obs_agent_dir,
            obs_human_dir
        ])

    def observe(self, agent_id):
        """Standard ParallelEnv observe method with consistent state space for all agents."""
        # Consistent observation for all agents: 
        # [current_agent_direction, all_agent_positions_flattened, door_states, key_states, box_positions]
        obs = []
        
        # Current agent's direction facing
        obs.append(self.agent_dirs.get(agent_id, 0))
        
        # All agent positions (consistent order: robots first, then humans)
        all_agent_ids = self.robot_agent_ids + self.human_agent_ids
        for aid in all_agent_ids:
            pos = self.agent_positions.get(aid, (-1, -1))
            obs.extend([pos[0], pos[1]])
        
        # Door states (open/closed for all doors)
        for door in self.doors:
            obs.append(1 if door['is_open'] else 0)
        
        # Key states: position if not picked up, or special encoding if picked up
        for i, key in enumerate(self.keys):
            # Key exists in world at its position
            obs.extend([key['pos'][0], key['pos'][1]])
        
        # Encode picked up keys (robot inventory) - use special position (-2, -2) for picked up
        for color in self.robot_has_keys:
            obs.extend([-2, -2])  # Special encoding for picked up keys
        
        # Pad to consistent size if needed (handle variable number of keys)
        max_keys = self.initial_num_keys + len(self.robot_has_keys)
        current_key_entries = len(self.keys) + len(self.robot_has_keys)
        for _ in range(max_keys - current_key_entries):
            obs.extend([-1, -1])  # Padding for non-existent keys
        
        # Box positions (flattened)
        for box in self.boxes:
            obs.extend([box['pos'][0], box['pos'][1]])
        
        return np.array(obs, dtype=np.int32)

    def _get_obs(self, agent_id):
        """Legacy method - redirects to observe for compatibility."""
        return self.observe(agent_id)

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
        # within bounds and not wall
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size): return False
        if self.grid[pos] == '#': return False
        # block locked doors
        for door in self.doors:
            if door['pos'] == pos and not door['is_open']: return False
        # block keys and boxes
        for key in self.keys:
            if key['pos'] == pos: return False
        for box in self.boxes:
            if box['pos'] == pos: return False
        return True

    def _handle_movement(self, agent_to_move_id, new_pos):
        # agent_to_move_id is the specific agent moving e.g. "robot_0", "human_0"
        
        # Get the character that represents this agent on the grid (simplified)
        agent_char_on_grid = "R" if agent_to_move_id.startswith("robot") else "H"

        current_pos = self.agent_positions[agent_to_move_id] # Use the definitive position

        # Logic to determine what character to leave behind on the grid
        # This should ideally revert to the static map feature at current_pos
        old_char_on_grid = " " # Default to empty space
        
        # A simple way: if current_pos is a goal tile for any human, leave 'G'
        is_goal_tile = any(current_pos == goal_pos for goal_pos in self.human_goals.values())
        if is_goal_tile:
            old_char_on_grid = "G"
        # More complex logic could check for keys, doors etc., if agents can temporarily occupy them
        # and the grid needs to reflect the object, not the agent.
        # For now, if it's not a goal, it becomes floor.
        
        self.grid[current_pos] = old_char_on_grid
        
        # Update the agent's position in the primary tracking dictionary
        self.agent_positions[agent_to_move_id] = new_pos
        
        # Update legacy single-agent trackers if the moving agent matches
        if agent_to_move_id in self.robot_agent_ids and agent_to_move_id == self.robot_agent_ids[0]:
            self.agent_pos = new_pos
        if agent_to_move_id in self.human_agent_ids and agent_to_move_id == self.human_agent_ids[0]:
            self.human_pos = new_pos
        
        # Mark the agent's new position on the grid if it's currently empty floor
        # This prevents overwriting 'G' or other important static items agents might move onto.
        if self.grid[new_pos] == " ":
            self.grid[new_pos] = agent_char_on_grid

    def step(self, actions: dict[str,int]):
        """
        actions: dict mapping agent_id to action enum/int
        Returns: obs, rewards, terminations, truncations, infos
        """
        # Store previous potentials for human agents for reward shaping
        prev_potentials_h = {
            hid: self.potential(hid) for hid in self.human_agent_ids
        }

        # reset per-step rewards 
        # Robot reward is 0 from env; IQL calculates its actual learning reward.
        # Human reward will be overwritten by shaping.
        self.rewards = {agent: 0 for agent in self.agents}
        
        # apply actions in random order
        order = self.agents[:]
        random.shuffle(order)
        # mapping from orientation idx to movement delta
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        for agent in order: # 'agent' is the agent_id string e.g. "robot_0", "human_0"
            act = actions.get(agent, self.ACTION_NO_OP)
            action_enum = Actions(act)

            # Enforce human action limitations (example for human_agent_ids[0])
            # This should be generalized if multiple humans have different action sets
            if agent in self.human_agent_ids: # Simplified: all humans have same restriction
                allowed_human_actions = {Actions.turn_left, Actions.turn_right, Actions.forward, Actions.done}
                if action_enum not in allowed_human_actions:
                    action_enum = Actions.done
            
            current_pos = self.agent_positions[agent] # CORRECTED: Use agent_positions for current agent
            dir_idx = self.agent_dirs[agent]
            dx, dy = deltas[dir_idx]
            front_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Determine other primary agent's position for collision (simplistic for 2 agents)
            # This needs a more general way to check collisions with ALL other agents
            other_agent_positions = [pos for aid, pos in self.agent_positions.items() if aid != agent]

            # Turn actions update orientation
            if action_enum == Actions.turn_left:
                self.agent_dirs[agent] = (dir_idx - 1) % 4
            elif action_enum == Actions.turn_right:
                self.agent_dirs[agent] = (dir_idx + 1) % 4
            # Forward moves the agent in its facing direction if valid
            elif action_enum == Actions.forward:
                collision_with_other_agent = any(front_pos == other_pos for other_pos in other_agent_positions)
                if (
                    self._is_valid_pos(front_pos)
                    and not collision_with_other_agent
                ):
                    self._handle_movement(agent, front_pos) # Pass the specific agent_id

            # Robot-specific actions
            if agent.startswith("robot"): # Check if the acting agent is a robot
                if action_enum == Actions.pickup:
                    for key in list(self.keys):
                        if key['pos'] == front_pos:
                            self.robot_has_keys.add(key['color']) # Assuming one robot's inventory for now
                            self.keys.remove(key)
                            self.grid[front_pos] = ' '
                            break
                elif action_enum == Actions.toggle:
                    for door in self.doors:
                        if door['pos'] == front_pos and door['is_locked'] and door['color'] in self.robot_has_keys:
                            door['is_locked'] = False
                            door['is_open'] = True
                            self.grid[front_pos] = ' ' # Door becomes passable, visually handled by Door object state
                            break
                elif action_enum == Actions.drop and self.robot_has_keys:
                    occupied_by_objects_or_agents = {tuple(d['pos']) for d in self.doors} | \
                                                    {tuple(k['pos']) for k in self.keys} | \
                                                    set(self.agent_positions.values())
                    if (0 <= front_pos[0] < self.grid_size and 0 <= front_pos[1] < self.grid_size
                        and self.grid[front_pos] == ' ' # Can only drop on empty floor
                        and front_pos not in occupied_by_objects_or_agents): # Check against all entities
                        color = self.robot_has_keys.pop()
                        self.keys.append({'pos': front_pos, 'color': color})
                        self.grid[front_pos] = 'K' # Mark key on grid


        # Calculate shaped rewards for human agents based on potential change
        for hid in self.human_agent_ids:
            current_potential_h = self.potential(hid)
            prev_potential = prev_potentials_h.get(hid, current_potential_h) # Default to current if no prev (e.g. first step)
            # Standard PBRS: P(s') - P(s)
            # R_env is 0 for non-terminal steps for human.
            shaped_reward = current_potential_h - prev_potential
            self.rewards[hid] = shaped_reward
            
            # Debug shaped reward calculation only for verbose level
            if self.debug_mode and self.debug_level == 'verbose' and abs(shaped_reward) > 0.01:
                print(f"🔄 SHAPED REWARD: Human {hid} got {shaped_reward:.3f} (potential: {prev_potential:.2f} → {current_potential_h:.2f})")

        # Check for humans reaching their goals and track which ones have completed
        humans_at_goals = []
        for hid in self.human_agent_ids:
            if self.agent_positions.get(hid) == self.human_goals.get(hid):
                # Check if this human just reached the goal (not already there)
                if not hasattr(self, '_humans_completed') or hid not in self._humans_completed:
                    # Initialize completed humans tracker if it doesn't exist
                    if not hasattr(self, '_humans_completed'):
                        self._humans_completed = set()
                    
                    # Mark this human as completed and give bonus reward
                    self._humans_completed.add(hid)
                    self.rewards[hid] += 500
                    
                    # Always print goal achievement regardless of debug level
                    if self.debug_mode or hasattr(self, '_minimal_debug_enabled'):
                        episode_info = f" (Episode {self._current_episode})" if hasattr(self, '_current_episode') and self._current_episode is not None else ""
                        print(f"🎯 GOAL REACHED: Human {hid} reached goal {self.human_goals.get(hid)} at step {self.timestep}{episode_info}")
                
                humans_at_goals.append(hid)
        
        # End episode only when ALL humans have reached their respective goals
        if len(humans_at_goals) == len(self.human_agent_ids):
            for a in self.agents:
                self.terminations[a] = True
            if self.debug_mode or hasattr(self, '_minimal_debug_enabled'):
                episode_info = f" (Episode {self._current_episode})" if hasattr(self, '_current_episode') and self._current_episode is not None else ""
                print(f"🏆 ALL GOALS COMPLETED: All humans reached their goals at step {self.timestep}{episode_info}")
        
        self.timestep += 1 # Increment timestep
        # Check for truncation based on max_steps
        if self.timestep >= self.max_steps:
            for agent_id in self.agents:
                self.truncations[agent_id] = True
            if self.debug_mode: # Conditional debug print
                print(f"DEBUG: Max steps {self.max_steps} reached. Truncating.") # DEBUG

        # Store debug info but don't print yet - will be printed after robot reward update
        self._debug_info_for_delayed_print = {
            'prev_potentials_h': prev_potentials_h,
            'timestep': self.timestep
        }

        # update global terminal/truncation if needed (as before)
        # return parallel outputs
        obs = {agent: self.observe(agent) for agent in self.possible_agents}
        return obs, self.rewards, self.terminations, self.truncations, self.infos

    def set_minimal_debug(self, enabled: bool):
        """Enable minimal debug mode for goal achievement notifications."""
        self._minimal_debug_enabled = enabled

    def set_current_episode(self, episode_num: int):
        """Set the current episode number for debug output."""
        self._current_episode = episode_num

    def print_debug_info(self):
        """Print debug information after robot reward has been updated."""
        if not self.debug_mode or not hasattr(self, '_debug_info_for_delayed_print'):
            return
            
        debug_info = self._debug_info_for_delayed_print
        prev_potentials_h = debug_info['prev_potentials_h']
        timestep = debug_info['timestep']
        
        print(f"\n--- Step {timestep} ---")
        for agent_id in self.possible_agents:
            pos = self.agent_positions.get(agent_id, "N/A")
            reward = self.rewards.get(agent_id, 0)
            termination = self.terminations.get(agent_id, False)
            truncation = self.truncations.get(agent_id, False)
            print(f"DEBUG: Agent {agent_id} | Pos: {pos} | Reward: {reward:.2f} | Term: {termination} | Trunc: {truncation}") # Formatted reward
            if agent_id.startswith("human"):
                goal = self.human_goals.get(agent_id, "N/A")
                
                # Enhanced logging for PrevPotential
                prev_potential_val_for_log = prev_potentials_h.get(agent_id) # Try direct get first
                if prev_potential_val_for_log is not None and isinstance(prev_potential_val_for_log, (float, int)):
                    prev_potential_str = f"{float(prev_potential_val_for_log):.2f}"
                elif agent_id not in prev_potentials_h:
                    prev_potential_str = f"N/A (Key '{agent_id}' not in prev_potentials_h. Keys: {list(prev_potentials_h.keys())})"
                else: # Key exists but value is problematic
                    prev_potential_str = f"N/A (Value for '{agent_id}' is {prev_potential_val_for_log}, type: {type(prev_potential_val_for_log)})"
                    
                current_potential_val_for_log = self.potential(agent_id) 
                print(f"DEBUG: Human {agent_id} | Goal: {goal} | PrevPotential: {prev_potential_str} | CurrPotential: {current_potential_val_for_log:.2f}")

    def render(self):
        if self.render_mode is None: 
            return

        if self.screen is None and self.render_mode == "human": 
            pygame.init()
            pygame.font.init() 
            pygame.display.set_caption("Locking Door Environment")
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()
            self.agent_char_font = pygame.font.SysFont("Arial", int(self.cell_size * 0.5))  # reduced size from 0.75
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
                # Base objects: walls, lava, and goal
                if char_on_grid == '#':
                    obj_to_render = Wall()
                elif char_on_grid == 'L':
                    obj_to_render = Lava()
                elif char_on_grid == 'G':
                    obj_to_render = Goal()
                # Doors
                if obj_to_render is None:
                    for door in self.doors:
                        if current_tile_coords == door['pos']:
                            obj_to_render = Door(color=door['color'], is_locked=door['is_locked'], is_open=door['is_open'])
                            break
                # Keys
                if obj_to_render is None:
                    for key in self.keys:
                        if current_tile_coords == key['pos']:
                            obj_to_render = Key(color=key['color'])
                            break
                # Boxes
                if obj_to_render is None:
                    for box in self.boxes:
                        if current_tile_coords == box['pos']:
                            obj_to_render = Box(color=box['color'])
                            break
                 
                if obj_to_render:
                    obj_to_render.render(tile_img_np)

                # Render agents from agent_positions (supports multiple humans)
                agent_char_to_render = None
                char_color = COLORS["black"]
                for aid in self.possible_agents:
                    pos = self.agent_positions.get(aid)
                    if pos == (r, c):
                        id_num = aid.split('_')[1]
                        dir_idx = self.agent_dirs.get(aid, 2)
                        arrow = ['↑','→','↓','←'][dir_idx]
                        if aid.startswith('robot'):
                            agent_char_to_render = f"R{id_num}{arrow}"
                            char_color = COLORS['blue']
                        else:
                            agent_char_to_render = f"H{id_num}{arrow}"
                            char_color = COLORS['purple']
                        break

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
            f"Keys: {len(self.keys)} | Robot Has Keys: {self.robot_has_keys}",
            f"Doors: {len(self.doors)} | Goal@: {self.human_goals.get(self.human_agent_ids[0])}"
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

    def update_robot_reward_for_logging(self, robot_agent_id: str, internal_reward: float):
        """Update the robot's reward in self.rewards for logging purposes."""
        if robot_agent_id in self.rewards:
            self.rewards[robot_agent_id] = internal_reward

    def potential(self, human_id: str) -> float:
        """
        Compute shaping potential: negative Manhattan distance for specified human to its goal.
        """
        goal = self.human_goals.get(human_id)
        pos = self.agent_positions.get(human_id)
        if goal is None or pos is None:
            return 0.0
        return - (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        return self._observation_spaces[agent_id]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        return self._action_spaces[agent_id]

class GridEnvironment(CustomEnvironment):
    """Environment with a locking door."""
    def __init__(self, map_name=DEFAULT_MAP, grid_size=None, debug_mode=False, debug_level='standard'): # Added debug_level
        super().__init__(map_name=map_name, grid_size=grid_size, debug_mode=debug_mode, debug_level=debug_level) # Pass debug_level
        # any specifics

# Generic alias
Environment = GridEnvironment