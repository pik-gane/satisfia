from __future__ import annotations

import functools
import random
from copy import copy
from typing import Tuple

import numpy as np
import pygame
from gymnasium.spaces import Discrete, MultiDiscrete # Correct import
from enum import IntEnum # Import IntEnum

from pettingzoo.utils.env import AECEnv
from pettingzoo.utils import AgentSelector

# don't use relative imports
from objects import WorldObj, Goal, Key, Wall, Door, Lava, Floor, CHAR_TO_OBJ_CLASS, COLORS, TILE_PIXELS
from rendering_utils import fill_coords, point_in_circle # Added point_in_circle for agent rendering

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
    # Permanently lock an object (e.g., a door)
    lock = 7 # New action

class CustomEnvironment(AECEnv):
    """Custom grid environment."""

    metadata = {
        "name": "custom_environment_v0",
        "render_modes": ["human"], # Added render_modes
        "is_parallelizable": False, # Added based on previous context
        "render_fps": 10, # Added based on previous context
    }

    # Action constants (mirrors Actions enum for direct use)
    ACTION_LEFT = Actions.left
    ACTION_RIGHT = Actions.right
    ACTION_UP = Actions.up
    ACTION_DOWN = Actions.down
    ACTION_PICKUP = Actions.pickup
    ACTION_DROP = Actions.drop
    ACTION_TOGGLE = Actions.toggle
    ACTION_LOCK = Actions.lock # New action constant

    def __init__(self, grid_size=15): # Increased grid size
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = TILE_PIXELS  # Use TILE_PIXELS from objects.py
        self.window_size = self.grid_size * self.cell_size
        self.screen = None
        self.clock = None

        self.timestep = 0

        self.possible_agents = ["robot", "human"]
        # self.agents = list(self.possible_agents) # agents will be set in reset
        self.agent_selector = AgentSelector(self.possible_agents)
        # self.current_agent = self.agent_selector.reset() # Replaced by agent_selection
        self.agent_selection = "" # Will be set by agent_selector.reset()

        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}

        # Door state variables
        self.door_is_open = False
        self.door_is_key_locked = False # True if door needs a key to be opened initially
        self.door_is_permanently_locked = False


        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='<U1')
        self.agent_pos = None
        self.human_pos = None
        self.goal_pos = None
        self.key_pos = None
        self.door_pos = None
        self.lava_positions = []

        # self.actions = Actions # Keep this for reference if needed, but direct constants are used
        # self.action_space = Discrete(len(self.actions)) # This is for single-agent Gym, PettingZoo uses _action_spaces

        self.grid_legend = {
            "#": "Wall", "G": "Goal", "K": "Key", "D": "Door",
            "L": "Lava", "R": "Robot", "H": "Human", " ": "Empty"
        }

        self.robot_has_key = False
        # self.door_locked = True # Replaced by new door state variables

        self._action_spaces = {agent: Discrete(len(Actions)) for agent in self.possible_agents}
        # Observation: agent_pos (2), human_pos (2), door_pos (2), goal_pos (2), 
        # robot_has_key (1), door_is_open (1), door_is_key_locked (1), door_is_permanently_locked (1)
        # Total 8 features, some are positions (grid_size), some are booleans (2)
        obs_shape = np.array(
            [grid_size, grid_size] * 4 +  # agent_pos, human_pos, door_pos, goal_pos
            [2] * 4                        # robot_has_key, door_is_open, door_is_key_locked, door_is_permanently_locked
        )
        self._observation_spaces = {
            agent: MultiDiscrete(obs_shape)
            for agent in self.possible_agents
        }
        
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}


        self.reset()


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = list(self.possible_agents)
        self.agent_selector.reinit(self.agents)
        self.agent_selection = self.agent_selector.reset()

        self.timestep = 0
        self.robot_has_key = False # No key in this scenario initially

        # Initialize door states for the scenario
        self.door_is_open = False
        self.door_is_key_locked = False # Door does not require a key to toggle
        self.door_is_permanently_locked = False

        # Setup grid
        self.grid = np.full((self.grid_size, self.grid_size), " ", dtype='<U1')
        
        # Outer walls
        self.grid[0, :] = "#"
        self.grid[-1, :] = "#"
        self.grid[:, 0] = "#"
        self.grid[:, -1] = "#"

        # Define a room
        room_wall_row = self.grid_size // 2
        for c in range(1, self.grid_size - 1):
            self.grid[room_wall_row, c] = "#"

        # Door position
        self.door_pos = (room_wall_row, self.grid_size // 2)
        self.grid[self.door_pos] = "D" # Place door in the wall

        # Goal position (inside the room)
        self.goal_pos = (room_wall_row + 2, self.grid_size // 2)
        if not (0 < self.goal_pos[0] < self.grid_size-1 and 0 < self.goal_pos[1] < self.grid_size-1):
             self.goal_pos = (self.grid_size - 2, self.grid_size - 2) # Fallback
        self.grid[self.goal_pos] = "G"
        
        # Key position (none for this scenario)
        self.key_pos = None # No key on the map

        # Lava positions (none for this scenario for simplicity)
        self.lava_positions = []

        # Agent and Human positions (outside the room)
        self.agent_pos = (room_wall_row - 2, self.grid_size // 2)
        if self.grid[self.agent_pos] == "#": self.agent_pos = (1,1) # Fallback
        self.grid[self.agent_pos] = "R"

        self.human_pos = (room_wall_row - 2, self.grid_size // 2 + 1)
        if self.grid[self.human_pos] == "#": self.human_pos = (1,2) # Fallback
        if self.human_pos == self.agent_pos: self.human_pos = (self.agent_pos[0], self.agent_pos[1]+1) # Ensure different
        self.grid[self.human_pos] = "H"


        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents} # Reset cumulative rewards
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # For PettingZoo, reset usually returns (observation, info) for the first agent
        # However, the main loop calls observe() separately.
        # So, just setting up the state is enough.
        # return self.observe(self.current_agent), self.infos[self.current_agent]


    def _get_obs(self, agent):
        # Ensure positions are valid, e.g., (-1,-1) if None
        # obs_key_pos = self.key_pos if self.key_pos is not None else (-1, -1) # Key not used
        obs_door_pos = self.door_pos if self.door_pos is not None else (-1,-1)
        obs_goal_pos = self.goal_pos if self.goal_pos is not None else (-1,-1)
        obs_agent_pos = self.agent_pos if self.agent_pos is not None else (-1,-1)
        obs_human_pos = self.human_pos if self.human_pos is not None else (-1,-1)

        return np.array([
            obs_agent_pos[0], obs_agent_pos[1],
            obs_human_pos[0], obs_human_pos[1],
            # obs_key_pos[0], obs_key_pos[1], # Key not part of this obs
            obs_door_pos[0], obs_door_pos[1],
            obs_goal_pos[0], obs_goal_pos[1],
            int(self.robot_has_key),
            int(self.door_is_open),
            int(self.door_is_key_locked),
            int(self.door_is_permanently_locked)
        ])

    def observe(self, agent):
        return self._get_obs(agent)

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

    def _handle_movement(self, agent_to_move, new_pos):
        old_char = " " 
        current_char_at_old_pos = self.grid[self.agent_pos if agent_to_move == "robot" else self.human_pos]

        if agent_to_move == "robot":
            # Key logic removed as key is not used in this scenario for movement
            # if self.agent_pos == self.key_pos and not self.robot_has_key:
            #     old_char = "K"
            
            self.grid[self.agent_pos] = old_char 
            self.agent_pos = new_pos
            # Do not overwrite D, G etc. with R if R is on them. Rendering handles agent presence.
            if self.grid[new_pos] == " ": self.grid[new_pos] = "R" 
        elif agent_to_move == "human":
            self.grid[self.human_pos] = old_char
            self.human_pos = new_pos
            if self.grid[new_pos] == " ": self.grid[new_pos] = "H"


    def step(self, action):
        agent = self.agent_selection # The agent whose turn it is

        # If agent was already terminated/truncated, PettingZoo expects step(None).
        # Its reward for this "pass" step should be 0.
        if self.terminations[agent] or self.truncations[agent]:
            # Action is None for dead agents, handled by _was_dead_step in AECEnv if not fully overridden
            # For a fully overridden step, ensure rewards are zero for this pass.
            self.rewards = {a: 0 for a in self.agents}
            # The agent remains in self.terminations/self.truncations
            # The rest of the step logic (advancing agent, accumulating rewards) still needs to run.
        else:
            # Agent is active, process the action.
            # Reset per-step rewards for all agents before processing the current action
            self.rewards = {a: 0 for a in self.agents} 

            if agent == "robot":
                current_pos = self.agent_pos
                if self.ACTION_LEFT <= action <= self.ACTION_DOWN: # Movement
                    new_pos = self._move_agent_vanilla(current_pos, action)
                    
                    can_move = True
                    if not self._is_valid_pos(new_pos): # Hits a wall
                        can_move = False
                        print(f"Robot tried to move into wall at {new_pos}")
                    elif new_pos == self.human_pos: # Collision with human
                        can_move = False
                        print("Robot tried to move into human.")
                    elif new_pos == self.door_pos and not self.door_is_open: # Collision with closed/locked door
                        can_move = False
                        if self.door_is_permanently_locked:
                            print("Robot tried to move through permanently locked door.")
                        else:
                            print("Robot tried to move through closed door.")
                    
                    if can_move:
                        self._handle_movement("robot", new_pos)

                elif action == self.ACTION_PICKUP:
                    # Simplified: No pickup in this scenario for now
                    # if self.agent_pos == self.key_pos and self.key_pos is not None:
                    #     self.robot_has_key = True
                    #     self.grid[self.key_pos] = " " 
                    #     self.key_pos = None 
                    #     print("Robot picked up key.")
                    print("Robot tried to pickup (action not fully configured for this scenario).")
                
                elif action == self.ACTION_DROP:
                    # Simplified: No drop
                    print("Robot tried to drop (action not fully configured).")

                elif action == self.ACTION_TOGGLE:
                    if self.agent_pos == self.door_pos:
                        if self.door_is_permanently_locked:
                            print("Robot tried to toggle a permanently locked door. No effect.")
                        elif self.door_is_key_locked:
                            if self.robot_has_key:
                                self.door_is_key_locked = False
                                self.door_is_open = True
                                # self.robot_has_key = False # Consume key
                                print("Robot unlocked and opened door with key.")
                            else:
                                print("Robot tried to toggle key-locked door, but has no key.")
                        else: # Not permanently locked, not key-locked
                            self.door_is_open = not self.door_is_open
                            print(f"Robot {'opened' if self.door_is_open else 'closed'} the door.")
                    else:
                        print("Robot tried to toggle, but not at the door position.")
                
                elif action == self.ACTION_LOCK: # New action logic
                    if self.agent_pos == self.door_pos:
                        if self.door_is_permanently_locked:
                            print("Robot tried to lock an already permanently locked door.")
                        else:
                            self.door_is_permanently_locked = True
                            self.door_is_open = False # Locking also closes it
                            print("Robot permanently locked the door.")
                    else:
                        print("Robot tried to lock, but not at the door position.")


            elif agent == "human":
                human_action_to_take = self._get_human_action() # Human chooses its own action
                current_pos = self.human_pos
                if self.ACTION_LEFT <= human_action_to_take <= self.ACTION_DOWN:
                    new_pos = self._move_agent_vanilla(current_pos, human_action_to_take)
                    
                    can_move_human = True
                    if not self._is_valid_pos(new_pos): # Hits a wall
                        can_move_human = False
                    elif new_pos == self.agent_pos: # Collision with robot
                        can_move_human = False
                    elif new_pos == self.door_pos and not self.door_is_open: # Collision with closed/locked door
                        can_move_human = False
                        # Human interaction with door can be made more complex if needed
                        # For now, human is blocked like robot by non-open doors.
                        # print(f"Human tried to move through {'permanently locked' if self.door_is_permanently_locked else 'closed'} door.")
                    
                    if can_move_human:
                        self._handle_movement("human", new_pos)
            
            self.timestep += 1 # Increment timestep only if an active agent took a step

        # Update game-ending conditions and rewards based on the action taken (or no action if dead)
        if self.human_pos == self.goal_pos:
            self.rewards["robot"] = 1 # Example: robot gets reward if human reaches goal
            self.rewards["human"] = 1
            self.terminations = {a: True for a in self.agents}
            print("Human reached the goal!")

        if self.agent_pos in self.lava_positions:
            self.rewards["robot"] = -1
            self.terminations["robot"] = True # Robot is done
            # self.truncations["robot"] = True # Use truncation if it's due to external factor like time limit
            print("Robot fell in lava!")
        if self.human_pos in self.lava_positions:
            self.rewards["human"] = -1
            self.terminations["human"] = True # Human is done
            # self.truncations["human"] = True
            print("Human fell in lava!")
        
        if self.timestep >= 200: # Max steps condition
            # self.truncations = {a: True for a in self.agents} # Mark all as truncated
            # For PettingZoo, if truncated, usually also set terminated for the agent_iter loop to stop
            self.terminations = {a: True for a in self.agents} 
            self.truncations = {a: True for a in self.agents}
            print("Max steps reached, episode truncated.")

        # PettingZoo end-of-step logic:
        # 1. Clear cumulative reward for the agent that just acted (or was skipped if dead)
        self._cumulative_rewards[agent] = 0

        # 2. Select next agent
        self.agent_selection = self.agent_selector.next()

        # 3. Accumulate rewards for all agents
        # self.rewards now contains the rewards from the current step (for the agent that acted, and any side-effects)
        for ag in self.agents:
            self._cumulative_rewards[ag] += self.rewards[ag]


    def _get_human_action(self):
        dx = self.goal_pos[1] - self.human_pos[1]
        dy = self.goal_pos[0] - self.human_pos[0]

        possible_actions = []
        if dy < 0: possible_actions.append(self.ACTION_UP)
        if dy > 0: possible_actions.append(self.ACTION_DOWN)
        if dx < 0: possible_actions.append(self.ACTION_LEFT)
        if dx > 0: possible_actions.append(self.ACTION_RIGHT)
        
        if not possible_actions:
            return random.choice([self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_UP, self.ACTION_DOWN])

        for act in possible_actions:
            temp_new_pos = self._move_agent_vanilla(self.human_pos, act)
            if self._is_valid_pos(temp_new_pos) and temp_new_pos != self.agent_pos:
                if temp_new_pos == self.door_pos and self.door_locked:
                    continue
                return act
        
        all_moves = [self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_UP, self.ACTION_DOWN]
        random.shuffle(all_moves)
        for act in all_moves:
            temp_new_pos = self._move_agent_vanilla(self.human_pos, act)
            if self._is_valid_pos(temp_new_pos) and temp_new_pos != self.agent_pos:
                if temp_new_pos == self.door_pos and self.door_locked:
                    continue
                return act
        return self.ACTION_DOWN


    def render(self):
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Locking Door Environment")
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        pygame.event.pump() # Add this line to process Pygame events

        grid_surface = pygame.Surface((self.window_size, self.window_size))
        grid_surface.fill(COLORS["grey"] * 0.5)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                char_on_grid = self.grid[r, c]
                tile_rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                
                tile_img_np = np.zeros((self.cell_size, self.cell_size, 3), dtype=np.uint8)
                
                floor_obj = Floor()
                floor_obj.render(tile_img_np)

                obj_to_render = None
                if (r,c) == self.door_pos: # Special handling for door state
                     obj_to_render = Door(
                         is_locked=self.door_is_key_locked, 
                         is_open=self.door_is_open, 
                         is_permanently_locked=self.door_is_permanently_locked
                     )
                elif char_on_grid in CHAR_TO_OBJ_CLASS:
                    ObjClass = CHAR_TO_OBJ_CLASS[char_on_grid]
                    # if ObjClass == Door: # Handled above
                    #     pass
                    if ObjClass != Floor and ObjClass != Door: # Avoid re-instantiating door/floor
                        obj_to_render = ObjClass()
                
                if obj_to_render:
                    obj_to_render.render(tile_img_np)

                if (r, c) == self.agent_pos:
                    fill_coords(tile_img_np, point_in_circle(0.5, 0.5, 0.4), COLORS["blue"])
                if (r, c) == self.human_pos:
                    fill_coords(tile_img_np, point_in_circle(0.5, 0.5, 0.4), COLORS["purple"])

                tile_surface = pygame.surfarray.make_surface(np.transpose(tile_img_np, (1,0,2)))
                grid_surface.blit(tile_surface, tile_rect.topleft)
                
                pygame.draw.rect(grid_surface, COLORS["black"], tile_rect, 1)


        self.screen.blit(grid_surface, (0,0))
        
        font = pygame.font.Font(None, 30)
        text_surface = font.render(f"Step: {self.timestep} Agent: {self.agent_selection}", True, COLORS["black"])
        self.screen.blit(text_surface, (10,10))
        
        door_status_text = f"Door: {'Open' if self.door_is_open else 'Closed'}, " \
                           f"{'KeyLck' if self.door_is_key_locked else 'NoKeyLck'}, " \
                           f"{'PermaLck' if self.door_is_permanently_locked else 'NoPermaLck'}"
        text_surface_door = font.render(door_status_text, True, COLORS["black"])
        self.screen.blit(text_surface_door, (10,40))
        
        key_status_text = f"Robot Has Key: {self.robot_has_key}"
        text_surface_key = font.render(key_status_text, True, COLORS["black"])
        self.screen.blit(text_surface_key, (10,70)) # Adjusted y-pos

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

class LockingDoorEnvironment(CustomEnvironment):
    """Environment with a locking door. Uses CustomEnvironment's logic."""
    def __init__(self, grid_size=7):
        super().__init__(grid_size)
        pass