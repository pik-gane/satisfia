from functools import cache, lru_cache

from . import MDPWorldModel

# based in large part on https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

import numpy as np
from numpy import random
import pygame

import gymnasium as gym
from gymnasium import spaces

unenterable_cell_types = ['#']
immovable_object_types = [',']
max_n_object_states = 2

class SimpleGridworld(MDPWorldModel):
    """A world model of a simple MDP-type Gridworld environment.
    
    A *state* here is a tuple of integers encoding the following sequence of items,
    each one encoded as one or two integers:

    - position 0: the current time step
    - positions 1+2: the current position x,y of the agent
    - positions 3+4: the previous position x,y of the agent
    - positions 5...4+k: for each of k immovable objects with variable state, its state
    - positions 5+k..4+k+2*l: for each of l movable objects without a variable state, its position x,y 
    - positions 5+k+2*l...4+k+2*l+3*m: for each of m movable objects with a variable state, 
                                       its state and its position x,y

    A *coordinate* in a position is encoded in as an integer, where:
    - -2 means the object is not present
    - -1 means the object is in the agent's inventory
    - >= 0 is a coordinate in the grid, counted from top-left to bottom-right

    Objects are *ordered* by their initial position in the ascii-art grid representation in row-major order.

    The *grid* and the agent's and all objects' *initial positions* are given as a 2d array of characters,
    each representing one cell of the grid, with the following character meanings: 

    - already implemented:   
        - '#' (hash): wall
        - ' ' (blank): empty space
        - '~': Uneven ground (Agents/boxes might fall off to any side except to where agent came from, 
               with equal probability)
        - '^': Pinnacle (Climbing on it will result in falling off to any side except to where agent came from, 
               with equal probability)
        - 'A': agent's initial position 

    - not yet implemented, but are planned to be implemented in the future:
        - ',': Empty tile that turns into a wall after leaving it (so that one cannot go back)
        - '-': Slippery ground (Agents and boxes might slide along in a straight line; after sliding by one tile, 
               a coin is tossed to decide whether we slide another tile, and this is repeated 
               until the coin shows heads or we hit an obstacle. All this motion takes place within a single time step.)
        - '%': Death trap (Episode ends when agent steps on it) 
        - 'B': Button (can be stepped on)
        - 'C': Collaborator (might move around)
        - 'D': Door (can only be entered after having collected a key)
        - 'E': Enemy (might move around on its own)
        - 'Δ': Delta (positive or negative, can be collected once, does not end the episode)
        - 'G': Goal or exit door (acting while on it ends the episode)
        - 'I': (Potential) interruption (agent might get stuck in forever)
        - 'K': Key (must be collected to be able to pass a door)
        - 'O': Ball (when pushed, will move straight until meeting an obstacle)
        - 'S': Supervisor (might move around on their own)
        - 'T': Teleporter (sends the agent to some destination t)
        - 't': Destination of a teleporter (stepping on it does nothing)
        - 'X': Box (can be pushed around but not pulled, can slide and fall off)
    (TODO: compare with pycolab asciiart conventions, try to harmonize them, and add things that are missing)

    *Deltas* (rewards) can accrue from the following events:
    - Time passing. This is specified by time_delta or a list time_deltas of length max_episode_length.
    - The agent stepping onto a certain object. This is specified by a list object_deltas
      ordered by the objects' initial positions in the ascii-art grid representation in row-major order.
    - The agent currently being in a certain position. This is specified by 
        - another 2d array of characters, delta_grid, of the same size as the grid, 
          containing cell_codes with the following character meanings:
            - ' ' (space): no Delta
            - '<character>': Delta as specified by cell_code2delta['<character>']
        - a dictionary cell_code2delta listing the actual Delta values for each cell_code in that grid
      Note that the delta accrues at each time point when the agent is in a cell, 
      not at the time point it steps onto it!
    """


    ## parameters:
    xygrid = None
    """(2d array of characters) The grid as an array of strings, each string representing one row of the grid,
    each character representing one cell of the grid"""
    delta_xygrid = None
    """(2d array of characters) codes for deltas (rewards) for each cell of the grid"""
    cell_code2delta = None
    """(dictionary) maps cell codes to deltas (rewards)"""

    n_immovable_objects = None
    """The number of immovable objects with variable state."""
    n_movable_constant_objects = None
    """The number of movable objects without a variable state."""
    n_movable_variable_objects = None
    """The number of movable objects with a variable state."""
    max_episode_length = None
    """The maximum number of steps in an episode."""
    initial_agent_location = None
    """(pair of ints) The initial location of the agent starting with zero."""
    time_deltas = None
    """(list of ints) The deltas (rewards) for each time step."""

    # additional attributes:
    _state = None
    """(singleton list of state) The current state encoded as a tuple of ints."""
    t = None
    """The current time step."""
    _agent_location = None
    """(pair of ints) The current location of the agent starting with zero."""
    _previous_agent_location = None
    """(pair of ints) The previous location of the agent starting with zero."""

    # TODO Jobst: adapt the following code to our needs:

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode = None, 
                 grid = [['A','G']],
                 delta_grid = [[' ','1']],
                 cell_code2delta = {'1': 1},
                 max_episode_length = 1e10,
                 time_deltas = [0],
                 uneven_ground_prob = 0.25,
                 fps = 4
                 ):

        self.xygrid = xygrid = np.array(grid).T
        self.delta_xygrid = delta_xygrid = np.array(delta_grid).T
        self.cell_code2delta = cell_code2delta
        self.max_episode_length = max_episode_length
        self.time_deltas = np.array(time_deltas).flatten()
        self.uneven_ground_prob = uneven_ground_prob
        self._fps = fps

        self._window_shape = 800 * np.array(xygrid.shape) / np.max(xygrid.shape)  # The size of the PyGame window in pixels

        # The initial agent location is the first occurrence of 'A' in the grid:
        wh = np.where(xygrid == 'A')
        self.initial_agent_location = (wh[0][0], wh[1][0])

        self.n_immovable_objects = self.n_movable_constant_objects = self.n_movable_variable_objects = 0  # TODO: extract from grid

        # Construct an auxiliary grid that contains a unique index of each immovable object 
        # (cells of a type in immovable_object_types), or None if there is none:
        self._immovable_object_indices = np.full(xygrid.shape, None)
        for x in range(xygrid.shape[0]):
            for y in range(xygrid.shape[1]):
                if xygrid[x, y] in immovable_object_types:
                    self._immovable_object_indices[x, y] = self.n_immovable_objects
                    self.n_immovable_objects += 1
        # Initialize all immovable object states to zero:
        self._immovable_object_states = np.zeros(self.n_immovable_objects, dtype = int)
    
        # The observation returned for reinforcement learning equals state, as described above.
        nx, ny = xygrid.shape[0], xygrid.shape[1]
        self.observation_space = spaces.MultiDiscrete(
            [max_episode_length,  # current time step
             nx, ny,  # current position
             nx, ny]  # previous position
            + [max_n_object_states] * self.n_immovable_objects 
            + [nx, ny] * self.n_movable_constant_objects 
            + [max_n_object_states, nx, ny] * self.n_movable_variable_objects
            , start = 
            [0,  # current time step
             -2, -2,  # current position
             -2, -2]  # previous position
            + [0] * self.n_immovable_objects 
            + [-2, -2] * self.n_movable_constant_objects 
            + [0, -2, -2] * self.n_movable_variable_objects
            )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "human":
            self._init_human_rendering()

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self._window = None
        self.clock = None

    def _get_target_location(self, location, action):
        """Return the next location of the agent if it takes the given action from the given location."""
        direction = self._action_to_direction[action]
        return (
            location[0] + direction[0],
            location[1] + direction[1]
        )

    def _can_move(self, location, target_location):
        """Return True if the agent can move from the given location to the given target_location."""
        can = (0 <= target_location[0] < self.xygrid.shape[0]
                and 0 <= target_location[1] < self.xygrid.shape[1]
                and not self.xygrid[target_location] in unenterable_cell_types)
        if can and self.xygrid[target_location] == ',':
            # can only move there if it hasn't turned into a wall yet:
            can = self._immovable_object_states[self._immovable_object_indices[target_location]] == 0
        # TODO: add other conditions for not being able to move, e.g. because of other objects
        return can 

    def opposite_action(self, action):
        """Return the opposite action to the given action."""
        return (action + 2) % 4
        
    @lru_cache(maxsize=None)
    def possible_actions(self, state):
        """Return a list of possible actions from the given state."""
        location = (state[1], state[2])
        return [action for action in range(4) 
                if self._can_move(location, self._get_target_location(location, action))]

    def _extract_state_attributes(self, state):
        """Return the individual attributes of a state."""
        return (state[0],  # time step
                (state[1], state[2]),  # current position
                (state[3], state[4]),  # previous position
                state[5:5+self.n_immovable_objects],  # immovable object states
         ) # TODO: extract object states and/or object locations

    def _set_state(self, state):
        """Set the current state to the last state encoded in the given result."""
        self._state = state
        self.t, loc, prev_loc, imm_states = self._extract_state_attributes(state)
        self._agent_location = loc
        self._previous_agent_location = prev_loc
        self._immovable_object_states = imm_states

    def _make_state(self, t, loc = (-2,-2), prev_loc = (-2, -2), imm_states = None):  # default locations are "not present"
        """Compile the given attributes into a state encoding that can be returned as an observation."""
        if imm_states is None:
            imm_states = np.zeros(self.n_immovable_objects, dtype = int)
        return (t, 
                loc[0], loc[1],
                prev_loc[0], prev_loc[1],
                *imm_states
                )

    @lru_cache(maxsize=None)
    def is_terminal(self, state):
        """Return True if the given state is a terminal state."""
        t, loc, prev_loc, imm_states = self._extract_state_attributes(state)
        was_at_goal = prev_loc[0] >= 0 and self.xygrid[prev_loc] == 'G'
        return (t == self.max_episode_length) or was_at_goal

    @lru_cache(maxsize=None)
    def transition_distribution(self, state, action, n_samples = None):
        if state is None and action is None:
            successor = self._make_state(0, self.initial_agent_location)
            return {successor: (1, True)}
        t, loc, prev_loc, imm_states = self._extract_state_attributes(state)
        cell_type = self.xygrid[loc]
        at_goal = cell_type == 'G'
        if at_goal:
            successor = self._make_state(t + 1, loc, loc, imm_states)
            return {successor: (1, True)} # TODO: probabilistic transitions!
        else:
            if cell_type == ',':
                # update the immovable object's state:
                l = list(imm_states)
                l[self._immovable_object_indices[loc]] = 1 
                imm_states = tuple(l)

            target_loc = self._get_target_location(loc, action)
            target_type = self.xygrid[target_loc]

            if target_type in ['^', '~']:
                # see what "falling-off" actions are possible:
                simulated_actions = [a for a in range(4) 
                      if a != self.opposite_action(action) # won't fall back to where we came from
                         and self._can_move(target_loc, self._get_target_location(target_loc, a))]
                if len(simulated_actions) > 0:
                    p0 = 1 if target_type == '^' else self.uneven_ground_prob  # probability of falling off
                    intermediate_state = self._make_state(t, target_loc, loc, imm_states)
                    trans_dist = {}
                    # compose the transition distribution recursively:
                    for simulate_action in simulated_actions:
                        for (successor, (probability, _)) in self.transition_distribution(intermediate_state, simulate_action, n_samples).items():
                            dp = p0 * probability / len(simulated_actions)
                            if successor in trans_dist:
                                trans_dist[successor] += dp
                            else:
                                trans_dist[successor] = dp
                    if target_type == '~':
                        trans_dist[intermediate_state] = 1 - p0
                    return { successor: (probability, True) for (successor,probability) in trans_dist.items() }

            # TODO: update object states and/or object locations, e.g. if the agent picks up an object or moves an object

            successor = self._make_state(t + 1, target_loc, loc, imm_states)
            return {successor: (1, True)} 

    @lru_cache(maxsize=None)
    def observation_and_reward_distribution(self, state, action, successor, n_samples = None):
        if state is None and action is None:
            return {(self._make_state(0, loc = self.initial_agent_location), 0): (1, True)}
        t, loc, prev_loc, imm_states = self._extract_state_attributes(state)
        delta = self.time_deltas[t % self.time_deltas.size]
        if self.delta_xygrid[loc] in self.cell_code2delta:
            delta += self.cell_code2delta[self.delta_xygrid[loc]]
        return {(successor, delta): (1, True)}

    # reset() and step() are inherited from MDPWorldModel and use the above transition_distribution():

    def reset(self, seed = None, options = None): 
        ret = super().reset(seed = seed, options = options)
        if self.render_mode == "human":
            self._render_frame()
        return ret

    def step(self, action):
        ret = super().step(action)
        if self.render_mode == "human":
            self._render_frame()
        return ret

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _init_human_rendering(self):
        pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
        self._cell_font = pygame.font.SysFont('Helvetica', 30)
        self._delta_font = pygame.font.SysFont('Helvetica', 10)

    def _render_frame(self):
        if self._window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode(
                self._window_shape
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self._window_shape)
        canvas.fill((255, 255, 255))
        pix_square_size = self._window_shape[0] / self.xygrid.shape[0]  # The size of a single grid square in pixels

        # Draw grid contents:
        for x in range(self.xygrid.shape[0]):
            for y in range(self.xygrid.shape[1]):
                cell_type = self.xygrid[x, y]
                if cell_type == "#" or (cell_type == "," and self._immovable_object_states[self._immovable_object_indices[x, y]] == 1):
                    pygame.draw.rect(
                        canvas,
                        (64, 64, 64),
                        (x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
                    )
                elif cell_type == "G":
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        (x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size),
                    )
                elif cell_type != " ":
                    canvas.blit(self._cell_font.render(cell_type, True, (0, 0, 0)),
                                      ((x+.3) * pix_square_size, (y+.3) * pix_square_size))
                cell_code = self.delta_xygrid[x, y]
                if cell_code in self.cell_code2delta:
                    canvas.blit(self._delta_font.render(cell_code + f" {self.cell_code2delta[cell_code]}", True, (0, 0, 0)),
                                      ((x+.1) * pix_square_size, (y+.1) * pix_square_size))

        # Now we draw the agent and its previous location:
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array(self._previous_agent_location) + 0.5) * pix_square_size,
            pix_square_size / 4,
            width = 3,
        )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array(self._agent_location) + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.xygrid.shape[0] + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self._window_shape[1]),
                width=3,
            )
        for y in range(self.xygrid.shape[1] + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * y),
                (self._window_shape[0], pix_square_size * y),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self._window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self._fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
