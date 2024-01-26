from . import MDPWorldModel

# based in large part on https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

CHAR_OFFSET = 63  # The ASCII code of '?' is 63
N_VALUES = 256 - CHAR_OFFSET  # The number of possible values of a character

class SimpleGridworld(MDPWorldModel):
    """A world model of a simple MDP-type Gridworld environment.
    
    A *state* here is a pair (num_state, str_state), where both entries contain the same data,
    once as a tuple of integers (which is convenient for reinforcement learning with PyTorch)
    and once as a string (which is more performant when used as a key for dictionaries).
    The data encoded in both versions of the state is following sequence of items,
    each one encoded as one or two integers (in num_state) and one or two characters (in str_state):

    - position 0: the current time step
    - positions 1+2: the current position x,y of the agent
    - positions 3+4: the previous position x,y of the agent
    - positions 5...4+k: for each of k immovable objects with variable state, its state
    - positions 5+k..4+k+2*l: for each of l movable objects without a variable state, its position x,y 
    - positions 5+k+2*l...4+k+2*l+3*m: for each of m movable objects with a variable state, 
                                       its state and its position x,y

    A *coordinate* in a position is encoded in num_state as a number from 0...192 
    and in str_state as a character from '@','A','B',... where:
    - 0 or '?' means the object is not present
    - 1 or '@' means the object is in the agent's inventory
    - 2...192 or 'A','B',... is a coordinate in the grid
    This way the numerical representation always equals the ASCII code of the character representation minus 64.

    Objects are *ordered* by their initial position in the ascii-art grid representation in row-major order.

    The *grid* and the agent's and all objects' *initial positions* are given as a 2d array of characters,
    each representing one cell of the grid, with the following character meanings: 
    
    - '#' (hash): wall
    - ' ' (blank): empty space
    - 'A': agent's initial position 
    - ... (TODO: take from gdoc, compare with pycolab asciiart conventions, 
          try to harmonize them, and add things that are missing)

    Remark: Don't confuse letters 'A'...'Z' occurring in grid ascii-art with coordinates 'A'...'Z' in state.

    *Deltas* (rewards) can accrue from the following events:
    - Time passing. This is specified by time_delta or a list time_deltas of length max_episode_length.
    - The agent stepping onto a certain object. This is specified by a list object_deltas
      ordered by the objects' initial positions in the ascii-art grid representation in row-major order.
    - The agent entering a certain position. This is specified by 
        - another 2d array of characters, delta_grid, of the same size as the grid, 
          containing cell_codes with the following character meanings:
            - ' ' (space): no Delta
            - '<character>': Delta as specified by cell_code2delta['<character>']
        - a dictionary cell_code2delta listing the actual Delta values for each cell_code in that grid
    """


    ## parameters:
    grid = None
    """(2d array of characters) The grid as an array of strings, each string representing one row of the grid,
    each character representing one cell of the grid"""
    delta_grid = None
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
    history = None
    """(singleton list of state) The current state as a singleton list."""
    t = None
    """The current time step."""
    _agent_location = None
    """(pair of ints) The current location of the agent starting with zero."""
    _previous_agent_location = None
    """(pair of ints) The previous location of the agent starting with zero."""

    # TODO Jobst: adapt the following code to our needs:

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, 
                 grid = [['A','G']],
                 delta_grid = [[' ','1']],
                 cell_code2delta = {'1': 1},
                 max_episode_length = 100,
                 time_deltas = [0],):
        self.window_size = 512  # The size of the PyGame window # TODO: understand

        self.grid = grid = np.array(grid)
        self.delta_grid = delta_grid = np.array(delta_grid)
        self.cell_code2delta = cell_code2delta
        self.max_episode_length = max_episode_length
        self.time_deltas = np.array(time_deltas).flatten()

        # the initial agent location is the first occurrence of 'A' in the grid:
        self.initial_agent_location = tuple(np.where(grid == 'A')[:,0])

        self.n_immovable_objects = self.n_movable_constant_objects = self.n_movable_variable_objects = 0  # TODO: extract from grid

        # The observation returned for reinforcement learning equals num_state, as described above.
        n1, n2 = 2+grid.shape[0], 2+grid.shape[1]
        self.observation_space = spaces.MultiDiscrete(
            [max_episode_length,  # current time step
             n1, n2,  # current position
             n1, n2]  # previous position
            + [N_VALUES] * self.n_immovable_objects 
            + [n1, n2] * self.n_movable_constant_objects 
            + [N_VALUES, n1, n2] * self.n_movable_variable_objects
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

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_target_location(self, location, action):
        direction = self._action_to_direction[action]
        return (
            self._agent_location[0] + direction[0],
            self._agent_location[1] + direction[1]
        )

    def _can_move(self, location, target_location):
        return 0 <= target_location[0] < self.grid.shape[0] \
            and 0 <= target_location[1] < self.grid.shape[1] \
            and self.grid[target_location] != '#'
        # TODO: add other conditions for not being able to move, e.g. because of other objects
    
    def possible_actions(self, state):
        location = (state[1]-2, state[2]-2)
        return [action for action in range(4) 
                if self._can_move(location, self._get_target_location(location, action))]

    def _extract_state(self, history):
        # history is either a singleton list [state] or ends in [state, delta, terminated]:
        state = history[0] if len(history) == 1 else history[-3]
        return (state[0],  # time step
                state[1]-2, state[2]-2,  # current position
                state[3]-2, state[4]-2  # previous position
         ) # TODO: extract object states and/or object locations

    def _make_observation(self, t, loc, prev_loc):
        return (t, 
                2+loc[0], 2+loc[1],
                2+prev_loc[0], 2+prev_loc[1]
                )

    def transition_distribution(self, history, action, n_samples = None):
        """Return a dictionary mapping results of calling step(action) after the given history,
        or, if history and action are None, of calling reset(),
        to tuples of the form (probability: float, exact: boolean)."""

        if history is None and action is None:
            result = self._make_observation(0, self.initial_agent_location, (-2,-2)), 0, False
            return {result: 1}

        t, loc, prev_loc = self._extract_state(history)

        # An episode is done iff the time is up or the agent is already (!) at a goal location:
        at_goal = self.grid[loc] == 'G'
        terminated = (t == self.max_episode_length) or at_goal

        # earn delta:
        delta = self.time_deltas[t % self.time_deltas.size]
        if self.delta_grid[loc] in self.cell_code2delta:
            delta += self.cell_code2delta[self.grid[loc]]
        # TODO: add object deltas
            
        # move:
        if not at_goal:
            target_loc = self._get_target_location(loc, action)
            new_loc = target_loc
            # TODO: update object states and/or object locations, e.g. if the agent picks up an object or moves an object
        else:
            new_loc = loc

        result = self._make_observation(t+1, new_loc, loc), delta, terminated

        return {result: 1} # TODO: probabilistic transitions!


    # reset() and step() are inherited from MDPWorldModel and use the above transition_distribution():

    def reset(self, seed = None, state = None): 
        ret = super().reset(seed = seed, state = state)
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

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()