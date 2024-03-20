from enum import Enum
from dataclasses import dataclass
import numpy as np
from typing import List
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete

# x is the horizontal axis, it is bigger on the right
# y is the vertical axis, it is bigger on the top
# (the convention is like in math)

Color = Enum("Color", ["white", "black"])
Cell = Enum("Cell", ["inaccessible", "empty", "white", "black"])

Side = Enum("Side", ["left", "right"])

def opposite_color(color):
    return {Color.white: Color.black, Color.black: Color.white}[color]

def piece_of_player(color):
    return {Color.white: Cell.white, Color.black: Cell.black}[color]

def x_direction(side):
    return {Side.left: -1, Side.right: +1}[side]

def y_direction(player_color):
    return {Color.white: +1, Color.black: -1}[player_color]

@dataclass
class Move:
    from_x: int
    from_y: int
    side: int
    player: int
    board_width: int
    board_height: int

    def __post_init__(self):
        assert (self.from_x + self.from_y) % 2 == 0 

        self.to_x           = (self.from_x +     x_direction(self.side)) % self.board_width
        self.to_x_if_eating = (self.from_x + 2 * x_direction(self.side)) % self.board_width

        self.to_y           = (self.from_y +     y_direction(self.player)) % self.board_height
        self.to_y_if_eating = (self.from_y + 2 * y_direction(self.player)) % self.board_height

def initial_board(board_width, board_height, num_rows_with_pieces_initially):
    assert board_height % 2 == 0
    assert board_width % 2 == 0
    assert 2 * num_rows_with_pieces_initially <= board_width

    board = np.full((board_width, board_height), Cell.empty)
    
    for x in range(board_width):
        for y in range(board_height):
            if (x + y) % 2 == 1:
                board[x, y] = Cell.inaccessible
            elif y < num_rows_with_pieces_initially:
                board[x, y] = Cell.white
            elif y >= board_height - num_rows_with_pieces_initially:
                board[x, y] = Cell.black
            else:
                board[x, y] = Cell.empty

    return board

@dataclass
class MoveResult:
    legal: bool
    eaten: bool = False

def play_move(board, move, modify_board=True) -> MoveResult:
    if board[move.from_x, move.from_y] != piece_of_player(move.player):
        return MoveResult(legal=False)
        
    non_eating_move_possible = board[move.to_x, move.to_y] == Cell.empty
    if non_eating_move_possible:
        if modify_board:
            board[move.from_x, move.from_y] = Cell.empty
            board[move.to_x, move.to_y] = piece_of_player(move.player)
        return MoveResult(legal=True, eaten=False)
        
    eating_move_possible = board[move.to_x, move.to_y] == piece_of_player(opposite_color(move.player)) \
                            and board[move.to_x_if_eating, move.to_y_if_eating] == Cell.empty
    if eating_move_possible:
        if modify_board:
            board[move.from_x, move.from_y] = Cell.empty
            board[move.to_x, move.to_y] = Cell.empty
            board[move.to_x_if_eating, move.to_y_if_eating] = piece_of_player(move.player)
        return MoveResult(legal=True, eaten=True)

    return MoveResult(legal=False)

def is_legal(board, move):
    return play_move(board, move, modify_board=False).legal

def all_moves(player, board_width, board_height):
    return ( Move(from_x=from_x, from_y=from_y, side=side, player=player, board_width=board_width, board_height=board_height)
             for from_x in range(board_width)
             for from_y in range(board_height)
             if (from_x + from_y) % 2 == 0
             for side in [Side.left, Side.right] )

def legal_moves(player, board, board_width, board_height):
    moves = all_moves(player=player, board_width=board_width, board_height=board_height)
    return [move for move in moves if is_legal(board, move)]

def side_to_id(side):
    return {Side.left: 0, Side.right: 1}[side]

def side_from_id(side_id):
    return {0: Side.left, 1: Side.right}[side_id]

def combine_ids(ids: List[int], bounds: List[int]) -> int:
    assert len(ids) == len(bounds)
    combined = 0
    for id, bound in zip(ids, bounds):
        combined = bound * combined + id
    return combined

def extract_ids(combined_ids: int, bounds: List[int]) -> List[int]:
    ids = []
    for bound in bounds[::-1]:
        ids = [combined_ids % bound] + ids
        combined_ids //= bound
    assert combined_ids == 0
    return ids

def move_to_id(move, board_width, board_height):
    nsides = 2
    return combine_ids( ids    = [move.from_x // 2, move.from_y,  side_to_id(move.side)],
                        bounds = [board_width // 2, board_height, nsides] )

def move_from_id(move_id, player, board_width, board_height):
    nsides = 2
    from_x, from_y, side = extract_ids(move_id, bounds=[board_width // 2, board_height, nsides])
    from_x *= 2
    side = side_from_id(side)

    if (from_x + from_y) % 2 != 0:
        from_x = from_x + 1

    return Move(from_x=from_x, from_y=from_y, side=side, player=player, board_width=board_width, board_height=board_height)

def cell_to_id(cell):
    return {Cell.empty: 0, Cell.white: 1, Cell.black: 2}[cell]

def board_to_ids(board):
    return np.array([cell_to_id(cell) for cell in board.flatten() if cell != Cell.inaccessible])

def num_accessible_cells(board_width, board_height):
    num_cells = board_width * board_height
    return num_cells // 2

def num_possible_moves(board_width, board_height):
    num_sides = 2
    return num_sides * num_accessible_cells(board_width=board_width, board_height=board_height)

def board_to_string(board, board_width, board_height):
    cell_to_string = {Cell.inaccessible: " ", Cell.empty: ".", Cell.white: "w", Cell.black: "b"}

    return "\n".join( "".join(cell_to_string[board[x, y]] for x in range(board_width))
                      for y in range(board_height-1, -1, -1) )

class SimplifiedCheckersAgainstRandomPlayer(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_width=8, board_height=8, num_rows_with_pieces_initially=3, agent_color=Color.white):
        self.board_width = board_width
        self.board_height = board_height
        self.num_rows_with_pieces_initially = num_rows_with_pieces_initially
        self.agent_color = agent_color
        
        self.action_space = Discrete(num_possible_moves(board_width=board_width, board_height=board_height))
        
        self.observation_space = MultiDiscrete([3] * num_accessible_cells(board_width=board_width, board_height=board_height))

    def reset(self, seed=None, options=None):
        # TO DO: use seed

        self.board = initial_board( board_width=self.board_width,
                                    board_height=self.board_height,
                                    num_rows_with_pieces_initially=self.num_rows_with_pieces_initially )

        self.reward_before_first_step = None

        if self.agent_color == Color.black:
            adversary_move = np.random.choice(legal_moves( player=Color.black,
                                                           board=self.board,
                                                           board_height=self.board_height,
                                                           board_width=self.board_width))
            result = play_move(self.board, adversary_move)
            if result.eaten:
                self.reward_before_first_step = -1.

        observation = board_to_ids(self.board)
        assert self.observation_space.contains(observation)
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0.

        if self.reward_before_first_step is not None:
            reward += self.reward_before_first_step
            self.reward_before_first_step = None

        agent_move = move_from_id( move_id=action,
                                   player=self.agent_color,
                                   board_width=self.board_width,
                                   board_height=self.board_height )
        if not is_legal(self.board, agent_move):
            reward -= 1.
            agent_legal_moves = legal_moves(self.agent_color, self.board, board_width=self.board_width, board_height=self.board_height)
            if agent_legal_moves == []:
                observation = self.action_space.sample()
                done = True
                truncated = False
                return observation, reward, done, truncated, {}
            agent_move = np.random.choice(agent_legal_moves)
        agent_move_result = play_move(self.board, agent_move)
        assert agent_move_result.legal
        if agent_move_result.eaten:
            reward += 1.

        adversary_legal_moves = legal_moves(opposite_color(self.agent_color), self.board, board_width=self.board_width, board_height=self.board_height)
        if adversary_legal_moves == []:
            observation = self.action_space.sample()
            done = True
            truncated = False
            return observation, reward, done, truncated, {}
        adversary_move = np.random.choice(adversary_legal_moves)
        adversary_move_result = play_move(self.board, adversary_move)
        assert adversary_move_result.legal
        if adversary_move_result.eaten:
            reward -= 1

        observation = board_to_ids(self.board)
        assert self.observation_space.contains(observation)
        done = False
        truncated = False
        return observation, reward, done, truncated, {}

    def render(self, mode="human", close=False):
        print(board_to_string(self.board, board_width=self.board_width, board_height=self.board_height))

gym.register( id="SimplifiedCheckersAgainstRandomPlayer-v0",
              entry_point="simplified_checkers_against_random_player:SimplifiedCheckersAgainstRandomPlayer" )