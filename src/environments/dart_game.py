from world_model.mdp_world_model import MDPWorldModel, State
from typing import Any, Optional
from gymnasium import spaces

class DartGame(MDPWorldModel):
    def __init__(self, *args, p_exact: float=0.2, total_throws: int=3) -> None:
        super().__init__()
        self.p_exact = p_exact
        self.total_throws = total_throws
        self.observation_space = spaces.Discrete(self.total_throws+1)
        self.action_space = spaces.Discrete(5)

    def transition_distribution(self, state: Any, action: Any | None, n_samples: int | None = None):
        if state is None and action is None:
            return {self.total_throws: (1.0, True)}
        return {state-1: (1.0, True)}
    
    def observation_and_reward_distribution(self, state: Any | None, action: Any | None, successor: Any, n_samples: int | None = None):
        if state is None and action is None:
            return {(successor, (0, 0)): (1.0, True)}
        aim_x, aim_y = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)][action]
        return {(successor, (aim_x, aim_y)): (self.p_exact, True),
               (successor, (aim_x-1, aim_y)): ((1-self.p_exact)/4, True),
               (successor, (aim_x+1, aim_y)): ((1-self.p_exact)/4, True),
               (successor, (aim_x, aim_y-1)): ((1-self.p_exact)/4, True),
               (successor, (aim_x, aim_y+1)): ((1-self.p_exact)/4, True)}
    
    def possible_actions(self, state: Optional[State] = None):
        """Return the list of all actions possible in a given state or in the current state if state is None.
        
        This default implementation assumes that the action space is of type gymnasium.spaces.Discrete,
        representing a range of integers."""
        #space = self.action_space
        return range(5)
    
    def is_terminal(self, state: Any) -> bool:
        return state == 0
    
    def _set_state(self, state: Any):
        self._state = state