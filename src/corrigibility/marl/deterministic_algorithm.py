import os
import json
from env import Actions

class DeterministicAlgorithm:
    def __init__(self, map_name="simple_map", algo_dir="saved"):
        # Align action constants with Actions enum
        self.ACTION_LEFT = int(Actions.turn_left)
        self.ACTION_RIGHT = int(Actions.turn_right)
        # forward is the only movement action in env
        self.ACTION_FORWARD = int(Actions.forward)
        self.ACTION_PICKUP = int(Actions.pickup)
        self.ACTION_DROP = int(Actions.drop)
        self.ACTION_TOGGLE = int(Actions.toggle)
        self.ACTION_NO_OP = int(Actions.done)

        # store map-specific sequences in JSON under algo_dir
        self.map_name = map_name
        os.makedirs(algo_dir, exist_ok=True)
        self.algo_path = os.path.join(algo_dir, f"deterministic_{map_name}.json")
        if os.path.exists(self.algo_path):
            self._load_algorithm()
        else:
            # default sequences: override per map as needed
            self.robot_hardcoded_actions = [self.ACTION_LEFT, self.ACTION_NO_OP, self.ACTION_FORWARD, self.ACTION_RIGHT, self.ACTION_TOGGLE, self.ACTION_RIGHT, self.ACTION_FORWARD]
            # Use forward for movement, default facing down
            self.human_hardcoded_actions = [self.ACTION_RIGHT, self.ACTION_NO_OP, self.ACTION_NO_OP, self.ACTION_NO_OP, self.ACTION_NO_OP, self.ACTION_NO_OP, self.ACTION_NO_OP, self.ACTION_FORWARD, self.ACTION_LEFT, self.ACTION_FORWARD,self.ACTION_FORWARD]
            self.save_algorithm()
        self.robot_action_index = 0
        self.human_action_index = 0

    def save_algorithm(self):
        data = {
            "robot_actions": self.robot_hardcoded_actions,
            "human_actions": self.human_hardcoded_actions
        }
        with open(self.algo_path, 'w') as f:
            json.dump(data, f)

    def _load_algorithm(self):
        with open(self.algo_path) as f:
            data = json.load(f)
        self.robot_hardcoded_actions = data.get("robot_actions", [])
        self.human_hardcoded_actions = data.get("human_actions", [])

    def choose_action(self, state, agent_id: str):
        """Chooses an action based on a hardcoded sequence for the given agent_id."""
        # The 'state' argument is kept for compatibility with existing call signatures in main.py,
        # but it is not used in this hardcoded action logic.
        action = self.ACTION_NO_OP # Default action

        if agent_id == f"robot_0":
            if self.robot_action_index < len(self.robot_hardcoded_actions):
                action = self.robot_hardcoded_actions[self.robot_action_index]
                self.robot_action_index += 1
        elif agent_id == f"human_0":
            if self.human_action_index < len(self.human_hardcoded_actions):
                action = self.human_hardcoded_actions[self.human_action_index]
                self.human_action_index += 1
        else:
            print(f"Algo: Unknown agent_id '{agent_id}'. Defaulting to NO_OP.")
        
        # Validate action against Actions enum, fallback to NO_OP if invalid
        try:
            Actions(action)
        except Exception:
            action = self.ACTION_NO_OP
        return action