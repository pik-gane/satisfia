class DeterministicAlgorithm:
    def __init__(self):
        # Actions from env.Actions enum (ensure these match your environment's definitions)
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1
        self.ACTION_UP = 2
        self.ACTION_DOWN = 3
        self.ACTION_PICKUP = 4
        # self.ACTION_DROP = 5 # If you need drop
        self.ACTION_TOGGLE = 6
        self.ACTION_NO_OP = 7 # MODIFIED: Ensure this matches Actions.no_op from env.py

        # Define hardcoded action sequences for robot and human
        # These are examples; adjust them for your specific scenario and environment layout.
        self.robot_hardcoded_actions = [
            self.ACTION_RIGHT,   # Example: Move towards a key
            self.ACTION_PICKUP,
            self.ACTION_TOGGLE   # Example: Toggle the door
        ]

        self.human_hardcoded_actions = [
            self.ACTION_RIGHT,
            self.ACTION_RIGHT,
            self.ACTION_LEFT,
            self.ACTION_DOWN,
            self.ACTION_DOWN
        ]

        self.robot_action_index = 0
        self.human_action_index = 0

    def choose_action(self, state, agent_id: str):
        """Chooses an action based on a hardcoded sequence for the given agent_id."""
        # The 'state' argument is kept for compatibility with existing call signatures in main.py,
        # but it is not used in this hardcoded action logic.

        action = self.ACTION_NO_OP # Default action

        if agent_id == "robot_0": # MODIFIED: Match AECEnv agent ID
            if self.robot_action_index < len(self.robot_hardcoded_actions):
                action = self.robot_hardcoded_actions[self.robot_action_index]
                # print(f"Algo (Robot): Executing hardcoded action index {self.robot_action_index}: {action}")
                self.robot_action_index += 1
            else:
                # print("Algo (Robot): Ran out of hardcoded actions. Defaulting to NO_OP.")
                pass # Keep returning NO_OP
        elif agent_id == "human_0": # MODIFIED: Match AECEnv agent ID
            if self.human_action_index < len(self.human_hardcoded_actions):
                action = self.human_hardcoded_actions[self.human_action_index]
                # print(f"Algo (Human): Executing hardcoded action index {self.human_action_index}: {action}")
                self.human_action_index += 1
            else:
                # print("Algo (Human): Ran out of hardcoded actions. Defaulting to NO_OP.")
                pass # Keep returning NO_OP
        else:
            print(f"Algo: Unknown agent_id '{agent_id}'. Defaulting to NO_OP.")
        
        return action