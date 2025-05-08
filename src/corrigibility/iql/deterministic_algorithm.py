class DeterministicAlgorithm:
    def __init__(self):
        # Actions from env.Actions enum
        self.ACTION_UP = 2
        self.ACTION_DOWN = 3
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1
        self.ACTION_TOGGLE = 6
        self.ACTION_LOCK = 7
        # self.ACTION_PICKUP = 4 # Not used in this scenario

    def choose_action(self, state):
        # Unpack state based on the new observation space in env.py
        agent_pos, human_pos, door_pos, goal_pos, \
        robot_has_key, door_is_open, door_is_key_locked, door_is_permanently_locked = state

        agent_r, agent_c = agent_pos
        door_r, door_c = door_pos
        goal_r, goal_c = goal_pos

        # Priority 1: If at door and door is not open, and not permanently locked, toggle it open.
        if agent_pos == door_pos and not door_is_open and not door_is_permanently_locked:
            print("Algo: At door, door is closed and not permalocked. Action: TOGGLE (to open)")
            return self.ACTION_TOGGLE 

        # Priority 2: If at door and door is open, and human is outside room, consider locking (for testing)
        # This part is to test the 'lock' action. 
        # A truly helpful agent might avoid this unless specifically beneficial.
        # For this test, let's assume the robot wants to reach the goal first.
        # if agent_pos == door_pos and door_is_open and not door_is_permanently_locked and human_pos[0] < door_pos[0]:
        #     print("Algo: At door, door is open, human outside. Action: LOCK (for testing)")
        #     return self.ACTION_LOCK

        # Priority 3: Navigate towards the door if outside the room and door is not permanently locked
        if agent_r > door_r and not door_is_permanently_locked: # Agent is below the room wall
            print(f"Algo: Navigating to door. Agent at {agent_pos}, Door at {door_pos}. Action: UP")
            return self.ACTION_UP
        
        # Navigate towards door (horizontal alignment first, then vertical)
        if agent_c < door_c and not door_is_permanently_locked and agent_r == door_r -1: # agent is to the left of door, on row above door wall
            print(f"Algo: Aligning with door. Agent at {agent_pos}, Door at {door_pos}. Action: RIGHT")
            return self.ACTION_RIGHT
        if agent_c > door_c and not door_is_permanently_locked and agent_r == door_r -1:
            print(f"Algo: Aligning with door. Agent at {agent_pos}, Door at {door_pos}. Action: LEFT")
            return self.ACTION_LEFT
        if agent_r < door_r -1 and not door_is_permanently_locked : # Agent is above the door approach row
             print(f"Algo: Moving to door approach row. Agent at {agent_pos}, Door at {door_pos}. Action: DOWN")
             return self.ACTION_DOWN


        # Priority 4: Once inside the room (agent_r > door_r) and door is open, navigate towards the goal.
        if agent_r > door_r and door_is_open:
            print(f"Algo: Inside room, door open. Navigating to goal. Agent at {agent_pos}, Goal at {goal_pos}")
            if agent_c < goal_c: return self.ACTION_RIGHT
            if agent_c > goal_c: return self.ACTION_LEFT
            if agent_r < goal_r: return self.ACTION_DOWN # Goal is typically lower (higher row index)
            if agent_r > goal_r: return self.ACTION_UP
        
        # Fallback: if on the door row but not at door (e.g. after opening it and moving away slightly)
        # and goal is inside, try to move towards goal.
        if agent_r == door_r and agent_pos != door_pos and agent_r < goal_r and door_is_open:
            print(f"Algo: On door row, door open. Navigating to goal. Agent at {agent_pos}, Goal at {goal_pos}")
            if agent_c < goal_c: return self.ACTION_RIGHT
            if agent_c > goal_c: return self.ACTION_LEFT
            return self.ACTION_DOWN # Move towards goal row

        print(f"Algo: No specific action. Defaulting. State: {state}")
        # Default action if none of the above (e.g., stay put or random, here just a default)
        return self.ACTION_DOWN # Example default