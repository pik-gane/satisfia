import numpy as np
import gymnasium as gym # Import gymnasium
from gymnasium import spaces # Import spaces
import random # Import random for goal selection

from minigrid.core.constants import COLOR_NAMES, OBJECT_TO_IDX, IDX_TO_OBJECT
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj, Ball, Door # Using Ball to represent Human, Import Door
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import fill_coords, point_in_rect # Ensure rendering utils are imported
from minigrid.core.constants import COLORS # Ensure COLORS are imported

# Define a new object type for the human NPC
# Find the next available index
next_idx = max(OBJECT_TO_IDX.values()) + 1
IDX_TO_OBJECT[next_idx] = "human"
OBJECT_TO_IDX["human"] = next_idx

class HumanNPC(WorldObj):
    def __init__(self, color="blue"):
        super().__init__("human", color)

    def can_overlap(self):
        return True # Agent can potentially occupy the same cell

    def render(self, img):
        # Render as a blue ball
        fill_coords(img, point_in_rect(0.25, 0.75, 0.25, 0.75), COLORS[self.color])

# Add new actions for toggle and lock
class ExtendedActions(MiniGridEnv.Actions):
    toggle = 6 # Use next available index after standard actions
    lock = 7   # Use next available index

class PowerGridEnv(MiniGridEnv):
    """
    MiniGrid environment with a Robot (agent) and a Human (NPC).
    Includes a room with a door that can be toggled or locked by the agent.
    Uses Dict observation space: {agent_pos, human_pos, goal_pos}.
    The agent's goal is implicitly to help the human reach its goal.
    """
    def __init__(
        self,
        size=9, # Increased size slightly for room
        agent_start_pos=None, # Place agent inside room
        human_start_pos=None, # Place human inside room
        # human_goal_pos is now selected from possible_goal_positions (outside room)
        max_steps=150, # Increased max steps slightly
        **kwargs,
    ):
        self.agent_start_pos_initial = agent_start_pos # Store initial preference
        self.human_start_pos_initial = human_start_pos # Store initial preference
        self._door_pos = None # Store door position

        # Define the set of possible goals the environment can use
        # Place goals outside the room
        self.possible_goal_positions = [
            (size - 2, size // 2), # Right side, middle
            (size // 2, size - 2), # Bottom side, middle
        ]
        # Filter out agent start position if it conflicts (unlikely now)
        self.possible_goal_positions = [
            p for p in self.possible_goal_positions if p != self.agent_start_pos_initial
        ]
        if not self.possible_goal_positions: # Fallback if all goals conflict
             self.possible_goal_positions = [(size // 2, size // 2)]

        # Store human position and the goal for the current episode
        self._human_pos = None
        self._human_goal_pos = None # Set in _gen_grid

        mission_space = MissionSpace(mission_func=lambda: "Help the human reach the goal outside the room")

        # Define observation space (Dict) - unchanged for now
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "human_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "goal_pos": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })

        # Action enumeration includes new actions
        self.actions = ExtendedActions

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )
        # print(f"Initialized PowerGridEnv. Observation Space: {self.observation_space}")


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Create a room
        room_w = width // 2 + 1
        room_h = height // 2 + 1
        self.grid.wall_rect(1, 1, room_w -1, room_h -1) # Inner walls

        # Place the door in the middle of the right wall of the room
        self._door_pos = (room_w - 1, room_h // 2)
        # Door starts closed and unlocked (default color is yellow)
        self.put_obj(Door(color="yellow", is_open=False, is_locked=False), self._door_pos[0], self._door_pos[1])

        # --- Placement Order: Goal -> Human -> Agent ---

        # 1. Place the human goal randomly from the possible set (outside the room)
        # Ensure goal positions are valid and outside the room
        valid_goals = []
        for gx, gy in self.possible_goal_positions:
            if gx >= room_w or gy >= room_h: # Check if outside room bounds
                 valid_goals.append((gx, gy))
        if not valid_goals: # Fallback if predefined goals are somehow inside
             valid_goals = [(width - 2, height - 2)] # Corner outside
        self.possible_goal_positions = valid_goals

        goal_idx = self.np_random.integers(0, len(self.possible_goal_positions))
        self._human_goal_pos = self.possible_goal_positions[goal_idx]
        self.put_obj(Goal(), self._human_goal_pos[0], self._human_goal_pos[1])
        # print(f"Placed goal at: {self._human_goal_pos}")

        # 2. Place the human NPC inside the room
        # Define placement area inside the room, excluding walls and door
        human_placement_kwargs = {
            'top': (2, 2),
            'size': (room_w - 3, room_h - 3),
            'reject_pos': [self._door_pos, self._human_goal_pos], # Avoid door and goal
            'max_tries': 100
        }
        if self.human_start_pos_initial is not None:
            # Check if specified start pos is valid (inside room, not door/goal)
            hx, hy = self.human_start_pos_initial
            if 1 < hx < room_w - 1 and 1 < hy < room_h - 1 and (hx, hy) != self._door_pos:
                 self._human_pos = self.human_start_pos_initial
                 self.put_obj(HumanNPC(), self._human_pos[0], self._human_pos[1])
            else:
                 print(f"Warning: Invalid human start pos {self.human_start_pos_initial}. Placing randomly inside room.")
                 start_pos = self.place_obj(HumanNPC(), **human_placement_kwargs)
                 if start_pos is None: raise RuntimeError("Failed to place human randomly in room")
                 self._human_pos = start_pos
        else:
            start_pos = self.place_obj(HumanNPC(), **human_placement_kwargs)
            if start_pos is None: raise RuntimeError("Failed to place human randomly in room")
            self._human_pos = start_pos
        # print(f"Placed human at: {self._human_pos}")


        # 3. Place the agent inside the room
        # Define placement area inside the room, excluding walls, door, human, goal
        agent_placement_kwargs = {
            'top': (2, 2),
            'size': (room_w - 3, room_h - 3),
            'reject_pos': [self._door_pos, self._human_goal_pos, self._human_pos], # Avoid door, goal, human
            'max_tries': 100
        }
        if self.agent_start_pos_initial is not None:
             ax, ay = self.agent_start_pos_initial
             if 1 < ax < room_w - 1 and 1 < ay < room_h - 1 and (ax, ay) != self._door_pos and (ax, ay) != self._human_pos:
                 self.agent_pos = self.agent_start_pos_initial
                 self.agent_dir = 0 # Start facing right (consistent)
             else:
                 print(f"Warning: Invalid agent start pos {self.agent_start_pos_initial}. Placing randomly inside room.")
                 self.place_agent(**agent_placement_kwargs)
        else:
             self.place_agent(**agent_placement_kwargs)

        # print(f"Placed agent at: {self.agent_pos}")
        self.mission = "Help the human reach the green goal square outside the room"


    def gen_obs(self):
        # Return the dictionary observation based on current state
        agent_pos = tuple(self.agent_pos) if self.agent_pos is not None else (-1, -1)
        human_pos = tuple(self._human_pos) if self._human_pos is not None else (-1, -1)
        goal_pos = tuple(self._human_goal_pos) if self._human_goal_pos is not None else (-1, -1)

        obs = {
            "agent_pos": agent_pos,
            "human_pos": human_pos,
            "goal_pos": goal_pos
        }
        # print(f"Generated obs: {obs}")
        return obs

    def reset(self, *, seed=None, options=None):
        # Call parent reset to set up grid, agent position etc.
        super().reset(seed=seed) # This calls _gen_grid

        # Generate the first Dict observation
        obs = self.gen_obs()

        # Get the custom info (human/goal pos)
        info = self._get_info()

        # Reset step count
        self.step_count = 0

        # print(f"Reset done. Obs: {obs}, Info: {info}")
        return obs, info

    def _get_info(self):
         # Provide human and goal positions in info dict for easy access by agent
         human_pos = tuple(self._human_pos) if self._human_pos is not None else (-1, -1)
         goal_pos = tuple(self._human_goal_pos) if self._human_goal_pos is not None else (-1, -1)
         return {"human_pos": human_pos, "goal_pos": goal_pos}

    def _move_human_npc(self):
        """Moves the human NPC one step towards its goal. Returns True if moved."""
        if self._human_pos is None or self._human_goal_pos is None or self._human_pos == self._human_goal_pos:
            return False # Cannot or no need to move

        hx, hy = self._human_pos
        gx, gy = self._human_goal_pos
        original_pos = self._human_pos

        # Simple greedy movement towards goal
        dx = np.sign(gx - hx)
        dy = np.sign(gy - hy)

        moved = False

        # Function to check if a cell is passable for the human
        def is_passable(cell):
            if cell is None or cell.type == 'goal' or cell.can_overlap():
                return True
            if isinstance(cell, Door):
                # Human can pass open doors or toggle unlocked closed doors
                return cell.is_open or not cell.is_locked
            return False

        # Try moving horizontally first if primary direction or only option
        if dx != 0 and abs(gx - hx) >= abs(gy - hy):
            next_pos_x = (hx + dx, hy)
            cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
            if is_passable(cell_x):
                 # If it's a closed door, human 'opens' it implicitly by moving
                 if isinstance(cell_x, Door) and not cell_x.is_open:
                     pass
                 self._human_pos = next_pos_x
                 moved = True

        # Try moving vertically if horizontal failed or is primary direction
        if not moved and dy != 0:
            next_pos_y = (hx, hy + dy)
            cell_y = self.grid.get(next_pos_y[0], next_pos_y[1])
            if is_passable(cell_y):
                 if isinstance(cell_y, Door) and not cell_y.is_open:
                     pass
                 self._human_pos = next_pos_y
                 moved = True

        # If primary direction failed, try the other direction (if not already tried)
        if not moved and dx != 0 and abs(gx - hx) < abs(gy - hy):
             next_pos_x = (hx + dx, hy)
             cell_x = self.grid.get(next_pos_x[0], next_pos_x[1])
             if is_passable(cell_x):
                 if isinstance(cell_x, Door) and not cell_x.is_open:
                     pass
                 self._human_pos = next_pos_x
                 moved = True

        # print(f"Human move attempt: {original_pos} -> {self._human_pos} (Goal: {self._human_goal_pos})")
        return moved


    def step(self, action):
        """Agent takes action, then Human NPC moves. Returns Dict obs."""
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # --- 1. Agent Action ---
        # Store previous agent pos/dir
        prev_agent_pos = self.agent_pos
        prev_agent_dir = self.agent_dir

        # Update agent state based on action
        # Rotate left/right
        if action == self.actions.left or action == self.actions.right:
            if action == self.actions.left:
                self.agent_dir -= 1
                if self.agent_dir < 0:
                    self.agent_dir += 4
            elif action == self.actions.right:
                self.agent_dir = (self.agent_dir + 1) % 4
        # Move forward
        elif action == self.actions.forward:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            # Allow moving into empty, goal, human, or OPEN doors
            can_move = False
            if fwd_cell is None or fwd_cell.can_overlap() or fwd_cell.type == 'goal':
                can_move = True
            elif isinstance(fwd_cell, Door) and fwd_cell.is_open:
                can_move = True

            if can_move:
                self.agent_pos = tuple(fwd_pos)
        # Toggle action
        elif action == self.actions.toggle:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and isinstance(fwd_cell, Door):
                # Can only toggle unlocked doors
                if not fwd_cell.is_locked:
                    fwd_cell.toggle(self, fwd_pos)
        # Lock action
        elif action == self.actions.lock:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is not None and isinstance(fwd_cell, Door):
                # Can only lock closed doors
                if not fwd_cell.is_open:
                    fwd_cell.is_locked = True
                    fwd_cell.color = "red" # Change color to indicate locked
        elif action == self.actions.done: # 'done' action exists but not used here
            pass
        else:
            # This case should ideally be prevented by the action mapping in main.py
            print(f"Warning: Unknown action {action} received in env.step. Agent stays.")

        # --- 2. Human NPC Movement ---
        # Clear human's old position on grid *before* moving
        # Only clear if it wasn't the goal cell
        if self._human_pos:
            cell_at_human = self.grid.get(self._human_pos[0], self._human_pos[1])
            if cell_at_human is None or cell_at_human.type != 'goal':
                 self.grid.set(self._human_pos[0], self._human_pos[1], None)

        # Move human
        human_moved = self._move_human_npc()

        # Place human object in new position on grid *after* moving
        # Only place if the new position isn't the goal cell
        if self._human_pos:
            cell_at_new_human = self.grid.get(self._human_pos[0], self._human_pos[1])
            if cell_at_new_human is None or cell_at_new_human.type != 'goal':
                 self.grid.set(self._human_pos[0], self._human_pos[1], HumanNPC())


        # --- 3. Calculate Rewards and Termination ---
        human_succeeded = False
        if self._human_pos == self._human_goal_pos:
            human_succeeded = True
            terminated = True # End episode if human reaches goal

        # Robot Reward (Proxy for Power): Reward agent if human succeeded
        if human_succeeded:
            reward = 1.0 # Positive reward for agent
        else:
            # Small step penalty for agent to encourage efficiency
            reward = -0.01

        # Base Human Reward (for agent's internal model update)
        # 1.0 if human reached goal *this step*, 0 otherwise
        r_h_obs = 1.0 if human_succeeded else 0.0

        # --- 4. Check Truncation ---
        if self.step_count >= self.max_steps:
            truncated = True

        # --- 5. Generate Observation and Info ---
        obs = self.gen_obs()
        info = self._get_info()

        # --- 6. Render ---
        if self.render_mode == "human":
            self.render()

        return obs, reward, r_h_obs, terminated, truncated, info


# Example usage (updated for new actions)
if __name__ == "__main__":
    env = PowerGridEnv(size=9, render_mode="human") # Use human render mode
    obs, info = env.reset()

    for i in range(150): # Max 150 steps example
        # Sample a valid environment action (0-7 now)
        # For testing, let's sample agent actions 0-5 and map them
        # Agent actions: 0:Up, 1:Down, 2:Left, 3:Right, 4:Toggle, 5:Lock
        # Env actions:   3:Up, 1:Down, 2:Left, 0:Right, 6:Toggle, 7:Lock
        agent_action = random.choice([0, 1, 2, 3, 4, 5]) # Sample semantic agent action

        action_map_test = { 0: 3, 1: 1, 2: 2, 3: 0, 4: 6, 5: 7 }
        env_action = action_map_test.get(agent_action, 0) # Default to right if map fails

        print(f"Step {i+1}, Agent Action: {agent_action}, Env Action: {env_action}")
        obs, reward, r_h_obs, terminated, truncated, info = env.step(env_action)
        print(f"Agent Reward: {reward:.2f}, Human Base Reward: {r_h_obs}, Term: {terminated}, Trunc: {truncated}")
        print(f"Agent Pos: {obs['agent_pos']}, Human Pos: {info['human_pos']}, Goal Pos: {info['goal_pos']}")
        # Check door state (optional, requires accessing grid)
        door_cell = env.grid.get(*env._door_pos)
        if door_cell: print(f"Door State: Open={door_cell.is_open}, Locked={door_cell.is_locked}")


        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

    env.close()
