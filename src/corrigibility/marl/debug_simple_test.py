#!/usr/bin/env python3
"""
Simple debug script to understand why tabular IQL is failing.
"""

import sys
import numpy as np
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from envs.simple_map import get_map as get_simple_map
from env import CustomEnvironment, Actions

def debug_single_episode():
    """Debug a single episode to see what's happening"""
    print("=== Debug Single Episode ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    # Setup agent IDs and action spaces
    robot_agent_ids = ["robot_0"]
    human_agent_ids = ["human_0"]
    action_space_dict = {
        "robot_0": list(range(6)),  # [turn_left, turn_right, forward, pickup, drop, toggle]
        "human_0": list(range(3))   # [turn_left, turn_right, forward]
    }
    
    # Get possible human goals
    human_goals = list(map_metadata["human_goals"].values())
    G = [np.array(goal) for goal in human_goals]
    mu_g = [1.0 / len(G)] * len(G)  # Uniform distribution over goals
    
    # Create IQL algorithm instance (tabular case)
    iql = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.1,
        alpha_r=0.1,
        alpha_p=0.1,
        gamma_h=0.9,
        gamma_r=0.9,
        beta_r_0=5.0,
        G=G,
        mu_g=mu_g,
        action_space_dict=action_space_dict,
        robot_agent_ids=robot_agent_ids,
        human_agent_ids=human_agent_ids,
        network=False,  # Use tabular case
        env=env
    )
    
    # Test without training first
    print("\n--- Testing before training ---")
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    # Test a few steps
    for step in range(5):
        goal = env.human_goals['human_0']
        actions = {}
        
        # Human action
        state_h = iql.get_human_state(env, 'human_0', iql.state_to_tuple(goal))
        actions['human_0'] = iql.sample_human_action_phase1('human_0', state_h, epsilon=0.0)
        
        # Robot action
        state_r = iql.get_full_state(env, 'robot_0')
        actions['robot_0'] = iql.sample_robot_action_phase2('robot_0', state_r, env, epsilon=0.0)
        
        print(f"Step {step}: Human action={actions['human_0']}, Robot action={actions['robot_0']}")
        
        # Execute actions
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"Rewards: {rewards}")
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("Episode ended")
            break
    
    # Now test with minimal training
    print("\n--- Training with minimal episodes ---")
    iql.train(
        environment=env,
        phase1_episodes=200,
        phase2_episodes=200,
        render=False
    )
    
    # Test after training
    print("\n--- Testing after training ---")
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    goal_reached = False
    for step in range(50):  # Max 50 steps for more complex cooperation
        goal = env.human_goals['human_0']
        actions = {}
        
        # Human action
        state_h = iql.get_human_state(env, 'human_0', iql.state_to_tuple(goal))
        actions['human_0'] = iql.sample_human_action_phase1('human_0', state_h, epsilon=0.0)
        
        # Robot action
        state_r = iql.get_full_state(env, 'robot_0')
        actions['robot_0'] = iql.sample_robot_action_phase2('robot_0', state_r, env, epsilon=0.0)
        
        print(f"Step {step}: Human action={actions['human_0']}, Robot action={actions['robot_0']}")
        
        # Execute actions
        obs, rewards, terms, truncs, _ = env.step(actions)
        
        print(f"After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"Keys: {[k['pos'] for k in env.keys]}, Robot has keys: {env.robot_has_keys}")
        print(f"Doors: {[(d['pos'], d['is_open']) for d in env.doors]}")
        print(f"Rewards: {rewards}")
        
        # Check if human reached goal
        human_pos = env.agent_positions['human_0']
        if tuple(human_pos) == tuple(goal):
            print("GOAL REACHED!")
            goal_reached = True
            break
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("Episode ended")
            break
    
    if not goal_reached:
        print("Goal NOT reached in 50 steps")
    
    return goal_reached

def test_basic_movement():
    """Test basic movement without ML to verify environment works"""
    print("\n=== Test Basic Movement ===")
    
    # Get map and create environment
    map_layout, map_metadata = get_simple_map()
    env = CustomEnvironment(
        grid_layout=map_layout,
        grid_metadata=map_metadata,
        render_mode=None
    )
    
    env.reset()
    print(f"Initial positions: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
    print(f"Initial directions: Human={env.agent_dirs['human_0']}, Robot={env.agent_dirs['robot_0']}")
    print(f"Human goal: {env.human_goals['human_0']}")
    
    # Print the grid to understand the layout
    print("\nGrid layout:")
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) == env.agent_positions['human_0']:
                print('H', end='')
            elif (i, j) == env.agent_positions['robot_0']:
                print('R', end='')
            elif (i, j) == tuple(env.human_goals['human_0']):
                print('G', end='')
            else:
                print(env.grid[i, j], end='')
        print()
    
    # Test if manual path to goal works
    # Human starts at (1,3), goal is at (3,2)
    # Need to go down 2 steps, left 1 step
    # Direction mapping: 0=up, 1=right, 2=down, 3=left
    human_path = [
        Actions.forward,  # Move down to (2,3)
        Actions.forward,  # Move down to (3,3)
        Actions.turn_left, # Turn to face left 
        Actions.forward,  # Move left to (3,2) - goal!
    ]
    
    goal_reached = False
    for step, human_action in enumerate(human_path):
        actions = {'human_0': human_action, 'robot_0': Actions.done}  # Robot does nothing
        print(f"\nStep {step}: Human action={human_action}")
        
        # Debug the movement logic before step
        human_pos = env.agent_positions['human_0']
        human_dir = env.agent_dirs['human_0']
        deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dx, dy = deltas[human_dir]
        front_pos = (human_pos[0] + dx, human_pos[1] + dy)
        print(f"  Before: Human at {human_pos}, facing {human_dir}, front_pos would be {front_pos}")
        
        # Check if position is valid
        if human_action == Actions.forward:
            is_valid = env._is_valid_pos(front_pos)
            print(f"  Is front_pos valid? {is_valid}")
            if not is_valid:
                print(f"  Grid at front_pos: '{env.grid[front_pos]}'")
                print(f"  Grid size: {env.grid_size}")
        
        obs, rewards, terms, truncs, _ = env.step(actions)
        print(f"After step: Human={env.agent_positions['human_0']}, Robot={env.agent_positions['robot_0']}")
        print(f"Directions: Human={env.agent_dirs['human_0']}, Robot={env.agent_dirs['robot_0']}")
        print(f"Rewards: {rewards}")
        
        # Check if human reached goal
        human_pos = env.agent_positions['human_0']
        goal_pos = env.human_goals['human_0']
        if tuple(human_pos) == tuple(goal_pos):
            print("GOAL REACHED!")
            goal_reached = True
            break
        
        # Check if episode ended
        if any(terms.values()) or any(truncs.values()):
            print("Episode ended")
            break
    
    return goal_reached

if __name__ == "__main__":
    success1 = debug_single_episode()
    success2 = test_basic_movement()
    print(f"\nFinal result: IQL={'SUCCESS' if success1 else 'FAILED'}, Manual={'SUCCESS' if success2 else 'FAILED'}")