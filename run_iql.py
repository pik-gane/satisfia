#!/usr/bin/env python3
"""
IQL Timescale Algorithm Runner with Network Support

This script provides an easy way to run the IQL algorithm with both tabular and neural network backends.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Now we can import from the marl package
from corrigibility.marl.env import CustomEnvironment as GridEnvironment, Actions
from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL
from corrigibility.marl.deterministic_algorithm import DeterministicAlgorithm

from corrigibility.marl.envs.map_loader import list_available_maps, DEFAULT_MAP

import numpy as np
import argparse
import pygame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IQL Timescale Algorithm with Network Support')
    parser.add_argument('--mode', type=str, choices=['train', 'visualize', 'test'], default='train',
                        help='Mode: train (train the model), visualize (run trained model), test (run deterministic test)')
    # parser.add_argument('--algorithm', type=str, choices=['timescale', 'standard'], default='timescale',
    #                     help='Algorithm to use: timescale (two-phase timescale IQL), standard (original IQL)')
    parser.add_argument('--save', type=str, default='q_values.pkl',
                        help='Path to save trained Q-values (default: q_values.pkl)')
    parser.add_argument('--load', type=str, default='q_values.pkl',
                        help='Path to load trained Q-values (default: q_values.pkl)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training (default: 1000)')
    parser.add_argument('--phase1-episodes', type=int, default=500,
                        help='Number of episodes for Phase 1 (timescale algorithm only, default: 500)')
    parser.add_argument('--phase2-episodes', type=int, default=500,
                        help='Number of episodes for Phase 2 (timescale algorithm only, default: 500)')
    parser.add_argument('--delay', type=int, default=100,
                        help='Delay in milliseconds between steps during visualization (default: 100)')
    parser.add_argument('--map', type=str, default=DEFAULT_MAP,
                        help=f'Map to use (default: {DEFAULT_MAP}). Available maps: {", ".join(list_available_maps())}')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size for the environment (default: derived from map)')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--debug_prints', action='store_true',
                        help='Enable detailed debug prints during training (if rendering)')
    parser.add_argument('--debug_level', type=str, choices=['minimal', 'standard', 'verbose'], default='standard',
                        help='Level of debug output: minimal (goal reached only), standard (step info), verbose (full IQL details)')
    parser.add_argument('--reward-function', type=str, 
                        choices=['power', 'log', 'bounded', 'generalized_bounded'], 
                        default='power',
                        help='Robot reward function: power (default), log, bounded, or generalized_bounded')
    parser.add_argument('--concavity-param', type=float, default=1.0,
                        help='Concavity parameter c for generalized_bounded function (default: 1.0)')
    parser.add_argument('--network', action='store_true',
                        help='Use neural network Q-learning instead of tabular (default: False for tabular mode)')
    parser.add_argument('--state-dim', type=int, default=4,
                        help='State vector dimension for neural network mode (default: 4)')
    args = parser.parse_args()

    # Create environment with specified map
    env_debug_mode = args.debug_prints and args.debug_level in ['standard', 'verbose']
    env = GridEnvironment(map_name=args.map, grid_size=args.grid_size, debug_mode=env_debug_mode, debug_level=args.debug_level)

    # Enable minimal debug for goal notifications if any debug level is active
    if args.debug_prints:
        env.set_minimal_debug(True)

    if args.mode == 'test':
        # Run deterministic algorithm for testing
        print(f"Running deterministic test on map: {args.map}")
        env.render_mode = "human"  # Enable rendering for visualization
        algo = DeterministicAlgorithm(map_name=args.map)
        obs = env.reset()
        env.render()
        done = False
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = algo.choose_action(agent_obs, agent_id)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Check if episode is done
            done = any(terminations.values()) or any(truncations.values())
            
            if not done:
                env.render()
                pygame.time.delay(args.delay)
                
        print("Deterministic test completed.")
        return

    # Create action space dictionary
    action_space_dict = {}
    for agent_id in env.possible_agents:
        action_space_dict[agent_id] = list(range(env.action_space(agent_id).n))
    
    # Get human goals and create goal distribution
    human_goals_dict = env.get_all_possible_human_goals()
    env.reset()  # Ensure goals are set
    human_goals_dict = env.get_all_possible_human_goals()
    
    # Extract goal positions as a list
    human_goals_list = []
    for human_id in env.human_agent_ids:
        if human_id in human_goals_dict:
            human_goals_list.append(human_goals_dict[human_id])
    
    # If no goals found, use a default goal
    if not human_goals_list:
        human_goals_list = [(3, 2)]  # Default goal position
    
    # Create uniform distribution over goals
    mu_g = np.ones(len(human_goals_list)) / len(human_goals_list)
    
    # Initialize the IQL algorithm
    iql_params = {
        'alpha_m': 0.1, 'alpha_e': 0.1, 'alpha_r': 0.01, 'alpha_p': 0.1,
        'gamma_h': 0.99, 'gamma_r': 0.99,
        'action_space_dict': action_space_dict,
        'robot_agent_ids': env.robot_agent_ids,
        'human_agent_ids': env.human_agent_ids,
        'network': args.network,
        'G': human_goals_list,
        'mu_g': mu_g
    }
    # Network mode now uses modern Q-learning backend
    if args.network:
        iql_params['env'] = env
    
    iql = TwoPhaseTimescaleIQL(**iql_params)


    if args.mode == 'visualize':
        # Visualize trained agent
        print(f"Loading trained agent from {args.load}")
        iql.load_models(args.load)

        obs = env.reset()
        env.render()

        done = False
        step_count = 0
        current_goals = {}
        for hid in env.human_agent_ids:
            goal_idx = np.random.choice(len(iql.G), p=iql.mu_g)
            current_goals[hid] = iql.state_to_tuple(iql.G[goal_idx])

        while not done and step_count < 1000:  # Prevent infinite loops
            actions = {}
            # Get actions from the trained IQL agent
            for agent_id, agent_obs in obs.items():
                if agent_id in iql.robot_agent_ids:
                    state_r = iql.get_full_state(env, agent_id)
                    actions[agent_id] = iql.sample_robot_action_phase2(agent_id, state_r)
                elif agent_id in iql.human_agent_ids:
                    goal = current_goals[agent_id]
                    state_h = iql.get_human_state(env, agent_id, goal)
                    actions[agent_id] = iql.sample_human_action_phase1(agent_id, state_h)

            obs, rewards, terminations, truncations, infos = env.step(actions)

            done = any(terminations.values()) or any(truncations.values())
            step_count += 1

            if not done:
                env.render()
                pygame.time.delay(args.delay)

        print(f"Visualization completed after {step_count} steps.")
        return

    elif args.mode == 'train':
        # Training mode
        mode_desc = "Neural Network" if args.network else "Tabular"
        print(f"Starting Two-Phase Timescale IQL training ({mode_desc} mode)")
        print(f"Map: {args.map}, Phase 1: {args.phase1_episodes} episodes, Phase 2: {args.phase2_episodes} episodes")

        # Train the agent
        if args.render:
            print("Training with rendering enabled...")
        
        iql.train(environment=env, 
                       phase1_episodes=args.phase1_episodes, 
                       phase2_episodes=args.phase2_episodes, 
                       render=args.render, 
                       render_delay=args.delay)
        
        print("Training completed!")
        
        # Save the trained Q-values
        print(f"Saving trained Q-values to {args.save}")
        iql.save_models(file_prefix=args.save)
        
        print("Training and saving completed successfully!")

if __name__ == '__main__':
    main()
