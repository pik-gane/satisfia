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
from corrigibility.marl.env import GridEnvironment, Actions
from corrigibility.marl.iql_timescale_algorithm import TwoPhaseTimescaleIQL
from corrigibility.marl.deterministic_algorithm import DeterministicAlgorithm
from corrigibility.marl.trained_agent import TrainedAgent
from corrigibility.marl.envs.map_loader import list_available_maps, DEFAULT_MAP

import numpy as np
import argparse
import pygame

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IQL Timescale Algorithm with Network Support')
    parser.add_argument('--mode', type=str, choices=['train', 'visualize', 'test'], default='train',
                        help='Mode: train (train the model), visualize (run trained model), test (run deterministic test)')
    parser.add_argument('--algorithm', type=str, choices=['timescale', 'standard'], default='timescale',
                        help='Algorithm to use: timescale (two-phase timescale IQL), standard (original IQL)')
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

    elif args.mode == 'visualize':
        # Visualize trained agent
        print(f"Loading trained agent from {args.load}")
        trained_agent = TrainedAgent(args.load)
        
        # Check if we need network mode for loading
        if args.network:
            print("Note: Visualization in network mode uses the saved model directly")
        
        obs = env.reset()
        env.render()
        
        done = False
        step_count = 0
        while not done and step_count < 1000:  # Prevent infinite loops
            actions = trained_agent.choose_actions(obs)
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
        
        # Set up algorithm parameters
        alpha_m = 0.1
        alpha_e = 0.1  
        alpha_r = 0.1
        beta_r_0 = 5.0
        eta = 0.1
        epsilon_h_0 = 0.1
        epsilon_r = 1
        gamma_h = 0.99
        gamma_r = 0.99

        # Goals and probabilities
        G = [(0, 0), (2, 2)]  # Simple goals - can be customized per map
        mu_g = np.array([0.5, 0.5])
        p_g = 0.1

        # Agent IDs
        robot_id_iql = "robot_0"
        human_agent_ids_iql = ["human_0"]

        # Action spaces
        robot_action_space = list(Actions)
        human_action_space = list(Actions)
        
        action_space_dict = {
            robot_id_iql: robot_action_space,
            "human_0": human_action_space
        }

        # Create IQL agent
        iql_agent = TwoPhaseTimescaleIQL(
            alpha_m=alpha_m,
            alpha_e=alpha_e,
            alpha_r=alpha_r,
            gamma_h=gamma_h,
            gamma_r=gamma_r,
            beta_r_0=beta_r_0,
            G=G,
            mu_g=mu_g,
            p_g=p_g,
            action_space_dict=action_space_dict,
            robot_agent_ids=[robot_id_iql],
            human_agent_ids=human_agent_ids_iql,
            eta=eta,
            epsilon_h_0=epsilon_h_0,
            epsilon_r=epsilon_r,
            reward_function=args.reward_function,
            concavity_param=args.concavity_param,
            network=args.network,  # Enable neural network mode if specified
            state_dim=args.state_dim,  # State dimension for networks
            debug=args.debug_level == 'verbose'
        )

        # Train the agent
        if args.render:
            print("Training with rendering enabled...")
        
        iql_agent.train(environment=env, 
                       phase1_episodes=args.phase1_episodes, 
                       phase2_episodes=args.phase2_episodes, 
                       render=args.render, 
                       render_delay=args.delay)
        
        print("Training completed!")
        
        # Save the trained Q-values
        print(f"Saving trained Q-values to {args.save}")
        iql_agent.save_models(filepath=args.save)
        
        print("Training and saving completed successfully!")

if __name__ == '__main__':
    main()
