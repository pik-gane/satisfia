from env import LockingDoorEnvironment, Actions
from iql_algorithm import TwoTimescaleIQL
from deterministic_algorithm import DeterministicAlgorithm
from trained_agent import TrainedAgent
from envs.map_loader import list_available_maps, DEFAULT_MAP
import numpy as np
import matplotlib.pyplot as plt
import pygame
import argparse
import os
import sys

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='IQL in custom gridworld environment')
    parser.add_argument('--mode', type=str, choices=['train', 'visualize', 'test'], default='train',
                        help='Mode: train (train the model), visualize (run trained model), test (run deterministic test)')
    parser.add_argument('--save', type=str, default='q_values.pkl',
                        help='Path to save trained Q-values (default: q_values.pkl)')
    parser.add_argument('--load', type=str, default='q_values.pkl',
                        help='Path to load trained Q-values (default: q_values.pkl)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training (default: 1000)')
    parser.add_argument('--delay', type=int, default=100,
                        help='Delay in milliseconds between steps during visualization (default: 100)')
    parser.add_argument('--map', type=str, default=DEFAULT_MAP,
                        help=f'Map to use (default: {DEFAULT_MAP}). Available maps: {", ".join(list_available_maps())}')
    parser.add_argument('--grid-size', type=int, default=None,
                        help='Grid size for the environment (default: derived from map)')
    args = parser.parse_args()

    # Create environment with specified map
    env = LockingDoorEnvironment(map_name=args.map, grid_size=args.grid_size)

    if args.mode == 'test':
        # Run deterministic algorithm for testing
        print(f"Running deterministic test on map: {args.map}")
        env.render_mode = "human"  # Enable rendering for visualization
        algo = DeterministicAlgorithm()
        
        # Standard AECEnv reset and agent iteration
        env.reset()

        for agent_id in env.agent_iter():
            observation, reward, terminated, truncated, info = env.last()

            action_to_take = None
            if terminated or truncated:
                action_to_take = None
            else:
                action_to_take = algo.choose_action(observation, agent_id)
            
            env.step(action_to_take)
            pygame.time.delay(args.delay)  # Slow down for visualization
        
        print("Deterministic test finished.")
        env.close()
    
    elif args.mode == 'visualize':
        # Visualize trained model
        if not os.path.exists(args.load):
            print(f"Error: Q-values file not found at {args.load}")
            print(f"Please train a model first using: python {sys.argv[0]} --mode train --save {args.load}")
            return

        print(f"Visualizing trained agent using Q-values from {args.load} on map: {args.map}")
        env.render_mode = "human"  # Enable rendering for visualization
        
        try:
            # Load the trained agent
            trained_agent = TrainedAgent(q_values_path=args.load)
            
            # Run the environment with the trained agent
            env.reset()

            for agent_id in env.agent_iter():
                observation, reward, terminated, truncated, info = env.last()

                action_to_take = None
                if terminated or truncated:
                    action_to_take = None
                else:
                    action_to_take = trained_agent.choose_action(observation, agent_id)
                
                env.step(action_to_take)
                pygame.time.delay(args.delay)  # Slow down for visualization
            
            print("Visualization finished.")
            
        except Exception as e:
            print(f"Error during visualization: {e}")
        
        env.close()
    
    else:  # args.mode == 'train'
        # Training mode
        print(f"Training IQL for {args.episodes} episodes on map: {args.map}")
        env.render_mode = None  # Disable rendering for training speed
        env.reset()

        # Agent IDs for IQL (consistent with env.possible_agents)
        robot_id_iql = env.robot_id_str
        human_id_iql = env.human_id_str

        # IQL hyperparameters
        alpha_h = 0.1
        alpha_r = 0.01
        gamma_h = 0.99
        gamma_r = 0.99
        beta_h = 5.0
        epsilon_r = 1.0
        
        # Goals: G should be a list of goals. Using env.goal_pos as the single goal.
        if not hasattr(env, 'goal_pos'):
            print("Error: Environment instance does not have 'goal_pos' attribute.")
            return
        G = [env.goal_pos] 
        mu_g = np.array([1.0])  # Prior probability for the single goal

        p_g = 0.01  # Probability of goal change per step
        E = args.episodes

        # Action spaces from the AECEnv for the specific agent IDs
        robot_action_space = list(range(env.action_space(robot_id_iql).n))
        human_action_space = list(range(env.action_space(human_id_iql).n))
        action_space_dict = {
            robot_id_iql: robot_action_space,
            human_id_iql: human_action_space
        }

        # Initialize IQL agent
        iql_agent = TwoTimescaleIQL(
            alpha_h=alpha_h,
            alpha_r=alpha_r,
            gamma_h=gamma_h,
            gamma_r=gamma_r,
            beta_h=beta_h,
            epsilon_r=epsilon_r,
            G=G,
            mu_g=mu_g,
            p_g=p_g,
            E=E,
            action_space_dict=action_space_dict,
            robot_agent_id=robot_id_iql,
            human_agent_id=human_id_iql,
            debug=True
        )

        # Train the agent
        print(f"Starting IQL training for {E} episodes...")
        iql_agent.train(environment=env, num_episodes=E)
        print("IQL training finished.")
        
        # Save the trained Q-values
        print(f"Saving trained Q-values to {args.save}")
        iql_agent.save_q_values(filepath=args.save)
        
        # Optionally visualize the trained agent right after training
        print("Training complete. To visualize the trained agent, run:")
        print(f"python {sys.argv[0]} --mode visualize --load {args.save} --map {args.map}")
        
        env.close()

if __name__ == "__main__":
    main()
