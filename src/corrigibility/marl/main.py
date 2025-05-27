from env import GridEnvironment, Actions
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
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
    args = parser.parse_args()

    # Create environment with specified map
    # Only enable env debug mode for standard and verbose levels
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
            env.render()
            done = any(terminations.values()) or any(truncations.values())
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
        
        trained_agent = TrainedAgent(q_values_path=args.load)
        obs = env.reset()
        env.render()
        done = False
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = trained_agent.choose_action(agent_obs, agent_id)
            obs, rewards, terminations, truncations, infos = env.step(actions)
            env.render()
            done = any(terminations.values()) or any(truncations.values())
            pygame.time.delay(args.delay)  # Slow down for visualization
        
        print("Visualization finished.")
        env.close()
    
    else:  # args.mode == 'train'
        # Training mode
        if args.algorithm == 'timescale':
            print(f"Training Two-Phase Timescale IQL: Phase1={args.phase1_episodes}, Phase2={args.phase2_episodes} episodes on map: {args.map}")
        else:
            print(f"Training Standard IQL for {args.episodes} episodes on map: {args.map}")
        if args.render:
            env.render_mode = "human"
            env.reset()
            env.render()
        else:
            env.render_mode = None  # Disable rendering for training speed
        env.reset()

        # Agent IDs for IQL (consistent with env.possible_agents)
        # Use first robot and human IDs from environment lists
        robot_id_iql = env.robot_agent_ids[0]
        human_id_iql = env.human_agent_ids[0]

        # IQL hyperparameters
        alpha_h = 0.1
        alpha_r = 0.01
        gamma_h = 0.99
        gamma_r = 0.99
        beta_h = 5.0
        epsilon_r = 1.0
        
        # Goals: G should be a list of goals. Retrieve the human goal using the selected human ID.
        human_goal = env.human_goals.get(human_id_iql)
        if human_goal is None:
            print(f"Error: No goal mapped for human agent '{human_id_iql}' in environment.")
            return
        G = [human_goal]
        mu_g = np.array([1.0])  # Prior probability for the single goal

        p_g = 0.01  # Probability of goal change per step
        E = args.episodes

        # Hard-coded action spaces: robot has actions 0-6, human has actions 0,1,2,6
        robot_action_space = [0, 1, 2, 3, 4, 5, 6]
        human_action_space = [0, 1, 2, 6]
        action_space_dict = {
            robot_id_iql: robot_action_space,
            human_id_iql: human_action_space
        }

        # Initialize algorithm based on choice
        if args.algorithm == 'timescale':
            # Timescale algorithm hyperparameters
            alpha_m = 0.1   # Phase 1 learning rate
            alpha_e = 0.2   # Phase 2 fast timescale learning rate
            beta_r = 5.0    # Robot softmax temperature
            eta = 0.1       # Power parameter for robot reward
            epsilon_h = 0.1 # Human epsilon-greedy parameter
            
            iql_agent = TwoPhaseTimescaleIQL(
                alpha_m=alpha_m,
                alpha_e=alpha_e,
                alpha_r=alpha_r,
                gamma_h=gamma_h,
                gamma_r=gamma_r,
                beta_h=beta_h,
                beta_r=beta_r,
                G=G,
                mu_g=mu_g,
                p_g=p_g,
                action_space_dict=action_space_dict,
                robot_agent_ids=[robot_id_iql],
                human_agent_ids=[human_id_iql],
                eta=eta,
                epsilon_h=epsilon_h,
                debug=args.debug_level == 'verbose'
            )
        else:
            # Standard IQL algorithm
            E = args.episodes
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
                human_agent_ids=[human_id_iql],
                debug=args.debug_level == 'verbose'
            )

        # Train the agent
        if args.algorithm == 'timescale':
            print(f"Starting Two-Phase Timescale IQL training...{' with rendering' if args.render else ''}{' and debug level: ' + args.debug_level if args.debug_prints else ''}")
        else:
            print(f"Starting Standard IQL training for {args.episodes} episodes...{' with rendering' if args.render else ''}{' and debug level: ' + args.debug_level if args.debug_prints else ''}")
        
        if args.render:
            # For timescale algorithm, use different training approach
            if args.algorithm == 'timescale':
                iql_agent.train(environment=env, 
                              phase1_episodes=args.phase1_episodes, 
                              phase2_episodes=args.phase2_episodes, 
                              render=args.render, 
                              render_delay=args.delay)
                print("Two-Phase Timescale IQL training finished (rendered mode).")
            else:
                # Keep existing rendered training for standard algorithm
                E = args.episodes
                # Track rewards for minimal debug reporting
                episode_reward_totals = []
                human_rewards_last_episodes = []
                robot_rewards_last_episodes = []
                
                for episode in range(E):
                    if args.debug_prints and args.debug_level != 'minimal':
                        print(f"\n--- Episode {episode + 1}/{E} ---")
                    
                    # Set current episode in environment for debug output
                    env.set_current_episode(episode + 1)
                    
                    obs_dict = env.reset()
                    # Initialize potential for human agent
                    env.render()
                    
                    episode_done = False
                    total_episode_rewards = {agent_id: 0 for agent_id in env.possible_agents}
                    step_count = 0

                    while not episode_done:
                        actions_dict = iql_agent.choose_actions_for_training(obs_dict) # Assuming this method exists for training
                        
                        next_obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(actions_dict)
                        
                        # Update Q-values and get the robot's calculated internal reward
                        r_robot_internal = iql_agent.update_q_values( 
                            obs_dict, actions_dict, rewards_dict, next_obs_dict, terminations_dict, truncations_dict
                        )
                        
                        # Update the environment's reward dictionary with the robot's internal reward for logging
                        if args.debug_prints and args.debug_level in ['standard', 'verbose']:
                            if args.debug_level == 'verbose':
                                print(f"DEBUG_MAIN: IQL returned robot reward: {r_robot_internal}")
                                print(f"DEBUG_MAIN: Human agent IDs: {env.human_agent_ids}")
                                print(f"DEBUG_MAIN: Robot agent IDs: {env.robot_agent_ids}")
                            env.update_robot_reward_for_logging(robot_id_iql, r_robot_internal)
                            # Now print the debug info with the updated robot reward
                            env.print_debug_info()
                        elif args.debug_prints and args.debug_level == 'minimal':
                            # For minimal level, only update robot reward but don't print step info
                            env.update_robot_reward_for_logging(robot_id_iql, r_robot_internal)
                            
                        obs_dict = next_obs_dict
                        env.render()  # This will now show the updated robot reward in debug prints
                        pygame.time.delay(args.delay)
                        
                        for agent_id in env.possible_agents:
                            total_episode_rewards[agent_id] += rewards_dict.get(agent_id, 0)

                        step_count += 1
                        if any(terminations_dict.values()) or any(truncations_dict.values()):
                            episode_done = True
                    
                    # Track total episode rewards
                    human_rewards_last_episodes.append(total_episode_rewards.get(human_id_iql, 0))
                    robot_rewards_last_episodes.append(total_episode_rewards.get(robot_id_iql, 0))
                    
                    if args.debug_prints:
                        if args.debug_level == 'minimal':
                            # Print total average rewards every 10 episodes for minimal level
                            if (episode + 1) % 10 == 0 or episode + 1 == E:
                                episodes_to_avg = min(len(human_rewards_last_episodes), 10)
                                avg_human_reward = sum(human_rewards_last_episodes[-episodes_to_avg:]) / episodes_to_avg
                                avg_robot_reward = sum(robot_rewards_last_episodes[-episodes_to_avg:]) / episodes_to_avg
                                print(f"🔍 EPISODE {episode + 1}/{E}: Last {episodes_to_avg} episodes total avg - Human: {avg_human_reward:.2f}, Robot: {avg_robot_reward:.2f}")
                        else:
                            print(f"Episode {episode + 1} finished after {step_count} steps. Total rewards: {total_episode_rewards}")
                            
                            # Also print averages for standard/verbose every 100 episodes
                            if (episode + 1) % 100 == 0:
                                episodes_to_avg = min(len(human_rewards_last_episodes), 100)
                                avg_human_reward = sum(human_rewards_last_episodes[-episodes_to_avg:]) / episodes_to_avg
                                avg_robot_reward = sum(robot_rewards_last_episodes[-episodes_to_avg:]) / episodes_to_avg
                                print(f"🔍 AVERAGE REWARDS (last {episodes_to_avg} episodes): Human: {avg_human_reward:.2f}, Robot: {avg_robot_reward:.2f}")
                                
                                # Add Q-value convergence monitoring for standard IQL
                                if hasattr(iql_agent, 'take_q_value_snapshot'):
                                    if not hasattr(iql_agent, 'last_q_snapshot_standard'):
                                        iql_agent.last_q_snapshot_standard = iql_agent.take_q_value_snapshot()
                                    else:
                                        current_snapshot = iql_agent.take_q_value_snapshot()
                                        changes = iql_agent.calculate_q_value_changes(iql_agent.last_q_snapshot_standard, current_snapshot)
                                        if args.debug_level == 'verbose':
                                            iql_agent.log_q_value_changes(episode + 1, "STANDARD", changes)
                                        iql_agent.last_q_snapshot_standard = current_snapshot
                    
                    # Add convergence monitoring methods to standard IQL if needed
                    if hasattr(iql_agent, 'end_of_episode_updates'):
                        iql_agent.end_of_episode_updates()
                print("Standard IQL training finished (rendered mode).")
        else:
            if args.algorithm == 'timescale':
                iql_agent.train(environment=env, 
                              phase1_episodes=args.phase1_episodes, 
                              phase2_episodes=args.phase2_episodes, 
                              render=args.render, 
                              render_delay=args.delay)
                print("Two-Phase Timescale IQL training finished.")
            else:
                iql_agent.train(environment=env, num_episodes=args.episodes, render=args.render, render_delay=args.delay)
                print("Standard IQL training finished.")
        
        # Save the trained Q-values
        print(f"Saving trained Q-values to {args.save}")
        if args.algorithm == 'timescale':
            iql_agent.save_models(filepath=args.save)
        else:
            iql_agent.save_q_values(filepath=args.save)
        
        # Optionally visualize the trained agent right after training
        print("Training complete. To visualize the trained agent, run:")
        print(f"python {sys.argv[0]} --mode visualize --load {args.save} --map {args.map}")
        
        env.close()

if __name__ == "__main__":
    main()
