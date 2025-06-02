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

def visualize_q_values_as_map(trained_agent, env, map_name):
    """Print a text-based visualization of the learned Q-values and policies."""
    print(f"\n{'='*80}")
    print(f"Q-VALUE MAP VISUALIZATION FOR: {map_name}")
    print(f"{'='*80}")
    
    # Get the map layout for reference
    from envs.map_loader import load_map
    try:
        map_layout, map_metadata = load_map(map_name)
        print(f"Map: {map_metadata.get('name', map_name)}")
        print(f"Description: {map_metadata.get('description', 'No description')}")
    except:
        map_layout = None
        print(f"Could not load map layout for {map_name}")
    
    print(f"\nAlgorithm Parameters:")
    iql = trained_agent.iql
    if hasattr(iql, 'beta_r_0'):
        print(f"  Final β_r: {iql.beta_r_0} (robot rationality)")
    if hasattr(iql, 'epsilon_h_0'):
        print(f"  Final ε_h: {iql.epsilon_h_0} (human exploration)")
    print(f"  Goals: {iql.G}")
    print(f"  Goal weights: {iql.mu_g}")
    
    # Get environment dimensions
    env.reset()
    
    # Action names for readable output
    action_names = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'PICK', 'DROP', 'TOGGLE', 'NOOP']
    direction_symbols = ['←', '→', '↑', '↓', 'P', 'D', 'T', '○']
    
    print(f"\n🤖 ROBOT POLICY VISUALIZATION:")
    print("Shows the robot's preferred action at each position")
    print("Symbols: ← → ↑ ↓ = movement, P=pickup, D=drop, T=toggle, ○=noop, ?=unknown")
    
    # Analyze robot Q-values
    robot_id = iql.robot_agent_ids[0]
    robot_q_table = iql.Q_r_dict[robot_id]
    
    # Find all unique positions that have been visited
    robot_positions = set()
    for state_tuple in robot_q_table.keys():
        # Assuming state tuple format includes robot position
        if len(state_tuple) >= 2:
            robot_positions.add((state_tuple[0], state_tuple[1]))
    
    if robot_positions:
        min_x = min(pos[0] for pos in robot_positions)
        max_x = max(pos[0] for pos in robot_positions)
        min_y = min(pos[1] for pos in robot_positions)
        max_y = max(pos[1] for pos in robot_positions)
        
        print(f"\nRobot visited positions: {len(robot_positions)}")
        print(f"Grid bounds: x=[{min_x},{max_x}], y=[{min_y},{max_y}]")
        
        # Create robot policy map
        print(f"\nRobot Policy Map:")
        print("   ", end="")
        for x in range(min_x, max_x + 1):
            print(f"{x:2}", end=" ")
        print()
        
        for y in range(min_y, max_y + 1):
            print(f"{y:2} ", end="")
            for x in range(min_x, max_x + 1):
                if (x, y) in robot_positions:
                    # Find best action for this position
                    best_action = None
                    best_value = float('-inf')
                    
                    # Look for states that match this position
                    for state_tuple, q_values in robot_q_table.items():
                        if len(state_tuple) >= 2 and state_tuple[0] == x and state_tuple[1] == y:
                            action_idx = np.argmax(q_values)
                            value = q_values[action_idx]
                            if value > best_value:
                                best_value = value
                                best_action = action_idx
                    
                    if best_action is not None and best_action < len(direction_symbols):
                        print(f" {direction_symbols[best_action]}", end=" ")
                    else:
                        print(" ?", end=" ")
                else:
                    print(" .", end=" ")
            print()
    
    print(f"\n👤 HUMAN POLICY VISUALIZATION:")
    print("Shows the human's preferred action at each position for their goal")
    
    # Analyze human Q-values
    human_id = iql.human_agent_ids[0]
    human_q_table = iql.Q_h_dict[human_id]
    
    # Get the goal the human is trying to reach
    goal = iql.G[0] if iql.G else None
    if goal:
        goal_tuple = iql.state_to_tuple(goal)
        print(f"Goal: {goal} -> {goal_tuple}")
        
        # Find all positions for this goal
        human_positions = set()
        for state_goal_tuple in human_q_table.keys():
            if len(state_goal_tuple) == 2:  # (state_tuple, goal_tuple)
                state_tuple, goal_part = state_goal_tuple
                if goal_part == goal_tuple and len(state_tuple) >= 2:
                    human_positions.add((state_tuple[0], state_tuple[1]))
        
        if human_positions:
            min_x = min(pos[0] for pos in human_positions)
            max_x = max(pos[0] for pos in human_positions)
            min_y = min(pos[1] for pos in human_positions)
            max_y = max(pos[1] for pos in human_positions)
            
            print(f"Human visited positions: {len(human_positions)}")
            print(f"Grid bounds: x=[{min_x},{max_x}], y=[{min_y},{max_y}]")
            
            # Create human policy map
            print(f"\nHuman Policy Map (for goal {goal}):")
            print("   ", end="")
            for x in range(min_x, max_x + 1):
                print(f"{x:2}", end=" ")
            print()
            
            for y in range(min_y, max_y + 1):
                print(f"{y:2} ", end="")
                for x in range(min_x, max_x + 1):
                    if (x, y) in human_positions:
                        # Find best action for this position toward the goal
                        best_action = None
                        best_value = float('-inf')
                        
                        for state_goal_tuple, q_values in human_q_table.items():
                            if len(state_goal_tuple) == 2:
                                state_tuple, goal_part = state_goal_tuple
                                if (goal_part == goal_tuple and 
                                    len(state_tuple) >= 2 and 
                                    state_tuple[0] == x and state_tuple[1] == y):
                                    
                                    # Get allowed actions for human
                                    allowed_actions = iql.action_space_dict.get(human_id, list(range(len(q_values))))
                                    action_values = [q_values[a] for a in allowed_actions]
                                    if action_values:
                                        best_action_idx = np.argmax(action_values)
                                        best_action = allowed_actions[best_action_idx]
                                        best_value = action_values[best_action_idx]
                                        break
                        
                        if best_action is not None and best_action < len(direction_symbols):
                            print(f" {direction_symbols[best_action]}", end=" ")
                        else:
                            print(" ?", end=" ")
                    else:
                        print(" .", end=" ")
                print()
    
    # Show Q-value statistics
    print(f"\n📊 Q-VALUE STATISTICS:")
    
    # Robot Q-value stats
    robot_q_values = []
    for q_values in robot_q_table.values():
        robot_q_values.extend(q_values)
    
    if robot_q_values:
        print(f"Robot Q-values:")
        print(f"  Range: [{np.min(robot_q_values):.3f}, {np.max(robot_q_values):.3f}]")
        print(f"  Mean: {np.mean(robot_q_values):.3f}")
        print(f"  Std: {np.std(robot_q_values):.3f}")
        print(f"  States learned: {len(robot_q_table)}")
    
    # Human Q-value stats
    human_q_values = []
    for q_values in human_q_table.values():
        human_q_values.extend(q_values)
    
    if human_q_values:
        print(f"Human Q-values:")
        print(f"  Range: [{np.min(human_q_values):.3f}, {np.max(human_q_values):.3f}]")
        print(f"  Mean: {np.mean(human_q_values):.3f}")
        print(f"  Std: {np.std(human_q_values):.3f}")
        print(f"  State-goal pairs learned: {len(human_q_table)}")
    
    # Show some example Q-values
    print(f"\n🔍 SAMPLE Q-VALUES:")
    
    # Show a few robot Q-values
    print("Robot Q-values (first 5 states):")
    for i, (state, q_vals) in enumerate(list(robot_q_table.items())[:5]):
        action_q_pairs = [(action_names[j] if j < len(action_names) else f"A{j}", q_vals[j]) 
                         for j in range(len(q_vals))]
        best_action_idx = np.argmax(q_vals)
        print(f"  State {state}: Best={action_names[best_action_idx] if best_action_idx < len(action_names) else f'A{best_action_idx}'}({q_vals[best_action_idx]:.3f})")
        print(f"    All: {', '.join([f'{name}:{val:.2f}' for name, val in action_q_pairs[:4]])}...")
    
    # Show a few human Q-values
    print("Human Q-values (first 5 state-goal pairs):")
    for i, (state_goal, q_vals) in enumerate(list(human_q_table.items())[:5]):
        if len(q_vals) > 0:
            allowed_actions = iql.action_space_dict.get(human_id, list(range(len(q_vals))))
            action_q_pairs = [(action_names[j] if j < len(action_names) else f"A{j}", q_vals[j]) 
                             for j in allowed_actions]
            best_action_idx = allowed_actions[np.argmax([q_vals[j] for j in allowed_actions])]
            print(f"  State-Goal {state_goal}: Best={action_names[best_action_idx] if best_action_idx < len(action_names) else f'A{best_action_idx}'}({q_vals[best_action_idx]:.3f})")
    
    print(f"\n{'='*80}")
    print("End of Q-value visualization")
    print(f"{'='*80}\n")

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
    parser.add_argument('--reward-function', type=str, 
                        choices=['power', 'log', 'bounded', 'generalized_bounded'], 
                        default='power',
                        help='Robot reward function: power (default), log, bounded, or generalized_bounded')
    parser.add_argument('--concavity-param', type=float, default=1.0,
                        help='Concavity parameter c for generalized_bounded function (default: 1.0)')
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

        print(f"Loading trained agent from {args.load} for map: {args.map}")
        trained_agent = TrainedAgent(q_values_path=args.load)
        
        # Print Q-value map visualization
        visualize_q_values_as_map(trained_agent, env, args.map)
        
        # Ask user if they want to see the pygame visualization too
        try:
            response = input("Would you like to see the pygame visualization as well? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print(f"Running pygame visualization...")
                env.render_mode = "human"  # Enable rendering for visualization
                
                obs = env.reset()
                env.render()
                done = False
                step_count = 0
                while not done and step_count < 100:  # Limit steps to prevent infinite loops
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        actions[agent_id] = trained_agent.choose_action(agent_obs, agent_id)
                    obs, rewards, terminations, truncations, infos = env.step(actions)
                    env.render()
                    done = any(terminations.values()) or any(truncations.values())
                    pygame.time.delay(args.delay)  # Slow down for visualization
                    step_count += 1
                
                print("Pygame visualization finished.")
            else:
                print("Skipping pygame visualization.")
        except KeyboardInterrupt:
            print("\nVisualization interrupted by user.")
        except:
            print("Invalid input, skipping pygame visualization.")
        
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

        # Timescale algorithm hyperparameters
        alpha_m = 0.1   # Phase 1 learning rate
        alpha_e = 0.1   # Phase 2 fast timescale learning rate
        alpha_r = 0.1
        beta_r_0 = 5.0  # Target robot softmax temperature for Phase 2
        eta = 0.1       # Power parameter for robot reward
        epsilon_h_0 = 0.1 # Final human epsilon-greedy parameter
        epsilon_r = 1   # Robot exploration in Phase 1
        gamma_h = 0.99
        gamma_r = 0.99

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
            human_agent_ids=[human_id_iql],
            eta=eta,
            epsilon_h_0=epsilon_h_0,
            epsilon_r=epsilon_r,
            reward_function=args.reward_function,  # NEW
            concavity_param=args.concavity_param,  # NEW
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
