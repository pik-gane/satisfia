import argparse
import os
import sys

import numpy as np
import pygame
from deterministic_algorithm import DeterministicAlgorithm
from env import Actions, GridEnvironment
from envs.map_loader import DEFAULT_MAP, list_available_maps
from iql_timescale_algorithm import TwoPhaseTimescaleIQL
from state_encoder import encode_full_observable_state
from trained_agent import TrainedAgent


def visualize_q_values_as_map(trained_agent, env, map_name):
    """Print a text-based visualization of the learned Q-values and policies. Also checks if the agent can reach the goal in a test run."""
    print(f"\n{'='*80}")
    print(f"Q-VALUE MAP VISUALIZATION FOR: {map_name}")
    print(f"{'='*80}")

    from envs.map_loader import load_map

    try:
        map_layout, map_metadata = load_map(map_name)
        print(f"Map: {map_metadata.get('name', map_name)}")
        print(f"Description: {map_metadata.get('description', 'No description')}")
    except:
        map_layout = None
        print(f"Could not load map layout for {map_name}")

    print("\nAlgorithm Parameters:")
    iql = trained_agent.iql
    is_network_based = getattr(iql, "network", False)
    learning_mode = "Neural Network" if is_network_based else "Tabular"
    print(f"  Learning Mode: {learning_mode}")
    if hasattr(iql, "beta_r_0"):
        print(f"  Final β_r: {iql.beta_r_0} (robot rationality)")
    if hasattr(iql, "epsilon_h_0"):
        print(f"  Final ε_h: {iql.epsilon_h_0} (human exploration)")
    print(f"  Goals: {iql.G}")
    print(f"  Goal weights: {iql.mu_g}")
    env.reset()
    action_names = ["LEFT", "RIGHT", "UP", "DOWN", "PICK", "DROP", "TOGGLE", "NOOP"]
    direction_symbols = ["←", "→", "↑", "↓", "P", "D", "T", "○"]
    if is_network_based:
        print("\n🧠 NEURAL NETWORK Q-VALUE VISUALIZATION:")
        print(
            "Network-based agents use neural networks to compute Q-values dynamically."
        )
        print("Showing sample Q-values for key positions...")
        visualize_network_q_values(iql, env, action_names, direction_symbols)
        print("\n📊 Q-VALUE STATISTICS:")
        print("Network-based agents don't store explicit Q-tables.")
        print("Q-values are computed dynamically by neural networks.")
        print("Statistics would require sampling across the entire state space.")
    else:
        print("\n📊 TABULAR Q-VALUE VISUALIZATION:")
        print("Shows the robot's preferred action at each position")
        print(
            "Symbols: ← → ↑ ↓ = movement, P=pickup, D=drop, T=toggle, ○=noop, ?=unknown"
        )
        visualize_tabular_q_values(iql, action_names, direction_symbols)
    print(f"\n{'='*80}")
    print("End of Q-value visualization")
    print(f"{'='*80}\n")

    # --- GOAL REACHABILITY CHECKER ---
    print("\n[Checker] Testing if agent can reach the goal in a rollout...")
    env.render_mode = None
    obs = env.reset()
    max_steps = 100
    reached_goal = False
    total_reward = 0
    robot_internal_rewards = []
    iql = trained_agent.iql
    robot_id = iql.robot_agent_ids[0] if hasattr(iql, "robot_agent_ids") else "robot_0"
    for step in range(max_steps):
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = trained_agent.choose_action(agent_obs, agent_id)
        obs, rewards, terminations, truncations, infos = env.step(actions)
        # Compute robot internal reward (power metric) for this step
        # Use the IQL's reward calculation method if available
        robot_state = None
        if hasattr(iql, "state_to_tuple"):
            robot_state = iql.state_to_tuple(obs.get(robot_id, {}))
        next_states = {robot_id: robot_state}
        # For multi-human, collect all next states
        if hasattr(iql, "human_agent_ids"):
            for hid in iql.human_agent_ids:
                next_states[hid] = iql.state_to_tuple(obs.get(hid, {}))
        # Use current goals if available
        current_goals = getattr(iql, "G", [])
        # Use the correct reward calculation method
        if hasattr(iql, "calculate_robot_reward_new"):
            robot_internal_reward = iql.calculate_robot_reward_new(
                next_states, current_goals, any(terminations.values())
            )
        else:
            robot_internal_reward = 0.0
        robot_internal_rewards.append(robot_internal_reward)
        print(
            f"[Checker] Step {step+1}: Robot internal reward (power metric): {robot_internal_reward:.4f}"
        )
        total_reward += sum(rewards.values())
        if any(terminations.values()):
            reached_goal = True
            print(
                f"[Checker] Agent reached a goal at step {step+1}. Total reward: {total_reward:.2f}"
            )
            break
    if not reached_goal:
        print(
            f"[Checker] Agent did NOT reach a goal in 100 steps. Total reward: {total_reward:.2f}"
        )
        print(
            "[Checker] Consider increasing reward for goal achievement or shaping rewards for progress."
        )
    else:
        print("[Checker] Success: Agent can reach the goal in this map.")
    print(
        f"[Checker] Robot internal rewards (first 10 steps): {robot_internal_rewards[:10]}"
    )


def visualize_network_q_values(iql, env, action_names, direction_symbols):
    """Visualize Q-values for network-based agents by sampling key positions."""
    print("\n🤖 ROBOT NETWORK Q-VALUES (Sample Positions):")

    robot_id = iql.robot_agent_ids[0]

    # Sample some key positions from the environment
    sample_positions = [(1, 1), (1, 2), (2, 1), (2, 2), (0, 0)]

    for pos in sample_positions:
        try:
            # Build a full observable state vector for the network
            state_tuple = (pos[0], pos[1], 1, 1, 0, 0)  # Basic state (legacy)
            state_vec = encode_full_observable_state(env, state_tuple)
            # Get Q-values from the network (pass state_vec instead of tuple)
            q_values = iql.robot_q_backend.get_q_values(robot_id, state_vec)

            # Find best action
            allowed_actions = iql.action_space_dict.get(
                robot_id, list(range(len(q_values)))
            )
            if allowed_actions:
                best_action_idx = allowed_actions[
                    np.argmax([q_values[a] for a in allowed_actions])
                ]
                best_action_name = (
                    action_names[best_action_idx]
                    if best_action_idx < len(action_names)
                    else f"A{best_action_idx}"
                )
                best_q_value = q_values[best_action_idx]

                print(
                    f"  Position {pos}: Best={best_action_name} (Q={best_q_value:.3f})"
                )
                # Fix f-string nesting issue
                action_list = [
                    f"{action_names[i] if i < len(action_names) else f'A{i}'}:{q_values[i]:.2f}"
                    for i in allowed_actions[:4]
                ]
                print(f"    All Q-values: {action_list}")
        except Exception as e:
            print(f"  Position {pos}: Could not compute Q-values ({e})")

    print("\n👤 HUMAN NETWORK Q-VALUES (Sample State-Goal Pairs):")

    human_id = iql.human_agent_ids[0]
    goal = iql.G[0] if iql.G else (2, 2)
    goal_tuple = iql.state_to_tuple(goal)

    for pos in sample_positions[:3]:  # Just show a few for humans
        try:
            state_tuple = (1, 1, pos[0], pos[1], 0, 0)  # Human at pos (legacy)
            state_vec = encode_full_observable_state(env, state_tuple)
            # Get Q-values from the network
            q_values = iql.human_q_m_backend.get_q_values(
                human_id, state_vec, goal_tuple
            )

            # Find best action
            allowed_actions = iql.action_space_dict.get(
                human_id, list(range(len(q_values)))
            )
            if allowed_actions:
                best_action_idx = allowed_actions[
                    np.argmax([q_values[a] for a in allowed_actions])
                ]
                best_action_name = (
                    action_names[best_action_idx]
                    if best_action_idx < len(action_names)
                    else f"A{best_action_idx}"
                )
                best_q_value = q_values[best_action_idx]

                print(
                    f"  Human@{pos} Goal={goal}: Best={best_action_name} (Q={best_q_value:.3f})"
                )
        except Exception as e:
            print(f"  Human@{pos}: Could not compute Q-values ({e})")


def visualize_tabular_q_values(iql, action_names, direction_symbols):
    """Visualize Q-values for tabular agents using the existing logic."""
    print("\n🤖 ROBOT POLICY VISUALIZATION:")

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
        print("\nRobot Policy Map:")
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
                    best_value = float("-inf")

                    # Look for states that match this position
                    for state_tuple, q_values in robot_q_table.items():
                        if (
                            len(state_tuple) >= 2
                            and state_tuple[0] == x
                            and state_tuple[1] == y
                        ):
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

    print("\n👤 HUMAN POLICY VISUALIZATION:")
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

                print(
                    f"\nHuman visited positions for goal {goal}: {len(human_positions)}"
                )
                print(f"Grid bounds: x=[{min_x},{max_x}], y=[{min_y},{max_y}]")

                # Create human policy map
                print(f"\nHuman Policy Map (Goal: {goal}):")
                print("   ", end="")
                for x in range(min_x, max_x + 1):
                    print(f"{x:2}", end=" ")
                print()

                for y in range(min_y, max_y + 1):
                    print(f"{y:2} ", end="")
                    for x in range(min_x, max_x + 1):
                        if (x, y) in human_positions:
                            # Find best action for this position and goal
                            best_action = None
                            best_value = float("-inf")

                            for state_goal_tuple, q_values in human_q_table.items():
                                state_tuple, goal_part = state_goal_tuple
                                if (
                                    goal_part == goal_tuple
                                    and len(state_tuple) >= 2
                                    and state_tuple[0] == x
                                    and state_tuple[1] == y
                                ):
                                    action_idx = np.argmax(q_values)
                                    value = q_values[action_idx]
                                    if value > best_value:
                                        best_value = value
                                        best_action = action_idx

                            if best_action is not None and best_action < len(
                                direction_symbols
                            ):
                                print(f" {direction_symbols[best_action]}", end=" ")
                            else:
                                print(" ?", end=" ")
                        else:
                            print(" .", end=" ")
                    print()

    # Show Q-value statistics
    print("\n📊 Q-VALUE STATISTICS:")

    # Robot Q-value stats
    robot_q_values = []
    for q_values in robot_q_table.values():
        robot_q_values.extend(q_values)

    if robot_q_values:
        print("Robot Q-values:")
        print(f"  Range: [{np.min(robot_q_values):.3f}, {np.max(robot_q_values):.3f}]")
        print(f"  Mean: {np.mean(robot_q_values):.3f}")
        print(f"  Std: {np.std(robot_q_values):.3f}")
        print(f"  States learned: {len(robot_q_table)}")

    # Human Q-value stats
    human_q_values = []
    for q_values in human_q_table.values():
        human_q_values.extend(q_values)

    if human_q_values:
        print("Human Q-values:")
        print(f"  Range: [{np.min(human_q_values):.3f}, {np.max(human_q_values):.3f}]")
        print(f"  Mean: {np.mean(human_q_values):.3f}")
        print(f"  Std: {np.std(human_q_values):.3f}")
        print(f"  State-goal pairs learned: {len(human_q_table)}")

    # Show some example Q-values
    print("\n🔍 SAMPLE Q-VALUES:")

    # Show a few robot Q-values
    print("Robot Q-values (first 5 states):")
    for i, (state, q_vals) in enumerate(list(robot_q_table.items())[:5]):
        action_q_pairs = [
            (action_names[j] if j < len(action_names) else f"A{j}", q_vals[j])
            for j in range(len(q_vals))
        ]
        best_action_idx = np.argmax(q_vals)
        print(
            f"  State {state}: Best={action_names[best_action_idx] if best_action_idx < len(action_names) else f'A{best_action_idx}'}({q_vals[best_action_idx]:.3f})"
        )
        print(
            f"    All: {', '.join([f'{name}:{val:.2f}' for name, val in action_q_pairs[:4]])}..."
        )

    # Show a few human Q-values
    print("Human Q-values (first 5 state-goal pairs):")
    for i, (state_goal, q_vals) in enumerate(list(human_q_table.items())[:5]):
        if len(q_vals) > 0:
            allowed_actions = iql.action_space_dict.get(
                human_id, list(range(len(q_vals)))
            )
            action_q_pairs = [
                (action_names[j] if j < len(action_names) else f"A{j}", q_vals[j])
                for j in allowed_actions
            ]
            best_action_idx = allowed_actions[
                np.argmax([q_vals[j] for j in allowed_actions])
            ]
            print(
                f"  State-Goal {state_goal}: Best={action_names[best_action_idx] if best_action_idx < len(action_names) else f'A{best_action_idx}'}({q_vals[best_action_idx]:.3f})"
            )


def main():
    print("[DEBUG] Entered main() function.")
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="IQL in custom gridworld environment")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "visualize", "test"],
        default="train",
        help="Mode: train (train the model), visualize (run trained model), test (run deterministic test)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["timescale", "standard"],
        default="timescale",
        help="Algorithm to use: timescale (two-phase timescale IQL), standard (original IQL)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="q_values.pkl",
        help="Path to save trained Q-values (default: q_values.pkl)",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="q_values.pkl",
        help="Path to load trained Q-values (default: q_values.pkl)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes for training (default: 1000)",
    )
    parser.add_argument(
        "--phase1-episodes",
        type=int,
        default=500,
        help="Number of episodes for Phase 1 (timescale algorithm only, default: 500)",
    )
    parser.add_argument(
        "--phase2-episodes",
        type=int,
        default=500,
        help="Number of episodes for Phase 2 (timescale algorithm only, default: 500)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="Delay in milliseconds between steps during visualization (default: 100)",
    )
    parser.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP,
        help=f'Map to use (default: {DEFAULT_MAP}). Available maps: {", ".join(list_available_maps())}',
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Grid size for the environment (default: derived from map)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during training"
    )
    parser.add_argument(
        "--debug_prints",
        action="store_true",
        help="Enable detailed debug prints during training (if rendering)",
    )
    parser.add_argument(
        "--debug_level",
        type=str,
        choices=["minimal", "standard", "verbose"],
        default="standard",
        help="Level of debug output: minimal (goal reached only), standard (step info), verbose (full IQL details)",
    )
    parser.add_argument(
        "--reward-function",
        type=str,
        choices=["power", "log", "bounded", "generalized_bounded"],
        default="power",
        help="Robot reward function: power (default), log, bounded, or generalized_bounded",
    )
    parser.add_argument(
        "--concavity-param",
        type=float,
        default=1.0,
        help="Concavity parameter c for generalized_bounded function (default: 1.0)",
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Use neural network Q-learning instead of tabular (default: False for tabular mode)",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=4,
        help="State vector dimension for neural network mode (default: 4)",
    )
    args = parser.parse_args()

    # Create environment with specified map
    # Only enable env debug mode for standard and verbose levels
    env_debug_mode = args.debug_prints and args.debug_level in ["standard", "verbose"]
    env = GridEnvironment(
        map_name=args.map,
        grid_size=args.grid_size,
        debug_mode=env_debug_mode,
        debug_level=args.debug_level,
    )

    # Enable minimal debug for goal notifications if any debug level is active
    if args.debug_prints:
        env.set_minimal_debug(True)

    if args.mode == "test":
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

    elif args.mode == "visualize":
        # Visualize trained model
        if not os.path.exists(args.load):
            print(f"Error: Q-values file not found at {args.load}")
            print(
                f"Please train a model first using: python {sys.argv[0]} --mode train --save {args.load}"
            )
            return

        print(f"Loading trained agent from {args.load} for map: {args.map}")
        trained_agent = TrainedAgent(q_values_path=args.load)

        # Print Q-value map visualization
        visualize_q_values_as_map(trained_agent, env, args.map)

        # Ask user if they want to see the pygame visualization too
        try:
            response = "y"
            if response in ["y", "yes"]:
                print("Running pygame visualization...")
                env.render_mode = "human"  # Enable rendering for visualization

                obs = env.reset()
                env.render()
                done = False
                step_count = 0
                while (
                    not done and step_count < 100
                ):  # Limit steps to prevent infinite loops
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        actions[agent_id] = trained_agent.choose_action(
                            agent_obs, agent_id
                        )
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
        # --- Variable initialization for training section ---
        gamma_h = 0.99
        gamma_r = 0.99
        eta = 0.1
        p_g = 0.1
        robot_id_iql = "robot_0"
        human_agent_ids_iql = ["human_0"]
        robot_action_space = list(Actions)
        human_action_space = list(Actions)
        action_space_dict = {
            robot_id_iql: robot_action_space,
            "human_0": human_action_space,
        }
        # Define goals and goal prior (can be customized per map)
        G = [(0, 0), (2, 2)]  # Example goals; replace with map-specific if needed
        mu_g = np.array([0.5, 0.5])

        # --- Hyperparameter tuning section (single loop, after all variables initialized) ---
        hyperparam_grid = [
            # (alpha_m, alpha_e, alpha_r, beta_r_0, epsilon_h_0, episodes)
            (0.1, 0.1, 0.1, 5.0, 0.1, 500),
            # (0.2, 0.2, 0.1, 5.0, 0.1, 500),
            # (0.1, 0.1, 0.1, 10.0, 0.05, 700),
            # (0.05, 0.05, 0.05, 7.0, 0.05, 1000),
            # (0.15, 0.15, 0.1, 8.0, 0.01, 800),
        ]
        best_success_rate = 0
        best_config = None
        best_save_path = None
        for idx, (
            alpha_m,
            alpha_e,
            alpha_r,
            beta_r_0,
            epsilon_h_0,
            episodes,
        ) in enumerate(hyperparam_grid):
            print(
                f"\n[Hyperparam Tuning] Config {idx+1}/{len(hyperparam_grid)}: alpha_m={alpha_m}, alpha_e={alpha_e}, alpha_r={alpha_r}, beta_r_0={beta_r_0}, epsilon_h_0={epsilon_h_0}, episodes={episodes}"
            )
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
                epsilon_r=1,
                reward_function=args.reward_function,
                concavity_param=args.concavity_param,
                network=args.network,
                state_dim=args.state_dim,
                debug=False,
            )
            # --- Phase 1: Learn conservative model for each human ---
            iql_agent.train_phase1(
                environment=env,
                phase1_episodes=episodes // 2,
                render=False,
                render_delay=args.delay,
            )
            # Save to a temp file for this config
            save_path = f"tune_config_{idx+1}.pkl"
            iql_agent.save_models(filepath=save_path)
            # Evaluate: run checker to see if agent reaches goal
            trained_agent = TrainedAgent(q_values_path=save_path)
            # Use the checker from visualize_q_values_as_map
            print("[Hyperparam Tuning] Running goal-reaching checker...")
            env.reset()
            obs = env.reset()
            max_steps = 100
            reached_goal = False
            # Only use agent IDs that exist in trained_agent and env
            valid_agent_ids = [
                aid for aid in obs.keys() if aid in trained_agent.agent_ids
            ]
            for step in range(max_steps):
                actions = {}
                for agent_id in valid_agent_ids:
                    actions[agent_id] = trained_agent.choose_action(
                        obs[agent_id], agent_id
                    )
                obs, rewards, terminations, truncations, infos = env.step(actions)
                if any(terminations.get(aid, False) for aid in valid_agent_ids):
                    reached_goal = True
                    print(f"[Checker] Agent reached a goal at step {step+1}.")
                    break
            if reached_goal:
                print(
                    f"[Hyperparam Tuning] Config {idx+1} SUCCESS: Agent reached the goal."
                )
                if best_success_rate < 1:
                    best_success_rate = 1
                    best_config = (
                        alpha_m,
                        alpha_e,
                        alpha_r,
                        beta_r_0,
                        epsilon_h_0,
                        episodes,
                    )
                    best_save_path = save_path
                    break  # Stop at first success for speed
            else:
                print(
                    f"[Hyperparam Tuning] Config {idx+1} FAIL: Agent did NOT reach the goal."
                )
        if best_success_rate == 1:
            print(
                f"\n[Hyperparam Tuning] Best config: alpha_m={best_config[0]}, alpha_e={best_config[1]}, alpha_r={best_config[2]}, beta_r_0={best_config[3]}, epsilon_h_0={best_config[4]}, episodes={best_config[5]}"
            )
            print(f"Saving best trained Q-values to {args.save}")
            import shutil

            shutil.move(best_save_path, args.save)
        else:
            print(
                "[Hyperparam Tuning] No config reliably reached the goal. Consider expanding the grid or increasing episodes."
            )
        print("Training complete.")
        # Always print the visualization command for both agent types
        print("\n[INFO] To visualize the trained agent, run:")
        if args.network:
            print(
                f"python {sys.argv[0]} --mode visualize --load {args.save} --map {args.map} --network"
            )
            print(
                "[INFO] For neural network agents, the policy is stored in the network weights (e.g., .pt files), not a Q-table. Ensure the .pt files are present in the same directory as the .pkl file."
            )
            print("[INFO] To visualize a tabular agent, rerun without --network.")
        else:
            print(
                f"python {sys.argv[0]} --mode visualize --load {args.save} --map {args.map}"
            )
            print("[INFO] For tabular agents, the Q-table is stored in the .pkl file.")
            print("[INFO] To visualize a neural network agent, rerun with --network.")

        # --- Post-training: verify both tabular and neural network agents if possible ---
        print(
            "\n[Verification] Checking that both Q-learning (tabular) and model-based (neural network) agents can be trained and reach the goal..."
        )
        if not args.network:
            print("[Verification] Tabular Q-learning agent check:")
            trained_agent = TrainedAgent(q_values_path=args.save)
            visualize_q_values_as_map(trained_agent, env, args.map)
        else:
            print("[Verification] Neural network agent check:")
            trained_agent = TrainedAgent(q_values_path=args.save)
            visualize_q_values_as_map(trained_agent, env, args.map)
        print("[Verification] RL training and checker completed for selected mode.")
        if args.network:
            print("[INFO] To verify tabular Q-learning, rerun with --network omitted.")
        else:
            print("[INFO] To verify neural network Q-learning, rerun with --network.")

        env.close()


if __name__ == "__main__":
    print("[DEBUG] __name__ == '__main__', starting main()...")
    main()
