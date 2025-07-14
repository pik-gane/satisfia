"""
Test script for curriculum learning: robot generalization across increasingly difficult auto-generated environments.

- Uses auto_env.generate_env_map to create deterministic, progressively harder environments.
- Each environment requires the human to pass through a door (with a key) to reach the goal.
- The robot's weights are reused and updated across all phases/environments.
- Difficulty increases only when the robot performs well in the current environment.
- Logs performance and environment details at each stage.

Assumes TwoTimescaleIQL or similar agent interface and environment API compatible with your codebase.
"""

import os
import sys

import numpy as np


import concurrent.futures

from env import Actions, GridEnvironment
from envs.auto_env import generate_env_map
from iql_timescale_algorithm import TwoPhaseTimescaleIQL

import wandb

# Placeholder for agent/environment imports


def make_door_barrier_env(size, seed, num_doors=1, num_keys=1):
    """Generate a deterministic env where human must pass through a door to reach goal."""
    rows, cols = size
    config = {
        "width": cols,
        "height": rows,
        "num_humans": 1,
        "num_robots": 1,
        "num_doors": num_doors,
        "num_keys": num_keys,
        "num_boxes": 0,
        "num_goals": 1,
        "lava_prob": 0.0,
        "seed": seed,
        "max_steps": 60,
    }
    grid, meta = generate_env_map(config)
    # Place a vertical wall with a door between human and goal
    # Human at (row_h, col_h), goal at (row_g, col_g)
    hpos = meta["human_starts"][0]
    gpos = meta["goal_positions"][0]
    barrier_col = (hpos[1] + gpos[1]) // 2
    for r in range(1, rows - 1):
        if grid[r][barrier_col] == "  ":
            grid[r][barrier_col] = "##"
    # Place a door in the barrier
    door_row = (hpos[0] + gpos[0]) // 2
    grid[door_row][barrier_col] = "YD"
    # Place a key on the human's side
    for c in range(1, barrier_col):
        if grid[hpos[0]][c] == "  ":
            grid[hpos[0]][c] = "YK"
            break
    return grid, meta


def evaluate_robot(env, agent, episodes=10):
    """Run episodes and return average reward and success rate."""
    total_reward = 0
    successes = 0
    robot_id = agent.robot_agent_ids[0]
    human_id = agent.human_agent_ids[0]
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                state_tuple = agent.state_to_tuple(agent_obs)
                if agent_id == robot_id:
                    actions[agent_id] = agent.sample_robot_action_phase2(
                        agent_id, state_tuple
                    )
                else:
                    # Use the first goal for evaluation
                    goal_tuple = agent.state_to_tuple(agent.G[0])
                    actions[agent_id] = agent.sample_human_action_phase2(
                        agent_id, state_tuple, goal_tuple
                    )
            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += sum(rewards.values())
            if any(terminations.values()):
                done = True
        total_reward += episode_reward
        if any(terminations.values()):
            successes += 1
    avg_reward = total_reward / episodes
    success_rate = successes / episodes
    return avg_reward, success_rate


def parallel_train_and_eval(
    env_args, agent, phase1_episodes, phase2_episodes, eval_episodes
):
    """Train and evaluate in a single environment instance (for parallel execution)."""
    size, num_doors, num_keys, stage = env_args
    grid, meta = make_door_barrier_env(
        size, seed=stage, num_doors=num_doors, num_keys=num_keys
    )
    grid_size = max(size)
    env = GridEnvironment(map_name="commit_map", grid_size=grid_size, debug_mode=False)
    env.map_layout = grid
    env.map_metadata = meta
    if hasattr(env, "reset_map_from_layout_and_metadata"):
        env.reset_map_from_layout_and_metadata()
    robot_id = "robot_0"
    human_id = "human_0"
    action_space_dict = {robot_id: list(Actions), human_id: list(Actions)}
    G = meta["goal_positions"]
    mu_g = np.ones(len(G)) / len(G)
    # Update agent's goals and priors for this env
    agent.G = G
    agent.mu_g = mu_g
    # PHASE 1: Human model learning
    agent.train_phase1(env, phase1_episodes=phase1_episodes)
    # PHASE 2: Robot learning
    agent.train_phase2(env, phase2_episodes=phase2_episodes)
    # Evaluate
    avg_reward, success_rate = evaluate_robot(env, agent, episodes=eval_episodes)
    return {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "size": size,
        "num_doors": num_doors,
        "num_keys": num_keys,
        "stage": stage + 1,
    }


def main():
    wandb.init(
        project="robot-generalization-curriculum",
        name="network-curriculum-parallel",
        config={"network": True},
    )
    # Curriculum: list of (size, num_doors, num_keys, stage)
    curriculum = []
    for stage in range(25):
        size = (7 + 2 * stage, 7 + 2 * stage)
        num_doors = min(1 + stage // 2, size[0] // 2)
        num_keys = num_doors
        curriculum.append((size, num_doors, num_keys, stage))
    # Create a single shared agent for all environments (weights reused)
    robot_agent = None
    robot_id = "robot_0"
    human_id = "human_0"
    action_space_dict = {robot_id: list(Actions), human_id: list(Actions)}
    # Use initial dummy G/mu_g, will be updated per env
    G = [(0, 0)]
    mu_g = np.ones(len(G)) / len(G)
    robot_agent = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.1,
        alpha_r=0.1,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=G,
        mu_g=mu_g,
        p_g=0.1,
        action_space_dict=action_space_dict,
        robot_agent_ids=[robot_id],
        human_agent_ids=[human_id],
        network=True,
        state_dim=4,
        debug=False,
    )
    phase1_episodes = 400
    phase2_episodes = 400
    eval_episodes = 20
    max_retries = 5
    for stage, env_args in enumerate(curriculum):
        retries = 0
        while retries < max_retries:
            # Run N parallel environments (N=4 for example)
            N = 4
            with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
                futures = [
                    executor.submit(
                        parallel_train_and_eval,
                        env_args,
                        robot_agent,
                        phase1_episodes,
                        phase2_episodes,
                        eval_episodes,
                    )
                    for _ in range(N)
                ]
                results = [f.result() for f in futures]
            # Aggregate results
            avg_success = np.mean([r["success_rate"] for r in results])
            avg_reward = np.mean([r["avg_reward"] for r in results])
            print(
                f"[Parallel Stage {stage+1}] Avg Success: {avg_success:.2f}, Avg Reward: {avg_reward:.2f} (retry {retries+1}/{max_retries})"
            )
            wandb.log(
                {
                    "stage": stage + 1,
                    "size": env_args[0][0],
                    "num_doors": env_args[1],
                    "num_keys": env_args[2],
                    "avg_reward": avg_reward,
                    "success_rate": avg_success,
                    "retry": retries + 1,
                }
            )
            if avg_success >= 0.8:
                print("Robot succeeded. Proceeding to harder environment.")
                break
            else:
                print(
                    "Robot not ready for next stage. Continuing training in this environment."
                )
                retries += 1
        if avg_success < 0.8:
            print(
                "Robot failed to generalize after maximum retries. Stopping curriculum."
            )
            break
    print("Curriculum finished.")
    wandb.finish()


if __name__ == "__main__":
    main()
