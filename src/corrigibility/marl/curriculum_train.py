import wandb

from .curriculum_envs import (
    env_01_simple,
    env_02_basic_door,
    env_03_multi_key,
    env_04_obstacles,
    env_05_larger_grid,
)
from .env import CustomEnvironment
from .iql_timescale_algorithm import TwoPhaseTimescaleIQL


# --- Evaluation function ---
def evaluate_agent(agent, env, eval_episodes=20, max_steps=200):
    successes = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            actions = {}
            for rid in agent.robot_agent_ids:
                state = agent.state_to_tuple(env.observe(rid))
                actions[rid] = agent.sample_robot_action_phase2(rid, state)
            for hid in agent.human_agent_ids:
                state = agent.state_to_tuple(env.observe(hid))
                # Use converged human policy; set goal as needed
                if hasattr(env, "goal_for_agent"):
                    goal = env.goal_for_agent(hid)
                else:
                    goal = (0, 0)
                actions[hid] = agent.sample_human_action_phase2(hid, state, goal)
            _, rewards, terms, truncs, _ = env.step(actions)
            done = any(terms.values()) or any(truncs.values())
            steps += 1
        # Check if the human reached their goal
        if any(
            r > 200
            for agent_id, r in rewards.items()
            if agent_id in agent.human_agent_ids
        ):
            successes += 1
    return successes / eval_episodes


# --- Curriculum training function ---
def curriculum_train(
    agent,
    curriculum,
    phase1_episodes=500,
    phase2_episodes=1000,
    max_retries=3,
    log_wandb=True,
    project="robot-curriculum",
):
    if log_wandb:
        wandb.init(project=project, config={"curriculum": len(curriculum)})

    for idx, env_config_loader in enumerate(curriculum):
        env_conf = env_config_loader.get_env_config()
        print(
            f"\n=== Curriculum Stage {idx+1}/{len(curriculum)}: {env_conf['description']} ==="
        )

        # Use the hardcoded map from the environment config
        grid_layout, grid_metadata = env_config_loader.get_map()

        env = CustomEnvironment(grid_layout=grid_layout, grid_metadata=grid_metadata)
        success_threshold = env_conf.get("training_config", {}).get(
            "success_threshold", 0.9
        )

        for attempt in range(max_retries):
            print(f"  Attempt {attempt+1}/{max_retries}")

            # Use training parameters from the environment config
            phase1_episodes = env_conf.get("training_config", {}).get(
                "phase1_episodes", 500
            )
            phase2_episodes = env_conf.get("training_config", {}).get(
                "phase2_episodes", 1000
            )

            agent.train_phase1(env, phase1_episodes)
            agent.train_phase2(env, phase2_episodes)
            success_rate = evaluate_agent(agent, env)
            print(f"    Success rate: {success_rate:.2f}")
            if log_wandb:
                wandb.log(
                    {
                        "stage": idx + 1,
                        "attempt": attempt + 1,
                        "success_rate": success_rate,
                    }
                )
            if success_rate >= success_threshold:
                print("    Success threshold reached, moving to next stage.")
                break
            elif attempt == max_retries - 1:
                print("    Max retries reached, flagging scenario.")

        # Save checkpoint after each stage
        agent.save_models(f"q_values_stage_{idx+1}.pkl")

    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    # Define curriculum by loading environment configs
    curriculum = [
        env_01_simple,
        env_02_basic_door,
        env_03_multi_key,
        env_04_obstacles,
        env_05_larger_grid,
    ]

    # Initialize agent ONCE
    agent = TwoPhaseTimescaleIQL(
        alpha_m=0.1,
        alpha_e=0.2,
        alpha_r=0.01,
        gamma_h=0.99,
        gamma_r=0.99,
        beta_r_0=5.0,
        G=[(4, 4)],
        mu_g=[1.0],
        p_g=0.0,
        action_space_dict={"robot_0": [0, 1, 2, 3], "human_0": [0, 1, 2, 3]},
        robot_agent_ids=["robot_0"],
        human_agent_ids=["human_0"],
        network=True,
        state_dim=4,
    )
    curriculum_train(agent, curriculum)
