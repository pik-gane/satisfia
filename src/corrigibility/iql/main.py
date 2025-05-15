# Tool code to modify main.py
from env import LockingDoorEnvironment, Actions # MODIFIED: Import Actions
from iql_algorithm import TwoTimescaleIQL
from deterministic_algorithm import DeterministicAlgorithm
import numpy as np # ADDED: Import numpy
import matplotlib.pyplot as plt
import pygame

def main():
    deterministic = False # MODIFIED: Set to True for testing deterministic path
    # MODIFIED: Instantiate the AECEnv-based LockingDoorEnvironment
    env = LockingDoorEnvironment()
    # env.render_mode = "human" # MODIFIED: Set render_mode conditionally
    
    if deterministic:
        env.render_mode = "human" # Enable rendering for deterministic visualization
        algo = DeterministicAlgorithm()
        # MODIFIED: Standard AECEnv reset and agent iteration
        env.reset() 
        # print(f"Initial agent: {env.agent_selection}")

        for agent_id in env.agent_iter():
            observation, reward, terminated, truncated, info = env.last()

            action_to_take = None
            if terminated or truncated:
                # If agent is done, PettingZoo expects a None action or it will be handled by env.step
                action_to_take = None # Or Actions.no_op if your env requires a valid action index
                # print(f"Main (Deterministic): Agent {agent_id} is done. Action: {action_to_take}")
            else:
                # Agent needs to act. Observation is already available from env.last()
                action_to_take = algo.choose_action(observation, agent_id) 
                # print(f"Main (Deterministic): Agent {agent_id}'s turn. Chosen action by algo: {action_to_take}")
            
            env.step(action_to_take)
            # env.render() # Render is called within env.step() if render_mode is human
            
            # print(f"Main (Deterministic): Agent {agent_id} took action. Next agent: {env.agent_selection}")
            # print(f"Robot@: {env.agent_pos}, Human@: {env.human_pos}, R_rew: {env.rewards.get(env.robot_id_str,0)}, H_rew: {env.rewards.get(env.human_id_str,0)}")
            # print(f"Terminations: {env.terminations}, Truncations: {env.truncations}")

            # Check if all agents are done to break the loop (optional, agent_iter handles this)
            # if not env.agents: # env.agents becomes empty when all are terminated/truncated
            #     print("Main (Deterministic): All agents are done. Ending episode.")
            #     break
            pygame.time.delay(100) # Slow down for visualization
        
        print("Deterministic run finished.")
        env.close()
    else:
        # Setup and run the TwoTimescaleIQL algorithm with the AECEnv environment
        print("Setting up IQL algorithm...")
        env.render_mode = None # MODIFIED: Disable rendering for IQL training for speed
        env.reset() # ADDED: Call reset() before accessing env.goal_pos for IQL setup

        # Agent IDs for IQL (consistent with env.possible_agents)
        robot_id_iql = env.robot_id_str # e.g., "robot_0"
        human_id_iql = env.human_id_str # e.g., "human_0"

        alpha_h = 0.1
        alpha_r = 0.01
        gamma_h = 0.99
        gamma_r = 0.99
        beta_h = 5.0
        epsilon_r = 1.0
        
        # Goals: G should be a list of goals. Using env.goal_pos as the single goal.
        # env.goal_pos is a tuple (x,y), which is hashable and suitable for Q_h keys.
        if not hasattr(env, 'goal_pos'):
            print("Error: Environment instance does not have 'goal_pos' attribute.")
            return
        G = [env.goal_pos] 
        mu_g = np.array([1.0]) # Prior probability for the single goal

        p_g = 0.01 # Probability of goal change per step
        E = 1000

        # Action spaces from the AECEnv for the specific agent IDs
        robot_action_space = list(range(env.action_space(robot_id_iql).n))
        human_action_space = list(range(env.action_space(human_id_iql).n))
        action_space_dict = {
            robot_id_iql: robot_action_space,
            human_id_iql: human_action_space
        }

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
            robot_agent_id = robot_id_iql, # ADDED: Pass agent IDs to IQL
            human_agent_id = human_id_iql,  # ADDED: Pass agent IDs to IQL
            debug=True
        )

        print(f"Starting IQL training for {E} episodes...")
        # Pass the AECEnv environment directly
        iql_agent.train(environment=env, num_episodes=E) # MODIFIED: Pass num_episodes
        print("IQL training finished.")
        
        env.close()

if __name__ == "__main__":
    main()
