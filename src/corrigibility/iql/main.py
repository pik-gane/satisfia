# Tool code to modify main.py
from env import LockingDoorEnvironment
from iql_algorithm import run_iql_algorithm
from deterministic_algorithm import DeterministicAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import pygame

def main():
    deterministic = True
    env = LockingDoorEnvironment()
    
    if deterministic:
        algo = DeterministicAlgorithm()
        env.reset() # Reset to get initial state for the first agent
        
        # episode_steps = 0 # Optional: limit steps per episode

        # The env.agent_iter() loop will run as long as there are active agents
        # and the environment has not ended the episode.
        for agent_id in env.agent_iter(): # Corrected: use env.agent_iter()
            # Get the observation, reward, termination, truncation, and info for the current agent
            observation, reward, terminated, truncated, info = env.last()

            action_to_take = None
            if terminated or truncated:
                # If the agent is terminated or truncated, pass None action.
                # env.step(None) is the standard way to handle this for a done agent.
                action_to_take = None
                # print(f"Main: Agent {agent_id} is done. Action: None.")
            else:
                # Agent needs to act
                if agent_id == "robot":
                    # Construct state tuple for the deterministic algorithm
                    # This state should reflect the current environment state for the robot's decision
                    state_tuple_for_algo = (
                        env.agent_pos, 
                        env.human_pos, 
                        env.door_pos, 
                        env.goal_pos,
                        env.robot_has_key, 
                        env.door_is_open,
                        env.door_is_key_locked,
                        env.door_is_permanently_locked
                    )
                    action_to_take = algo.choose_action(state_tuple_for_algo)
                    print(f"Main: Robot's turn ({agent_id}). Chosen action: {action_to_take}")
                elif agent_id == "human":
                    # Human action is determined internally by env._get_human_action()
                    # when env.step() is called for the human agent with action=None.
                    action_to_take = None 
                    print(f"Main: Human's turn ({agent_id}). Action: None (determined in env.step).")
            
            env.step(action_to_take) # Step the environment with the action for agent_id
                                     # env.step() will process the action and advance to the next agent internally.
            env.render()
            pygame.time.delay(300) # Slow down for visualization

            # episode_steps +=1 # Optional step counting
            # if episode_steps > 300: # Example of external max step limit
            #     print("Max episode steps reached by external counter.")
            #     # To end the episode here, you would typically need to set all agents' 
            #     # termination/truncation flags in the environment and then break,
            #     # or rely on the environment's own max step handling.
            #     break 

        print("Episode finished.")
        env.close()
        # pygame.quit() # Pygame is quit within env.close() if screen was initialized
    else:
        # run_iql_algorithm() # This would need similar PettingZoo integration
        print("IQL algorithm execution would start here (not implemented for this loop structure yet).")

if __name__ == "__main__":
    main()
