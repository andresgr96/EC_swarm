import numpy as np
from hebbian_env_test import HebbianEnv

# Initialize the environment
env = HebbianEnv(n_envs=5, n_individuals=10,  headless=True)  # Set headless to False if you want to render

# Reset the environment to get the initial observations
obs = env.reset()

# Number of steps to run
num_steps = 1000000

print("Starting simulation...")
i = 0
try:
    for step in range(num_steps):

        # Create actions for each environment and robot
        # Each action should be a list of two elements: [velocity_left, velocity_right]
        actions = [
            [np.random.random(2) for _ in range(len(env.robot_handles_list[i_env]))]
            for i_env in range(env.n_envs)
        ]
        # print("Number of environments:", env.n_envs)
        # print("Robot handles list shape:", [len(handles) for handles in env.robot_handles_list])
        # print("Actions shape:", len(actions), [len(action) for action in actions])
        
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        # print(obs[0][-1])
        i+=1

        if  i % 600 == 0:
            env.reset()
        # env.render()
        
        # Print rewards at each step
        # print(f"Step {step + 1}: Rew = {rewards}")
        # print(f"Step {step + 1}: Inf = {infos}")
        # print(f"Hnadle Lenght: {len(env.robot_handles_list[0])}")
finally:
    # Ensure the environment is properly closed
    env.close()

print("Simulation complete.")
