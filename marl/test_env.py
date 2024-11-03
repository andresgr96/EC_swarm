import numpy as np
from hebbian_env import HebbianEnv  # Make sure this is the correct import path for HebbianEnv

# Initialize the environment
env = HebbianEnv(n_envs=1, headless=False)  # Set headless to False if you want to render

# Reset the environment to get the initial observations
obs = env.reset()

# Number of steps to run
num_steps = 100

print("Starting simulation...")

try:
    for step in range(num_steps):
        # Example actions: using random actions for demonstration
        actions = [[np.random.rand(9) for _ in range(len(env.robot_handles_list[0]))]]
        
        # Step the environment
        obs, rewards, dones, infos = env.step(actions)
        
        # Print rewards at each step
        print(f"Step {step + 1}: Rewards = {rewards}")
finally:
    # Ensure the environment is properly closed
    env.close()

print("Simulation complete.")
