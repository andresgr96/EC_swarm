import os
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import psutil 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from hebbian_env import HebbianEnv

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--n-envs", type=int, default=30, help="number of parallel environments")
    parser.add_argument("--n-individuals", type=int, default=20, help="number of unique agents per environment")
    parser.add_argument("--max-episode-len", type=int, default=1000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=1800, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=30, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the MLP")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="experiment", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./results/", help="directory to save model")
    parser.add_argument("--save-rate", type=int, default=18, help="save model every this many episodes")
    parser.add_argument("--load-dir", type=str, default="./results/", help="directory to load model")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.compat.v1.layers.dense(out, num_units, activation=tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, num_units, activation=tf.nn.relu)
        out = tf.compat.v1.layers.dense(out, num_outputs, activation=tf.nn.tanh)
        return out

def make_env(n_envs, n_individuals, headless=True):
    return HebbianEnv(n_envs=n_envs, n_individuals=n_individuals, headless=headless)

def get_trainers(env, n_individuals, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(n_individuals):
        trainers.append(trainer(
            "agent_%d" % i, model, [env.observation_space.shape] * n_individuals,
            [env.action_space] * n_individuals, i, arglist, local_q_func=False
        ))
    return trainers

def train(arglist):
    script_dir = os.path.dirname(os.path.abspath(__file__))  
    base_save_dir = os.path.join(script_dir, arglist.save_dir)
    exp_save_dir = os.path.join(base_save_dir, arglist.exp_name)
    tracking_int = 1
    unique_save_dir = exp_save_dir

    # Increment the tracking integer until we find a unique directory name
    while os.path.exists(unique_save_dir):
        unique_save_dir = f"{exp_save_dir}_{tracking_int}"
        tracking_int += 1
    
    arglist.save_dir = unique_save_dir

    if not arglist.restore:
            os.makedirs(arglist.save_dir, exist_ok=True)


    with U.single_threaded_session():
        env = make_env(arglist.n_envs, arglist.n_individuals, headless=not arglist.display)
        trainers = get_trainers(env, arglist.n_individuals, arglist)
        print("Using MADDPG with", arglist.n_individuals, "unique agents")
        U.initialize()

        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.restore:
            print("Loading previous state...")
            U.load_state(arglist.load_dir)

        episodes_per_generation = int(arglist.num_episodes / 100)
        episode_rewards = [0.0]
        agent_rewards = [[0.0] for _ in range(arglist.n_individuals)]
        saver = tf.compat.v1.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        current_episode = 1
        current_generation = 1
        episode_start_time = time.time()
        losses = []
        episode_light_values = []

        # Create a unique log directory
        log_base_dir = os.path.join(script_dir, "logs", arglist.exp_name)
        log_unique_dir = log_base_dir
        log_tracking_int = 1

        while os.path.exists(log_unique_dir):
            log_unique_dir = f"{log_base_dir}_{log_tracking_int}"
            log_tracking_int += 1

        os.makedirs(log_unique_dir, exist_ok=True)

        loss_file_path = os.path.join(log_unique_dir, "mean_loss_values.txt")
        fitness_file_path = os.path.join(log_unique_dir, "mean_fitness_values.txt")

        with open(loss_file_path, "a") as loss_file, open(fitness_file_path, "a") as fitness_file:
            print("Starting training iterations...")
            while True:
                all_actions = []
                for env_idx in range(arglist.n_envs):
                    observations = obs_n[env_idx]
                    action_n = [agent.action(observations[i]) for i, agent in enumerate(trainers)]
                    all_actions.append(action_n)

                new_obs_n, rew_n, done_n, info_n = env.step(all_actions)

                episode_step += 1

                if episode_step % 250 == 0:
                    mid_episode_duration = time.time() - episode_start_time
                    mid_episode_duration_minutes = mid_episode_duration / 60
                    print(f"Generation: {current_generation}, Episode: {current_episode}/{arglist.num_episodes}, Ep Step: {episode_step}")
                    print(f"Time elapsed since episode start: {mid_episode_duration_minutes:.2f}")

                done = all([all(agent_done) for agent_done in done_n])
                terminal = (episode_step >= arglist.max_episode_len)

                for env_idx in range(arglist.n_envs):
                    for agent_idx, agent in enumerate(trainers):
                        agent.experience(
                            obs_n[env_idx][agent_idx], 
                            all_actions[env_idx][agent_idx], 
                            rew_n[env_idx][agent_idx], 
                            new_obs_n[env_idx][agent_idx], 
                            done_n[env_idx][agent_idx], 
                            terminal
                        )
                obs_n = new_obs_n

                # Accumulate rewards for logging
                for env_idx in range(arglist.n_envs):
                    for i, rew in enumerate(rew_n[env_idx]):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew

                # Track mean light intensity during the last episode of each generation
                if current_episode % episodes_per_generation == 0:
                    episode_light_values.extend([info["mean_light"] for info in info_n])

                if done or terminal:
                    episode_duration = time.time() - episode_start_time
                    episode_duration_minutes = episode_duration / 60
                    episode_start_time = time.time()
                    print(f"Episode {current_episode} done. Episode time: {episode_duration_minutes:.2f} minutes")

                    obs_n = env.reset()
                    episode_step = 0
                    current_episode += 1
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)

                    if current_episode % episodes_per_generation == 1:  
                        mean_light_intensity = np.mean(episode_light_values)
                        fitness_file.write(f"{mean_light_intensity}\n")
                        fitness_file.flush()  
                        print(f"Generation {current_generation} mean light intensity: {mean_light_intensity}")
                        episode_light_values.clear()  
                        current_generation += 1

                train_step += 1

                if train_step % 100 == 0:
                    q_losses, p_losses, mean_target_qs, mean_rewards, mean_target_q_nexts, std_target_qs = [], [], [], [], [], []
                    for agent in trainers:
                        agent.preupdate()
                    for agent in trainers:
                        loss = agent.update(trainers, train_step)
                        if loss is not None:
                            q_loss, p_loss, mean_target_q, mean_rew, mean_target_q_next, std_target_q = loss
                            q_losses.append(q_loss)
                            p_losses.append(p_loss)
                            mean_target_qs.append(mean_target_q)
                            mean_rewards.append(mean_rew)
                            mean_target_q_nexts.append(mean_target_q_next)
                            std_target_qs.append(std_target_q)

                    if q_losses:  
                        # Calculate the mean for each loss component
                        mean_q_loss = np.mean(q_losses)
                        mean_p_loss = np.mean(p_losses)
                        mean_target_q = np.mean(mean_target_qs)
                        mean_reward = np.mean(mean_rewards)
                        mean_target_q_next = np.mean(mean_target_q_nexts)
                        std_target_q = np.mean(std_target_qs)

                        loss_output = (
                            f"q_loss: {mean_q_loss}, p_loss: {mean_p_loss}, "
                            f"mean_target_q: {mean_target_q}, mean_reward: {mean_reward}, "
                            f"mean_target_q_next: {mean_target_q_next}, std_target_q: {std_target_q}\n"
                        )
                        loss_file.write(loss_output)
                        loss_file.flush()  # Ensure it's written immediately

                # Save model
                if terminal and (current_episode % arglist.save_rate == 0):
                    episode_save_dir = os.path.join(arglist.save_dir, f"episode_{current_episode}")
                    os.makedirs(episode_save_dir, exist_ok=True)  # Create directory if it doesn't exist
                    U.save_state(episode_save_dir, saver=saver)

                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[:-1]),
                        round(time.time() - episode_start_time, 3)
                    ))
                    print(f"Model saved for episode {current_episode} in {episode_save_dir}")


                # Stop if the number of episodes is reached
                if len(episode_rewards) > arglist.num_episodes:
                    print("...Finished total of {} episodes.".format(len(episode_rewards)))
                    break



if __name__ == "__main__":
    arglist = parse_args()
    train(arglist)
