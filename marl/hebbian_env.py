import copy
import re
import time
from multiprocessing import shared_memory, Process
from typing import AnyStr, TypedDict, List
from isaacgym import gymapi
from isaacgym import gymutil
from scipy.spatial.transform import Rotation as R
import gym
import numpy as np
from numpy.random import default_rng

from utils.Fitnesses import FitnessCalculator
from utils.Individual import Individual, thymio_genotype
from utils.Sensors import Sensors
from utils.plot_swarm import swarm_plotter
from utils.Simulate_swarm_population import EnvSettings

class HebbianEnv(gym.Env):
    def __init__(self, n_envs=1, headless=True):
        super(HebbianEnv, self).__init__()
        self.env_settings = EnvSettings
        self.n_envs = n_envs
        self.life_timeout = 600
        self.headless = headless
        self.individuals = self.create_individuals()
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.env_list = []
        self.robot_handles_list = []
        self.controller_list = []

        # Initialize Isaac Gym and environments
        self.initialize_simulator()

    def calc_vel_targets(self, controller, states):
        velocity_target = controller.velocity_commands(np.array(states))
        n_l = ((velocity_target[0] + 0.025) - (velocity_target[1] / 2) * 0.085) / 0.021
        n_r = ((velocity_target[0] + 0.025) + (velocity_target[1] / 2) * 0.085) / 0.021
        return [n_l, n_r]

    def create_individuals(self):
        # Create example individuals, replace this with your setup
        return [[Individual(thymio_genotype("hNN", 9, 2), i) for i in range(20)]]

    def initialize_simulator(self):
        # Parse arguments
        args = gymutil.parse_arguments(description="Loading and testing")

        # Configure simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.1
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = True

        # Create the simulator
        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
        )
        if self.sim is None:
            raise RuntimeError("*** Failed to create sim")

        # Add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up
        plane_params.distance = 0
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

        # Asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.flip_visual_attachments = True
        asset_options.armature = 0.0001

        # Set up the environment grid
        num_envs = self.n_envs
        arena = self.env_settings['arena_type']
        spacing = int(re.findall(r'\d+', arena)[-1])
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # Initialize environment and robot handles
        self.env_list = []
        self.robot_handles_list = []
        self.controller_list = []

        print(f"Creating {num_envs} {arena} environments")
        for i_env in range(num_envs):
            individual = self.individuals[i_env]
            controllers = [member.controller for member in copy.deepcopy(individual)]
            self.controller_list.append(controllers)

            # Create environment
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_envs)
            self.env_list.append(env)
            robot_handles = []

            # Initialize robots
            num_robots = len(individual)
            initial_positions = np.zeros((2, num_robots))  # Allocation to save initial positions
            rng = default_rng()

            if self.env_settings['random_start']:
                arena_length = spacing
                init_area = arena_length / 10
                arena_center = arena_length / 2
                r_distance = arena_center - init_area
                init_failure = True

                # Generate initial positions until valid
                while init_failure:
                    ixs = init_area * (2 * rng.random(num_robots) - 1) + arena_center
                    iys = init_area * (2 * rng.random(num_robots) - 1) + arena_center
                    x_diff = np.subtract.outer(ixs, ixs)
                    y_diff = np.subtract.outer(iys, iys)
                    distances = np.hypot(x_diff, y_diff)
                    init_failure = np.any(distances[np.triu_indices(num_robots, k=1)] < 0.2)

                ihs = 2 * np.pi * rng.random(num_robots)

                # Set robot poses
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0, 0, 0.032)
                pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)
                for i in range(num_robots):
                    pose.p = gymapi.Vec3(ixs[i], iys[i], 0.033)
                    initial_positions[:, i] = [pose.p.x, pose.p.y]
                    rotation = R.from_euler('zyx', [ihs[i], 0.0, 0.0]).as_quat()
                    pose.r = gymapi.Quat(rotation[0], rotation[1], rotation[2], rotation[3])

                    robot_asset_file = individual[i].body
                    robot_asset = self.gym.load_asset(self.sim, "./", robot_asset_file, asset_options)
                    robot_handle = self.gym.create_actor(env, robot_asset, pose, f"robot_{i}", 0, 0)
                    robot_handles.append(robot_handle)
            else:
                # Place robots in a grid
                distance = 0.5
                rows = int(np.ceil(np.sqrt(num_robots)))
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0, 0, 0.032)
                pose.r = gymapi.Quat(0, 0.0, 0.0, 0.707107)

                for i in range(num_robots):
                    pose.p = gymapi.Vec3(
                        (i % rows) * distance - (rows - 1) * distance / 2,
                        (i // rows) * distance - (rows - 1) * distance / 2,
                        0.033
                    )
                    initial_positions[:, i] = [pose.p.x, pose.p.y]
                    robot_asset_file = individual[i].body
                    robot_asset = self.gym.load_asset(self.sim, "./", robot_asset_file, asset_options)
                    robot_handle = self.gym.create_actor(env, robot_asset, pose, f"robot_{i}", 0, 0)
                    robot_handles.append(robot_handle)

            self.robot_handles_list.append(robot_handles)

            # Set up properties for robot DOFs
            for handle in robot_handles:
                props = self.gym.get_actor_dof_properties(env, handle)
                props['driveMode'].fill(gymapi.DOF_MODE_VEL)
                props['stiffness'].fill(0.05)
                props['damping'].fill(0.025)
                self.gym.set_actor_dof_properties(env, handle, props)

            print(f"Initialized {num_robots} robots in environment {i_env}")

        # Create viewer if headless mode is off
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("*** Failed to create viewer")


    def reset(self):
        # Close the current simulator to avoid multiple instances, we should instead reposition the robots
        self.close()
        self.initialize_simulator()
        initial_obs = self.get_obs()
        return initial_obs

    def step(self, actions):
        # Apply actions to robots
        for i_env, env in enumerate(self.env_list):
            for i_robot, action in enumerate(actions[i_env]):
                velocity_command = self.calc_vel_targets(self.controller_list[i_env][i_robot], action)
                self.gym.set_actor_dof_velocity_targets(env, self.robot_handles_list[i_env][i_robot], velocity_command)

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Collect new observations and rewards
        obs = self.get_obs()
        rewards = self.calculate_rewards()
        dones = self.check_termination()
        infos = self.collect_infos()

        return obs, rewards, dones, infos

    def render(self, mode='human'):
        if self.viewer is not None:
            self.gym.step_graphics(self.gym)
            self.gym.draw_viewer(self.viewer, self.gym, False)
        else:
            raise RuntimeError("Viewer is not initialized. Set headless=False to use render.")

    def get_obs(self):
        # Collect observations for each agent
        obs = []
        for i_env, env in enumerate(self.env_list):
            env_obs = []
            for i_robot in range(len(self.robot_handles_list[i_env])):
                # Example observation: placeholder values
                env_obs.append([0.0, 0.0, 0.0])
            obs.append(env_obs)
        return obs

    def calculate_rewards(self):
        # Placeholder rewards
        rewards = []
        for i_env in range(self.n_envs):
            rewards.append([0.0] * len(self.robot_handles_list[i_env]))
        return rewards

    def check_termination(self):
        # Placeholder termination status
        dones = []
        for i_env in range(self.n_envs):
            dones.append([False] * len(self.robot_handles_list[i_env]))
        return dones

    def collect_infos(self):
        # Placeholder info
        infos = []
        for i_env in range(self.n_envs):
            infos.append({})
        return infos

    def close(self):
        # Check if the viewer exists and destroy it
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

        # Destroy the simulator
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None