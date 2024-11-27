import copy
import re
from typing import List
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
import numpy as np
from numpy.random import default_rng

from utils.Fitnesses import FitnessCalculator
from utils.Individual import Individual, thymio_genotype
from utils.Sensors import Sensors
from utils.Simulate_swarm_population import EnvSettings
import torch

class HebbianEnv(gym.Env):
    def __init__(self, n_envs=1, n_individuals=10, headless=True):
        super(HebbianEnv, self).__init__()
        self.env_settings = EnvSettings
        self.n_envs = n_envs
        self.n_individuals = n_individuals
        self.life_timeout = 600
        self.headless = headless
        self.individuals = self.create_individuals()  
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.env_list = []
        self.robot_handles_list = []
        self.controller_list = []
        self.fitness_list = []
        self.fitness_list2 = []
        self.sensor_list = []
        self.sensor_list2 = []
        self.current_rewards = [[] for _ in range(n_envs)]
        self.current_light_values = [[] for _ in range(n_envs)]
        self.last_light_values = [[] for _ in range(n_envs)]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.initialize_simulator()

    def calc_vel_targets(self, actions):
        """
        Vectorized conversion of the actions into velocity commands suitable for the robots.
        :param actions: List of actions, where each action is a [linear_velocity, angular_velocity] pair
        :return: List of transformed actions as [n_l, n_r] pairs
        """
        actions_array = np.array(actions)
        linear_velocities = actions_array[:, :, 0]
        angular_velocities = actions_array[:, :, 1]

        n_l = ((linear_velocities + 0.025) - (angular_velocities / 2) * 0.085) / 0.021
        n_r = ((linear_velocities + 0.025) + (angular_velocities / 2) * 0.085) / 0.021
        transformed_actions = np.stack((n_l, n_r), axis=-1)

        return transformed_actions


    def create_individuals(self):
        individuals = []
        for _ in range(self.n_envs):
            # Each environment gets its own list of 20 individuals
            env_individuals = [Individual(thymio_genotype("hNN", 9, 2), i) for i in range(self.n_individuals)]
            individuals.append(env_individuals)
        return individuals

    
    def get_pos_and_headings(self, env, robot_handles):
        """
        Get the positions and headings of all robots in the specified environment.

        :param env: The Isaac Gym environment instance.
        :param robot_handles: List of robot handles in the environment.
        :return: headings, positions_x, positions_y
        """
        num_robots = len(robot_handles)
        headings = np.zeros((num_robots,))
        positions_x = np.zeros_like(headings)
        positions_y = np.zeros_like(headings)

        for i in range(num_robots):
            body_pose = self.gym.get_actor_rigid_body_states(env, robot_handles[i], gymapi.STATE_POS)["pose"][0]
            body_angle_mat = np.array(body_pose[1].tolist())

            try:
                r = R.from_quat(body_angle_mat)
                if r.magnitude() < 1e-6:
                    print("Zero-magnitude quaternion detected, setting to default.")
                    body_angle_mat = np.array([0.0, 0.0, 0.0, 1.0])
                    r = R.from_quat(body_angle_mat)
                headings[i] = r.as_euler('zyx')[0]  # yaw
            except ValueError:
                print("Invalid quaternion encountered; defaulting to identity rotation.")

                # Set up initial root states and DOF states after loading assets
                actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
                root_states = gymtorch.wrap_tensor(actor_root_state)
                print(root_states[3:7])




                print(body_angle_mat)
                body_angle_mat = np.array([0.0, 0.0, 0.0, 1.0])  # Default valid quaternion
                r = R.from_quat(body_angle_mat)
                headings[i] = r.as_euler('zyx')[0]  # yaw

            positions_x[i] = body_pose[0][0]
            positions_y[i] = body_pose[0][1]

        return headings, positions_x, positions_y




    def initialize_simulator(self):
        self.env_list.clear()
        self.robot_handles_list.clear()
        self.controller_list.clear()
        self.fitness_list.clear()
        self.fitness_list2.clear()
        self.sensor_list.clear()
        self.sensor_list2.clear()
        self.current_rewards = [[] for _ in range(self.n_envs)]
        self.current_light_values = [[] for _ in range(self.n_envs)]

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

        # Increase GPU memory allocation?
        sim_params.physx.max_gpu_contact_pairs = 2_000_000 
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
            initial_positions = np.zeros((2, num_robots))
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

            # Add Fitness and Sensor objects
            self.fitness_list.append(FitnessCalculator(individual, initial_positions, 10, arena=arena,
                                                       objectives=self.env_settings['objectives']))
            self.fitness_list2.append(FitnessCalculator(individual, initial_positions, 10, arena="circle_corner_30x30",
                                                        objectives=self.env_settings['objectives']))
            self.sensor_list.append(Sensors([controller.controller_type for controller in controllers], arena=arena))
            self.sensor_list2.append(Sensors([controller.controller_type for controller in controllers], arena="circle_corner_30x30"))

            print(f"Initialized {num_robots} robots in environment {i_env}")

        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("*** Failed to create viewer")

        # Set up initial root states and DOF states after loading assets
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # Wrap tensors for easier manipulation
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # print(self.root_states)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        # print(self.dof_states)

        # Clone initial states
        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        # Zero out velocities for initial states
        self.initial_root_states[:, 7:13] = 0  # zero linear and angular velocity in initial state
        self.initial_dof_states[:, 1] = 0  # zero out dof velocity
            

    def reset(self):
        rng = np.random.default_rng()
        num_robots = self.n_individuals
        arena_length = int(re.findall(r'\d+', self.env_settings['arena_type'])[-1])
        init_area = arena_length / 10
        arena_center = arena_length / 2

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        for i_env, env in enumerate(self.env_list):
            if self.env_settings['random_start']:
                init_failure = True

                # Generate initial positions until they are valid
                while init_failure:
                    ixs = init_area * (2 * rng.random(num_robots) - 1) + arena_center
                    iys = init_area * (2 * rng.random(num_robots) - 1) + arena_center
                    x_diff = np.subtract.outer(ixs, ixs)
                    y_diff = np.subtract.outer(iys, iys)
                    distances = np.hypot(x_diff, y_diff)
                    init_failure = np.any(distances[np.triu_indices(num_robots, k=1)] < 0.2)

                # Generate random yaw angles for each robot
                ihs = rng.uniform(-np.pi, np.pi, size=(num_robots,))
            else:
                distance = 0.5
                rows = int(np.ceil(np.sqrt(num_robots)))
                ixs = [(i % rows) * distance - (rows - 1) * distance / 2 for i in range(num_robots)]
                iys = [(i // rows) * distance - (rows - 1) * distance / 2 for i in range(num_robots)]
                ihs = [0.0] * num_robots  # All robots face the same direction in the grid

            # Apply perturbations to root states for each robot in this environment
            for i, handle in enumerate(self.robot_handles_list[i_env]):
                # Set up initial position and orientation
                position = torch.tensor([ixs[i], iys[i], 0.033], device=self.initial_root_states.device)
                yaw = ihs[i]
                random_quat = R.from_euler('zyx', [yaw, 0.0, 0.0]).as_quat()
                quaternion = torch.tensor(random_quat, device=self.initial_root_states.device)

                # Update initial_root_states with position and orientation
                self.initial_root_states[handle, 0:3] = position
                self.initial_root_states[handle, 3:7] = quaternion
                self.initial_root_states[handle, 7:13] = 0  # zero linear and angular velocity

            # Convert the actor indices for Isaac Gym compatibility
            env_actor_handles = torch.tensor(self.robot_handles_list[i_env], dtype=torch.int32, device=self.initial_root_states.device)
            env_actor_handles_gym = gymtorch.unwrap_tensor(env_actor_handles)

            # Apply initial state changes
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.initial_root_states),
                env_actor_handles_gym,
                len(self.robot_handles_list[i_env])
            )

            # Apply initial DOF states as well
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.initial_dof_states),
                env_actor_handles_gym,
                len(self.robot_handles_list[i_env])
            )

        # Simulate one step to apply reset states
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # Refresh and return new observations
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        return self.get_obs()

    def step(self, actions):
        np_actions = np.array(actions)
        # print(f"Actions Shape: {np_actions.shape}")
        transformed_actions = self.calc_vel_targets(actions)
        # np_trans = np.array(transformed_actions)
        # print(f"Trans Shape: {np_trans.shape}")

        for i_env, env in enumerate(self.env_list):
            env_commands = transformed_actions[i_env]
            for i_robot, velocity_command in enumerate(env_commands):
                self.gym.set_actor_dof_velocity_targets(
                    env, self.robot_handles_list[i_env][i_robot], velocity_command.astype(np.float32)
                )

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)

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
        obs = []

        for i_env, env in enumerate(self.env_list):

            robot_handles = self.robot_handles_list[i_env]
            headings, positions_x, positions_y = self.get_pos_and_headings(env, robot_handles)
            positions = np.array([positions_x, positions_y])

            if self.life_timeout >= 300 and self.env_settings['env_perturb']:
                self.sensor_list2[i_env].calculate_states(positions, headings)
                states = self.sensor_list2[i_env].get_current_state()
            else:
                self.sensor_list[i_env].calculate_states(positions, headings)
                states = self.sensor_list[i_env].get_current_state()

            for state in states:
                state[-1] = state[-1] / 255.0 

            obs.append(states)

        return obs


    def calculate_rewards(self):
        rewards = []

        for i_env in range(self.n_envs):
            robot_handles = self.robot_handles_list[i_env]
            headings, positions_x, positions_y = self.get_pos_and_headings(self.env_list[i_env], robot_handles)
            positions = np.array([positions_x, positions_y])

            if self.life_timeout >= 300 and self.env_settings['env_perturb']:
                self.sensor_list2[i_env].calculate_states(positions, headings)
                states = self.sensor_list2[i_env].get_current_state()
            else:
                self.sensor_list[i_env].calculate_states(positions, headings)
                states = self.sensor_list[i_env].get_current_state()

            grad_sensor_outputs = np.array([state[-1] for state in states])  
            normalized_rewards = grad_sensor_outputs / 255.0  

            env_rewards = []
            for i, light_value in enumerate(grad_sensor_outputs):  
                if self.last_light_values[i_env] and light_value > self.last_light_values[i_env][i]:
                    env_rewards.append(normalized_rewards[i])  
                else:
                    env_rewards.append(0.0)

            # Update light values and rewards
            self.current_light_values[i_env] = grad_sensor_outputs.tolist()  
            self.last_light_values[i_env] = grad_sensor_outputs.tolist()  
            self.current_rewards[i_env] = env_rewards
            rewards.append(env_rewards)

        return rewards

    def check_termination(self):
        # Always false since dont have termination conditions other than time
        dones = []
        for i_env in range(self.n_envs):
            dones.append([False] * len(self.robot_handles_list[i_env]))
        return dones

    def collect_infos(self):
        infos = []

        for i_env in range(self.n_envs):
            mean_reward = np.mean(self.current_rewards[i_env]) if self.current_rewards[i_env] else 0.0
            std_reward = np.std(self.current_rewards[i_env]) if self.current_rewards[i_env] else 0.0
            mean_light = np.mean(self.current_light_values[i_env]) if self.current_light_values[i_env] else 0.0
            std_light = np.std(self.current_light_values[i_env]) if self.current_light_values[i_env] else 0.0

            infos.append({
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_light": mean_light,
                "std_light": std_light
            })

        return infos

    def close(self):
        print(f"Environment Killed")
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None

        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None