import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import src.config as config
import src.config as config
from src.real_robot_api import RealRobotAPI
import trimesh
import pandas as pd
from scipy.spatial import Delaunay
from tests.test_blob_sampling import generate_surface, sample_point_in_hull, is_inside_hull, load_point_cloud_from_csv


class RobotEnv(gym.Env):
    """
    A real robot environment.
    Observation: concatenated 3D coordinates of the tip wrt to the base (shape: 3,)
    Action: a 3D discrete vector (each element is an integer from 0 to config.max_stroke, inclusive)
    Reward: negative Euclidean distance from the tip to the goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobotEnv, self).__init__()

        # Load the point cloud from a CSV file and get the mesh 
        print("Loading point cloud...")
        file_path = r"data/exp_2025-04-02_11-43-59/output_exp_2025-04-02_11-43-59.csv"  # Change this to your actual CSV file path
        point_cloud = load_point_cloud_from_csv(file_path)
        self.alpha_shape, self.convex_hull = generate_surface(point_cloud, alpha=1.0)
        
        # Action space: 4 discrete actions (config.steps) for each motor and elongation
        self.action_space = spaces.MultiDiscrete([config.steps] * 4)
        self.action_mapping = {
            # For each dimension, map 0-9 to your desired values
            0: np.linspace(0, config.max_stroke, config.steps),  # First motor
            1: np.linspace(0, config.max_stroke, config.steps),  # Second motor
            2: np.linspace(0, config.max_stroke, config.steps),  # Third motor
            3: np.linspace(0, config.max_stroke, config.steps),  # Elongation (all motors)
        }
        
        # Observation space: 3D coordinates of the tip wrt to the base (shape: 3,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.tip = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if config.pick_random_goal:
            self.goal = self.pick_goal()
        else:
            self.goal = config.rl_goal
        self.max_steps = 100
        self.current_step = 0
        self.robot_api = RealRobotAPI()

    def get_goal(self):
        return self.goal

    def pick_goal(self):
        """
        Pick a random goal within the workspace.
        """
        return sample_point_in_hull(self.convex_hull, num_samples=1)[0]

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        # Map discrete actions to continuous values
        mapped_action = np.array([
            self.action_mapping[0][action[0]],
            self.action_mapping[1][action[1]],
            self.action_mapping[2][action[2]],
            self.action_mapping[3][action[3]]
        ])
        
        # Use mapped_action instead of action
        elongation = mapped_action[3]
        
        # Create a new command vector with the first 3 actions adjusted by elongation
        command = np.zeros(3, dtype=np.float32)
        for i in range(3):
            # Combine each action with elongation, ensuring it stays within bounds
            combined_action = min(mapped_action[i] + elongation, config.max_stroke)
            command[i] = max(0, combined_action)  # Ensure non-negative
        
        # Send the modified command to the robot
        self.robot_api.send_command(command)
        self.tip = self.robot_api.get_current_tip()
        distance = np.linalg.norm(self.tip - self.goal)
        terminated = False
        self.current_step += 1
        reward = 0

        ## Old way
        # distance_threshold = 1.9
        
        # if distance > distance_threshold:
        #     # terminated = True
        #     # reward = 0
        #     reward -= 1  # Negative reward for being too far from the goal
        # bonus = 0

        # # Check 3d points alignment with the goal
        # # Compute perpendicular distance from goal to the line defined by the origin and the tip.
        # dist_to_line = np.linalg.norm(np.cross(self.tip, self.goal)) / np.linalg.norm(self.tip)

        # # If the goal lies nearly on the line, add a bonus reward.
        # if dist_to_line < 0.7 and np.linalg.norm(self.goal) > np.linalg.norm(self.tip):
        #     bonus += 5
        #     if distance < 0.5:
        #         bonus += 10
        #         if distance < 0.2:
        #             bonus += 20
        #             if distance < 0.1:
        #                 bonus += 50

        # if not terminated:
        #     # reward = 1/(distance*(self.current_step**2)) + bonus
        #     reward = 1/(0.5*distance+0.0001) + bonus

        ## New way
        if distance < 0.65:
            reward += 1
            self.goal = self.pick_goal()
            print(f"Setting new goal at {self.goal}")

        print(f"Step {self.current_step}: Distance {distance} ")
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {"step": self.current_step, "distance": distance}


        return self.tip, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)
        self.robot_api.reset_robot()
        self.tip = self.robot_api.get_current_tip()
        self.current_step = 0

        if config.pick_random_goal:
            # Change goal with 20% probability.
            if np.random.random() < 0.4:
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")
        return self.tip, {}

    def render(self, mode='human'):
        pass

