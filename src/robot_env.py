import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.config as config
import math
import src.config as config
from src.real_robot_api import RealRobotAPI
import pandas as pd


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
        # Change action space to a discrete multi-dimensional space.
        self.action_space = spaces.MultiDiscrete([config.max_stroke + 1] * 3)
        # Observation space remains the same.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.tip = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if config.pick_random_goal:
            self.goal = self.pick_goal()
        else:
            self.goal = config.rl_goal
        self.max_steps = 200
        self.current_step = 0
        self.previous_distace = 0
        self.robot_api = RealRobotAPI()

    def get_goal(self):
        return self.goal

    def pick_goal(self):
        """
        Pick a random goal within the workspace.
        """
        exp_name = "exp_2025-03-17_15-26-06"
        csv_name = "output_exp_2025-03-17_15-26-06.csv"
        csv_path = os.path.join(config.data_dir, exp_name, csv_name)

        data = pd.read_csv(csv_path)
        tip_columns = ['tip_x', 'tip_y', 'tip_z']
        base_columns = ['base_x', 'base_y', 'base_z']

        base_x_avg = data[base_columns[0]].mean()
        base_y_avg = data[base_columns[1]].mean()
        base_z_avg = data[base_columns[2]].mean()

        tip_min_x = data[tip_columns[0]].min() - base_x_avg
        tip_max_x = data[tip_columns[0]].max() - base_x_avg
        tip_min_y = data[tip_columns[1]].min() - base_y_avg
        tip_max_y = data[tip_columns[1]].max() - base_y_avg
        tip_min_z = data[tip_columns[2]].min() - base_z_avg
        tip_max_z = data[tip_columns[2]].max() - base_z_avg
        
        lower_bound = np.array([tip_min_x, tip_min_y, tip_min_z])
        upper_bound = np.array([tip_max_x, tip_max_y, tip_max_z])

        print(f"Picked goal: {np.random.uniform(lower_bound, upper_bound).astype(np.float32)} within workspace bounds: {lower_bound} and {upper_bound}")
        return np.random.uniform(lower_bound, upper_bound).astype(np.float32)

    def step(self, action):
        self.robot_api.send_command(action)
        self.tip = self.robot_api.get_current_tip()
        distance = np.linalg.norm(self.tip - self.goal)
        terminated = False
        self.current_step += 1
        distance_threshold = 5
        
        if distance > distance_threshold:
            terminated = True
        # print(f"Step {self.current_step}: Terminated {terminated} Distance {distance} ", end="\r", flush=True)
        print(f"Step {self.current_step}: Terminated {terminated} Distance {distance} ")
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {}
        if not terminated:
            reward = 1/10 * (math.exp(-2*distance+3))

            # reward = (1/(distance_threshold-0.3))*distance
        else:
            reward = 0
        self.previous_distace = distance
        return self.tip, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)
        self.robot_api.reset_robot()
        self.tip = self.robot_api.get_current_tip()
        self.current_step = 0
        return self.tip, {}

    def render(self, mode='human'):
        pass

