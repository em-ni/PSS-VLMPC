import numpy as np
import gymnasium as gym
from gymnasium import spaces
import src.config as config
import math
import src.config as config
from src.real_robot_api import RealRobotAPI


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
        self.goal = np.array([4.5, -1.1, 0.0], dtype=np.float32)
        self.max_steps = 200
        self.current_step = 0
        self.previous_distace = 0
        self.robot_api = RealRobotAPI()

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

