import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.nn_model import VolumeNet
from tests.test_blob_sampling import generate_surface, load_point_cloud_from_csv
from utils.nn_functions import predict_tip
from utils.traj_functions import pick_goal, generate_snapped_trajectory
from src.config import MODEL_PATH, SCALERS_PATH, POINT_CLOUD_PATH
class SimRobotEnv(gym.Env):
    """
    A simulated robot environment using a pre-trained neural network model.
    """
    def __init__(self):
        super(SimRobotEnv, self).__init__()

        # Load the point cloud from a CSV file and get the mesh 
        self.point_cloud = load_point_cloud_from_csv(POINT_CLOUD_PATH)
        self.alpha_shape, self.convex_hull = generate_surface(self.point_cloud, alpha=1.0)
        print("Loaded point cloud")

        # Continuous action space: 4 continuous values from 0 to max_stroke, 3 bending + 1 elongation
        self.action_space = spaces.Box(
            low=0.0, 
            high=config.max_stroke, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Observation space: tip(3) + goal(3) + distance(1) + last_action(4)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), 
            dtype=np.float32
        )
        
        # Store the last action for observation
        self.last_action = np.zeros(4, dtype=np.float32)
        
        # Initialize neural network model and scalers
        self.init_nn()
        print("Neural network initialized.")

        # Initialize tip position with commands at initial position
        self.tip = predict_tip(np.zeros(3, dtype=np.float32), self.scaler_volumes, self.scaler_deltas, self.model, self.device)

        # Set the maximum number of steps for the episode
        self.max_steps = 1000
        self.current_step = 0
        self.distances = []

        if config.N_POINTS > 0:
            self.current_trajectory_index = 0
            # Generate a smooth trajectory with the required number of points
            trajectory_points = generate_snapped_trajectory(
                self.point_cloud,
                num_waypoints=config.N_POINTS,
                T_sim=config.N_POINTS * 10.0,
                N_steps=self.max_steps,
                from_initial_pos=True  # Start from the robot's initial position
            )
            
            # Use these trajectory points as our goals
            self.goals = trajectory_points
            print(f"Generated smooth trajectory with {len(self.goals)} points")
            
            # Set the first goal as the initial goal
            self.goal = self.goals[self.current_trajectory_index]
            print(f"Setting initial goal at {self.goal}")
        else:
            self.goal = pick_goal(self.point_cloud)
            print(f"Setting new goal at {self.goal}")

    def init_nn(self):
        """
        Initialize the neural network model and scalers.
        """
        # Load scalers
        scalers_path = SCALERS_PATH
        scalers = np.load(scalers_path)

        # Recreate scalers
        self.scaler_volumes = MinMaxScaler()
        self.scaler_volumes.min_ = scalers['volumes_min']
        self.scaler_volumes.scale_ = scalers['volumes_scale']

        self.scaler_deltas = MinMaxScaler()
        self.scaler_deltas.min_ = scalers['deltas_min']
        self.scaler_deltas.scale_ = scalers['deltas_scale']

        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        self.model = VolumeNet(input_dim=3, output_dim=3)

        # Load state dict to the appropriate device
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_goal(self):
        return self.goal

    def step(self, action):
        # Actions [motor1, motor2, motor3, elongation (3 motors combined)]
        elongation = action[3]
        
        # Create a new command vector with the first 3 actions adjusted by elongation
        command = np.zeros(3, dtype=np.float32)
        for i in range(3):
            # Combine each action with elongation, ensuring it stays within bounds
            combined_action = min(action[i] + elongation, config.max_stroke)
            command[i] = max(0, combined_action)  # Ensure non-negative
        
        self.tip = predict_tip(command, self.scaler_volumes, self.scaler_deltas, self.model, self.device)
        distance = np.linalg.norm(self.tip - self.goal)
        self.distances = np.append(self.distances, distance)
        terminated = False
        self.current_step += 1
        
        # Base reward is negative distance
        reward = -distance
        
        # Truncate episode if we exceed max steps
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {"step": self.current_step, "distance": float(distance)}

        # Update goal
        if config.N_POINTS > 0:
            if distance < 0.5:
                # Move to the next goal if the current one is reached
                reward += 10  # Bonus for reaching the goal
                self.current_trajectory_index += 1
                if self.current_trajectory_index < len(self.goals):
                    self.goal = self.goals[self.current_trajectory_index]
                else:
                    # If all goals are reached, terminate the episode
                    terminated = True
                    reward += 100  # Bonus for reaching all goals
                    print("All goals reached. Ending episode.")

        
        # Store the current action for the next observation
        self.last_action = action.copy()
        
        # Create the enhanced observation
        observation = np.concatenate([
            self.tip,                # Current tip position (3)
            self.goal,               # Current goal position (3)
            [distance],              # Distance to goal (1)
            self.last_action,        # Last action taken (4)
        ])
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        # print("Resetting environment...")
        super().reset(seed=seed)
        self.current_step = 0
        self.distances = []
        
        # Reset the last action
        self.last_action = np.zeros(4, dtype=np.float32)
        
        if config.N_POINTS > 0:
            self.goals = generate_snapped_trajectory(
                self.point_cloud,
                num_waypoints=config.N_POINTS,
                T_sim=config.N_POINTS * 10.0,
                N_steps=self.max_steps,
                from_initial_pos=True  
            )
            self.current_trajectory_index = 0
            self.goal = self.goals[self.current_trajectory_index]
            # print(f"Resetting trajectory")
        else:
            self.goal = pick_goal(self.point_cloud)
            self.goals = [self.goal] # Treat as a 1-point trajectory
            # print(f"Resetting to single goal: {self.goal}")


        # Reset the tip position to the initial position
        self.tip = predict_tip(np.zeros(3, dtype=np.float32), self.scaler_volumes, self.scaler_deltas, self.model, self.device)
        distance = np.linalg.norm(self.tip - self.goal)
        self.last_action = np.zeros(4, dtype=np.float32)

        observation = np.concatenate([
            self.tip,
            self.goal,
            [distance],
            self.last_action,
        ]).astype(np.float32)

        return observation, {}

    def render(self, mode='human'):
        pass