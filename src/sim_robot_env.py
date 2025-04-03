import os
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.lstm_model import LSTMModel
from tests.test_blob_sampling import generate_surface, sample_point_in_hull, is_inside_hull, load_point_cloud_from_csv

class SimRobotEnv(gym.Env):
    """
    A simulated robot environment using a pre-trained LSTM model for dynamics.
    Observation: 3D coordinates of the simulated tip wrt to the base (shape: 3,)
    Action: A 3D discrete vector representing desired volumes (tau).
            (NOTE: Modified from RobotEnv's 4D action for clarity, assuming
             the 4th 'elongation' action is handled externally or incorporated
             into how the 3 base actions are chosen by the RL agent policy)
             Alternatively, retain 4D action and map internally as before.
             Let's retain the 4D action for consistency with original env.
    Reward: Negative Euclidean distance from the simulated tip to the goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, model_path=r"data/exp_2025-04-02_11-43-59/lstm_model.pth"):
        super(SimRobotEnv, self).__init__()

        # --- Constants ---
        self.sequence_length = config.sequence_length 
        self.n_features_tau = config.n_features_tau
        self.n_features_x = config.n_features_x
        self.total_features = config.total_features
        self.output_dim = config.output_dim
        self.lstm_hidden_units = config.lstm_hidden_units
        self.lstm_num_layers = config.lstm_num_layers

        # --- Action Space (Consistent with RobotEnv) ---
        self.action_space = spaces.MultiDiscrete([config.steps] * 4)

        # Mapping from discrete action indices to continuous tau values
        self.action_mapping = {
            i: np.linspace(0, config.max_stroke, config.steps) for i in range(4)
        }

        # --- Observation Space ---
        # Denormalized relative tip position [x, y, z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # --- Load Model and Scaler ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SimEnv using device: {self.device}")
        try:
            # Load the saved dictionary that contains both model and scaler
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create and load the model
            self.model = LSTMModel(self.total_features, self.lstm_hidden_units, self.output_dim, self.lstm_num_layers)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Loaded trained LSTM model from {model_path}")
            
            # Get the scaler from the same file
            self.scaler = checkpoint['scaler']
            print(f"Loaded data scaler from model checkpoint")
            
            if not isinstance(self.scaler, MinMaxScaler) or not hasattr(self.scaler, 'n_features_in_'):
                raise ValueError("Loaded object is not a fitted scikit-learn scaler.")
            if self.scaler.n_features_in_ != self.total_features:
                raise ValueError(f"Scaler expected {self.scaler.n_features_in_} features, but environment requires {self.total_features}.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. SimEnv cannot function.")
            raise
        except Exception as e:
            print(f"Error loading model or scaler: {e}")
            raise

        # --- Goal and Workspace ---
        print("Loading point cloud for goal sampling...")
        # Use the same file path as in RobotEnv or make it configurable
        point_cloud_file = r"data/exp_2025-04-02_11-43-59/output_exp_2025-04-02_11-43-59.csv"
        point_cloud = load_point_cloud_from_csv(point_cloud_file)
        self.alpha_shape, self.convex_hull = generate_surface(point_cloud, alpha=1.0)
        self.goal = np.zeros(3, dtype=np.float32) # Initialized in reset

        # --- Internal State ---
        self.history_tau_norm = np.zeros((self.sequence_length, self.n_features_tau), dtype=np.float32)
        self.history_x_norm = np.zeros((self.sequence_length, self.n_features_x), dtype=np.float32)
        self.current_tip_denorm = np.zeros(3, dtype=np.float32) # Stores the latest denormalized tip pos [x,y,z]

        self.max_steps = 100 # Max steps per episode
        self.current_step = 0

        # Optional: Keep track of LSTM hidden state if needed for stateful prediction
        self.lstm_hidden_state = None

    def _map_action_to_tau(self, action):
        """Maps the 4D discrete action to 3D continuous tau (volume command)."""
        mapped_action = np.array([
            self.action_mapping[0][action[0]],
            self.action_mapping[1][action[1]],
            self.action_mapping[2][action[2]],
            self.action_mapping[3][action[3]]
        ], dtype=np.float32)

        elongation = mapped_action[3]
        command = np.zeros(3, dtype=np.float32)
        for i in range(3):
            combined_action = min(mapped_action[i] + elongation, config.max_stroke)
            command[i] = max(0, combined_action)
        return command

    def _normalize_features(self, tau=None, x=None):
        """Normalizes tau or x using the loaded scaler."""
        # Scaler expects input shape (n_samples, n_features)
        # n_features = total_features = n_features_tau + n_features_x
        dummy_input = np.zeros((1, self.total_features))
        if tau is not None:
            dummy_input[0, :self.n_features_tau] = tau
        if x is not None:
            dummy_input[0, self.n_features_tau:] = x

        scaled_dummy = self.scaler.transform(dummy_input)

        tau_norm = scaled_dummy[0, :self.n_features_tau] if tau is not None else None
        x_norm = scaled_dummy[0, self.n_features_tau:] if x is not None else None

        if tau is not None and x is not None:
             return tau_norm, x_norm
        elif tau is not None:
             return tau_norm
        elif x is not None:
             return x_norm
        else:
             return None

    def _denormalize_x(self, x_norm):
        """Denormalizes predicted x using the loaded scaler."""
        dummy_input = np.zeros((1, self.total_features))
        dummy_input[0, self.n_features_tau:] = x_norm
        denormalized_dummy = self.scaler.inverse_transform(dummy_input)
        return denormalized_dummy[0, self.n_features_tau:]

    def _update_history(self, tau_norm, x_norm):
        """Updates the internal history buffers."""
        # Shift history back by one step
        self.history_tau_norm = np.roll(self.history_tau_norm, -1, axis=0)
        self.history_x_norm = np.roll(self.history_x_norm, -1, axis=0)
        # Add new values at the end
        self.history_tau_norm[-1, :] = tau_norm
        self.history_x_norm[-1, :] = x_norm

    def pick_goal(self):
        """Pick a random goal within the workspace."""
        print("Picking new random goal...")
        if self.convex_hull:
             return sample_point_in_hull(self.convex_hull, num_samples=1)[0]
        else:
             print("Warning: No convex hull available, returning default goal.")
             # Example: return a random point within some reasonable bounds
             return np.random.uniform(-2, 2, size=3).astype(np.float32)

    def set_goal(self, goal):
         self.goal = np.array(goal, dtype=np.float32)
         print(f"Goal set to: {self.goal}")

    def get_goal(self):
         return self.goal

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        print("Resetting simulated environment...")
        super().reset(seed=seed)
        
        # Reset internal state trackers
        self.current_step = 0
        self.history_tau_norm = np.zeros((self.sequence_length, self.n_features_tau), dtype=np.float32)
        self.history_x_norm = np.zeros((self.sequence_length, self.n_features_x), dtype=np.float32)
        self.current_tip_denorm = np.zeros(3, dtype=np.float32)
        
        # Reset LSTM hidden state
        self.lstm_hidden_state = None
        
        # Pick a new goal with some probability, similar to RobotEnv
        if config.pick_random_goal:
            # Change goal with 40% probability, matching RobotEnv
            if np.random.random() < 0.4:
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")
        
        # Return observation and empty info dict
        return self.current_tip_denorm.copy(), {}

    def step(self, action):
        """Execute one step in the environment."""
        # 1. Map discrete action -> continuous tau command
        tau_t_plus_1 = self._map_action_to_tau(action) # This is the command for the *next* step

        # 2. Normalize the command
        tau_t_plus_1_norm = self._normalize_features(tau=tau_t_plus_1)

        # 3. Prepare LSTM input sequence
        # Input: (tau_t, tau_t-1, tau_t-2, tau_t-3, x_t-1, x_t-2, x_t-3)
        sequence = np.column_stack((self.history_tau_norm, self.history_x_norm))
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # 4. Predict delta_x using the LSTM model
        with torch.no_grad():
            predicted_x_norm = self.model(sequence_tensor)
            predicted_x_norm = predicted_x_norm[0, -1].cpu().numpy()  # Get last prediction

        # 5. Denormalize prediction to get actual delta_x
        delta_x_denorm = self._denormalize_x(predicted_x_norm)

        # 6. Update the current tip position
        self.current_tip_denorm = delta_x_denorm  # Replace with absolute position (like RobotEnv)

        # 7. Update history with new values
        self._update_history(tau_t_plus_1_norm, predicted_x_norm)

        # 8. Calculate distance and reward 
        distance = np.linalg.norm(self.current_tip_denorm - self.goal)
        reward = -distance 
        
        # # Similar reward logic to RobotEnv
        # if distance < 0.65:
        #     reward += 1
        #     self.goal = self.pick_goal()
        #     # print(f"Setting new goal at {self.goal}")

        # 9. Update step counter and check termination
        self.current_step += 1
        terminated = False  # Only terminate if we hit max steps
        truncated = self.current_step >= self.max_steps
        
        # print(f"Step {self.current_step}: Distance {distance}")
        
        # 10. Return observation, reward, done flags, and info
        observation = self.current_tip_denorm.copy()
        info = {
            'distance': distance,
            'goal': self.goal,
            'step': self.current_step
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Render the environment. In simulation, we could add matplotlib visualization.
        For consistency with the real environment, this is a placeholder.
        """
        # For actual visualization, you could add:
        if mode == 'human' and hasattr(self, 'fig'):
            plt.figure(self.fig)
            plt.plot([0, self.current_tip_denorm[0]], 
                     [0, self.current_tip_denorm[1]], 
                     [0, self.current_tip_denorm[2]], 'r-')
            plt.plot(self.goal[0], self.goal[1], self.goal[2], 'go')
            plt.draw()
            plt.pause(0.01)
