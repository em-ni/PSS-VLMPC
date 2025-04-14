from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from tests.test_blob_sampling import load_point_cloud_from_csv 
from utils.mpc_functions import predict_delta_from_volume, solve_for_optimal_volume
from utils.traj_functions import generate_snapped_trajectory
from utils.nn_functions import load_model_and_scalers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from src.config import (
    INITIAL_POS_VAL,
    U_MAX_CMD,
    U_MIN_CMD,
    V_REST,
    V_MIN_PHYSICAL,
    V_MAX_PHYSICAL,
    VOLUME_BOUNDS_LIST,
    CONTROL_DIM,
    STATE_DIM,
    DT,
    T_SIM,
    N_sim_steps,
    Q_matrix,
    R_matrix,
    OPTIMIZER_METHOD,
    PERTURBATION_SCALE
)

class MpcSimEnv(gym.Env):
    """
    Gymnasium environment for learning GP parameters to generate a control correction factor (u_correction)
    on top of a baseline optimal control (u_opt) derived from single-step
    volume optimization using a pre-trained NN model (VolumeNet: volume -> delta_pos).
    """

    def __init__(self, render_mode=None):
        super().__init__()

        # --- Essential Parameters ---
        self.initial_pos_val = INITIAL_POS_VAL
        self.u_max_cmd = U_MAX_CMD
        self.u_min_cmd = U_MIN_CMD
        self.v_rest = V_REST.copy()
        self.v_min_physical = V_MIN_PHYSICAL
        self.v_max_physical = V_MAX_PHYSICAL
        self.volume_bounds_list = VOLUME_BOUNDS_LIST

        # --- Paths ---
        # Use absolute paths based on this file's location for robustness
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir) # Assumes env file is in 'src'
        self.model_path = os.path.join(self.project_root, "data/exp_2025-04-04_19-17-42/volume_net.pth")
        self.scalers_path = os.path.join(self.project_root, "data/exp_2025-04-04_19-17-42/volume_net_scalers.npz")
        self.point_cloud_path = os.path.join(self.project_root, "data/exp_2025-04-04_19-17-42/output_exp_2025-04-04_19-17-42.csv")

        # --- Load Model and Scalers ---
        self.nn_model, self.scaler_volumes, self.scaler_deltas, self.nn_device = load_model_and_scalers(self.model_path, self.scalers_path)
        if self.nn_model is None:
            raise RuntimeError("Failed to load NN model or scalers. Environment cannot initialize.")

        # --- Load Point Cloud ---
        try:
            self.point_cloud = load_point_cloud_from_csv(self.point_cloud_path)
            if len(self.point_cloud) == 0: raise ValueError("Point cloud file is empty.")
            print(f"Loaded point cloud with {len(self.point_cloud)} points for environment.")
        except Exception as e:
            print(f"ERROR loading point cloud '{self.point_cloud_path}': {e}. Using random points.")
            self.point_cloud = np.random.rand(100, 3) * 5 - 2.5 # Fallback

        # --- Gaussian Process Setup ---
        # Initialize GP for each dimension of control
        self.gp_models = [None for _ in range(CONTROL_DIM)]
        
        # --- RL Action Space (GP Parameters) ---
        # For simplicity, we'll use: [length_scale, signal_variance, noise_variance] per dimension
        # Total parameters: 3 * CONTROL_DIM
        gp_params_per_dim = 3  # length_scale, signal_variance, noise_variance
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.1, 0.01] * CONTROL_DIM),  # min values for each parameter
            high=np.array([10.0, 10.0, 1.0] * CONTROL_DIM),  # max values for each parameter
            shape=(gp_params_per_dim * CONTROL_DIM,),
            dtype=np.float32
        )

        # --- RL Observation Space ---
        # [current_delta(3), target_delta(3), pred_delta_opt(3), last_u_correction(3)]
        obs_dim = STATE_DIM + STATE_DIM + STATE_DIM + CONTROL_DIM # 3+3+3+3 = 12
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # --- Simulation Parameters ---
        self.dt = DT
        self.t_sim = T_SIM
        self.n_sim_steps = N_sim_steps
        self.max_steps_per_episode = self.n_sim_steps # Timeout after T_SIM

        # --- Environment State Variables ---
        self.x_current_delta = None # Current actual delta state
        self.current_actual_volume = None # Current volume corresponding to x_current_delta
        self.delta_ref_trajectory = None # Full reference trajectory for the episode
        self.current_traj_index = 0 # Index for the *next* target in the trajectory
        self.last_u_correction = np.zeros(CONTROL_DIM, dtype=np.float32) # Last correction applied
        self.current_step = 0
        
        # --- GP Training Memory ---
        # Store data for GP training: (state, optimal_u, error)
        self.gp_memory_x = [[] for _ in range(CONTROL_DIM)]
        self.gp_memory_y = [[] for _ in range(CONTROL_DIM)]
        self.max_memory_size = 100  # Limit the memory size for computational efficiency

        # --- Optimization Settings (passed to solver) ---
        # These are fixed for the environment instance but could be made init args
        self.q_matrix = Q_matrix
        self.r_matrix = R_matrix
        self.optimizer_method = OPTIMIZER_METHOD
        self.perturbation_scale = PERTURBATION_SCALE

        # --- Rendering ---
        self.render_mode = render_mode
        # Add any variables needed for rendering if 'human' mode is implemented

    def _update_gaussian_process(self, gp_params):
        """
        Update the Gaussian Process parameters based on RL action
        
        Parameters:
        - gp_params: array of shape (3*CONTROL_DIM,) containing parameters for each dimension's GP
        """
        for i in range(CONTROL_DIM):
            # Extract parameters for this dimension
            start_idx = i * 3
            length_scale = gp_params[start_idx]
            signal_variance = gp_params[start_idx + 1]
            noise_variance = gp_params[start_idx + 2]
            
            # Create kernel with these parameters
            kernel = ConstantKernel(constant_value=signal_variance) * RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_variance)
            
            # Create or update the GP model
            if self.gp_models[i] is None:
                self.gp_models[i] = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
            else:
                self.gp_models[i].kernel = kernel
            
            # Train the GP if we have data
            if len(self.gp_memory_x[i]) > 0:
                X = np.array(self.gp_memory_x[i])
                y = np.array(self.gp_memory_y[i])
                self.gp_models[i].fit(X, y)

    def _get_gp_correction(self, state_features):
        """
        Generate correction using the Gaussian Process models
        
        Parameters:
        - state_features: Features used for GP prediction (e.g., current state, target, prediction error)
        
        Returns:
        - u_correction: Array of corrections for each control dimension
        """
        u_correction = np.zeros(CONTROL_DIM, dtype=np.float32)
        
        # Make prediction for each dimension if GP is trained
        for i in range(CONTROL_DIM):
            if self.gp_models[i] is not None and len(self.gp_memory_x[i]) > 0:
                # Reshape for sklearn's GP which expects 2D array
                X = np.array([state_features])
                u_correction[i] = self.gp_models[i].predict(X)[0]
            
        return u_correction

    def _add_to_gp_memory(self, state_features, optimal_u, actual_delta, target_delta):
        """
        Add a data point to the GP training memory
        
        Parameters:
        - state_features: Features used for GP input
        - optimal_u: The optimal control applied
        - actual_delta: The resulting state delta
        - target_delta: The target state delta
        """
        # Calculate error (difference between actual and target)
        error = actual_delta - target_delta
        
        # Add data for each control dimension
        for i in range(CONTROL_DIM):
            # Add to memory
            self.gp_memory_x[i].append(state_features)
            self.gp_memory_y[i].append(error[i])
            
            # Limit memory size by removing oldest entries if needed
            if len(self.gp_memory_x[i]) > self.max_memory_size:
                self.gp_memory_x[i].pop(0)
                self.gp_memory_y[i].pop(0)

    def reset(self, *, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        # 1. Generate a new target trajectory
        num_waypoints = 1
        self.delta_ref_trajectory = generate_snapped_trajectory(self.point_cloud, num_waypoints, self.t_sim, self.n_sim_steps)
        self.current_traj_index = 0 # Points to the *next* target index (starts at 1)

        # 2. Reset internal state variables
        self.current_step = 0
        self.last_u_correction = np.zeros(CONTROL_DIM, dtype=np.float32)
        self.current_actual_volume = self.v_rest.copy()

        # 3. Calculate initial state (delta) based on resting volume
        self.x_current_delta = predict_delta_from_volume(
            self.current_actual_volume,
            self.nn_model,
            self.scaler_volumes,
            self.scaler_deltas,
            self.nn_device
        )
        if np.isnan(self.x_current_delta).any():
            print("FATAL ERROR during reset: Initial state prediction failed.")
            # Handle error appropriately, maybe return a default observation
            # or raise an exception depending on how SB3 handles it.
            # For now, return zeros, but this should be investigated.
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # 4. Prepare the initial observation
        # Target for the first step is the point at index 1
        delta_ref_next = self.delta_ref_trajectory[self.current_traj_index + 1]
        # Calculate the u_opt that *would* be applied to reach this target
        v_opt_init = solve_for_optimal_volume(
            delta_ref_next, 
            self.q_matrix, 
            self.r_matrix, 
            self.volume_bounds_list,
            self.v_rest, 
            self.nn_model, 
            self.scaler_volumes, 
            self.scaler_deltas, 
            self.nn_device,
            v_guess_init=self.current_actual_volume,
            method=self.optimizer_method,
            perturbation_scale=self.perturbation_scale
        )
        u_opt_init = v_opt_init - self.v_rest
        # Predict the state if only u_opt was used (from resting state)
        x_pred_opt_init = predict_delta_from_volume(
            v_opt_init,
            self.nn_model,
            self.scaler_volumes,
            self.scaler_deltas,
            self.nn_device
        )

        # Reset GP models
        self.gp_models = [None for _ in range(CONTROL_DIM)]
        self.gp_memory_x = [[] for _ in range(CONTROL_DIM)]
        self.gp_memory_y = [[] for _ in range(CONTROL_DIM)]

        observation = self._get_observation(delta_ref_next, x_pred_opt_init)

        info = {} # No extra info needed for reset usually

        return observation, info

    def step(self, action: np.ndarray):
        """Executes one time step within the environment."""
        # 1. Update GP parameters from RL action
        gp_params = action.astype(np.float32)
        self._update_gaussian_process(gp_params)

        # 2. Determine the target state for this step
        self.current_traj_index += 1 # Move to the next target index
        # Ensure we don't go past the trajectory end
        target_idx = min(self.current_traj_index, len(self.delta_ref_trajectory) - 1)
        delta_ref_current_target = self.delta_ref_trajectory[target_idx]

        # 3. Calculate the baseline optimal control u_opt for the current target
        # Warm start solve_for_optimal_volume with the previous actual volume
        v_opt = solve_for_optimal_volume(
            delta_ref_current_target, 
            self.q_matrix, 
            self.r_matrix, 
            self.volume_bounds_list,
            self.v_rest, 
            self.nn_model, 
            self.scaler_volumes, 
            self.scaler_deltas, 
            self.nn_device,
            v_guess_init=self.current_actual_volume,
            method=self.optimizer_method,
            perturbation_scale=self.perturbation_scale
        )
        u_opt = v_opt - self.v_rest

        # --- Calculate the state prediction based *only* on u_opt for observation ---
        # Note: Use v_opt directly as it corresponds to u_opt
        x_pred_opt = predict_delta_from_volume(
            v_opt,
            self.nn_model,
            self.scaler_volumes,
            self.scaler_deltas,
            self.nn_device
        )
        # ----------------------------------------------------------------------------

        # 4. Generate GP correction
        state_features = np.concatenate([
            self.x_current_delta.flatten(),
            delta_ref_current_target.flatten(),
            x_pred_opt.flatten()
        ])
        u_correction = self._get_gp_correction(state_features)

        # 5. Combine optimal control and GP correction
        u_final = u_opt + u_correction

        # 6. Clip the final command and determine actual volume
        u_clipped = np.clip(u_final, self.u_min_cmd, self.u_max_cmd)
        v_actual = self.v_rest + u_clipped

        # 7. Simulate the system: Predict the actual next state delta
        self.x_current_delta = predict_delta_from_volume(
            v_actual, 
            self.nn_model, 
            self.scaler_volumes, 
            self.scaler_deltas, 
            self.nn_device
        )
        if np.isnan(self.x_current_delta).any():
            print(f"FATAL: NaN prediction during step {self.current_step + 1}. Ending episode.")
            # Return high penalty, mark as terminated, provide default observation
            observation = np.zeros(self.observation_space.shape, dtype=np.float32) # Or last valid obs?
            reward = -1000.0 # High penalty
            terminated = True
            truncated = False
            info = {"error": "NaN prediction"}
            return observation, reward, terminated, truncated, info

        # 8. Update internal state tracking
        self.current_actual_volume = v_actual # Use actual volume for next warm start
        self.last_u_correction = u_correction # Store the applied correction

        # 9. Add data to GP memory
        self._add_to_gp_memory(state_features, u_opt, self.x_current_delta, delta_ref_current_target)

        # 10. Calculate reward (based on distance to the current target)
        distance = np.linalg.norm(self.x_current_delta - delta_ref_current_target)
        # Simple negative distance reward, can be made more complex
        reward = -distance

        # 11. Check for termination conditions
        self.current_step += 1
        terminated = (self.current_traj_index >= len(self.delta_ref_trajectory) - 1) # Reached end
        truncated = (self.current_step >= self.max_steps_per_episode) # Max steps timeout

        # 12. Construct the next observation
        observation = self._get_observation(delta_ref_current_target, x_pred_opt, distance)

        # 13. Info dictionary
        info = {"distance": distance, "target_delta": delta_ref_current_target, "u_opt": u_opt, "u_corr": u_correction}

        return observation, reward, terminated, truncated, info

    def _get_observation(self, target_delta, pred_delta_opt, distance=None):
        """Constructs the observation array."""
        # Observation: [current_delta(3), target_delta(3), pred_delta_opt(3), last_u_correction(3)]
        obs = np.concatenate([
            self.x_current_delta.flatten(),
            target_delta.flatten(),
            pred_delta_opt.flatten(),
            self.last_u_correction.flatten()
        ]).astype(np.float32)

        # --- Verification ---
        # Ensure the observation has the expected shape
        expected_shape = self.observation_space.shape
        if obs.shape != expected_shape:
            print(f"FATAL ERROR: Observation shape mismatch!")
            print(f"  Expected: {expected_shape}, Got: {obs.shape}")
            print(f"  Components:")
            print(f"    Current Delta: {self.x_current_delta.shape}")
            print(f"    Target Delta: {target_delta.shape}")
            print(f"    Pred Delta Opt: {pred_delta_opt.shape}")
            print(f"    Last U Correction: {self.last_u_correction.shape}")
            # Handle this critical error, maybe raise exception or return default
            # For debugging, let's return zeros, but this needs fixing.
            return np.zeros(expected_shape, dtype=np.float32)
            # raise ValueError(f"Observation shape mismatch: Expected {expected_shape}, Got {obs.shape}")
        # --- End Verification ---

        return obs

    def render(self):
        """Renders the environment (optional)."""
        if self.render_mode == 'human':
            # Implement visualization here if needed (e.g., using matplotlib 3D plot)
            # This would likely involve plotting self.x_current_delta and self.delta_ref_trajectory
            print(f"Step: {self.current_step}, Current Delta: {np.round(self.x_current_delta, 3)}, Target Delta: {np.round(self.delta_ref_trajectory[self.current_traj_index], 3)}")
        else:
            pass # No rendering for other modes

    def close(self):
        """Performs any necessary cleanup."""
        # e.g., close plotting windows if created
        plt.close('all') # Close any matplotlib figures
        print("MpcSimEnv closed.")