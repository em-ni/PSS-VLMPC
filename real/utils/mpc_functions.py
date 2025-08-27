import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
from scipy.optimize import minimize
from src.config import STATE_DIM
import time
from src.config import MPC_DEBUG

# --- Prediction Function ---
def predict_delta_from_volume(volumes_np: np.ndarray, nn_model: nn.Module, scaler_volumes: MinMaxScaler, scaler_deltas: MinMaxScaler, nn_device: torch.device) -> np.ndarray:
    if MPC_DEBUG: start_time = time.time()
    if nn_model is None: raise RuntimeError("NN Model is not loaded.")
    volumes_np = volumes_np.astype(np.float32).reshape(1, -1)
    try:
        volumes_scaled = scaler_volumes.transform(volumes_np)
    except Exception as e: print(f"ERROR during volume scaling: {e}\n  Input: {volumes_np}\n  Scaler min: {scaler_volumes.min_}\n  Scaler scale: {scaler_volumes.scale_}"); return np.full(STATE_DIM, np.nan)
    volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32).to(nn_device)
    with torch.no_grad():
        predictions_scaled = nn_model(volumes_tensor).cpu().numpy()
    try:
        if predictions_scaled.ndim == 1: predictions_scaled = predictions_scaled.reshape(1, -1)
        elif predictions_scaled.shape[0] != 1: predictions_scaled = predictions_scaled.reshape(1, -1)
        predicted_delta_np = scaler_deltas.inverse_transform(predictions_scaled)
    except Exception as e: print(f"ERROR during delta unscaling: {e}\n  Scaled pred: {predictions_scaled}\n  Scaler min: {scaler_deltas.min_}\n  Scaler scale: {scaler_deltas.scale_}"); return np.full(STATE_DIM, np.nan)
    if MPC_DEBUG: end_time = time.time()
    if MPC_DEBUG: elapsed_time = end_time - start_time
    if MPC_DEBUG: print(f"Prediction took {elapsed_time:.6f} seconds for input: {volumes_np.flatten()}")
    return predicted_delta_np.flatten()

# --- MPC Cost Function ---
def mpc_cost_function(
    v_sequence_flat: np.ndarray, # Flattened sequence [v_k[0]..v_k[d], v_k+1[0]..v_k+1[d], ...]
    delta_ref_sequence: np.ndarray, # Reference states [delta_ref_{k+1}, ..., delta_ref_{k+N}]
    current_state_delta: np.ndarray, # The state x_k (unused currently)
    v_previous_applied: np.ndarray, 
    Q_matrix: np.ndarray,
    R_matrix: np.ndarray,
    R_delta_matrix: np.ndarray,    
    Q_terminal_matrix: np.ndarray,
    v_rest_np: np.ndarray,
    nn_model: nn.Module,
    scaler_volumes: MinMaxScaler,
    scaler_deltas: MinMaxScaler,
    nn_device: torch.device,
    horizon_N: int,
    volume_dim: int,
    state_dim: int
    ):
    """
    Calculates the total cost over the prediction horizon N, including
    state tracking, control effort (deviation from rest), and control rate penalty.
    """
    total_cost = 0.0
    v_sequence = v_sequence_flat.reshape(horizon_N, volume_dim) 

    # Check if rate penalty should be applied
    apply_rate_penalty = np.any(np.abs(R_delta_matrix) > 1e-12)

    # --- Calculate penalty for the first step's change (v_k - v_{k-1}) ---
    if apply_rate_penalty:
        delta_v_first = v_sequence[0] - v_previous_applied
        rate_cost_first = delta_v_first @ R_delta_matrix @ delta_v_first
        total_cost += rate_cost_first

    # --- Loop through the horizon for state, control, and subsequent rate costs ---
    for j in range(horizon_N): # Loop from j=0 to N-1
        v_current_step = v_sequence[j] # This is v_{k+j}

        # Predict the state resulting from this volume: x_{k+j+1}
        predicted_delta_next = predict_delta_from_volume(v_current_step, nn_model, scaler_volumes, scaler_deltas, nn_device)

        if np.isnan(predicted_delta_next).any():
            return 1e20 

        # Get the corresponding reference state: delta_ref_{k+j+1}
        delta_ref_current = delta_ref_sequence[j] 

        # --- Calculate Stage Cost (State Tracking) ---
        state_error = predicted_delta_next - delta_ref_current
        if j == horizon_N - 1: # Use terminal cost for the last state
            stage_cost = state_error @ Q_terminal_matrix @ state_error
        else: # Use running cost for intermediate states
            stage_cost = state_error @ Q_matrix @ state_error

        # --- Add Control Effort Cost (deviation from rest) ---
        if np.any(np.abs(R_matrix) > 1e-12):
            volume_deviation = v_current_step - v_rest_np
            control_cost = volume_deviation @ R_matrix @ volume_deviation
            stage_cost += control_cost

        # --- Add Control Rate Cost (v_{k+j} - v_{k+j-1}) for j > 0 ---
        if apply_rate_penalty and j > 0:
            delta_v_step = v_sequence[j] - v_sequence[j-1]
            rate_cost_step = delta_v_step @ R_delta_matrix @ delta_v_step
            stage_cost += rate_cost_step

        total_cost += stage_cost

    return total_cost

# --- MPC Optimization Solver ---
def solve_mpc_optimization(
    delta_ref_sequence: np.ndarray, # Reference states [delta_ref_{k+1}, ..., delta_ref_{k+N}]
    current_state_delta: np.ndarray, # State x_k (unused by current cost func, but good practice)
    v_previous_applied: np.ndarray, 
    Q_matrix: np.ndarray,
    R_matrix: np.ndarray,
    R_delta_matrix: np.ndarray,     
    Q_terminal_matrix: np.ndarray,
    volume_bounds_list: list, 
    v_rest_np: np.ndarray,
    nn_model: nn.Module,
    scaler_volumes: MinMaxScaler,
    scaler_deltas: MinMaxScaler,
    nn_device: torch.device,
    horizon_N: int,
    v_sequence_guess_init: np.ndarray, # Initial guess for the *entire sequence* V = [v_k, ..., v_{k+N-1}]
    method='trust-constr', 
    perturbation_scale=0.005
    ):
    """
    Solves the MPC optimization problem for the optimal volume sequence.
    Returns the *entire* optimal sequence V*.
    """
    volume_dim = len(v_rest_np)
    state_dim = Q_matrix.shape[0]

    # --- Prepare arguments for the cost function ---
    cost_args = (
        delta_ref_sequence, current_state_delta,
        v_previous_applied,
        Q_matrix, R_matrix,
        R_delta_matrix,   
        Q_terminal_matrix,
        v_rest_np, nn_model, scaler_volumes, scaler_deltas, nn_device,
        horizon_N, volume_dim, state_dim
    )

    objective = lambda v_flat: mpc_cost_function(v_flat, *cost_args)

    # --- Prepare initial guess ---
    v_guess_flat = v_sequence_guess_init.flatten()
    if perturbation_scale > 0.0:
        v_guess_flat += np.random.randn(horizon_N * volume_dim) * perturbation_scale

    # --- Prepare bounds for the sequence ---
    bounds_per_step = volume_bounds_list # Assume bounds are same for all dimensions for now
    bounds_sequence = []
    # Ensure bounds_per_step matches volume_dim
    if len(bounds_per_step) != volume_dim:
        # If only one bound pair is given, replicate it for all dims
        if len(bounds_per_step) == 1:
            bounds_per_step = bounds_per_step * volume_dim
            # print(f"Warning: Replicating single volume bound pair for {volume_dim} dimensions.") # Keep commented unless debugging
        else:
            raise ValueError(f"Length of volume_bounds_list ({len(bounds_per_step)}) does not match VOLUME_DIM ({volume_dim})")

    for _ in range(horizon_N):
        bounds_sequence.extend(bounds_per_step)

    # Apply initial clipping to guess based on sequence bounds
    for i in range(len(v_guess_flat)):
        min_b, max_b = bounds_sequence[i]
        v_guess_flat[i] = np.clip(v_guess_flat[i], min_b, max_b)


    # --- Set optimizer options ---
    optim_options = {'disp': False, 'maxiter': 1000*1000}
    if method == 'trust-constr':
        optim_options['gtol'] = 1e-10
        optim_options['xtol'] = 1e-10
        optim_options.pop('eps', None)
    elif method in ['SLSQP', 'L-BFGS-B', 'TNC']:
        optim_options['ftol'] = 1e-8
        optim_options['eps'] = 1.49e-08
        optim_options.pop('gtol', None)
        optim_options.pop('xtol', None)
    elif method == 'COBYQA':
        # optim_options['gtol'] = 1e-10
        # optim_options['xtol'] = 1e-10
        # optim_options.pop('eps', None)
        pass

    # --- Run Optimization ---
    if MPC_DEBUG: start = time.time()
    result = minimize(
        objective,
        v_guess_flat,
        method=method,
        bounds=bounds_sequence,
        options=optim_options
    )
    if MPC_DEBUG: end = time.time()
    if MPC_DEBUG: print(f"minimize took {end - start:.6f} seconds using method '{method}' with {len(bounds_sequence)} bounds.")

    # --- Process result ---
    optimal_v_sequence_flat = result.x
    if not result.success:
        print(f"Warning: MPC optimization failed! ...") # Keep commented unless debugging
        optimal_v_sequence_flat = v_guess_flat
    else:
         for i in range(len(optimal_v_sequence_flat)):
             min_b, max_b = bounds_sequence[i]
             optimal_v_sequence_flat[i] = np.clip(optimal_v_sequence_flat[i], min_b, max_b)

    optimal_v_sequence = optimal_v_sequence_flat.reshape(horizon_N, volume_dim)

    return optimal_v_sequence


def load_trajectory_data(file_path="planned_trajectory.csv"):
    """
    Load the trajectory and control data from CSV file.
    
    Returns:
        tuple: (reference_trajectory, control_inputs)
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded trajectory data with {len(df)} steps")
        
        # Extract reference trajectory
        ref_cols = [col for col in df.columns if col.startswith('ref_delta_')]
        ref_trajectory = df[ref_cols].values
        
        # Extract control inputs
        control_cols = [col for col in df.columns if col.startswith('control_')]
        control_inputs = df[control_cols].values
        
        return ref_trajectory, control_inputs
    except Exception as e:
        print(f"Error loading trajectory data: {e}")
        return None, None