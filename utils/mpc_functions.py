import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import src.config as config
import torch.nn as nn
from scipy.optimize import minimize
from src.config import STATE_DIM, VOLUME_DIM

# --- Prediction Function ---
def predict_delta_from_volume(volumes_np: np.ndarray, nn_model: nn.Module, scaler_volumes: MinMaxScaler, scaler_deltas: MinMaxScaler, nn_device: torch.device) -> np.ndarray:
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
    return predicted_delta_np.flatten()

# --- Cost Function ---
def volume_cost_function(v_target_np: np.ndarray, delta_ref_np: np.ndarray, Q_matrix: np.ndarray, R_matrix: np.ndarray, v_rest_np: np.ndarray, nn_model: nn.Module, scaler_volumes: MinMaxScaler, scaler_deltas: MinMaxScaler, nn_device: torch.device):
    predicted_delta_np = predict_delta_from_volume(v_target_np, nn_model, scaler_volumes, scaler_deltas, nn_device)
    if np.isnan(predicted_delta_np).any(): return 1e20 # Return large cost if prediction failed
    state_error = predicted_delta_np - delta_ref_np
    state_cost = state_error @ Q_matrix @ state_error
    total_cost = state_cost
    if np.any(np.abs(R_matrix) > 1e-12): # Only add if R is non-negligible
        volume_deviation = v_target_np - v_rest_np
        control_cost = volume_deviation @ R_matrix @ volume_deviation
        total_cost += control_cost
    return total_cost

# --- Optimization Function (Adjusted Options) ---
def solve_for_optimal_volume(delta_ref_np: np.ndarray, Q_matrix: np.ndarray, R_matrix: np.ndarray, volume_bounds: list, v_rest_np: np.ndarray, nn_model: nn.Module, scaler_volumes: MinMaxScaler, scaler_deltas: MinMaxScaler, nn_device: torch.device, v_guess_init=None, method='SLSQP', perturbation_scale=0.0):
    objective = lambda v: volume_cost_function(v, delta_ref_np, Q_matrix, R_matrix, v_rest_np, nn_model, scaler_volumes, scaler_deltas, nn_device)
    if v_guess_init is not None:
        v_guess = v_guess_init + np.random.randn(VOLUME_DIM) * perturbation_scale
        v_guess = np.clip(v_guess, volume_bounds[0][0], volume_bounds[0][1])
    else:
        v_guess = v_rest_np

    # --- Set options based on method ---
    optim_options = {'disp': False, 'maxiter': 200} # Common options
    if method == 'trust-constr':
        optim_options['gtol'] = 1e-8
        optim_options['xtol'] = 1e-8
        # Remove 'eps' if present from previous attempts for other methods
        optim_options.pop('eps', None)
    elif method in ['SLSQP', 'L-BFGS-B', 'TNC']:
        optim_options['ftol'] = 1e-9
        optim_options['eps'] = 1.49e-08 # Default step size okay for these
        # Remove 'gtol'/'xtol' if present
        optim_options.pop('gtol', None)
        optim_options.pop('xtol', None)
    # ----------------------------------

    result = minimize(objective, v_guess, method=method, bounds=volume_bounds, options=optim_options)

    final_v = result.x
    if not result.success:
        # print(f"Warning: Volume optimization failed! Method={method}, Reason: {result.message}. Cost={result.fun:.4e}.") # Keep commented unless debug
        if not np.allclose(v_guess, v_rest_np, atol=1e-5):
             # print("  Retrying optimization from v_rest...") # Keep commented unless debug
             result_retry = minimize(objective, v_rest_np, method=method, bounds=volume_bounds, options=optim_options)
             if result_retry.success: final_v = result_retry.x
             else: final_v = v_rest_np
        else: final_v = v_rest_np
    final_v = np.clip(final_v, volume_bounds[0][0], volume_bounds[0][1]) # Final safety clip
    return final_v