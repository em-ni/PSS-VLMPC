import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tests.test_blob_sampling import load_point_cloud_from_csv
import src.config as config
from utils.traj_functions import generate_denser_point_cloud, generate_snapped_trajectory, pick_goal, generate_straight_trajectory, generate_circle_trajectory, generate_spiral_trajectory
from utils.nn_functions import load_model_and_scalers
from utils.mpc_functions import predict_delta_from_volume, solve_mpc_optimization 
import warnings
from scipy.optimize import OptimizeWarning
import pandas as pd

# --- Suppress specific SciPy optimization warnings ---
# This filters warnings where the message contains "delta_grad == 0.0"
warnings.filterwarnings(
    "ignore",
    message=r".*delta_grad == 0\.0.*", 
    category=UserWarning
)
warnings.filterwarnings(
    "ignore", 
    message=r".*delta_grad == 0\.0.*",
    category=OptimizeWarning
)

# --- Import constants from config ---
STATE_DIM = config.STATE_DIM
CONTROL_DIM = config.CONTROL_DIM # Control u = v - v_rest
VOLUME_DIM = config.VOLUME_DIM   # Optimization variable is v
DT = config.DT
T_SIM = config.T_SIM
N_sim_steps = config.N_sim_steps
V_REST = config.V_REST
VOLUME_BOUNDS_LIST = config.VOLUME_BOUNDS_LIST # e.g., [(v_min, v_max)] or [(v1_min, v1_max), (v2_min,..)]
U_MIN_CMD = config.U_MIN_CMD
U_MAX_CMD = config.U_MAX_CMD
Q_matrix = config.Q_matrix
R_matrix = config.R_matrix
Q_TERMINAL_matrix = config.Q_terminal_matrix 
OPTIMIZER_METHOD = config.OPTIMIZER_METHOD
PERTURBATION_SCALE = config.PERTURBATION_SCALE 
MODEL_PATH = config.MODEL_PATH
SCALERS_PATH = config.SCALERS_PATH
POINT_CLOUD_PATH = config.POINT_CLOUD_PATH
N_HORIZON = config.N_HORIZON 

def run_simulation():
    """Runs the MPC control loop following a generated trajectory."""
    nn_model, scaler_volumes, scaler_deltas, nn_device = load_model_and_scalers(MODEL_PATH, SCALERS_PATH)
    if nn_model is None: return

    # 1. Load Point Cloud 
    try:
        original_point_cloud = load_point_cloud_from_csv(POINT_CLOUD_PATH)
        point_cloud = generate_denser_point_cloud(original_point_cloud, density_factor=10)
        if len(point_cloud) == 0: raise ValueError("Point cloud file is empty.")
        print(f"Loaded point cloud with {len(original_point_cloud)} points. Increased density to {len(point_cloud)} points for smoother trajectories.")
    except Exception as e:
        print(f"FATAL: Error loading point cloud '{POINT_CLOUD_PATH}': {e}. Cannot generate trajectory.")
        return

    # 2. Generate Trajectory 
    num_traj_waypoints = config.N_WAYPOINTS 
    if num_traj_waypoints > 1:
        # Generate N_sim_steps + N_horizon steps to have references for the final horizons
        total_ref_steps = N_sim_steps + N_HORIZON
        # delta_ref_trajectory = generate_snapped_trajectory(
        #     point_cloud, num_traj_waypoints, T_SIM * (total_ref_steps / N_sim_steps), total_ref_steps, from_initial_pos=True
        # )
        # delta_ref_trajectory = generate_straight_trajectory(num_points=total_ref_steps, distance=0.5)
        # delta_ref_trajectory = generate_circle_trajectory(num_points=total_ref_steps, radius=0.65, plane='yz')
        delta_ref_trajectory = generate_spiral_trajectory(num_points=total_ref_steps, plane='yz', height_change=0.5)
    else:
        # Use a single goal for all steps
        single_goal = pick_goal(point_cloud)
        print("Generating a single goal trajectory: ", single_goal)
        delta_ref_trajectory = np.tile(single_goal, (N_sim_steps + N_HORIZON, 1)) 
    print(f"Generated reference trajectory with {len(delta_ref_trajectory)} points.")

    # 3. Initialize state and history
    x_current_delta = predict_delta_from_volume(V_REST, nn_model, scaler_volumes, scaler_deltas, nn_device)
    if np.isnan(x_current_delta).any(): print("FATAL: Failed to predict initial state."); return
    print(f"Initial State Delta (predicted for v_rest): {x_current_delta}")
    print(f"Cost Weights: Q_diag={np.diag(Q_matrix)}, R_diag={np.diag(R_matrix)}, Q_term_diag={np.diag(Q_TERMINAL_matrix)}")
    print(f"Volume Bounds: {VOLUME_BOUNDS_LIST}")
    print(f"Prediction Horizon N: {N_HORIZON}")
    print(f"Starting MPC Simulation Loop (Optimizer: {OPTIMIZER_METHOD})...")

    x_current_delta = predict_delta_from_volume(V_REST, nn_model, scaler_volumes, scaler_deltas, nn_device)
    state_history = [x_current_delta]; control_history = []; volume_history = [V_REST]; computation_times = []
    # Initialize MPC sequence guess (repeat V_REST for the horizon)
    v_sequence_guess = np.tile(V_REST, (N_HORIZON, 1))
    v_previous_applied = V_REST.copy() 

    # --- Simulation Loop ---
    for i in range(N_sim_steps): # Loop from 0 to N_sim_steps-1
        step_start_time = time.time()

        # --- MPC Core Logic ---
        # 1. Get reference trajectory over the horizon [ref_{i+1}, ..., ref_{i+N}]
        # Ensure we don't exceed bounds of the generated reference trajectory
        ref_start_idx = i + 1
        ref_end_idx = min(ref_start_idx + N_HORIZON, len(delta_ref_trajectory))
        delta_ref_horizon = delta_ref_trajectory[ref_start_idx:ref_end_idx]

        # If horizon extends beyond trajectory, pad with the last reference point
        actual_horizon_len = len(delta_ref_horizon)
        if actual_horizon_len < N_HORIZON:
            padding_needed = N_HORIZON - actual_horizon_len
            last_ref = delta_ref_horizon[-1]
            padding = np.tile(last_ref, (padding_needed, 1))
            delta_ref_horizon = np.vstack([delta_ref_horizon, padding])
            # Also adjust the guess sequence length if it was based on previous optimal
            if v_sequence_guess.shape[0] > actual_horizon_len:
                 v_sequence_guess = v_sequence_guess[:actual_horizon_len] # Trim guess
                 last_guess_v = v_sequence_guess[-1] if len(v_sequence_guess) > 0 else V_REST
                 guess_padding = np.tile(last_guess_v, (padding_needed, 1))
                 v_sequence_guess = np.vstack([v_sequence_guess, guess_padding])
            elif v_sequence_guess.shape[0] < N_HORIZON: # Should not happen if initialized correctly
                 last_guess_v = v_sequence_guess[-1] if len(v_sequence_guess) > 0 else V_REST
                 guess_padding = np.tile(last_guess_v, (N_HORIZON - v_sequence_guess.shape[0], 1))
                 v_sequence_guess = np.vstack([v_sequence_guess, guess_padding])


        # 2. Solve the optimization problem for the sequence V* = [v*_k, ..., v*_{k+N-1}]
        optimal_v_sequence = solve_mpc_optimization(
                delta_ref_horizon,
                x_current_delta,
                v_previous_applied,     
                Q_matrix, R_matrix,
                config.R_delta_matrix,       
                Q_TERMINAL_matrix,
                VOLUME_BOUNDS_LIST, V_REST,
                nn_model, scaler_volumes, scaler_deltas, nn_device,
                N_HORIZON,
                v_sequence_guess_init=v_sequence_guess,
                method=OPTIMIZER_METHOD,
                perturbation_scale=PERTURBATION_SCALE
            )
        # 3. Apply only the *first* volume command from the optimal sequence
        v_command = optimal_v_sequence[0] # This is v*_k
        u_command = v_command - V_REST

        # Clip the *control command* u, then find the actual volume applied
        u_command_clipped = np.clip(u_command, U_MIN_CMD, U_MAX_CMD)
        v_actual_applied = V_REST + u_command_clipped

        # --- End MPC Core Logic ---

        comp_time = time.time() - step_start_time
        computation_times.append(comp_time)
        control_history.append(u_command_clipped) 
        volume_history.append(v_actual_applied)   

        # Simulate the system's next state (state at step i+1) based on the actual volume applied at step i
        x_next_delta = predict_delta_from_volume(v_actual_applied, nn_model, scaler_volumes, scaler_deltas, nn_device)
        if np.isnan(x_next_delta).any(): print(f"FATAL: NaN prediction at step {i+1}. Stopping."); break

        # Update state for the next iteration (this is now state at i+1)
        x_current_delta = x_next_delta
        state_history.append(x_current_delta)

        # --- Prepare Warm Start for Next Iteration ---
        # Shift the optimal sequence and append a guess for the last element
        v_sequence_guess = np.vstack((optimal_v_sequence[1:], optimal_v_sequence[-1]))
        v_previous_applied = v_actual_applied.copy() 

        # Print progress: Compare state at i+1 with target at i+1
        delta_ref_current = delta_ref_trajectory[i+1] # Target state for the *end* of this step
        current_error_norm = np.linalg.norm(x_current_delta - delta_ref_current)
        print(f"Step {i+1}/{N_sim_steps} | State Delta: [{x_current_delta[0]:.3f}, {x_current_delta[1]:.3f}, {x_current_delta[2]:.3f}] | Target: [{delta_ref_current[0]:.3f},{delta_ref_current[1]:.3f},{delta_ref_current[2]:.3f}] | ErrNorm: {current_error_norm:.3f} | Cmd u*: [{u_command_clipped[0]:.3f}, {u_command_clipped[1]:.3f}, {u_command_clipped[2]:.3f}] | Time: {comp_time:.4f}s", end="\r", flush=True)

    # --- Final Summary ---
    print("MPC Simulation finished.")
    if computation_times: print(f"Average/Maximum computation time per optimization step: {np.mean(computation_times):.4f}s / {np.max(computation_times):.4f}s")
    final_target = delta_ref_trajectory[N_sim_steps] # Target is at index N_sim_steps
    final_error = np.linalg.norm(x_current_delta - final_target)
    print(f"Final State Delta: {x_current_delta}"); print(f"Final Target Delta: {final_target}"); print(f"Final Error Norm (Delta): {final_error:.4f}")

    # Compute smoothed control and resulting trajectory
    smoothed_control = smooth_control(control_history)

    # Re-simulate with smoothed control to get the resulting trajectory
    smoothed_state_history = [state_history[0]] 
    for i in range(len(smoothed_control)):
        # Convert smoothed control back to volume
        smoothed_v = V_REST + smoothed_control[i]
        # Predict next state using the smoothed volume
        next_state_delta = predict_delta_from_volume(
            smoothed_v, nn_model, scaler_volumes, scaler_deltas, nn_device)
        smoothed_state_history.append(next_state_delta)

    smoothed_state_history = np.array(smoothed_state_history)

    # Save trajectory and control data to CSV
    save_trajectory_to_csv(delta_ref_trajectory, smoothed_control, config.TRAJ_DIR)
    

    # Pass the smoothed data to the plotting function
    plot_results(state_history, control_history, delta_ref_trajectory, point_cloud, 
                 DT, U_MIN_CMD, U_MAX_CMD, smoothed_control, smoothed_state_history)

def save_trajectory_to_csv(delta_ref_trajectory, smoothed_control, output_path="planned_trajectory.csv"):
    """
    Save the reference trajectory and smoothed control inputs to a CSV file.
    
    Args:
        delta_ref_trajectory: Reference trajectory points [N, state_dim]
        smoothed_control: Smoothed control inputs [M, control_dim]
        output_path: Output CSV file path
    """
    import pandas as pd
    import os
    
    # Ensure arrays are numpy arrays
    delta_ref = np.array(delta_ref_trajectory)
    smooth_ctrl = np.array(smoothed_control)
    
    # Create column headers
    ref_headers = [f"ref_delta_{i+1}" for i in range(delta_ref.shape[1])]
    ctrl_headers = [f"control_{i+1}" for i in range(smooth_ctrl.shape[1])]
    
    # Ensure they have matching length (control is one shorter than states)
    # We'll use the reference points that correspond to the applied controls
    matching_length = len(smooth_ctrl)
    matched_ref = delta_ref[:matching_length+1]  # Include target points
    
    # Create DataFrame with both trajectory and control data
    # Each row contains the reference delta and the control to reach it
    data = {}
    
    # Add reference trajectory data
    for i, header in enumerate(ref_headers):
        data[header] = matched_ref[1:, i]  # Skip initial state to align with controls
    
    # Add control data
    for i, header in enumerate(ctrl_headers):
        data[header] = smooth_ctrl[:, i]
    
    # Create and save DataFrame
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\nTrajectory and control data saved to {output_path}")

def plot_results(state_history, control_history, delta_ref_trajectory, point_cloud, 
                dt, u_min, u_max, smoothed_control=None, smoothed_state_history=None):
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    num_plot_steps = len(state_history)
    # Adjust reference history length for plotting against state history
    reference_history = delta_ref_trajectory[:num_plot_steps] # Show reference matching the actual run length

    time_axis_state = np.arange(num_plot_steps) * dt
    time_axis_control = np.arange(len(control_history)) * dt

    # --- Figure 1: 2D Plots (State vs Time, Control vs Time) ---
    fig1, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle('MPC Trajectory Following Control Results')

    # Plot State Deltas vs Reference Trajectory (Time Domain)
    axs[0].set_title('State Delta vs. Time')
    labels_delta = ['Actual Delta X', 'Actual Delta Y', 'Actual Delta Z']
    labels_ref = ['Ref Delta X', 'Ref Delta Y', 'Ref Delta Z']
    colors = plt.cm.viridis(np.linspace(0, 0.8, STATE_DIM))
    for i in range(STATE_DIM):
        axs[0].plot(time_axis_state, state_history[:, i], 'o-', color=colors[i], markersize=3, linewidth=1.5, label=labels_delta[i])
        axs[0].plot(time_axis_state, reference_history[:, i], '--', color=colors[i], linewidth=2.0, label=labels_ref[i])
        # Add smoothed state trajectory if provided
        if smoothed_state_history is not None:
            axs[0].plot(time_axis_state, smoothed_state_history[:, i], '*-', color=colors[i], alpha=0.7, 
                      markersize=2, linewidth=1.0, label=f'Smoothed Delta {i+1}')
    axs[0].set_ylabel('State Delta Value')
    axs[0].legend(ncol=2)
    axs[0].grid(True)

    # Plot Control Commands (u*) (Time Domain)
    axs[1].set_title('Control Inputs (Commands u*) vs. Time')
    labels_cmd = ['Command u1', 'Command u2', 'Command u3']
    if len(control_history) > 0:
        for i in range(CONTROL_DIM):
            axs[1].plot(time_axis_control, control_history[:, i], label=labels_cmd[i], drawstyle='steps-post')
            # Add smoothed control if provided
            if smoothed_control is not None:
                axs[1].plot(time_axis_control, smoothed_control[:, i], '--', alpha=0.7,
                          label=f'Smoothed {labels_cmd[i]}')
        axs[1].axhline(u_min, color='r', linestyle='--', label=f'Min/Max Cmd ({u_min:.1f}, {u_max:.1f})')
        axs[1].axhline(u_max, color='r', linestyle='--')
        axs[1].set_ylabel('Command Value')
        axs[1].legend(loc='best')
    else:
        axs[1].text(0.5, 0.5, 'No control history', ha='center', va='center')
    axs[1].grid(True)
    axs[1].set_xlabel('Time (s)')

    fig1.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

    # --- Figure 2: 3D Trajectory Plot ---
    fig2 = plt.figure(figsize=(10, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_xlim(-1, 5)
    ax3d.set_ylim(-5, 5)
    ax3d.set_zlim(-5, 5)
    ax3d.view_init(elev=-50, azim=-100, roll=-80)

    # Plot Point Cloud (optional, can be slow if large)
    if point_cloud is not None and len(point_cloud) > 0:
         # Downsample point cloud for plotting if too large
         plot_pc_step = max(1, len(point_cloud) // 5000) # Aim for ~5000 points
         ax3d.scatter(point_cloud[::plot_pc_step, 0], point_cloud[::plot_pc_step, 1], point_cloud[::plot_pc_step, 2],
                      c='gray', marker='.', s=1, label='Workspace (Sampled)')

    # Plot Reference Trajectory (plot the part corresponding to the simulation length)
    plot_ref_len = num_plot_steps # Match length of state history
    ax3d.plot(reference_history[:plot_ref_len, 0], reference_history[:plot_ref_len, 1], reference_history[:plot_ref_len, 2],
              'r--', linewidth=2, label='Reference Trajectory')
    ax3d.scatter(reference_history[0, 0], reference_history[0, 1], reference_history[0, 2],
                 c='red', marker='o', s=50, label='Reference Start')
    # Use the reference point corresponding to the last state for the end marker
    ax3d.scatter(reference_history[plot_ref_len-1, 0], reference_history[plot_ref_len-1, 1], reference_history[plot_ref_len-1, 2],
                 c='red', marker='x', s=100, label='Reference End (Sim)')

    # Plot Actual Trajectory
    ax3d.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2],
              'b-', linewidth=2, label='Actual Trajectory')
    ax3d.scatter(state_history[0, 0], state_history[0, 1], state_history[0, 2],
                 c='blue', marker='o', s=50, label='Actual Start')
    ax3d.scatter(state_history[-1, 0], state_history[-1, 1], state_history[-1, 2],
                 c='blue', marker='x', s=100, label='Actual End')
                 
    # Plot Smoothed Trajectory if provided
    if smoothed_state_history is not None:
        ax3d.plot(smoothed_state_history[:, 0], smoothed_state_history[:, 1], smoothed_state_history[:, 2],
                'g-', linewidth=1.5, alpha=0.7, label='Smoothed Trajectory')
        ax3d.scatter(smoothed_state_history[-1, 0], smoothed_state_history[-1, 1], smoothed_state_history[-1, 2],
                   c='green', marker='x', s=80, label='Smoothed End')

    # Setting Labels and Title
    ax3d.set_xlabel('Delta X')
    ax3d.set_ylabel('Delta Y')
    ax3d.set_zlabel('Delta Z')
    ax3d.set_title('3D Trajectory Comparison (MPC)') # Updated title

    # Adjust plot limits to fit data (optional, helps visualization)
    all_points = np.vstack((state_history, reference_history[:plot_ref_len])) # Use plotted reference part
    if smoothed_state_history is not None:
        all_points = np.vstack((all_points, smoothed_state_history))
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    center = (max_coords + min_coords) / 2
    max_range = (max_coords - min_coords).max() / 2.0 * 1.1 # 10% padding
    if max_range < 1e-6: max_range = 1.0 # Avoid zero range if points overlap
    ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
    ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
    ax3d.set_zlim(center[2] - max_range, center[2] + max_range)

    ax3d.legend()
    ax3d.grid(True)

    plt.show() # Show both figures

def smooth_control(control_history, window_size=10):
    """
    Smooth control trajectory using moving average filter.
    
    Args:
        control_history: Original control inputs [N, control_dim]
        window_size: Size of moving average window
    
    Returns:
        Smoothed control trajectory with same dimensions
    """
    control_history = np.array(control_history)
    smoothed_control = np.copy(control_history)
    
    # Ensure window size is valid
    window_size = min(window_size, len(control_history))
    if window_size < 2:
        return control_history  # No smoothing possible
    
    # Apply moving average for each control dimension
    for i in range(control_history.shape[1]):
        # Using convolution for moving average
        kernel = np.ones(window_size) / window_size
        # Pad to prevent edge effects
        padded = np.pad(control_history[:, i], (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        # Ensure output length matches input length
        smoothed_control[:, i] = smoothed[:len(control_history)]
    
    return smoothed_control


# --- Script Execution ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    elif not os.path.exists(SCALERS_PATH): print(f"FATAL ERROR: Scalers file not found at {SCALERS_PATH}")
    else: run_simulation()