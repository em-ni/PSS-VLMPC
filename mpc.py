import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tests.test_blob_sampling import load_point_cloud_from_csv
import src.config as config 
from utils.traj_functions import generate_snapped_trajectory
from utils.nn_functions import load_model_and_scalers
from utils.mpc_functions import predict_delta_from_volume, solve_for_optimal_volume

# --- Import constants from config ---
STATE_DIM = config.STATE_DIM
CONTROL_DIM = config.CONTROL_DIM
VOLUME_DIM = config.VOLUME_DIM
DT = config.DT
T_SIM = config.T_SIM
N_sim_steps = config.N_sim_steps
V_REST = config.V_REST
VOLUME_BOUNDS_LIST = config.VOLUME_BOUNDS_LIST
U_MIN_CMD = config.U_MIN_CMD
U_MAX_CMD = config.U_MAX_CMD
Q_matrix = config.Q_matrix
R_matrix = config.R_matrix
OPTIMIZER_METHOD = config.OPTIMIZER_METHOD
PERTURBATION_SCALE = config.PERTURBATION_SCALE
MODEL_PATH = config.MODEL_PATH
SCALERS_PATH = config.SCALERS_PATH
POINT_CLOUD_PATH = config.POINT_CLOUD_PATH

# --- Main Simulation ---
def run_simulation():
    """Runs the single-step optimization control loop following a generated trajectory."""
    nn_model, scaler_volumes, scaler_deltas, nn_device = load_model_and_scalers(MODEL_PATH, SCALERS_PATH)
    if nn_model is None: return

    # 1. Load Point Cloud
    try:
        point_cloud = load_point_cloud_from_csv(POINT_CLOUD_PATH)
        if len(point_cloud) == 0: raise ValueError("Point cloud file is empty.")
        print(f"Loaded point cloud with {len(point_cloud)} points.")
    except Exception as e:
        print(f"FATAL: Error loading point cloud '{POINT_CLOUD_PATH}': {e}. Cannot generate trajectory.")
        return

    # 2. Generate Trajectory
    num_traj_waypoints = 3
    delta_ref_trajectory = generate_snapped_trajectory(
        point_cloud, num_traj_waypoints, T_SIM, N_sim_steps, from_initial_pos=True
    )

    # 3. Initialize state and history
    x_current_delta = predict_delta_from_volume(V_REST, nn_model, scaler_volumes, scaler_deltas, nn_device)
    if np.isnan(x_current_delta).any(): print("FATAL: Failed to predict initial state."); return
    print(f"Initial State Delta (predicted for v_rest): {x_current_delta}")
    print(f"Cost Weights: Q_diag={np.diag(Q_matrix)}, R_diag={np.diag(R_matrix)}")
    print(f"Volume Bounds: {VOLUME_BOUNDS_LIST[0]}")
    print("-" * 60)
    print(f"Starting Simulation Loop (Optimizer: {OPTIMIZER_METHOD}, Following Trajectory)...")
    print("-" * 30)

    state_history = [x_current_delta]; control_history = []; volume_history = [V_REST]; computation_times = []
    current_actual_volume = V_REST.copy()

    # --- Simulation Loop ---
    for i in range(N_sim_steps): # Loop from 0 to N_sim_steps-1
        step_start_time = time.time()

        # *** The target for control u_i applied at step i should be the reference at step i+1 ***
        delta_ref_current = delta_ref_trajectory[i+1] # Target state for the *end* of this step

        v_star = solve_for_optimal_volume(
            delta_ref_current, Q_matrix, R_matrix, VOLUME_BOUNDS_LIST, V_REST,
            nn_model, scaler_volumes, scaler_deltas, nn_device,
            v_guess_init=current_actual_volume,
            method=OPTIMIZER_METHOD,
            perturbation_scale=PERTURBATION_SCALE
        )
        u_star = v_star - V_REST
        u_star_clipped = np.clip(u_star, U_MIN_CMD, U_MAX_CMD)
        v_actual = V_REST + u_star_clipped
        current_actual_volume = v_actual # Update warm start base

        comp_time = time.time() - step_start_time
        computation_times.append(comp_time)
        control_history.append(u_star_clipped)
        volume_history.append(v_actual)

        # Simulate the system's next state (state at step i+1) based on the actual volume/command applied at step i
        x_next_delta = predict_delta_from_volume(v_actual, nn_model, scaler_volumes, scaler_deltas, nn_device)
        if np.isnan(x_next_delta).any(): print(f"FATAL: NaN prediction at step {i+1}. Stopping."); break

        # Update state for the next iteration (this is now state at i+1)
        x_current_delta = x_next_delta
        state_history.append(x_current_delta)

        # Print progress: Compare state at i+1 with target at i+1
        current_error_norm = np.linalg.norm(x_current_delta - delta_ref_current)
        print(f"Step {i+1}/{N_sim_steps} | State Delta: [{x_current_delta[0]:.3f}, {x_current_delta[1]:.3f}, {x_current_delta[2]:.3f}] | Target: [{delta_ref_current[0]:.3f},{delta_ref_current[1]:.3f},{delta_ref_current[2]:.3f}] | ErrNorm: {current_error_norm:.3f} | Cmd u*: [{u_star_clipped[0]:.3f}, {u_star_clipped[1]:.3f}, {u_star_clipped[2]:.3f}] | Time: {comp_time:.4f}s")

    # --- Final Summary ---
    print("-" * 30); print("Simulation finished.")
    if computation_times: print(f"Average/Maximum computation time per optimization step: {np.mean(computation_times):.4f}s / {np.max(computation_times):.4f}s")
    final_target = delta_ref_trajectory[-1]
    final_error = np.linalg.norm(x_current_delta - final_target)
    print(f"Final State Delta: {x_current_delta}"); print(f"Final Target Delta: {final_target}"); print(f"Final Error Norm (Delta): {final_error:.4f}")

    # --- Plotting Results ---
    plot_results(state_history, control_history, delta_ref_trajectory, point_cloud, DT, U_MIN_CMD, U_MAX_CMD)

# --- Plotting Function ---
def plot_results(state_history, control_history, delta_ref_trajectory, point_cloud, dt, u_min, u_max):
    """Plots the state delta (2D and 3D) and control command history."""
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    num_plot_steps = len(state_history)
    reference_history = delta_ref_trajectory[:num_plot_steps]

    time_axis_state = np.arange(num_plot_steps) * dt
    time_axis_control = np.arange(len(control_history)) * dt

    # --- Figure 1: 2D Plots (State vs Time, Control vs Time) ---
    fig1, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig1.suptitle('Trajectory Following Control Results')

    # Plot State Deltas vs Reference Trajectory (Time Domain)
    axs[0].set_title('State Delta vs. Time')
    labels_delta = ['Actual Delta X', 'Actual Delta Y', 'Actual Delta Z']
    labels_ref = ['Ref Delta X', 'Ref Delta Y', 'Ref Delta Z']
    colors = plt.cm.viridis(np.linspace(0, 0.8, STATE_DIM))
    for i in range(STATE_DIM):
        axs[0].plot(time_axis_state, state_history[:, i], 'o-', color=colors[i], markersize=3, linewidth=1.5, label=labels_delta[i])
        axs[0].plot(time_axis_state, reference_history[:, i], '--', color=colors[i], linewidth=2.0, label=labels_ref[i])
    axs[0].set_ylabel('State Delta Value')
    axs[0].legend(ncol=2)
    axs[0].grid(True)

    # Plot Control Commands (u*) (Time Domain)
    axs[1].set_title('Control Inputs (Commands u*) vs. Time')
    labels_cmd = ['Command u1', 'Command u2', 'Command u3']
    if len(control_history) > 0:
        for i in range(CONTROL_DIM):
            axs[1].plot(time_axis_control, control_history[:, i], label=labels_cmd[i], drawstyle='steps-post')
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

    # Plot Point Cloud (optional, can be slow if large)
    if point_cloud is not None and len(point_cloud) > 0:
         # Downsample point cloud for plotting if too large
         plot_pc_step = max(1, len(point_cloud) // 5000) # Aim for ~5000 points
         ax3d.scatter(point_cloud[::plot_pc_step, 0], point_cloud[::plot_pc_step, 1], point_cloud[::plot_pc_step, 2],
                      c='lightgray', marker='.', s=1, label='Workspace (Sampled)')

    # Plot Reference Trajectory
    ax3d.plot(reference_history[:, 0], reference_history[:, 1], reference_history[:, 2],
              'r--', linewidth=2, label='Reference Trajectory')
    ax3d.scatter(reference_history[0, 0], reference_history[0, 1], reference_history[0, 2],
                 c='red', marker='o', s=50, label='Reference Start')
    ax3d.scatter(reference_history[-1, 0], reference_history[-1, 1], reference_history[-1, 2],
                 c='red', marker='x', s=100, label='Reference End')

    # Plot Actual Trajectory
    ax3d.plot(state_history[:, 0], state_history[:, 1], state_history[:, 2],
              'b-', linewidth=2, label='Actual Trajectory')
    ax3d.scatter(state_history[0, 0], state_history[0, 1], state_history[0, 2],
                 c='blue', marker='o', s=50, label='Actual Start')
    ax3d.scatter(state_history[-1, 0], state_history[-1, 1], state_history[-1, 2],
                 c='blue', marker='x', s=100, label='Actual End')

    # Setting Labels and Title
    ax3d.set_xlabel('Delta X')
    ax3d.set_ylabel('Delta Y')
    ax3d.set_zlabel('Delta Z')
    ax3d.set_title('3D Trajectory Comparison')

    # Adjust plot limits to fit data (optional, helps visualization)
    all_points = np.vstack((state_history, reference_history))
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    center = (max_coords + min_coords) / 2
    max_range = (max_coords - min_coords).max() / 2.0 * 1.1 # 10% padding
    ax3d.set_xlim(center[0] - max_range, center[0] + max_range)
    ax3d.set_ylim(center[1] - max_range, center[1] + max_range)
    ax3d.set_zlim(center[2] - max_range, center[2] + max_range)

    # Make axes equal for better 3D perception
    # ax3d.set_aspect('equal', adjustable='box') # Matplotlib might struggle with this

    ax3d.legend()
    ax3d.grid(True)

    plt.show() # Show both figures

# --- Script Execution ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    elif not os.path.exists(SCALERS_PATH): print(f"FATAL ERROR: Scalers file not found at {SCALERS_PATH}")
    else: run_simulation()