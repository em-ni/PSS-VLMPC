import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import threading
import signal
import time
import os

from utils.mpc_functions import load_trajectory_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pressure_loader import PressureLoader
import src.config as config
from src.robot_env import RobotEnv
from utils.circle_arc import calculate_circle_through_points

def umeyama_alignment(source_points, target_points):
    """
    Computes the similarity transformation (rotation, translation, scale)
    that aligns source_points to target_points using the Umeyama algorithm.

    Args:
        source_points (np.ndarray): NxM array of source points (M dimensions).
        target_points (np.ndarray): NxM array of target points (M dimensions).

    Returns:
        np.ndarray: (M+1)x(M+1) transformation matrix, or None if alignment fails.
    """
    if source_points.shape != target_points.shape:
        print("Error: Source and target point sets must have the same shape.")
        return None
    if source_points.shape[0] < source_points.shape[1] + 1:  # Need at least M+1 points for M dimensions for non-degenerate solution
        print(f"Error: Not enough points ({source_points.shape[0]}) for robust alignment in {source_points.shape[1]} dimensions. Need at least {source_points.shape[1]+1}.")
        return None

    N, M = source_points.shape

    mu_source = source_points.mean(axis=0)
    mu_target = target_points.mean(axis=0)

    source_centered = source_points - mu_source
    target_centered = target_points - mu_target

    Sigma_target_source = (target_centered.T @ source_centered) / N

    U, D_singular_vec, Vt = np.linalg.svd(Sigma_target_source)

    S_diag_matrix = np.eye(M)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_diag_matrix[-1, -1] = -1
    
    R = U @ S_diag_matrix @ Vt

    trace_Sigma_source_source = np.sum(np.mean(source_centered**2, axis=0)) 
    
    if trace_Sigma_source_source < 1e-9: 
        print("Warning: Variance of source points is close to zero. Assuming scale = 1.")
        c = 1.0
    else:
        c = np.trace(np.diag(D_singular_vec) @ S_diag_matrix) / trace_Sigma_source_source
        
    t = mu_target - c * R @ mu_source

    T_matrix = np.eye(M + 1)
    T_matrix[:M, :M] = c * R
    T_matrix[:M, M] = t
    
    return T_matrix

def main():
    trajectory_file = config.TRAJ_DIR
    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file '{trajectory_file}' not found.")
        print("Please run mpc.py first to generate the trajectory file.")
        return

    ref_trajectory, control_inputs = load_trajectory_data(trajectory_file)
    if ref_trajectory is None or control_inputs is None:
        return
    
    print(f"Loaded reference trajectory with shape {ref_trajectory.shape}")
    print(f"Loaded control inputs with shape {control_inputs.shape}")

    offsets = []
    pressure_loader = PressureLoader()
    offsets = pressure_loader.load_pressure()
    
    env = RobotEnv()
    
    signal_handler = env.robot_api.get_signal_handler()
    signal.signal(signal.SIGINT, signal_handler)
    
    tracker_thread = threading.Thread(
        target=env.robot_api.update_tracker, 
        args=(env.robot_api.get_tracker(),)
    )
    tracker_thread.daemon = True
    tracker_thread.start()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("MPC Trajectory Following (Live)")
    ax.view_init(elev=-50, azim=-100, roll=-80)
    
    base_scatter = ax.scatter([], [], [], s=40, c="yellow", label="Base")
    tip_scatter = ax.scatter([], [], [], s=60, c="red", label="Current Tip")
    body_scatter = ax.scatter([], [], [], s=5, c="blue", label="Body")
    
    ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
            'g--', linewidth=2, label="Planned Trajectory")
    ax.scatter([ref_trajectory[0, 0]], [ref_trajectory[0, 1]], [ref_trajectory[0, 2]], 
              s=80, c="green", label="Start")
    ax.scatter([ref_trajectory[-1, 0]], [ref_trajectory[-1, 1]], [ref_trajectory[-1, 2]], 
              s=80, c="green", marker='x', label="Goal")
    ax.legend()
    
    total_steps = len(control_inputs)
    step_time = config.DT
    
    step_info = {"current_display_step_idx": 0, "completed": False}
    error_history_display = [] 
    position_history_plot = [] 

    executed_trajectory_points = []
    planned_trajectory_points_for_alignment = []
    
    step_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    error_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)
    
    def control_loop_thread_fn(env_instance, ctrl_inputs, ref_traj, press_offsets, num_steps, dt, shared_step_info, exec_pts_list, planned_pts_list):
        current_step_idx = 0
        print("Starting trajectory following control loop...")
        
        while current_step_idx < num_steps:
            shared_step_info["current_display_step_idx"] = current_step_idx
            
            # Skip command if step number is even (0-indexed)
            # This means actual control is applied on odd steps 1, 3, 5...
            if current_step_idx % 2 == 0: 
                time.sleep(dt) # Still wait for the step time
                current_step_idx += 1
                continue
            
            start_time_of_step = time.time()
            
            u_command_current = ctrl_inputs[current_step_idx]
            u_command_current = u_command_current + press_offsets
            env_instance.robot_api.send_command(u_command_current)
            
            elapsed_for_command = time.time() - start_time_of_step
            sleep_for_dt = max(0, dt - elapsed_for_command)
            time.sleep(sleep_for_dt) 

            base_state = env_instance.robot_api.get_tracker().get_current_base()
            tip_state = env_instance.robot_api.get_tracker().get_current_tip()

            if base_state is not None and tip_state is not None:
                base_pos_vec = base_state.ravel()
                tip_pos_vec = tip_state.ravel()
                current_executed_delta = tip_pos_vec - base_pos_vec
                
                exec_pts_list.append(current_executed_delta)
                planned_pts_list.append(ref_traj[current_step_idx]) 
            
            current_step_idx += 1
            
        print("Trajectory control loop complete. Close the window to stop the animation.")
        shared_step_info["completed"] = True
    
    ctrl_thread = threading.Thread(
        target=control_loop_thread_fn, 
        args=(env, control_inputs, ref_trajectory, offsets, total_steps, step_time, 
              step_info, executed_trajectory_points, planned_trajectory_points_for_alignment)
    )
    ctrl_thread.daemon = True
    ctrl_thread.start()
    
    # Store lines for actual path and circle to remove them more reliably
    actual_path_line = None
    circle_line = None

    def animate(frame_num):
        nonlocal actual_path_line, circle_line
        try:
            base = env.robot_api.get_tracker().get_current_base()
            tip = env.robot_api.get_tracker().get_current_tip()
            body = env.robot_api.get_tracker().get_current_body()
            
            base_x, base_y, base_z = [], [], []
            tip_x_abs, tip_y_abs, tip_z_abs = [], [], [] # Absolute tip coordinates
            body_x_abs, body_y_abs, body_z_abs = [], [], [] # Absolute body coordinates
            
            current_pos_delta = None

            if base is not None:
                base = base.ravel()
                base_x, base_y, base_z = [base[0]], [base[1]], [base[2]]
                
            if tip is not None:
                tip = tip.ravel()
                tip_x_abs, tip_y_abs, tip_z_abs = [tip[0]], [tip[1]], [tip[2]]
                
            if body is not None:
                body = body.ravel()
                body_x_abs, body_y_abs, body_z_abs = [body[0]], [body[1]], [body[2]]
            
            # Calculate delta positions for plotting (relative to base)
            delta_tip_x, delta_tip_y, delta_tip_z = [], [], []
            if base_x and tip_x_abs:
                delta_tip_x = [tip_x_abs[0] - base_x[0]]
                delta_tip_y = [tip_y_abs[0] - base_y[0]]
                delta_tip_z = [tip_z_abs[0] - base_z[0]]
                current_pos_delta = np.array([delta_tip_x[0], delta_tip_y[0], delta_tip_z[0]])
                if len(position_history_plot) == 0 or not np.allclose(position_history_plot[-1], current_pos_delta):
                    position_history_plot.append(current_pos_delta)

            anim_step_idx = step_info["current_display_step_idx"]
            if not step_info["completed"] and anim_step_idx < total_steps and current_pos_delta is not None:
                # Use the target for the current display step for error calculation
                target_pos_display = ref_trajectory[anim_step_idx] 
                error_val = np.linalg.norm(current_pos_delta - target_pos_display)
                if len(error_history_display) == 0 or error_history_display[-1] != error_val :
                     error_history_display.append(error_val)
                
                step_text.set_text(f"Step: {anim_step_idx+1}/{total_steps}")
                error_text.set_text(f"Error: {error_val:.4f}")
            elif step_info["completed"]:
                step_text.set_text(f"Step: {total_steps}/{total_steps} (Done)")
                if error_history_display: error_text.set_text(f"Final Error: {error_history_display[-1]:.4f}")
            else:
                step_text.set_text(f"Step: {anim_step_idx+1}/{total_steps}") # Handles pre-start or if current_pos_delta is None
                error_text.set_text(f"Error: N/A")

            delta_body_x, delta_body_y, delta_body_z = [], [], []
            if base_x and body_x_abs:
                delta_body_x = [body_x_abs[0]-base_x[0]]
                delta_body_y = [body_y_abs[0]-base_y[0]]
                delta_body_z = [body_z_abs[0]-base_z[0]]
            
            base_scatter._offsets3d = ([0], [0], [0]) 
            tip_scatter._offsets3d = (delta_tip_x, delta_tip_y, delta_tip_z)
            body_scatter._offsets3d = (delta_body_x, delta_body_y, delta_body_z)
            
            if actual_path_line:
                actual_path_line.pop(0).remove()
                actual_path_line = None
            if circle_line:
                circle_line.pop(0).remove()
                circle_line = None
            
            if base is not None and tip is not None and body is not None:
                body_delta_for_circle = body - base
                tip_delta_for_circle = tip - base
                origin_delta_for_circle = np.array([0,0,0]) # Base is origin in delta space
                if body_delta_for_circle.ndim == 1 and tip_delta_for_circle.ndim == 1:
                     circle_points = calculate_circle_through_points(body_delta_for_circle, tip_delta_for_circle, origin_delta_for_circle)
                     circle_line = ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
                                           color="cyan", linewidth=1, alpha=0.5) 
            
            if len(position_history_plot) > 1:
                positions_np = np.array(position_history_plot)
                actual_path_line = ax.plot(positions_np[:, 0], positions_np[:, 1], positions_np[:, 2], 
                                           'r-', linewidth=1, alpha=0.7, label="Actual Path (Live)") # Give it a unique label
                # Ensure legend is updated if new items are added dynamically (though "Actual Path (Live)" is added once)
                handles, labels = ax.get_legend_handles_labels()
                # Filter out duplicate "Actual Path (Live)" if any before re-creating legend
                unique_labels = {}
                unique_handles = []
                for handle, label in zip(handles, labels):
                    if label not in unique_labels or label not in ["Actual Path (Live)"]:
                        unique_labels[label] = handle
                        unique_handles.append(handle)
                if not any(label == "Actual Path (Live)" for label in unique_labels) and actual_path_line:
                     # This logic might be tricky with FuncAnimation; often better to set all labels initially
                     pass # ax.legend() might be called once outside if all labels are known

            if frame_num % 50 == 0:
                import gc
                gc.collect()
            
            return base_scatter, tip_scatter, body_scatter, step_text, error_text, *(actual_path_line or []), *(circle_line or [])
            
        except Exception as e:
            # print(f"Animation error: {e}") 
            return base_scatter, tip_scatter, body_scatter, step_text, error_text
    
    anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, interval=50, blit=False)
    
    plt.tight_layout()
    plt.show() 
    
    ctrl_thread.join(timeout=10.0) 
    if ctrl_thread.is_alive():
        print("Warning: Control thread did not terminate cleanly.")

    if error_history_display:
        print("\nTrajectory Following Displayed Stats (Live Animation):")
        print(f"Average Displayed Error: {np.mean(error_history_display):.4f}")
        print(f"Maximum Displayed Error: {np.max(error_history_display):.4f}")
        if error_history_display:
            print(f"Final Displayed Error: {error_history_display[-1]:.4f}")

    transformation_matrix = None
    P_executed = np.array([])
    P_planned_targets = np.array([])

    if executed_trajectory_points and planned_trajectory_points_for_alignment:
        P_executed = np.array(executed_trajectory_points)
        P_planned_targets = np.array(planned_trajectory_points_for_alignment)

        print(f"\nPerforming Umeyama alignment with {P_executed.shape[0]} point pairs.")
        
        if P_executed.shape[0] > 0 and P_executed.shape == P_planned_targets.shape and P_executed.shape[0] >= P_executed.shape[1] + 1 :
            transformation_matrix = umeyama_alignment(P_executed, P_planned_targets)
            
            if transformation_matrix is not None:
                print("\nTransformation Matrix (from executed to planned):")
                print(transformation_matrix)
                
                output_filename = "transformation_matrix.npy"
                np.save(output_filename, transformation_matrix)
                print(f"Transformation matrix saved to {output_filename}")

                P_executed_homogeneous = np.hstack((P_executed, np.ones((P_executed.shape[0], 1))))
                P_executed_aligned_homogeneous = (transformation_matrix @ P_executed_homogeneous.T).T
                P_executed_aligned = P_executed_aligned_homogeneous[:, :3]
                
                alignment_errors = np.linalg.norm(P_executed_aligned - P_planned_targets, axis=1)
                print(f"Mean alignment error after transformation: {np.mean(alignment_errors):.4f}")
                print(f"Max alignment error after transformation: {np.max(alignment_errors):.4f}")
            else:
                print("Umeyama alignment failed.")
        else:
            print("Not enough valid points or mismatched shapes for Umeyama alignment.")
            print(f"Shape of collected executed points: {P_executed.shape}")
            print(f"Shape of collected planned target points: {P_planned_targets.shape}")
            if P_executed.shape[0] > 0 :
                 print(f"Required minimum points for {P_executed.shape[1]} dimensions: {P_executed.shape[1]+1}")
    else:
        print("\nNo points collected for Umeyama alignment.")

    # Plotting the alignment result
    if transformation_matrix is not None and P_executed.shape[0] > 0:
        P_executed_homogeneous = np.hstack((P_executed, np.ones((P_executed.shape[0], 1))))
        P_executed_aligned_homogeneous = (transformation_matrix @ P_executed_homogeneous.T).T
        P_executed_aligned = P_executed_aligned_homogeneous[:, :3]

        fig_align = plt.figure(figsize=(12, 10))
        ax_align = fig_align.add_subplot(111, projection='3d')
        ax_align.set_title("Trajectory Alignment Visualization")
        ax_align.set_xlabel("X (delta)")
        ax_align.set_ylabel("Y (delta)")
        ax_align.set_zlabel("Z (delta)")

        ax_align.plot(P_planned_targets[:, 0], P_planned_targets[:, 1], P_planned_targets[:, 2],
                      'g--', linewidth=2, label="Planned Trajectory (Targets)")
        if P_planned_targets.shape[0] > 0:
            ax_align.scatter(P_planned_targets[0, 0], P_planned_targets[0, 1], P_planned_targets[0, 2],
                             s=100, c='green', marker='o', label="Planned Start", depthshade=False)
            ax_align.scatter(P_planned_targets[-1, 0], P_planned_targets[-1, 1], P_planned_targets[-1, 2],
                             s=100, c='darkgreen', marker='X', label="Planned End", depthshade=False)

        ax_align.plot(P_executed[:, 0], P_executed[:, 1], P_executed[:, 2],
                      'r-', linewidth=1.5, alpha=0.7, label="Original Executed Trajectory")
        if P_executed.shape[0] > 0:
            ax_align.scatter(P_executed[0, 0], P_executed[0, 1], P_executed[0, 2],
                             s=100, c='red', marker='o', label="Executed Start", depthshade=False)
            ax_align.scatter(P_executed[-1, 0], P_executed[-1, 1], P_executed[-1, 2],
                             s=100, c='darkred', marker='X', label="Executed End", depthshade=False)
        
        ax_align.plot(P_executed_aligned[:, 0], P_executed_aligned[:, 1], P_executed_aligned[:, 2],
                      'b-', linewidth=1.5, alpha=0.9, label="Aligned Executed Trajectory")
        if P_executed_aligned.shape[0] > 0:
            ax_align.scatter(P_executed_aligned[0, 0], P_executed_aligned[0, 1], P_executed_aligned[0, 2],
                             s=100, c='blue', marker='o', label="Aligned Start", depthshade=False)
            ax_align.scatter(P_executed_aligned[-1, 0], P_executed_aligned[-1, 1], P_executed_aligned[-1, 2],
                             s=100, c='darkblue', marker='X', label="Aligned End", depthshade=False)

        all_points_for_limits = np.vstack((P_planned_targets, P_executed, P_executed_aligned))
        if all_points_for_limits.shape[0] > 0:
            min_coords = all_points_for_limits.min(axis=0)
            max_coords = all_points_for_limits.max(axis=0)
            ax_align.set_xlim(min_coords[0] - 0.5, max_coords[0] + 0.5)
            ax_align.set_ylim(min_coords[1] - 0.5, max_coords[1] + 0.5)
            ax_align.set_zlim(min_coords[2] - 0.5, max_coords[2] + 0.5)
        
        ax_align.legend()
        ax_align.view_init(elev=-50, azim=-100, roll=-80) # Match initial view
        plt.tight_layout()
        plt.show()
    elif P_executed.shape[0] == 0:
        print("Skipping alignment plot as no executed points were collected.")
    else:
        print("Skipping alignment plot as transformation matrix was not computed.")


    if tracker_thread.is_alive():
        # Assuming RobotAPI's update_tracker loop checks a stop flag similar to control_loop_thread_fn
        # or relies on daemon=True. For explicit cleanup, RobotAPI would need a stop method for its tracker loop.
        print("Tracker thread is still alive. Relaying on daemon property for exit.")
    print("Robot calibration finished.")

    # Move to initial position after completing trajectory
    env.robot_api.reset_robot()

if __name__ == "__main__":
    main()