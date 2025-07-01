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

def apply_transformation(points_3d, matrix_4x4):
    """
    Applies a 4x4 transformation matrix to a list of 3D points.
    Args:
        points_3d (np.ndarray): Nx3 array of 3D points.
        matrix_4x4 (np.ndarray): 4x4 transformation matrix.
    Returns:
        np.ndarray: Nx3 array of transformed 3D points.
    """
    if points_3d.ndim == 1: # Single point
        points_3d = points_3d.reshape(1, -1)
    if points_3d.shape[1] != 3:
        raise ValueError("Input points must be Nx3.")
    if matrix_4x4.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4.")

    num_points = points_3d.shape[0]
    points_homogeneous = np.hstack((points_3d, np.ones((num_points, 1)))) # Nx4
    
    transformed_points_homogeneous = (matrix_4x4 @ points_homogeneous.T).T # (4x4 @ 4xN).T = Nx4
    
    transformed_points_3d = transformed_points_homogeneous[:, :3]
    if transformed_points_3d.shape[0] == 1: # Single point back to 1D
        return transformed_points_3d.ravel()
    return transformed_points_3d


def main():
    # Check if trajectory file exists
    trajectory_file = config.TRAJ_DIR
    if not os.path.exists(trajectory_file):
        print(f"Error: Trajectory file '{trajectory_file}' not found.")
        print("Please run mpc.py first to generate the trajectory file.")
        return

    # Load trajectory data
    ref_trajectory, control_inputs = load_trajectory_data(trajectory_file)
    if ref_trajectory is None or control_inputs is None:
        return
    
    # Load transformation matrix
    transformation_matrix_file = "transformation_matrix.npy"
    T_exec_to_plan = np.load(transformation_matrix_file)
    print(f"Loaded transformation matrix (T_exec_to_plan) from {transformation_matrix_file}:")
    print(T_exec_to_plan)

    # # Apply transformation to reference trajectory
    # ref_trajectory = apply_transformation(ref_trajectory, T_exec_to_plan)
    # print("Transformed reference trajectory")

    
    print(f"Loaded reference trajectory with shape {ref_trajectory.shape}")
    print(f"Loaded control inputs with shape {control_inputs.shape}")

    # Load pressure 
    offsets = []
    pressure_loader = PressureLoader()
    offsets = pressure_loader.load_pressure()
    
    # Create robot environment
    env = RobotEnv()
    
    # Register signal handler for Ctrl+C
    signal_handler = env.robot_api.get_signal_handler()
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start tracker thread
    tracker_thread = threading.Thread(
        target=env.robot_api.update_tracker, 
        args=(env.robot_api.get_tracker(),)
    )
    tracker_thread.daemon = True
    tracker_thread.start()
    
    # Set up the figure for 3D plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("MPC Trajectory Following")
    ax.view_init(elev=-50, azim=-100, roll=-80)
    
    # Create scatter plots for visualization
    base_scatter = ax.scatter([], [], [], s=40, c="yellow", label="Base")
    tip_scatter = ax.scatter([], [], [], s=60, c="red", label="Current Tip")
    body_scatter = ax.scatter([], [], [], s=5, c="blue", label="Body")
    
    # Plot the reference trajectory
    ax.plot(ref_trajectory[:, 0], ref_trajectory[:, 1], ref_trajectory[:, 2], 
            'g--', linewidth=2, label="Planned Trajectory")
    ax.scatter([ref_trajectory[0, 0]], [ref_trajectory[0, 1]], [ref_trajectory[0, 2]], 
              s=80, c="green", label="Start")
    ax.scatter([ref_trajectory[-1, 0]], [ref_trajectory[-1, 1]], [ref_trajectory[-1, 2]], 
              s=80, c="green", marker='x', label="Goal")
    
    # Add legend
    ax.legend()
    
    # Initialize variables for tracking
    current_step = 0
    total_steps = len(control_inputs)
    step_time = config.DT  # Time between steps from config
    
    # Create a mutable container for step info that can be updated in the animation function
    step_info = {"current": 0}
    error_history = []
    position_history = []
    
    # Text objects for displaying info
    step_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    error_text = ax.text2D(0.02, 0.90, "", transform=ax.transAxes)
    
    # Start a separate thread to apply control inputs
    def control_thread():
        step = 0
        v_rest = np.array(config.V_REST)
        
        print("Starting trajectory following...")
        
        while step < total_steps:
            # Skip command if step number is even
            if step % 2 == 0:
                step_info["current"] = step
                time.sleep(step_time)
                step += 1
                continue
            start_time = time.time()
            
            # Get control input for current step
            u_command = control_inputs[step]

            # Add offset to control input
            u_command = u_command + offsets

            # Apply control to the robot
            env.robot_api.send_command(u_command)
            print(f"Step {step+1}/{total_steps}: Applying control input {u_command}")            
            # Update step info for display
            step_info["current"] = step
            
            # Wait for next step (accounting for computation time)
            elapsed = time.time() - start_time
            sleep_time = max(0, step_time - elapsed)
            time.sleep(sleep_time)
            
            step += 1
            
        print("Trajectory complete")
    
    # Start control thread
    ctrl_thread = threading.Thread(target=control_thread)
    ctrl_thread.daemon = True
    ctrl_thread.start()
    
    # Animation function to update the plot
    def animate(frame):
        try:
            # Get current state from robot tracker
            base = env.robot_api.get_tracker().get_current_base()
            tip = env.robot_api.get_tracker().get_current_tip()
            body = env.robot_api.get_tracker().get_current_body()
            
            # Initialize position arrays
            base_x, base_y, base_z = [], [], []
            tip_x, tip_y, tip_z = [], [], []
            body_x, body_y, body_z = [], [], []
            
            # Process base position
            if base is not None:
                base = base.ravel()
                base_x, base_y, base_z = [base[0]], [base[1]], [base[2]]
                
            # Process tip position
            if tip is not None:
                tip = tip.ravel()
                tip_x, tip_y, tip_z = [tip[0]], [tip[1]], [tip[2]]
                
            # Process body position
            if body is not None:
                body = body.ravel()
                body_x, body_y, body_z = [body[0]], [body[1]], [body[2]]
            
            # Calculate delta positions relative to base
            if base_x and base_y and base_z and tip_x and tip_y and tip_z:
                dif_x, dif_y, dif_z = [tip_x[0] - base_x[0]], [tip_y[0] - base_y[0]], [tip_z[0] - base_z[0]]
                
                # Track actual position for analysis
                current_pos = np.array([dif_x[0], dif_y[0], dif_z[0]])

                # Apply transformation to current position
                current_pos = apply_transformation(current_pos, T_exec_to_plan)

                # Append to position history
                position_history.append(current_pos)
                
                # Calculate error if we're following the trajectory
                current_step = step_info["current"]
                if current_step < total_steps:
                    target_pos = ref_trajectory[current_step]
                    error = np.linalg.norm(current_pos - target_pos)
                    error_history.append(error)
                    
                    # Update UI with current step and error
                    step_text.set_text(f"Step: {current_step+1}/{total_steps}")
                    error_text.set_text(f"Error: {error:.4f}")
            else:
                dif_x, dif_y, dif_z = [], [], []
                
            if body_x and body_y and body_z:
                body_dif_x, body_dif_y, body_dif_z = [body_x[0]-base_x[0]], [body_y[0]-base_y[0]], [body_z[0]-base_z[0]]
            else:
                body_dif_x, body_dif_y, body_dif_z = [], [], []
            
            # Transform the points before plotting
            origin_base = apply_transformation(np.array([0, 0, 0]), T_exec_to_plan) # origin_base is a (3,) array

            # Transform dif_x, dif_y, dif_z if they contain data
            if dif_x: # Check if dif_x list is not empty (i.e., data was available for tip)
                point_to_transform_tip = np.array([dif_x[0], dif_y[0], dif_z[0]]) # Create (3,) array
                transformed_tip_point = apply_transformation(point_to_transform_tip, T_exec_to_plan) # Returns (3,) array
                # Update dif_x, dif_y, dif_z to be lists of single transformed coords
                dif_x = [transformed_tip_point[0]]
                dif_y = [transformed_tip_point[1]]
                dif_z = [transformed_tip_point[2]]
            # else: dif_x, dif_y, dif_z remain empty lists, e.g. [], [], []

            # Transform body_dif_x, body_dif_y, body_dif_z if they contain data
            if body_dif_x: # Check if body_dif_x list is not empty
                point_to_transform_body = np.array([body_dif_x[0], body_dif_y[0], body_dif_z[0]]) # Create (3,) array
                transformed_body_point = apply_transformation(point_to_transform_body, T_exec_to_plan) # Returns (3,) array
                # Update body_dif_x, body_dif_y, body_dif_z to be lists of single transformed coords
                body_dif_x = [transformed_body_point[0]]
                body_dif_y = [transformed_body_point[1]]
                body_dif_z = [transformed_body_point[2]]
            # else: body_dif_x, body_dif_y, body_dif_z remain empty lists
            
            # Update scatter plot positions
            base_scatter._offsets3d = ([origin_base[0]], [origin_base[1]], [origin_base[2]])
            tip_scatter._offsets3d = (dif_x, dif_y, dif_z)
            body_scatter._offsets3d = (body_dif_x, body_dif_y, body_dif_z)
            
            # Clear previous circle line
            for line in list(ax.get_lines()):
                if not line.get_label() == "Planned Trajectory":
                    line.remove()
            
            # Draw circle if we have all three points
            if base is not None and tip is not None and body is not None:
                circle_points = calculate_circle_through_points(body-base, tip-base, [0,0,0])
                ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
                        color="blue", linewidth=1, alpha=0.5)
            
            # If we have position history, plot the actual trajectory
            if len(position_history) > 1:
                positions = np.array(position_history)
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                        'r-', linewidth=1, alpha=0.7, label="Actual Path")
                
            # Garbage collection every 50 frames
            if frame % 50 == 0:
                import gc
                gc.collect()
            
            return base_scatter, tip_scatter, body_scatter, step_text, error_text
            
        except Exception as e:
            print(f"Animation error: {e}")
            return base_scatter, tip_scatter, body_scatter
    
    # Create the animation
    anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, interval=50)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # After animation ends, print statistics
    if error_history:
        print("\nTrajectory Following Results:")
        print(f"Average Error: {np.mean(error_history):.4f}")
        print(f"Maximum Error: {np.max(error_history):.4f}")
        print(f"Final Error: {error_history[-1]:.4f}")

if __name__ == "__main__":
    main()