# control.py
# 
# VLM Visual Control Integration:
# The VLM now receives visual context from the simulation, including:
# - Multi-view robot visualization (XY, XZ, 3D)
# - Current robot tip position and trajectory history
# - Target position and trajectory history  
# - Distance visualization between tip and target
#
# To use visual VLM control:
# 1. Start VLM server: ./llama-server -m ./models/smolvlm-500m-instruct-q4_k_m.gguf -ngl 99 --port 8080
# 2. Set CONTROL_MODE = "vlm" 
# 3. Run this script and type commands like "go right", "move up", etc.
# 4. The VLM will see the current scene and respond accordingly
#
import sys
import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
import signal
import atexit
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface
import numpy as np

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim
from src.VLM import VLM
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
mpc_project_path = os.path.join(project_root, 'sorolearn', 'generic-neural-mpc')
if mpc_project_path not in sys.path:
    sys.path.append(mpc_project_path)
from mpc_casadi_sim import MPCController

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
matplotlib.use('Qt5Agg')  # Use interactive backend

# Cleanup function
def cleanup():
    plt.close('all')

# Register cleanup
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda sig, frame: (cleanup(), sys.exit(0)))

# Simulation settings
REAL_TIME_PLOT = True
DEBUG_STATE_PREDICTION = False  # Enable to plot state prediction error
PLOT_EVERY_N_STEPS = 500
FINAL_TIME = 10
# CONTROL_MODE = "spr" # set point regulation
# CONTROL_MODE = "tt"  # trajectory tracking
CONTROL_MODE = "vlm"  # VLM control
APPROXIMATION_ORDER = 1

# rod parameters
simulation_params = {
    'n_elem': 30,
    'base_length': 0.4,
    'base_radius': 0.02,
    'density': 500,
    'youngs_modulus': 1e5,
    'poisson_ratio': 0.5,
    'damping_constant': 1e-1,
    'dt': 1e-4,
    'double_rod': True,
    'max_torque': 9e-2,
    'mpc_dt': 0.02
}
vlm_dt = 1.0

def plot_predictions(y_true, y_pred, save_path):
    """Generates Predicted vs. Actual plots and saves the figure to a file."""
    # These labels match the output of our new model, so they don't need to change.
    state_labels = [
        'Tip Position X', 'Tip Position Y', 'Tip Position Z',
        'Tip Velocity X', 'Tip Velocity Y', 'Tip Velocity Z'
    ]
    
    num_states = y_true.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(num_states):
        ax = axes[i]
        # Use a smaller subset of points for plotting if the test set is very large
        sample_size = min(len(y_true), 2000)
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        
        ax.scatter(y_true[indices, i], y_pred[indices, i], alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(state_labels[i], fontsize=14)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(pad=3.0)
    plt.suptitle("Predicted vs. Actual State Values during simulation", fontsize=20, y=1.02)
    
    # Save the plot to a file.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")
    plt.close(fig) # Close the figure to free up memory

def main():
    global CONTROL_MODE
    # init sim
    print("Initializing Constant Curvature Simulation...")
    cc_sim = Sim(**simulation_params)
    
    # Get simulation components
    rods_list = cc_sim.get_rods()
    callback_params = cc_sim.get_callback_params()
    
    # Print simulation info
    sim_info = cc_sim.get_simulation_info()
    print("\nSimulation Configuration:")
    for key, value in sim_info.items():
        print(f"  {key}: {value}")
        
    # Set real-time plot
    plotter = None
    if REAL_TIME_PLOT:
        print("Setting up real-time plotter...")
        plotter = RTPlotter(rods_list)
    
    # Initialize MPC controller
    mpc = MPCController(nn_approximation_order=APPROXIMATION_ORDER)
    
    if CONTROL_MODE == "vlm":
        # Initialize VLM for user input (set text_only_mode=False to enable images)
        vlm = VLM(vlm_dt=vlm_dt, mpc_dt=simulation_params['mpc_dt'], text_only_mode=True)
        
        # Check if VLM server is running
        if not vlm.check_server():
            print("Warning: VLM server not running!")
            print("To use VLM features, start the server with:")
            print("./llama-server -m ./models/smolvlm-500m-instruct-q4_k_m.gguf -ngl 99 --port 8080")
            print("Continuing without VLM control...")
            CONTROL_MODE = 'tt'
        else:
            print("VLM server connected successfully!")
            if vlm.text_only_mode:
                print("VLM running in TEXT-ONLY mode for debugging")
            else:
                print("VLM running in VISUAL mode with image processing")
            vlm.start_input_thread()
            
        # VLM trajectory variables
        vlm_trajectory = None
        vlm_trajectory_index = 0

    # --- Phase 3: The Simulation Loop ---
    print("\nStarting MPC simulation")
    
    # Define targets
    # default (downward position)
    x_target = np.array([0.0, 0.0, -len(rods_list) * simulation_params['base_length'], 0.0, 0.0, 0.0])
    # right
    x_target_right = np.array([0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
    # left
    x_target_left = np.array([-0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
    # up
    x_target_up = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
    # down
    x_target_down = np.array([0.0, -0.5, -0.5, 0.0, 0.0, 0.0])

    # Get initial state
    rods = cc_sim.get_rods()
    current_tip_pos = rods[-1].position_collection[:, -1]
    current_tip_vel = rods[-1].velocity_collection[:, -1]
    x_current = np.array(current_tip_pos.tolist() + current_tip_vel.tolist())

    if CONTROL_MODE == "tt":
        # Generate a reference trajectory of FINAL_TIME/simulation_params['mpc_dt'] steps
        hold_time = FINAL_TIME / 2
        hold_steps = int(hold_time / simulation_params['mpc_dt'])
        reference_trajectory = np.linspace(x_current, x_target, num=int((FINAL_TIME - hold_time) / simulation_params['mpc_dt']))
        trajctory_tail = np.tile(x_target, (hold_steps, 1))
        reference_trajectory = np.vstack((reference_trajectory, trajctory_tail))
        print(f"Generated reference trajectory with {len(reference_trajectory)} steps.")
        ref_index = 0
        x_target = reference_trajectory[ref_index] 

    # History variables
    current_control = np.zeros(4)  
    history_x, history_u = [], []
    state_pred_error_history = []
    history_x_current_test = []
    history_x_current_pred = []
    history_x_target = []
    
    # VLM visual history for better context
    tip_position_history = []
    target_position_history = []

    # Run simulation
    total_steps = int(FINAL_TIME / simulation_params['dt'])
    step_skip_mpc = int(simulation_params['mpc_dt'] / simulation_params['dt'])
    step_skip_vlm = int(vlm_dt / simulation_params['dt'])
    time = 0.0
    i = 0
    try:
        # main loop
        # for i in tqdm(range(total_steps)):        
        while True:
            # print(f"\nStep {i}, Time: {time:.2f}s, Target: {x_target[:3]}, Current: {x_current[:3]}", end='\r', flush=True)
            # VLM control updates
            if CONTROL_MODE == 'vlm' and i % step_skip_vlm == 0:
                # Get current state for VLM
                rods_vlm = cc_sim.get_rods()
                current_tip_pos_vlm = rods_vlm[-1].position_collection[:, -1]
                current_tip_vel_vlm = rods_vlm[-1].velocity_collection[:, -1]
                x_current_vlm = np.array(current_tip_pos_vlm.tolist() + current_tip_vel_vlm.tolist())
                
                # Update position histories for VLM context
                tip_position_history.append(current_tip_pos_vlm.copy())
                if len(history_x_target) > 0:
                    target_position_history.append(history_x_target[-1][:3])  # Only position, not velocity
                
                # Generate scene image for VLM (unless in text-only mode)
                scene_image = None
                if not vlm.text_only_mode:
                    scene_image = vlm.ingest_info(
                        sim_data=cc_sim,
                        current_target=x_target[:3] if len(history_x_target) > 0 else None,
                        tip_history=tip_position_history[-50:],  # Last 50 positions for history
                        target_history=target_position_history[-50:] if target_position_history else None
                    )

                # Process VLM input with visual context
                new_trajectory, target_name = vlm.process_user_input(x_current_vlm, scene_image)
                
                if new_trajectory is not None:
                    vlm_trajectory = new_trajectory
                    vlm_trajectory_index = 0
                    print(f"New VLM trajectory activated: {target_name}")
                    
                    # Optional: Save scene image for debugging (uncomment to enable)
                    # vlm.save_scene_image()
                    
                    # Print VLM status
                    status = vlm.get_status()
                    if i % (step_skip_vlm * 5) == 0:  # Print status less frequently
                        print(f"VLM Status: {status['last_response']}")
                
            # Update the MPC controller every step_skip_mpc
            if i % step_skip_mpc == 0:
                # Get current state
                rods = cc_sim.get_rods()
                current_tip_pos = rods[-1].position_collection[:, -1]
                current_tip_vel = rods[-1].velocity_collection[:, -1]
                x_current_mpc = np.array(current_tip_pos.tolist() + current_tip_vel.tolist())
                
                # Get MPC control input
                u_mpc = mpc.step(x_target, x_current_mpc)
                if u_mpc is None:
                    print(f"MPC failed at step {i}")
                    break
                current_control = np.array(u_mpc)
                
                # Determine target
                if CONTROL_MODE == 'vlm' and vlm_trajectory is not None:
                    # Use VLM trajectory
                    if vlm_trajectory_index < len(vlm_trajectory):
                        x_target = vlm_trajectory[vlm_trajectory_index]
                        vlm_trajectory_index += 1
                    else:
                        # VLM trajectory finished, stay at final target
                        x_target = vlm_trajectory[-1]
                
                if CONTROL_MODE == "tt":
                    # Get the next target state from the reference trajectory
                    if ref_index < len(reference_trajectory):
                        x_target = reference_trajectory[ref_index]
                        ref_index += 1
                    else:
                        x_target = reference_trajectory[-1]
                
                # Store history
                history_x.append(x_current_mpc)
                history_u.append(u_mpc)
                history_x_target.append(x_target)
                
            # Apply the control input to the simulation
            cc_sim.set_torque(0, np.array([current_control[0], current_control[1], 0.0]))
            cc_sim.set_torque(1, np.array([current_control[2], current_control[3], 0.0]))
            
            # step sim
            time = cc_sim.step(time)
            
            if DEBUG_STATE_PREDICTION:
                # Save states for plotting
                if i % step_skip_mpc == 0:
                    # Check the next state prediction of the network the mpc uses
                    x_current_pred = mpc.simulate_system(x_current_mpc, u_mpc)
                    rods_test = cc_sim.get_rods()
                    current_tip_pos_test = rods_test[-1].position_collection[:, -1]
                    current_tip_vel_test = rods_test[-1].velocity_collection[:, -1]
                    x_current_test = np.array(current_tip_pos_test.tolist() + current_tip_vel_test.tolist())
                    state_pred_error = np.linalg.norm(x_current_pred - x_current_test)

                    history_x_current_test.append(x_current_test)
                    history_x_current_pred.append(x_current_pred)
                    state_pred_error_history.append(state_pred_error)
                    # print(f"\tNN prediction error: {state_pred_error:.4f}")

            # Check for numerical instability
            if not cc_sim.check_stability():
                break
            
            if REAL_TIME_PLOT and plotter and i % PLOT_EVERY_N_STEPS == 0:
                # Check if plot window is still open
                if not plt.get_fignums():
                    print("Plot window closed, stopping simulation")
                    break
                    
                plotter.update_plot(time, target_position=x_target[:3])                
                
            i += 1
        
        print("Simulation complete!")
        
        # Plot results
        mpc.history_x = history_x  # Set for plotting
        mpc.history_u = history_u  # Set for plotting
        mpc.plot_results(history_x_target=history_x_target)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup VLM
        if CONTROL_MODE == "vlm" and 'vlm' in locals():
            print("Stopping VLM input thread...")
            vlm.stop()
            
        # Close all plots
        plt.close('all')
        
    # Post-processing
    print("Saving simulation data and plotting results...")

    if DEBUG_STATE_PREDICTION:
        # Plot and save the state prediction error history
        if state_pred_error_history:
            plt.figure(figsize=(10, 6))
            plt.plot(state_pred_error_history)
            plt.title('State Prediction Error Over Time')
            plt.xlabel('MPC Step')
            plt.ylabel('Prediction Error (L2 Norm)')
            plt.grid(True)
            plt.savefig('results/state_pred_error.png')
            print("State prediction error plot saved to results/state_pred_error.png")
            plt.close()
            
        # Save the history of current states and predictions
        if history_x_current_test and history_x_current_pred:
            history_x_current_test = np.array(history_x_current_test)
            history_x_current_pred = np.array(history_x_current_pred)        
            plot_predictions(history_x_current_test, history_x_current_pred, 'results/predictions_plot.png')
        
    print("\nAll done!")


if __name__ == "__main__":
    main()