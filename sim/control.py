# main.py
import sys
import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface
import numpy as np

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim
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

# Simulation settings
REAL_TIME_PLOT = False
plot_every_n_steps = 1000
final_time = 10
control_mode = "spr" # set point regulation
# control_mode = "tt"  # trajectory tracking

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
    mpc = MPCController(nn_approximation_order=2)
    
    # --- Phase 3: The Simulation Loop ---
    print("\nStarting MPC simulation")
    
    # Sample initial and target states from the simulation data
    # sample = mpc.df[mpc.state_cols].dropna().sample(2, random_state=42)
    # x_target = sample.iloc[1].values
    x_target = np.array([0.01, -0.42, -0.62, 0.0, 0.0, 0.0])

    # Get initial state
    rods = cc_sim.get_rods()
    current_tip_pos = rods[-1].position_collection[:, -1]
    current_tip_vel = rods[-1].velocity_collection[:, -1]
    x_current = np.array(current_tip_pos.tolist() + current_tip_vel.tolist())

    if control_mode == "tt":
        # Generate a reference trajectory of final_time/simulation_params['mpc_dt'] steps
        reference_trajectory = np.linspace(x_current, x_target, num=int(final_time / simulation_params['mpc_dt']))
        ref_index = 0
        x_target = reference_trajectory[ref_index] 

    # History variables
    current_control = np.zeros(4)  
    history_x, history_u = [], []
    state_pred_error_history = []
    history_x_current_test = []
    history_x_current_pred = []

    # Run simulation
    total_steps = int(final_time / simulation_params['dt'])
    step_skip = int(simulation_params['mpc_dt'] / simulation_params['dt'])
    time = 0.0
    try:
        # main loop
        for i in tqdm(range(total_steps)):            
            # Update the MPC controller every step_skip
            if i % step_skip == 0:
                # Get current state
                rods = cc_sim.get_rods()
                current_tip_pos = rods[-1].position_collection[:, -1]
                current_tip_vel = rods[-1].velocity_collection[:, -1]
                x_current = np.array(current_tip_pos.tolist() + current_tip_vel.tolist())
                
                # Get MPC control input
                u_mpc = mpc.step(x_target, x_current)
                if u_mpc is None:
                    print(f"MPC failed at step {i}")
                    break
                current_control = np.array(u_mpc)
                
                if control_mode == "tt":
                    # Get the next target state from the reference trajectory
                    if ref_index < len(reference_trajectory):
                        x_target = reference_trajectory[ref_index]
                        ref_index += 1
                    else:
                        x_target = reference_trajectory[-1]
                
                # Store history
                history_x.append(x_current)
                history_u.append(u_mpc)
                
            # Apply the control input to the simulation
            cc_sim.set_torque(0, np.array([current_control[0], current_control[1], 0.0]))
            cc_sim.set_torque(1, np.array([current_control[2], current_control[3], 0.0]))
            
            # step sim
            time = cc_sim.step(time)
            
            if i % step_skip == 0:
                # Check the next state prediction of the network the mpc uses
                x_current_pred = mpc.simulate_system(x_current, u_mpc)
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
            
            if REAL_TIME_PLOT and plotter and i % plot_every_n_steps == 0:
                # Check if plot window is still open
                if not plt.get_fignums():
                    print("Plot window closed, stopping simulation")
                    break
                    
                plotter.update_plot(time)
        
        print("Simulation complete!")
        
        # Plot results
        mpc.history_x = history_x  # Set for plotting
        mpc.history_u = history_u  # Set for plotting
        mpc.plot_results(x_target)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Keep the final plot open
        if REAL_TIME_PLOT and plt.get_fignums():
            print("Press any key in terminal to close...")
            input()
        plt.close('all')
        
    # Post-processing
    print("Saving simulation data and plotting results...")

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