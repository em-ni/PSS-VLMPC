# collect.py
import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface
import numpy as np
import pandas as pd

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path

# Try to set a working interactive backend, fallback to non-interactive if needed
def set_matplotlib_backend():
    backends_to_try = ['TkAgg', 'Qt5Agg', 'QtAgg', 'Agg']  # Agg is non-interactive fallback
    
    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            print(f"Successfully set matplotlib backend to: {backend}")
            return backend
        except Exception as e:
            print(f"Failed to set backend {backend}: {e}")
            continue
    
    # If all backends fail, use Agg (non-interactive)
    matplotlib.use('Agg')
    print("Warning: Using non-interactive backend 'Agg'. Plots will be saved but not displayed.")
    return 'Agg'

current_backend = set_matplotlib_backend()

# Simulation settings
SAVE_RESULTS = True
SAVE_VIDEO = False
REAL_TIME_PLOT = False
N_TRAJECTORIES = 5000  # 1000 trajectories ~ 500k rows
TRAJECTORY_TIME = 10    # Total time for each trajectory
HOLD_TIME = 2           # Time to hold the last torque before pausing
PAUSE_TIME = 1        # Time to pause with zero torque
plot_every_n_steps = 1000

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
step_skip = simulation_params['mpc_dt'] / simulation_params['dt']

# Time parameters
if N_TRAJECTORIES > 0:
    final_time = N_TRAJECTORIES * (TRAJECTORY_TIME)
else:
    final_time = 10
    
    
def build_trajectories(n_trajectories, n_rods, trajectory_time, max_torque, hold_time, pause_time, mpc_dt):
    """
    Each trajectory is a sequence of total_time/mpc_dt control inputs built as follows:
    - sample 4 random torques for each rod (tau_x_start, tau_y_start, tau_x_end, tau_y_end)
    - generate a linespace between the two torques of trajectory_time/mpc_dt steps
    - append hold_time/mpc_dt steps of the last sampled torque
    - append pause_time/mpc_dt steps of zero torque
        
    Returns:
        list: A list of trajectories, each trajectory is a list of torques for each rod
        rod_trajectory = [[tau_x1, tau_y1], [tau_x2, tau_y2], ..., [tau_xn, tau_yn]] 
        trajectory = [rod_1_trajectory, rod_2_trajectory, ..., rod_n_rods_trajectory]
        trajectories = [trajectory_1, trajectory_2, ..., trajectory_n_trajectories]

    """
    trajectories = []
    steps_execution = int((trajectory_time - hold_time - pause_time) / mpc_dt)
    steps_hold = int(hold_time / mpc_dt)
    steps_pause = int(pause_time / mpc_dt)
    
    for _ in range(n_trajectories):
        trajectory = []
        for _ in range(n_rods):            
            # Sample random torques
            tau_x_start = np.random.uniform(-max_torque, max_torque)
            tau_y_start = np.random.uniform(-max_torque, max_torque)
            tau_x_end = np.random.uniform(-max_torque, max_torque)
            tau_y_end = np.random.uniform(-max_torque, max_torque)
            
            # Generate linearly spaced torques (execution)
            execution_torques_x = np.linspace(tau_x_start, tau_x_end, steps_execution)
            execution_torques_y = np.linspace(tau_y_start, tau_y_end, steps_execution)
            
            # Hold the last torque for hold_time
            hold_torques_x = np.full(steps_hold, tau_x_end)
            hold_torques_y = np.full(steps_hold, tau_y_end)
            
            # Pause with zero torque for pause_time
            pause_torques_x = np.zeros(steps_pause)
            pause_torques_y = np.zeros(steps_pause)
            
            # Combine all torques
            rod_trajectory = []
            rod_trajectory.extend(zip(execution_torques_x, execution_torques_y))
            rod_trajectory.extend(zip(hold_torques_x, hold_torques_y))
            rod_trajectory.extend(zip(pause_torques_x, pause_torques_y))
            trajectory.append(rod_trajectory)

        trajectories.append(trajectory)
    return trajectories

def build_dataset(applied_torque_rod_1_x, applied_torque_rod_1_y,
                  applied_torque_rod_2_x, applied_torque_rod_2_y,
                  tip_x, tip_y, tip_z, tip_vx, tip_vy, tip_vz,
                  times):
    # Create a DataFrame for the dataset
    data = {
        'T': times,
        'rod1_torque_x': applied_torque_rod_1_x,
        'rod1_torque_y': applied_torque_rod_1_y,
        'rod2_torque_x': applied_torque_rod_2_x,
        'rod2_torque_y': applied_torque_rod_2_y,
        'tip_position_x': tip_x,
        'tip_position_y': tip_y,
        'tip_position_z': tip_z,
        'tip_velocity_x': tip_vx,
        'tip_velocity_y': tip_vy,
        'tip_velocity_z': tip_vz,
        
    }
    dataset = pd.DataFrame(data)
    return dataset

def main():

    # init sim
    print("Initializing Constant Curvature Simulation...")
    cc_sim = Sim(**simulation_params)
    
    # Get simulation components
    rods_list = cc_sim.get_rods()
    torque_objects = cc_sim.get_torque_objects()
    callback_params = cc_sim.get_callback_params()
    
    # Build trajectories
    trajectories = []
    if N_TRAJECTORIES > 0:
        trajectories = build_trajectories(
            n_trajectories=N_TRAJECTORIES,
            n_rods=len(rods_list),
            trajectory_time=TRAJECTORY_TIME,
            max_torque=simulation_params['max_torque'],
            hold_time=HOLD_TIME,
            pause_time=PAUSE_TIME,
            mpc_dt=simulation_params['mpc_dt']
        )
        
    # Dataset variables (csv columns)
    applied_torque_rod_1_x = []
    applied_torque_rod_1_y = []
    applied_torque_rod_2_x = []
    applied_torque_rod_2_y = []
    tip_x = []
    tip_y = []
    tip_z = []
    tip_vx = []
    tip_vy = []
    tip_vz = []
    times = []
        
    # Print simulation info
    sim_info = cc_sim.get_simulation_info()
    print("\nSimulation Configuration:")
    for key, value in sim_info.items():
        print(f"  {key}: {value}")
        
    if trajectories:
        print(f"\nTrajectory Execution Plan:")
        print(f"  Total trajectories: {len(trajectories)}")
        print(f"  Trajectory duration: {TRAJECTORY_TIME}s")
        print(f"  Pause between trajectories: {PAUSE_TIME}s")
        print(f"  Total simulation time: {final_time}s")
    
    # Set real-time plot
    plotter = None
    if REAL_TIME_PLOT:
        print("Setting up real-time plotter...")
        plotter = RTPlotter(rods_list)
    
    # Run simulation
    total_steps = int(final_time / simulation_params['dt'])
    print(f"\nStarting simulation with {total_steps} steps")
    if REAL_TIME_PLOT:
        print(f"Real-time plotting every {plot_every_n_steps} steps")
        print("Close the plot window to stop the simulation")
    
    try:
        time = 0.0
        torque_index = 0
        trajectory_index = 0
        rod_1_torque = np.array([0.0, 0.0])
        rod_2_torque = np.array([0.0, 0.0])
        # sim loop
        for i in tqdm(range(total_steps)):
            # Update torque value with the mpc frequency (mpc loop)
            if i % step_skip == 0:
                # print(f"\nStep {i+1}/{total_steps}, Time: {time:.2f}s, Trajectory: {trajectory_index+1}/{len(trajectories)}, torque index: {torque_index}/{len(trajectories[trajectory_index][0])}")
                times.append(time)
                if trajectory_index < len(trajectories):
                    # Get the current trajectory and rod torques
                    trajectory = trajectories[trajectory_index]
                    rod_1_trajectory = trajectory[0]
                    rod_2_trajectory = trajectory[1]
                    rod_1_torque = rod_1_trajectory[torque_index]
                    rod_2_torque = rod_2_trajectory[torque_index]

                    # Increment indexes
                    torque_index += 1
                    if torque_index >= len(rod_1_trajectory):
                        torque_index = 0
                        trajectory_index += 1
                else:
                    # If no more trajectories, set zero torque
                    cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0]))
                    cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))
            
            # Set the torques for the rods (this is held constant for step_skip steps)
            cc_sim.set_torque(0, np.array([rod_1_torque[0], rod_1_torque[1], 0.0]))
            cc_sim.set_torque(1, np.array([rod_2_torque[0], rod_2_torque[1], 0.0]))
            
            if i % step_skip == 0:
                # Save applied torque
                applied_torque_rod_1_x.append(rod_1_torque[0])
                applied_torque_rod_1_y.append(rod_1_torque[1])
                applied_torque_rod_2_x.append(rod_2_torque[0])
                applied_torque_rod_2_y.append(rod_2_torque[1])
                
                # Get current state
                rods = cc_sim.get_rods()
                current_tip_pos = rods[-1].position_collection[:, -1]
                current_tip_vel = rods[-1].velocity_collection[:, -1]
                tip_x.append(current_tip_pos[0])
                tip_y.append(current_tip_pos[1])
                tip_z.append(current_tip_pos[2])
                tip_vx.append(current_tip_vel[0])
                tip_vy.append(current_tip_vel[1])
                tip_vz.append(current_tip_vel[2])
            
            # step sim
            time = cc_sim.step(time)
            
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
    if SAVE_VIDEO and callback_params:
        print("\nGenerating video...")
        try:
            rendering_fps = 40
            plot_video_with_surface(
                callback_params,
                video_name="plot.mp4",
                fps=rendering_fps,
                step=1,
                x_limits=(-1.5, 1.5),
                y_limits=(-1.5, 1.5),
                z_limits=(-1.5, 1.5),
                dpi=100,
                vis3D=False,
                vis2D=True,
            )
            print("Video saved successfully!")
        except Exception as e:
            print(f"Error generating video: {e}")
    

    # Save results    
    if SAVE_RESULTS and callback_params:
        print("\nSaving simulation results...")
        
        # Define the target directory
        results_dir = "results"

        # Ensure the results directory exists
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        else:
            # Iterate over all the items in the directory
            for filename in os.listdir(results_dir):
                # Check if the item's name ends with .dat
                if filename.endswith(".dat"):
                    file_path = os.path.join(results_dir, filename)
                    try:
                        # Extra check to ensure it's a file before trying to delete
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            print(f"Deleted data file: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
            
        # Save data for each rod
        for i, params in enumerate(callback_params):
            filename = f"results/rod{i+1}.dat"
            with open(filename, "wb") as file:
                pickle.dump(params, file)
            print(f"Saved rod {i+1} data to {filename}")
            
        # Build the csv dataset
        print("Building dataset from simulation data...")
        dataset = build_dataset(
            applied_torque_rod_1_x,
            applied_torque_rod_1_y,
            applied_torque_rod_2_x,
            applied_torque_rod_2_y,
            tip_x, tip_y, tip_z, tip_vx, tip_vy, tip_vz,
            times
        )
        dataset_filename = os.path.join(results_dir, "sim_dataset.csv")
        dataset.to_csv(dataset_filename, index=False)
        print(f"Dataset saved to {dataset_filename}")
    
    print("\nAll done!\n")


if __name__ == "__main__":
    main()