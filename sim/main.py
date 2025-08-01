import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
import shutil
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface
import numpy as np

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim
from utils.trajectories import build_trajectories, get_current_torques

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
matplotlib.use('Qt5Agg')  # Use interactive backend

# Simulation settings
SAVE_RESULTS = True
SAVE_VIDEO = False
REAL_TIME_PLOT = False
N_TRAJECTORIES = 1000
TRAJECTORY_TIME = 10
PAUSE_TIME = 5
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

# Time parameters
if N_TRAJECTORIES > 0:
    final_time = N_TRAJECTORIES * (TRAJECTORY_TIME + PAUSE_TIME) + 1
else:
    final_time = 10

def main():

    # init sim
    print("Initializing Constant Curvature Simulation...")
    cc_sim = Sim(**simulation_params)
    
    # Get simulation components
    rods_list = cc_sim.get_rods()
    torque_objects = cc_sim.get_torque_objects()
    callback_params = cc_sim.get_callback_params()
    
    # Build trajectories by sampling random torques
    trajectories = []
    if N_TRAJECTORIES > 0:
        trajectories = build_trajectories(
            N_TRAJECTORIES, 
            TRAJECTORY_TIME, 
            simulation_params['max_torque'],
            PAUSE_TIME
        )
    
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
    
    time = 0.0
    current_trajectory = -1
    try:
        # main loop
        for i in tqdm(range(total_steps)):
            current_time = i * simulation_params['dt']
            
            # --- control for data collection ---
            # Warm up for 0.5 s
            if current_time < 0.5:
                cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0])) 
                cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))  
            else:
                # Execute trajectories
                if trajectories:
                    # Determine which trajectory we're in
                    trajectory_cycle_time = TRAJECTORY_TIME + PAUSE_TIME
                    new_trajectory_idx = int((current_time - 0.5) // trajectory_cycle_time)
                    
                    # Print trajectory transitions
                    if new_trajectory_idx != current_trajectory and new_trajectory_idx < len(trajectories):
                        current_trajectory = new_trajectory_idx
                        cycle_time = (current_time - 0.5) % trajectory_cycle_time
                        if cycle_time < TRAJECTORY_TIME:
                            print(f"\nStarting trajectory {current_trajectory + 1} at time {current_time:.2f}s")
                    
                    # Get current torques based on trajectory execution
                    rod1_torque, rod2_torque = get_current_torques(
                        trajectories, 
                        current_time - 0.5,  # Subtract warm-up time
                        TRAJECTORY_TIME,
                        PAUSE_TIME
                    )
                    
                    cc_sim.set_torque(0, rod1_torque)
                    cc_sim.set_torque(1, rod2_torque)
                else:
                    # No trajectories defined, use zero torques
                    cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0])) 
                    cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))
            # --- end data collection ---
            
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
                    
                plotter.update_plot(current_time)
        
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
        
        # Clean results folder
        if not os.path.exists("results"):
            os.makedirs("results")
        else:
            for file in os.listdir("results"):
                file_path = os.path.join("results", file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        
        # Save data for each rod
        for i, params in enumerate(callback_params):
            filename = f"results/rod{i+1}.dat"
            with open(filename, "wb") as file:
                pickle.dump(params, file)
            print(f"Saved rod {i+1} data to {filename}")
    
    print("\nAll done!")


if __name__ == "__main__":
    main()