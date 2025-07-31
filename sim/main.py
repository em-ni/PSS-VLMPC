import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
import shutil
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
matplotlib.use('Qt5Agg')  # Use interactive backend


def main():
    # Simulation settings
    SAVE_RESULTS = True
    SAVE_VIDEO = True
    REAL_TIME_PLOT = True
    
    # rod parameters
    simulation_params = {
        'n_elem': 30,
        'base_length': 0.4,
        'base_radius': 0.02,
        'density': 1000,
        'youngs_modulus': 1e5,
        'poisson_ratio': 0.5,
        'damping_constant': 1e-1,
        'dt': 1e-4,
        'double_rod': True,
        'max_torque': 5e-2,
        'mpc_dt': 0.02
    }
    
    # Time parameters
    final_time = 10
    plot_every_n_steps = 200
    
    # init sim
    print("Initializing Constant Curvature Simulation...")
    cc_sim = Sim(**simulation_params)
    
    # Get simulation components
    rods_list = cc_sim.get_rods()
    torque_objects = cc_sim.get_torque_objects()
    callback_params = cc_sim.get_callback_params()
    
    # Print simulation info
    sim_info = cc_sim.get_simulation_info()
    print("Simulation Configuration:")
    for key, value in sim_info.items():
        print(f"  {key}: {value}")
    
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
    
    try:
        # main loop
        for i in tqdm(range(total_steps)):
            current_time = i * simulation_params['dt']
            
            # =================================================================
            # DYNAMIC TORQUE CONTROL (Optional)
            # =================================================================
            # Uncomment and modify as needed for dynamic control
            
            # Example: Time-based torque control
            # if current_time < 2.0:
            #     cc_sim.set_torque(0, np.array([0.0, -2e-2, 0.0]))  # Rod 1
            #     if len(torque_objects) > 1:
            #         cc_sim.set_torque(1, np.array([0.0, 7e-3, 0.0]))  # Rod 2
            # elif current_time < 5.0:
            #     cc_sim.set_torque(0, np.array([0.0, -5e-2, 0.0]))
            #     if len(torque_objects) > 1:
            #         cc_sim.set_torque(1, np.array([0.0, 1e-2, 0.0]))
            # elif current_time < 8.0:
            #     cc_sim.set_torque(0, np.array([0.0, 3e-2, 0.0]))
            #     if len(torque_objects) > 1:
            #         cc_sim.set_torque(1, np.array([0.0, -8e-3, 0.0]))
            # else:
            #     cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0]))
            #     if len(torque_objects) > 1:
            #         cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))
            
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