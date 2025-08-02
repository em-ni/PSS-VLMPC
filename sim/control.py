# main_with_mpc.py
import matplotlib
import matplotlib.pyplot as plt
import imageio_ffmpeg
import os
import pickle
import shutil
import time
from elastica.timestepper import tqdm
from examples.MuscularSnake.post_processing import plot_video_with_surface
import numpy as np

# Local imports
from src.RTPlotter import RTPlotter
from src.Sim import Sim
from src.MPCController import MPCController

# Set path for FFMPEG for saving video animations
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
matplotlib.use('Qt5Agg')  # Use interactive backend

# Simulation settings
SAVE_RESULTS = True
SAVE_VIDEO = False
REAL_TIME_PLOT = True
USE_MPC_CONTROL = True  # NEW: Enable MPC control
MPC_UPDATE_RATE = 200   # NEW: MPC runs every 50 simulation steps (50 * 1e-4 = 5ms)
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
final_time = 10.0  # Reduced for MPC demo


class WorkspaceConstrainedTrajectory:
    """
    Generate trajectories that respect the robot's workspace constraints.
    
    Based on the output ranges:
    - X: [-0.708747, 0.707358]
    - Y: [-0.707138, 0.704363] 
    - Z: [-0.800254, 0.503983]
    """
    
    def __init__(self):
        # Define workspace bounds with safety margins
        self.workspace_bounds = {
            'x_min': -0.65,   # Safety margin from -0.708747
            'x_max': 0.65,    # Safety margin from 0.707358
            'y_min': -0.65,   # Safety margin from -0.707138
            'y_max': 0.65,    # Safety margin from 0.704363
            'z_min': -0.75,   # Safety margin from -0.800254
            'z_max': 0.45     # Safety margin from 0.503983
        }
        
        # Calculate workspace center and safe operating region
        self.x_center = (self.workspace_bounds['x_max'] + self.workspace_bounds['x_min']) / 2
        self.y_center = (self.workspace_bounds['y_max'] + self.workspace_bounds['y_min']) / 2
        self.z_center = (self.workspace_bounds['z_max'] + self.workspace_bounds['z_min']) / 2
        
        # Safe radii for circular motions
        self.max_xy_radius = min(
            self.workspace_bounds['x_max'] - abs(self.x_center),
            self.workspace_bounds['y_max'] - abs(self.y_center)
        ) * 0.8  # 80% of maximum to be safe
        
        # Variables to track trajectory initialization
        self.trajectory_initialized = False
        self.start_position = None
        self.trajectory_start_time = None
        
        print(f"Workspace center: ({self.x_center:.3f}, {self.y_center:.3f}, {self.z_center:.3f})")
        print(f"Maximum safe XY radius: {self.max_xy_radius:.3f}")
    
    def initialize_trajectory(self, current_tip_position, start_time):
        """Initialize trajectory to start from current tip position"""
        self.start_position = current_tip_position.copy()
        self.trajectory_start_time = start_time
        self.trajectory_initialized = True
        print(f"Trajectory initialized from position: [{self.start_position[0]:.3f}, {self.start_position[1]:.3f}, {self.start_position[2]:.3f}]")
    
    def create_target_trajectory(self, current_time, trajectory_type="circle", current_tip_position=None):
        """
        Generate workspace-constrained target trajectory that starts from current tip position.
        
        Args:
            current_time: Current simulation time
            trajectory_type: Type of trajectory ("circle", "figure8", "point", "step", "vertical_circle", "lemniscate")
            current_tip_position: Current tip position for initialization
        
        Returns:
            target_position: Target tip position [x, y, z] within workspace bounds
        """
        
        # Initialize trajectory if not done yet
        if not self.trajectory_initialized and current_tip_position is not None:
            self.initialize_trajectory(current_tip_position, current_time)
        
        # If not initialized, return workspace center
        if not self.trajectory_initialized:
            return np.array([self.x_center, self.y_center, self.z_center])
        
        # Calculate relative time from trajectory start
        relative_time = current_time - self.trajectory_start_time
        
        if trajectory_type == "circle":
            # Horizontal circular trajectory in XY plane
            radius = min(0.25, self.max_xy_radius)  # Safe radius
            frequency = 0.2  # Hz
            
            # Calculate circle center - offset from start position
            circle_center_x = self.start_position[0]
            circle_center_y = self.start_position[1] + radius  # Offset so trajectory starts from current position
            center_height = self.start_position[2]
            
            angle = 2 * np.pi * frequency * relative_time - np.pi/2  # Start from bottom of circle
            target_x = circle_center_x + radius * np.cos(angle)
            target_y = circle_center_y + radius * np.sin(angle)
            target_z = center_height
            
        elif trajectory_type == "figure8":
            # Figure-8 trajectory (lemniscate)
            radius = min(0.2, self.max_xy_radius * 0.8)  # Smaller for figure-8
            frequency = 0.15  # Hz
            
            # Start from current position
            angle = 2 * np.pi * frequency * relative_time
            target_x = self.start_position[0] + radius * np.sin(angle)
            target_y = self.start_position[1] + radius * np.sin(2 * angle)
            target_z = self.start_position[2]
            
        elif trajectory_type == "vertical_circle":
            # Circular trajectory in XZ plane (vertical circle)
            radius_x = min(0.25, self.max_xy_radius)
            radius_z = min(0.3, (self.workspace_bounds['z_max'] - self.workspace_bounds['z_min']) / 3)
            frequency = 0.2  # Hz
            
            # Calculate circle center
            circle_center_x = self.start_position[0] + radius_x  # Offset so trajectory starts from current position
            circle_center_z = self.start_position[2]
            
            angle = 2 * np.pi * frequency * relative_time - np.pi  # Start from left of circle
            target_x = circle_center_x + radius_x * np.cos(angle)
            target_y = self.start_position[1]
            target_z = circle_center_z + radius_z * np.sin(angle)
            
        elif trajectory_type == "lemniscate":
            # 3D lemniscate (infinity symbol in 3D)
            radius_xy = min(0.2, self.max_xy_radius * 0.8)
            radius_z = min(0.15, (self.workspace_bounds['z_max'] - self.workspace_bounds['z_min']) / 4)
            frequency = 0.12  # Hz
            
            angle = 2 * np.pi * frequency * relative_time
            target_x = self.start_position[0] + radius_xy * np.cos(angle)
            target_y = self.start_position[1] + radius_xy * np.sin(2 * angle)
            target_z = self.start_position[2] + radius_z * np.sin(angle)
            
        elif trajectory_type == "point":
            # Fixed point target - move to a point relative to start position
            target_x = self.start_position[0] + 0.15
            target_y = self.start_position[1] + 0.2
            target_z = self.start_position[2] + 0.1
            
        elif trajectory_type == "step":
            # Step changes in target - all relative to start position
            if relative_time < 3.0:
                target_x = self.start_position[0] + 0.2
                target_y = self.start_position[1]
                target_z = self.start_position[2] + 0.1
            elif relative_time < 6.0:
                target_x = self.start_position[0] - 0.2
                target_y = self.start_position[1] + 0.25
                target_z = self.start_position[2] - 0.1
            else:
                target_x = self.start_position[0]
                target_y = self.start_position[1] - 0.2
                target_z = self.start_position[2] + 0.2
                
        elif trajectory_type == "spiral":
            # Spiral trajectory expanding and contracting
            max_radius = min(0.3, self.max_xy_radius)
            frequency = 0.1  # Hz
            spiral_frequency = 0.05  # Spiral expansion frequency
            
            angle = 2 * np.pi * frequency * relative_time
            radius = max_radius * (0.3 + 0.7 * (np.sin(2 * np.pi * spiral_frequency * relative_time) + 1) / 2)
            
            target_x = self.start_position[0] + radius * np.cos(angle)
            target_y = self.start_position[1] + radius * np.sin(angle)
            target_z = self.start_position[2] + 0.1 * np.sin(angle * 2)
            
        elif trajectory_type == "wave":
            # Wave motion along X with Y and Z variations
            frequency = 0.15  # Hz
            
            # X moves back and forth from start position
            target_x = self.start_position[0] + 0.4 * np.sin(2 * np.pi * frequency * relative_time)
            # Y oscillates at different frequency
            target_y = self.start_position[1] + 0.2 * np.sin(2 * np.pi * frequency * 1.5 * relative_time)
            # Z oscillates at yet another frequency
            target_z = self.start_position[2] + 0.15 * np.sin(2 * np.pi * frequency * 0.7 * relative_time)
            
        else:
            # Default to start position
            target_x = self.start_position[0]
            target_y = self.start_position[1]
            target_z = self.start_position[2]
        
        # Final safety check - clamp to workspace bounds
        target_position = np.array([
            np.clip(target_x, self.workspace_bounds['x_min'], self.workspace_bounds['x_max']),
            np.clip(target_y, self.workspace_bounds['y_min'], self.workspace_bounds['y_max']),
            np.clip(target_z, self.workspace_bounds['z_min'], self.workspace_bounds['z_max'])
        ])
        
        return target_position
    
    def reset(self):
        """Reset trajectory initialization"""
        self.trajectory_initialized = False
        self.start_position = None
        self.trajectory_start_time = None


def create_target_trajectory(current_time, trajectory_type="circle", current_tip_position=None):
    """
    Generate target trajectory for the MPC controller that respects workspace constraints
    and starts from the current tip position.
    
    Args:
        current_time: Current simulation time
        trajectory_type: Type of trajectory ("circle", "figure8", "point", "step", etc.)
        current_tip_position: Current tip position for initialization
    
    Returns:
        target_position: Target tip position [x, y, z] within workspace bounds
    """
    # Initialize trajectory generator as a function attribute for persistence
    if not hasattr(create_target_trajectory, 'generator'):
        create_target_trajectory.generator = WorkspaceConstrainedTrajectory()
    
    return create_target_trajectory.generator.create_target_trajectory(
        current_time, trajectory_type, current_tip_position
    )


def main():
    # init sim
    global USE_MPC_CONTROL
    print("Initializing Continuum Robot Simulation with MPC Control...")
    cc_sim = Sim(**simulation_params)
    
    # Get simulation components
    rods_list = cc_sim.get_rods()
    torque_objects = cc_sim.get_torque_objects()
    callback_params = cc_sim.get_callback_params()
    
    # Initialize MPC controller
    mpc_controller = None
    if USE_MPC_CONTROL:
        print("Initializing MPC Controller...")
        try:
            mpc_controller = MPCController()
            print("MPC Controller initialized successfully!")
            
        except Exception as e:
            print(f"Failed to initialize MPC Controller: {e}")
            print("Falling back to zero torque control")
            USE_MPC_CONTROL = False
    
    # Print simulation info
    sim_info = cc_sim.get_simulation_info()
    print("\nSimulation Configuration:")
    for key, value in sim_info.items():
        print(f"  {key}: {value}")
    
    if USE_MPC_CONTROL:
        print(f"\nMPC Configuration:")
        print(f"  Update rate: Every {MPC_UPDATE_RATE} simulation steps")
        print(f"  MPC frequency: {1000/(MPC_UPDATE_RATE * simulation_params['dt'])} Hz")
        print(f"  Prediction horizon: {mpc_controller.config['N']} steps")
    
    # Set real-time plot
    plotter = None
    if REAL_TIME_PLOT:
        print("Setting up real-time plotter with reference trajectory...")
        plotter = RTPlotter(rods_list, show_reference=True, trajectory_history_length=100)
    
    # Initialize MPC tracking variables
    mpc_solve_times = []
    target_history = []
    tip_position_history = []
    control_history = []
    
    # Choose trajectory type - now workspace constrained!
    TRAJECTORY_TYPE = "point"  # Options: "circle", "figure8", "vertical_circle", "lemniscate", "spiral", "wave", "step", "point"
    
    # Run simulation
    total_steps = int(final_time / simulation_params['dt'])
    print(f"\nStarting simulation with {total_steps} steps")
    print(f"Using trajectory type: {TRAJECTORY_TYPE}")
    if REAL_TIME_PLOT:
        print(f"Real-time plotting every {plot_every_n_steps} steps")
        print("Close the plot window to stop the simulation")
    
    time_sim = 0.0
    trajectory_started = False
    
    try:
        # main loop
        for i in tqdm(range(total_steps)):
            current_time = i * simulation_params['dt']
            
            # --- MPC Control Logic ---
            if USE_MPC_CONTROL and current_time > 0.5:  # Allow warm-up period
                
                # Get current tip position for trajectory initialization
                current_tip = mpc_controller._get_tip_state(rods_list)[:3]
                
                # Update MPC at specified rate
                if i % MPC_UPDATE_RATE == 0:
                    # Generate target trajectory - now starting from current tip position!
                    # target_position = create_target_trajectory(
                    #     current_time - 0.5, 
                    #     trajectory_type=TRAJECTORY_TYPE,
                    #     current_tip_position=current_tip
                    # )
                    target_position = [0.0008206369730857777,-0.00022698138237233265,-0.7999965849872518]
                    target_state = mpc_controller.set_target(target_position)
                    
                    if not trajectory_started:
                        print(f"Trajectory started from tip position: [{current_tip[0]:.3f}, {current_tip[1]:.3f}, {current_tip[2]:.3f}]")
                        trajectory_started = True
                    
                    # Solve MPC
                    mpc_start_time = time.time()
                    rod1_torque, rod2_torque = mpc_controller.compute_control(
                        rods_list, target_state
                    )
                    mpc_solve_time = time.time() - mpc_start_time
                    mpc_solve_times.append(mpc_solve_time)
                    
                    # Store current torques for reuse until next MPC update
                    current_rod1_torque = rod1_torque
                    current_rod2_torque = rod2_torque
                    
                    # Log MPC performance periodically
                    if len(mpc_solve_times) % 20 == 0:
                        avg_solve_time = np.mean(mpc_solve_times[-20:])
                        max_solve_time = np.max(mpc_solve_times[-20:])
                        print(f"\nMPC Performance (last 20 solves):")
                        print(f"  Avg solve time: {1000*avg_solve_time:.1f} ms")
                        print(f"  Max solve time: {1000*max_solve_time:.1f} ms")
                        
                        # Get current tip position for tracking
                        distance_to_target = np.linalg.norm(current_tip - target_position)
                        print(f"  Distance to target: {distance_to_target:.4f} m")
                        print(f"  Target position: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
                        print(f"  Current tip: [{current_tip[0]:.3f}, {current_tip[1]:.3f}, {current_tip[2]:.3f}]")
                
                # Apply current MPC torques
                if 'current_rod1_torque' in locals():
                    cc_sim.set_torque(0, current_rod1_torque)
                    cc_sim.set_torque(1, current_rod2_torque)
                else:
                    # Fallback to zero torques if MPC hasn't computed anything yet
                    cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0]))
                    cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))
                
                # Store data for analysis
                if i % (MPC_UPDATE_RATE * 5) == 0:  # Store every 5 MPC updates
                    tip_pos = mpc_controller._get_tip_state(rods_list)[:3]
                    target_pos = create_target_trajectory(
                        current_time - 0.5, 
                        TRAJECTORY_TYPE, 
                        current_tip_position=tip_pos
                    )
                    
                    target_history.append(target_pos.copy())
                    tip_position_history.append(tip_pos.copy())
                    if 'current_rod1_torque' in locals():
                        control_history.append([
                            current_rod1_torque[0], current_rod1_torque[1],
                            current_rod2_torque[0], current_rod2_torque[1]
                        ])
            
            else:
                # Warm-up period or no MPC - use zero torques
                cc_sim.set_torque(0, np.array([0.0, 0.0, 0.0])) 
                cc_sim.set_torque(1, np.array([0.0, 0.0, 0.0]))
            
            # --- End Control Logic ---
            
            # step sim
            time_sim = cc_sim.step(time_sim)
            
            # Check for numerical instability
            if not cc_sim.check_stability():
                print("Simulation became unstable!")
                break
            
            if REAL_TIME_PLOT and plotter and i % plot_every_n_steps == 0:
                # Check if plot window is still open
                if not plt.get_fignums():
                    print("Plot window closed, stopping simulation")
                    break
                
                # Get current target for visualization
                current_target = None
                if USE_MPC_CONTROL and current_time > 0.5:
                    current_tip_viz = mpc_controller._get_tip_state(rods_list)[:3]
                    current_target = create_target_trajectory(
                        current_time - 0.5, 
                        TRAJECTORY_TYPE,
                        current_tip_position=current_tip_viz
                    )
                
                plotter.update_plot(current_time, current_target)
        
        print("Simulation complete!")
        
        # Print MPC performance summary
        if USE_MPC_CONTROL and mpc_solve_times:
            print(f"\nMPC Performance Summary:")
            print(f"  Total MPC solves: {len(mpc_solve_times)}")
            print(f"  Avg solve time: {1000*np.mean(mpc_solve_times):.1f} ms")
            print(f"  Max solve time: {1000*np.max(mpc_solve_times):.1f} ms")
            print(f"  Min solve time: {1000*np.min(mpc_solve_times):.1f} ms")
            
            if target_history and tip_position_history:
                target_history = np.array(target_history)
                tip_position_history = np.array(tip_position_history)
                tracking_errors = np.linalg.norm(
                    tip_position_history - target_history, axis=1
                )
                print(f"  Final tracking error: {tracking_errors[-1]:.4f} m")
                print(f"  Average tracking error: {np.mean(tracking_errors):.4f} m")
                print(f"  Maximum tracking error: {np.max(tracking_errors):.4f} m")
                
                # Workspace compliance check
                print(f"\nWorkspace Compliance Check:")
                x_min, x_max = tip_position_history[:, 0].min(), tip_position_history[:, 0].max()
                y_min, y_max = tip_position_history[:, 1].min(), tip_position_history[:, 1].max()
                z_min, z_max = tip_position_history[:, 2].min(), tip_position_history[:, 2].max()
                print(f"  Actual X range: [{x_min:.3f}, {x_max:.3f}]")
                print(f"  Actual Y range: [{y_min:.3f}, {y_max:.3f}]")
                print(f"  Actual Z range: [{z_min:.3f}, {z_max:.3f}]")
        
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
                video_name=f"mpc_control_{TRAJECTORY_TYPE}_plot.mp4",
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
            filename = f"results/rod{i+1}_mpc_{TRAJECTORY_TYPE}.dat"
            with open(filename, "wb") as file:
                pickle.dump(params, file)
            print(f"Saved rod {i+1} data to {filename}")
        
        # Save MPC performance data
        if USE_MPC_CONTROL and mpc_solve_times:
            mpc_data = {
                'solve_times': mpc_solve_times,
                'target_history': np.array(target_history) if len(target_history) > 0 else None,
                'tip_position_history': np.array(tip_position_history) if len(tip_position_history) > 0 else None,
                'control_history': np.array(control_history) if len(control_history) > 0 else None,
                'config': mpc_controller.config,
                'trajectory_type': TRAJECTORY_TYPE,
                'workspace_bounds': create_target_trajectory.generator.workspace_bounds
            }
            
            with open(f"results/mpc_performance_{TRAJECTORY_TYPE}.pkl", "wb") as file:
                pickle.dump(mpc_data, file)
            print(f"Saved MPC performance data to results/mpc_performance_{TRAJECTORY_TYPE}.pkl")
            
            # Create a simple tracking plot
            if len(target_history) > 0 and len(tip_position_history) > 0:
                create_tracking_plot(target_history, tip_position_history, control_history, TRAJECTORY_TYPE)
    
    print("\nAll done!")


def create_tracking_plot(target_history, tip_history, control_history, trajectory_type):
    """Create plots showing MPC tracking performance with workspace bounds"""
    target_history = np.array(target_history)
    tip_history = np.array(tip_history)
    control_history = np.array(control_history) if control_history else None
    
    # Get workspace bounds from the trajectory generator
    workspace_bounds = create_target_trajectory.generator.workspace_bounds
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 3D trajectory plot
    ax = axes[0, 0]
    ax.plot(target_history[:, 0], target_history[:, 1], 'g--', 
            label='Target', linewidth=2)
    ax.plot(tip_history[:, 0], tip_history[:, 1], 'b-', 
            label='Actual', linewidth=2)
    
    # Draw workspace bounds
    workspace_rect = plt.Rectangle(
        (workspace_bounds['x_min'], workspace_bounds['y_min']),
        workspace_bounds['x_max'] - workspace_bounds['x_min'],
        workspace_bounds['y_max'] - workspace_bounds['y_min'],
        fill=False, edgecolor='gray', linestyle=':', alpha=0.7, linewidth=2
    )
    ax.add_patch(workspace_rect)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'XY Trajectory Tracking - {trajectory_type.title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Position tracking over time
    ax = axes[0, 1]
    time_axis = np.arange(len(target_history)) * 0.1  # Approximate time
    ax.plot(time_axis, target_history[:, 0], 'g--', label='Target X', linewidth=2)
    ax.plot(time_axis, tip_history[:, 0], 'b-', label='Actual X', linewidth=1.5)
    ax.plot(time_axis, target_history[:, 1], 'g:', label='Target Y', linewidth=2)
    ax.plot(time_axis, tip_history[:, 1], 'm-', label='Actual Y', linewidth=1.5)
    
    # Add workspace bounds as horizontal lines
    ax.axhline(y=workspace_bounds['x_min'], color='gray', linestyle=':', alpha=0.5, label='X bounds')
    ax.axhline(y=workspace_bounds['x_max'], color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=workspace_bounds['y_min'], color='lightgray', linestyle=':', alpha=0.5, label='Y bounds')
    ax.axhline(y=workspace_bounds['y_max'], color='lightgray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Position Tracking vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Tracking error
    ax = axes[1, 0]
    tracking_error = np.linalg.norm(tip_history - target_history, axis=1)
    ax.plot(time_axis, tracking_error, 'k-', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('Tracking Error Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_error = np.mean(tracking_error)
    max_error = np.max(tracking_error)
    ax.text(0.02, 0.98, f'Mean Error: {mean_error:.4f} m\nMax Error: {max_error:.4f} m', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Control inputs
    ax = axes[1, 1]
    if control_history is not None:
        control_time = np.arange(len(control_history)) * 0.1
        ax.plot(control_time, control_history[:, 0], label='Rod1 Torque X', linewidth=1.5)
        ax.plot(control_time, control_history[:, 1], label='Rod1 Torque Y', linewidth=1.5)
        ax.plot(control_time, control_history[:, 2], label='Rod2 Torque X', linewidth=1.5)
        ax.plot(control_time, control_history[:, 3], label='Rod2 Torque Y', linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Torque (Nm)')
        ax.set_title('MPC Control Inputs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No control data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(f'MPC Tracking Performance - {trajectory_type.title()} Trajectory', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/mpc_tracking_performance_{trajectory_type}.png', dpi=150, bbox_inches='tight')
    print(f"Saved MPC tracking plot to results/mpc_tracking_performance_{trajectory_type}.png")
    plt.close()


if __name__ == "__main__":
    main()