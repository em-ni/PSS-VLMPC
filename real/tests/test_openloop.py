from zaber_motion.ascii import Connection
import sys
import os
import pandas as pd
import numpy as np
import time
from zaber_motion import Units

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pressure_loader import PressureLoader
import src.config as config

def load_control_inputs(file_path="planned_trajectory.csv"):
    """
    Load the control inputs from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        numpy.ndarray: Control inputs, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded trajectory data with {len(df)} steps from {file_path}")
        
        # Extract control inputs
        control_cols = [col for col in df.columns if col.startswith('control_')]
        control_inputs = df[control_cols].values
        
        return control_inputs
    except FileNotFoundError:
        print(f"Error: Trajectory file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading control inputs: {e}")
        return None
    
def send_command(action, axis_1, axis_2, axis_3):
    # Move each axis by the action vector
    axis_1.move_absolute(config.initial_pos + float(action[0]), Units.LENGTH_MILLIMETRES, False)
    axis_2.move_absolute(config.initial_pos + float(action[1]), Units.LENGTH_MILLIMETRES, False)
    axis_3.move_absolute(config.initial_pos + float(action[2]), Units.LENGTH_MILLIMETRES, False)
    time.sleep(1)

def init_axis():
    # Open connection on COM3
    connection = Connection.open_serial_port('COM3')
    connection.enable_alerts()

    # Detect devices
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    
    # Get the axis
    axis_1 = device_list[0].get_axis(1)
    axis_2 = device_list[1].get_axis(1)
    axis_3 = device_list[2].get_axis(1)

    return axis_1, axis_2, axis_3


def main():
    # Define the trajectory file path using config
    trajectory_file = config.TRAJ_DIR
    
    # Load control inputs
    control_inputs = load_control_inputs(trajectory_file)
    if control_inputs is None:
        print("Exiting due to error in loading control inputs.")
        return
    
    print(f"Loaded control inputs with shape {control_inputs.shape}")

    # Load pressure offsets
    pressure_loader = PressureLoader()
    offsets = pressure_loader.load_pressure()
    if offsets is None:
        print("Warning: Failed to load pressure offsets. Continuing without them.")
        # Assuming offsets should be zeros if not loaded, matching number of control channels
        if control_inputs.shape[1] > 0:
             offsets = np.zeros(control_inputs.shape[1])
        else: # Default to 6 if control_inputs is empty for some reason
            print("Warning: Control inputs are empty. Assuming 6 channels for zero offsets.")
            offsets = np.zeros(6) # Default number of channels if not determinable

    # init axis
    axis_1, axis_2, axis_3 = init_axis()
    
    # Time to allow tracker to initialize if needed
    time.sleep(1.0) 

    total_steps = len(control_inputs)
    step_time = config.DT  # Time between steps from config
    
    print("Starting to apply trajectory commands...")
    
    try:
        for step in range(total_steps):
            start_time = time.time()
            
            # Get control input for current step
            u_command_original = control_inputs[step]

            # Add offset to control input
            # Ensure offsets array is compatible with u_command_original
            if len(offsets) == len(u_command_original):
                u_command_with_offset = u_command_original + offsets
            else:
                print(f"Warning: Mismatch in offset length ({len(offsets)}) and control input length ({len(u_command_original)}). Using original command.")
                u_command_with_offset = u_command_original

            # Apply control to the robot
            send_command(u_command_with_offset, axis_1, axis_2, axis_3)
            print(f"Step {step+1}/{total_steps}: Applying control input {u_command_with_offset}")
            
            # Wait for next step (accounting for computation time)
            elapsed = time.time() - start_time
            sleep_duration = max(0, step_time - elapsed)
            time.sleep(sleep_duration)

        print("Trajectory command application complete. Moving to inital position.")
        # Move to initial position after completing trajectory
        send_command([0, 0, 0], axis_1, axis_2, axis_3)

            
    except Exception as e:
        print(f"An error occurred during command execution: {e}")
    finally:
        print("Trajectory command application complete or interrupted.")
        # Ensure robot is stopped in case of error or completion
        print("Robot stopped.")

if __name__ == "__main__":
    main()