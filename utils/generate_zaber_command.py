# Function that takes the planned_trajectory from the CSV file and generates the Zaber command using pvt

# Load the trajectory file
import os
import sys

from zaber_motion import CommandFailedException, ConnectionFailedException, Measurement, MotionLibException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from utils.mpc_functions import load_trajectory_data
from zaber_motion.ascii import Connection, Device, Axis


def build_sequence_data():

    print(f"Building sequence data from {os.path.abspath(config.TRAJ_DIR)}")
    _, control_inputs = load_trajectory_data(os.path.abspath(config.TRAJ_DIR))

    # Build 3 csv tables for each axis
    axis_1 = control_inputs[:, 0]
    axis_2 = control_inputs[:, 1]
    axis_3 = control_inputs[:, 2]

    # Set the timestamp from config
    start = 0
    end = config.T_SIM
    step = config.DT

    # Create the time vector
    time_vector = [start + i * step for i in range(len(axis_1))]

    # Create the CSV files
    with open('axis_1.csv', 'w') as f:
        f.write('Time (s),X Position (mm)\n')
        for t, pos in zip(time_vector, axis_1):
            f.write(f'{t},{pos}\n')

    with open('axis_2.csv', 'w') as f:
        f.write('Time (s),Y Position (mm)\n')
        for t, pos in zip(time_vector, axis_2):
            f.write(f'{t},{pos}\n')

    with open('axis_3.csv', 'w') as f:
        f.write('Time (s),Z Position (mm)\n')
        for t, pos in zip(time_vector, axis_3):
            f.write(f'{t},{pos}\n')

def send_pvt_command():

    print("Initializing motors...")
    connection = None  # Initialize connection to None for finally block
    try:
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()

        device_list = connection.detect_devices()
        print("Found {} devices.".format(len(device_list)))
        
        if not device_list:
            print("No devices found. Exiting.")
            return
        
        print(device_list)
        
        device = device_list[0] # Using the first detected device
        
        # Assuming control of one axis of this device with PVT.
        # The pvt_sequence.point call uses a single-element list for position and velocity,
        # implying one axis is being controlled by this PVT sequence.
        num_controlled_axes = 1 

        pvt_sequence_id = 1
        pvt_buffer_id = 1 # Using buffer 1 for sequence 1

        pvt_sequence = device.pvt.get_sequence(pvt_sequence_id)
        pvt_buffer = device.pvt.get_buffer(pvt_buffer_id)
        
        print(f"Setting up PVT sequence {pvt_sequence_id} to store in buffer {pvt_buffer_id} for {num_controlled_axes} axis/axes.")
        pvt_buffer.erase() # Erase buffer before storing new sequence
        pvt_sequence.setup_store(pvt_buffer, num_controlled_axes)
        
        # Define the path with (position, velocity, time_duration)
        # IMPORTANT: Time duration (the third element) MUST be greater than 0.
        # Your original path had 0s durations for the first and third points.
        # Example corrected path (adjust values as needed for your application):
        corrected_path = [
            (0.0, 0.0, 0.1),  # Pos (mm), Vel (mm/s), Time (s) - e.g., short time for initial point
            (0.1552202572759232, None, 1.0), # Velocity can be None
            (0.0, 0.0, 0.1)   # e.g., short time for final point
        ]

        print("Adding points to PVT sequence...")
        for i, point_data in enumerate(corrected_path):
            pos_val, vel_val, time_val = point_data
            
            if time_val <= 0:
                # This check is crucial as Zaber devices require positive time for PVT points.
                print(f"Error: PVT point time duration must be positive. Point {i+1} has time {time_val} s.")
                # Optionally raise an error or handle it:
                # raise ValueError(f"PVT point time duration must be positive. Point {i+1} has time {time_val} s.")
                continue # Skip this invalid point or handle error

            print(f"  Point {i+1}: Pos={pos_val} mm, Vel={vel_val if vel_val is not None else 'auto'} mm/s, Time={time_val} s")
            pvt_sequence.point(
                [Measurement(pos_val, "mm")], 
                [Measurement(vel_val, "mm/s") if vel_val is not None else None], 
                Measurement(time_val, "s")
            )
        
        print("Finished writing points to PVT buffer.")
        pvt_sequence.disable() # Finalize writing to the buffer

        # To execute the stored sequence (currently commented out):
        # print(f"Calling PVT sequence {pvt_sequence_id} from buffer {pvt_buffer_id}.")
        # pvt_sequence.call(pvt_buffer)
        # print("PVT sequence called. If motion was expected, check device.")
        # You might need to add a wait here if you want the script to pause until motion is complete.

        # The pvt_sequence.setup_live(1, 2) call from your original script is for a different mode
        # and would typically not be mixed this way with setup_store and call.
        # If you intend to use live mode, the structure would be different:
        # pvt_sequence.setup_live(1) # For one axis (e.g. axis 1 of the device)
        # Then send points with pvt_sequence.point(...) which execute immediately.

        print("PVT command sequence generation logic complete. Execution part is commented out.")

    except CommandFailedException as e:
        print(f"Zaber Command failed: {e}")
    except MotionLibException as e:
        print(f"Zaber Motion library error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if connection:
            print("Closing connection.")
            connection.close()

if __name__ == "__main__":
    # build_sequence_data()
    send_pvt_command()
