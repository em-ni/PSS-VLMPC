import time
import numpy as np
from src.tracker import Tracker
import src.config as config
from zaber_motion import Units
from zaber_motion.ascii import Connection
import threading
import time
import sys
import src.config as config

# Robot API for real-time interaction.
class RealRobotAPI:
    def __init__(self):
        print("Initializing motors...")
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()

        # connection.enableAlerts()  # (commented out as in MATLAB)
        device_list = connection.detect_devices()
        print("Found {} devices.".format(len(device_list)))
        print(device_list)
        
        # Get the axis
        self.axis_1 = device_list[0].get_axis(1)
        self.axis_2 = device_list[1].get_axis(1)
        self.axis_3 = device_list[2].get_axis(1)

        # Home each axis if home_first is True
        if config.home_first: 
            self.axis_1.home()
            self.axis_2.home()
            self.axis_3.home()

        input("Press Enter to move each axis to the initial position...")
        self.reset_robot()
        print("Motors initialized.")

        print("Initializing tracker...")
        self.tracker = Tracker(config.experiment_name, config.save_dir, config.csv_path)
        print("Tracker initialized.")

        # Global stop event to allow threads to exit cleanly
        self.stop_event = threading.Event()

        print("Robot API initialized.\n")
    
    def send_command(self, action):
        # Move each axis by the action vector
        self.axis_1.move_absolute(config.initial_pos + float(action[0]), Units.LENGTH_MILLIMETRES, True)
        self.axis_2.move_absolute(config.initial_pos + float(action[1]), Units.LENGTH_MILLIMETRES, True)
        self.axis_3.move_absolute(config.initial_pos + float(action[2]), Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.1)
    
    def get_current_tip(self):
        tip = self.tracker.get_current_tip()
        base = self.tracker.get_current_base()
        if tip is None or base is None:
            print("Tracker data not available, returning default zero vector.")
            return np.zeros(3, dtype=np.float32)
        # Convert to numpy arrays and subtract
        dif = np.array(tip) - np.array(base)
        # Flatten the result to match shape (3,)
        dif = np.squeeze(dif)
        return dif.astype(np.float32)

    def get_signal_handler(self):
        return self.signal_handler
    
    def get_tracker(self):
        return self.tracker

    def reset_robot(self):
        # Move each axis to the initial position
        self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        time.sleep(1)

    # Signal handler to allow graceful exit on Ctrl+C
    def signal_handler(self, sig, _frame):
        print("Exiting... Signal:", sig)
        self.stop_event.set()  # Signal threads to stop
        time.sleep(0.5)   # Allow some time for cleanup
        sys.exit(0)

    # Function to update tracker data in a separate thread.
    def update_tracker(self, tracker):
        while not self.stop_event.is_set():
            self.tracker.real_time_tracking()
            time.sleep(1)
