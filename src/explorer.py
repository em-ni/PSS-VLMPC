import csv
import os
import socket
import struct
import threading
import time
import numpy as np
import pandas as pd
from zaber_motion import Units
from zaber_motion.ascii import Connection
import cv2
import src.config as config
# import config as config

class Explorer:
    def __init__(self, save_dir, csv_path, offsets, realtime=False):
        print("Initializing Explorer...")
        self.cam_left_index = config.cam_left_index
        self.cam_right_index = config.cam_right_index
        self.save_dir = save_dir
        self.output_file = csv_path
        self.pressure_values = []
        self.offsets = offsets
        self.initial_pos_1 = config.initial_pos + self.offsets[0]
        self.initial_pos_2 = config.initial_pos + self.offsets[1]
        self.initial_pos_3 = config.initial_pos + self.offsets[2]

        if realtime:
            self.quit = False
            self.dt = config.SCOPE_RECORD_DT
            self.start_delay = 0.01 # ms
            self.save_signal = False
            self.temp_csv_path = config.exp_temp_csv
            if os.path.exists(self.temp_csv_path):
                os.remove(self.temp_csv_path)
            today = time.strftime("%Y-%m-%d")
            time_now = time.strftime("%H-%M-%S")
            experiment_name = "exp_" + today + "_" + time_now
            save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
            self.rt_data_csv_path = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Init axis and oscilloscopes
            self.axis_1 = None
            self.axis_2 = None
            self.axis_3 = None
            self.scope_1 = None
            self.scope_2 = None
            self.scope_3 = None

            # Init devices and scopes
            self.connect_devices()

        print(f"Initial positions: {self.initial_pos_1}, {self.initial_pos_2}, {self.initial_pos_3}")
        print(f"Explorer initialized\n")

        return 

    def connect_devices(self):
        """
        Connect to the Zaber devices and initialize the oscilloscopes.
        """
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()
        device_list = connection.detect_devices()
        print("Found {} devices.".format(len(device_list)))
        print(device_list)

        # Get the axis
        self.axis_1 = device_list[0].get_axis(1)
        self.axis_2 = device_list[1].get_axis(1)
        self.axis_3 = device_list[2].get_axis(1)

        # Initialize oscilloscopes
        self.scope_1 = device_list[0].oscilloscope
        self.scope_2 = device_list[1].oscilloscope
        self.scope_3 = device_list[2].oscilloscope
        print(f'Oscilloscope 1 can store {self.scope_1.get_max_buffer_size()} samples.')
        print(f'Oscilloscope 2 can store {self.scope_2.get_max_buffer_size()} samples.')
        print(f'Oscilloscope 3 can store {self.scope_3.get_max_buffer_size()} samples.')
        return

    def get_image(self, cam_index, timestamp):
        """
        Input: cam_index - Index of the camera
        Output: img - Image from the camera
        """

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # cap = cv2.VideoCapture(cam_index)
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Error: Cannot access the camera")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return

        photo_name = f"cam_{cam_index}_{timestamp}.png"
        photo_path = os.path.join(self.save_dir, photo_name)
        cv2.imwrite(photo_path, frame)
        # print(f"Photo saved at {photo_path}")

        cap.release()
        cv2.destroyAllWindows()

        return frame, photo_path

    def get_pressure_values(self):
        return self.pressure_values

    def join_temp_files(self):
        """
        Join the temporary files into the main csv file.
        explorer fills the columns: 'trajectory', 'T (ms)', 'volume_1 (mm)', 'volume_2 (mm)', 'volume_3 (mm)',
        tracker fills the columns: 'k', 'pressure_1', 'pressure_2', 'pressure_3', 'base_x', 'base_y', 'base_z', 'tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z', 'tip_acceleration_x', 'tip_acceleration_y', 'tip_acceleration_z'
        They are joint in a single new file appending the columns from the tracker to the explorer csv file.

        """
        try:
            # Load both CSV files
            exp_df = pd.read_csv(config.exp_temp_csv)
            trk_df = pd.read_csv(config.track_temp_csv)
            exp_length = len(exp_df)
            trk_length = len(trk_df)

            if exp_df.empty or trk_df.empty:
                print("One of the DataFrames is empty.")
                return

            print("Explorer DataFrame:")
            print(exp_df.head())
            print("Tracker DataFrame:")
            print(trk_df.head())

            # Use time columns for alignment
            explorer_time = exp_df['T (ms)'].to_numpy() / 1000.0 # 'T (ms)' in seconds
            tracker_time = trk_df['timestamp'].to_numpy() - trk_df['timestamp'].iloc[0] # difference between the last and first timestamp
            tracker_time = tracker_time.astype(float)

            print(f"Explorer total time: {explorer_time[-1]} seconds")
            print(f"Tracker total time: {tracker_time[-1]} seconds")

            # Interpolate Explorer data to Tracker timestamps if lengths do not match 
            # TODO:check this ai code may be shit 
            if exp_length != trk_length:
                # Set unique index for interpolation
                exp_df_interp = exp_df.copy()
                exp_df_interp['time_s'] = explorer_time
                # Interpolate each column except 'trajectory' and 'time_s'
                interp_cols = [col for col in exp_df_interp.columns if col not in ['trajectory', 'T (ms)', 'time_s']]
                # Build new DataFrame with interpolated values
                interp_dict = {'trajectory': exp_df_interp['trajectory'].iloc[0] if 'trajectory' in exp_df_interp.columns else 0,
                              'T (ms)': tracker_time * 1000.0}
                for col in interp_cols:
                    interp_dict[col] = np.interp(tracker_time, exp_df_interp['time_s'], exp_df_interp[col])
                exp_df_resampled = pd.DataFrame(interp_dict)
                print(f"Resampled Explorer DataFrame to {len(exp_df_resampled)} rows")
            else:
                exp_df_resampled = exp_df.copy()


            # Reset index for joining
            exp_df_resampled = exp_df_resampled.reset_index(drop=True)
            trk_df = trk_df.reset_index(drop=True)

            # Join the two dataframes on the index
            joined_df = pd.concat([exp_df_resampled, trk_df], axis=1)

            if not os.path.exists(self.rt_data_csv_path):
                # Save the joined dataframe to the output file
                joined_df.to_csv(self.rt_data_csv_path, index=False)
                print(f"Joined data saved to {self.rt_data_csv_path}")
            else:
                # Append to the existing file
                joined_df.to_csv(self.rt_data_csv_path, mode='a', header=False, index=False)
                print(f"Appended data to {self.rt_data_csv_path}")

            # Remove the temporary files
            os.remove(config.track_temp_csv)
            os.remove(config.exp_temp_csv)
            print("Temporary files removed.")

            # Reset the save signal after joining files
            self.save_signal = False

        except Exception as e:
            print("Error reading CSV files:")
            print(e)
            return

    def listen_pressure_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_PRESSURE_PORT))
        sock.settimeout(1.0)
        print(f"Listening on {config.UDP_IP}:{config.UDP_PRESSURE_PORT}")
        try:
            while not self.quit:
                try:
                    data, addr = sock.recvfrom(1024)
                    try:
                        values_double = struct.unpack('ddd', data)
                        cur_pressure_1 = values_double[0]
                        cur_pressure_2 = values_double[1]
                        cur_pressure_3 = values_double[2]
                        cur_pressure_values = [cur_pressure_1, cur_pressure_2, cur_pressure_3]
                        self.set_pressure_values(cur_pressure_values)
                    except:
                        print(f"Received unhandled data type: {data} from {addr}")
                        pass
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("Stopping receiver.")
        finally:
            sock.close()

    def listen_save_signal(self):
        """
        Listen to the save signal from the tracker to explorer via UDP.
        This is used to waiting for the tracker to write the data to the csv file before joining the two csv files.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_T2E_TRACK_SIGNAL_PORT))
        sock.settimeout(1.0)
        print(f"Explorer listening for save signal on {config.UDP_IP}:{config.UDP_T2E_TRACK_SIGNAL_PORT}")
        try:
            while not self.quit:
                try:
                    data, addr = sock.recvfrom(1024)
                    save_signal = struct.unpack('?', data)[0]
                    self.save_signal = save_signal
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("Stopping save signal listener.")
        finally:
            sock.close()

    def move(self):
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()

        try:
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

            # Move each axis to the initial position
            self.axis_1.move_absolute(self.initial_pos_1, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(self.initial_pos_2, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(self.initial_pos_3, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            
            userInput = input("Enter 2 to continue:\n")
            
            if userInput == '2':
                i = 0
                j = 0
                k = 0
                j_flipFlag = 1
                k_flipFlag = 1
                stepCounter = 0  
                windowsteps = config.window_steps # Total steps in each window
                elongationstepSize = config.elongationstepSize  # Step size for elongation
                elongationstep = 0
                position_i = self.initial_pos_1
                position_j = self.initial_pos_2
                position_k = self.initial_pos_3
                while elongationstep <= (config.steps-config.window_steps):
                    print("\r", elongationstep)
                    while i <= windowsteps + elongationstep:
                        while j <= windowsteps + elongationstep:
                            while k <= windowsteps + elongationstep:
                                if k == elongationstep and k_flipFlag == -1:
                                    k_flipFlag = -k_flipFlag
                                    print("\r", i, j, k, " ", end="", flush=True)
                                    self.step_and_save(position_i, position_j, position_k)
                                    position_k = self.initial_pos_3 + k * config.stepSize
                                    stepCounter += 1
                                    break
                                if k == windowsteps+elongationstep and k_flipFlag == 1:
                                    k_flipFlag = -k_flipFlag
                                    print("\r", i, j, k, " ", end="", flush=True)
                                    self.step_and_save(position_i, position_j, position_k)
                                    position_k = self.initial_pos_3 + k * config.stepSize
                                    stepCounter += 1
                                    break
                                print("\r", i, j, k, " ", end="", flush=True)
                                self.step_and_save(position_i, position_j, position_k)  
                                position_k = self.initial_pos_3 + k * config.stepSize
                                k = k + k_flipFlag
                                stepCounter += 1
                            if j == elongationstep and j_flipFlag == -1:
                                j_flipFlag = -j_flipFlag
                                position_j = self.initial_pos_2 + j * config.stepSize
                                break
                            if j == windowsteps+elongationstep and j_flipFlag == 1:
                                j_flipFlag = -j_flipFlag
                                position_j = self.initial_pos_2 + j * config.stepSize
                                break
                            j = j + j_flipFlag
                            position_j = self.initial_pos_2 + j * config.stepSize
                        position_i = self.initial_pos_1 + i * config.stepSize
                        i = i + 1
                    j += elongationstepSize
                    k += elongationstepSize
                    elongationstep += windowsteps
            
            self.axis_1.move_absolute(self.initial_pos_1, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(self.initial_pos_2, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(self.initial_pos_3, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.2)
            print("Finished explorer")
            
        except Exception as exception:
            connection.close()
            raise exception
            
        connection.close()

    def run(self):
        # Listen to the UDP connection in a separate thread
        udp_thread = threading.Thread(target=self.listen_pressure_udp)
        udp_thread.start()

        try:
            # Execute the movement
            self.move()
            # self.move_from_csv()
        except Exception as exception:
            print("An error occurred, stopping the motors")
            print(exception)
            

        # Probably I should join the thread here

    def run_realtime(self):
        """
        Run the explorer in real-time mode.
        Run random trajectories and signal the tracker to track the movements.
        """

        # Start UDP listener for save signal
        udp_listener_thread = threading.Thread(target=self.listen_save_signal)
        udp_listener_thread.start()
        time.sleep(1)  # Small delay to ensure the listener thread has time to start

        traj_n = 0
        while traj_n < config.N_TRAJECTORIES:
            print(f"\nRunning trajectory {traj_n + 1}/{config.N_TRAJECTORIES}")
            # Clear, setup and start scopes
            self.setup_scopes()
            self.start_scopes()

            # Get the next position
            next_points = np.random.uniform(0, config.max_stroke, size=(3,))

            # Send the track signal to the tracker
            self.send_track_signal(True)
            time.sleep(0.5) # With this delay the tracking duration is almost always 0.1 s longer than the trajectory duration
            start = time.time()

            # Move the motors to the next points
            self.axis_1.move_absolute(self.initial_pos_1 + next_points[0], Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(self.initial_pos_2 + next_points[1], Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(self.initial_pos_3 + next_points[2], Units.LENGTH_MILLIMETRES, False)
            self.axis_1.wait_until_idle()
            self.axis_2.wait_until_idle()    
            self.axis_3.wait_until_idle()
            time.sleep(0.1)

            # Stop the track signal
            end_time = time.time()
            traj_time = (end_time - start)
            print(f"Time taken for trajectory: {traj_time} s")
            time.sleep(0.5)
            self.send_track_signal(False)
            time.sleep(0.001)

            # Write volumes to csv
            self.write_volumes_to_csv(traj_n)

            # Wait for the save signal to be true before joining the two csv
            while not self.save_signal:
                time.sleep(0.1)

            # Save the data in the csv file
            self.join_temp_files()

            # Increase the traj_n
            traj_n += 1

            time.sleep(0.01)  # Small delay to ensure the tracker has time to process the data

        # After all trajectories are done, send the quit signal to the tracker
        print("\nAll trajectories completed. Sending quit signal to tracker.")
        self.send_quit_signal()

        # Quit
        self.quit = True
        time.sleep(0.1)  # Small delay to ensure the quit signal is sent
        udp_listener_thread.join()
        time.sleep(0.1)  # Small delay to ensure the listener thread has time to finish

        # Move to the initial position
        self.axis_1.move_absolute(self.initial_pos_1, Units.LENGTH_MILLIMETRES, False)
        self.axis_2.move_absolute(self.initial_pos_2, Units.LENGTH_MILLIMETRES, False)
        self.axis_3.move_absolute(self.initial_pos_3, Units.LENGTH_MILLIMETRES, False)
        self.axis_1.wait_until_idle()
        self.axis_2.wait_until_idle()
        self.axis_3.wait_until_idle()

        return

    def save_data(self, volume_values, pressure_values, frame_1_name, frame_2_name, timestamp):
        """
        Save data in a csv with columns:
        timestamp - volume_1 - volume_2 - volume_3 - frame_1 - frame_2 - tip_x - tip_y - tip_z - base_x - base_y - base_z
        """
        # Not checking if file exists since it should be created in config.py
        with open(self.output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp] + volume_values + pressure_values + [frame_1_name, frame_2_name])

    def send_quit_signal(self):
        """
        Send a quit signal to the tracker via UDP.
        This is used to stop the execution in real-time mode.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Pack the boolean as a single byte
        data = struct.pack('?', True)
        sock.sendto(data, (config.UDP_IP, config.UDP_QUIT_TRACK_PORT))
        sock.close()
        print("Sent quit signal to tracker.")

    def set_pressure_values(self, pressure_values):
        self.pressure_values = pressure_values

    def setup_scopes(self):
        """
        Setup the oscilloscopes for data collection.
        """
        self.scope_1.clear()
        self.scope_2.clear()
        self.scope_3.clear()
        self.scope_1.add_channel(1, 'encoder.pos')  
        self.scope_2.add_channel(1, 'encoder.pos')
        self.scope_3.add_channel(1, 'encoder.pos')
        self.scope_1.set_timebase(self.dt, Units.TIME_MILLISECONDS)
        self.scope_1.set_delay(self.start_delay, Units.TIME_MILLISECONDS)
        self.scope_2.set_timebase(self.dt, Units.TIME_MILLISECONDS)
        self.scope_2.set_delay(self.start_delay, Units.TIME_MILLISECONDS)
        self.scope_3.set_timebase(self.dt, Units.TIME_MILLISECONDS)
        self.scope_3.set_delay(self.start_delay, Units.TIME_MILLISECONDS)

    def start_scopes(self):
        """
        Start the oscilloscopes.
        """
        self.scope_1.start()
        self.scope_2.start()
        self.scope_3.start()
        time.sleep(self.start_delay / 1000)  # Convert ms to seconds for sleep

    def write_volumes_to_csv(self, trajectory):
        """
        Write the volumes to a csv file.
        """
        if not os.path.exists(self.temp_csv_path):
            # Init csv file
            with open(self.temp_csv_path, mode="w", newline="") as csvfile:
                fieldnames = ['trajectory', 
                            'T (ms)', 
                            'volume_1 (mm)', 
                            'volume_2 (mm)', 
                            'volume_3 (mm)',
                            ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        data_1 = self.scope_1.read()
        data_2 = self.scope_2.read()
        data_3 = self.scope_3.read()
        encoder_1 = data_1[0]
        encoder_samples_1 = encoder_1.get_data(Units.LENGTH_MILLIMETRES)
        encoder_1_times = encoder_1.get_sample_times(Units.TIME_MILLISECONDS)
        encoder_2 = data_2[0]   
        encoder_samples_2 = encoder_2.get_data(Units.LENGTH_MILLIMETRES)
        encoder_2_times = encoder_2.get_sample_times(Units.TIME_MILLISECONDS)
        encoder_3 = data_3[0]
        encoder_samples_3 = encoder_3.get_data(Units.LENGTH_MILLIMETRES)
        encoder_3_times = encoder_3.get_sample_times(Units.TIME_MILLISECONDS)

        # Verify encoder samples lengths and times
        if len(encoder_samples_1) != len(encoder_samples_2) or len(encoder_samples_1) != len(encoder_samples_3):
            print("Error: Encoder samples lengths do not match.")
            return
        if len(encoder_1_times) != len(encoder_2_times) or len(encoder_1_times) != len(encoder_3_times):
            print("Error: Encoder times lengths do not match.")
            return

        with open(self.temp_csv_path, 'a') as file:
            for i in range(min(len(encoder_samples_1), len(encoder_samples_2), len(encoder_samples_3))):
                file.write(f'{trajectory},')
                file.write(f'{round(encoder_1_times[i],4)},')
                file.write(f'{encoder_samples_1[i]},{encoder_samples_2[i]},{encoder_samples_3[i]}\n')
        print(f"Written {i} rows to {self.temp_csv_path}")

    def send_track_signal(self, track_signal):
        """
        Send the track signal to the tracker via UDP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Convert boolean to int (1 or 0), then pack as a single byte
        data = struct.pack('?', bool(track_signal))
        sock.sendto(data, (config.UDP_IP, config.UDP_E2T_TRACK_SIGNAL_PORT))
        sock.close()

    def step_and_save(self, position_i, position_j, position_k):
        """
        Move the motors to the position and save the data
        """

        # Skip if positions are above the maximum volume
        if position_i > (config.max_vol_1 + self.offsets[0])  or position_j > (config.max_vol_2 + self.offsets[1]) or position_k > (config.max_vol_3 + self.offsets[2]):
            print("Position is above the maximum volume, skipping")
            return

        # Move the motors
        self.axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
        self.axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
        self.axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)

        # Wait for the motors to move
        time.sleep(0.2)

        # Volume values
        volume_values = [position_i, position_j, position_k]
        pressure_values = self.get_pressure_values()

        # Take images from the cameras
        timestamp = time.time()
        _, img1_path = self.get_image(self.cam_left_index, timestamp)
        _, img2_path = self.get_image(self.cam_right_index, timestamp)

        # Save data
        self.save_data(volume_values, pressure_values, img1_path, img2_path, timestamp)

if __name__ == "__main__":
    # save_dir = config.save_dir
    # csv_path = config.csv_path
    save_dir = os.path.abspath(os.path.join(os.getcwd(), "test_realtime"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_path = os.path.abspath(os.path.join(save_dir, f"test_rt.csv"))
    explorer = Explorer(save_dir, csv_path, [0,0,0], realtime=True)
    # explorer.run()
    explorer.run_realtime()