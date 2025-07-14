import os
import socket
import struct
import threading
import time
import cv2
import numpy as np
import yaml
import csv
import src.config as config
# import config as config

# Threaded camera stream for fast frame grabbing
class CameraStream:
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class Tracker:
    def __init__(self, experiment_name, save_dir, csv_path, realtime=False):
        print("Initializing Tracker...")
        self.experiment_name = experiment_name
        self.save_dir = save_dir
        self.csv_path = csv_path

        # Initialize the projection matrices
        self.P_right_matrix = None
        self.P_left_matrix = None

        # Load the projection matrices
        self.P_right_matrix = self.load_projection_matrix(config.P_right_yaml)
        # print("Projection Matrix for right camera:\n", self.P_right_matrix)
        self.P_left_matrix = self.load_projection_matrix(config.P_left_yaml)
        # print("Projection Matrix for left camera:\n", self.P_left_matrix)

        # Backup base positions
        self.base_left_bck = None
        self.base_right_bck = None
        self.first_base = False

        # Camera indices
        self.cam_left_index = config.cam_left_index
        self.cam_right_index = config.cam_right_index

        # Current tip and base positions
        self.cur_base_3d = None
        self.cur_tip_3d = None

        # Current body position
        self.cur_body_3d = None
        self.alpha = 0.2  # Smoothing factor (0.0-1.0) - lower means more smoothing
        self.filtered_body_3d = None  # Initialize filtered coordinates

        # If realtime tracking track also velocity and acceleration
        if realtime:
            # Track boolean set by explorer
            self.quit = False # Used to exit the program
            self.track = False # Used to start/stop tracking
            self.temp_csv_path = config.track_temp_csv

            # Data buffer for real-time tracking
            self.data_buffer = []  # Buffer to store data in memory before writing to file

            # Create a temporary csv file for real-time tracking    
            os.makedirs(os.path.dirname(self.temp_csv_path), exist_ok=True)

            # input variables
            self.cur_pressure_1 = None
            self.cur_pressure_2 = None
            self.cur_pressure_3 = None

            # Tracking variables
            self.prev_tip_3d = None
            self.prev_base_3d = None
            self.cur_tip_vel_3d = None
            self.prev_tip_vel_3d  = None
            self.cur_tip_acc_3d = None

            # Sampling frequency
            self.dt = config.TRACK_RECORD_DT

            # Initialize threads
            self.init_rt_threads()

        print(f"\nTracker initialized")

    # These three could be a single function...
    def detect_tip(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the red tip of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make masks for red color
        mask_red1 = cv2.inRange(hsv, config.lower_red1, config.upper_red1)
        mask_red2 = cv2.inRange(hsv, config.lower_red2, config.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Find contours in the mask.
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No red tip detected.")
            return None

        # Choose the largest contour.
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        return cx, cy

    def detect_base(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the base of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make mask for yellow color
        mask_yellow = cv2.inRange(hsv, config.lower_yellow, config.upper_yellow)
        
        # Find contours in the mask.
        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No yellow base detected.")
            return None
        
        # Choose the largest contour.
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        return cx, cy
    
    def detect_body(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the body of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make mask for green color
        mask_blue = cv2.inRange(hsv, config.lower_blue, config.upper_blue)
        
        # Find contours in the mask.
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No green body detected.")
            return None
        
        # Choose the largest contour.
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        return cx, cy

    def filter_coordinates(self, new_coords):
        """Apply exponential moving average filter to smooth coordinates"""
        if self.filtered_body_3d is None:
            # First measurement - initialize filter
            self.filtered_body_3d = new_coords
        else:
            # Apply EMA filter
            self.filtered_body_3d = self.alpha * new_coords + (1 - self.alpha) * self.filtered_body_3d
        
        return self.filtered_body_3d

    def get_current_tip(self):
        return self.cur_tip_3d
    
    def get_current_base(self):
        return self.cur_base_3d
    
    def get_current_body(self):
        return self.filtered_body_3d

    def get_image_from_csv(self, img_path):
        """
        Input: row - Row of the csv file
                column - Column of the csv file
        Output: img - Image from the camera
        """
        if img_path is None:
            print("Error: Could not read frame")
            return
        
        frame = cv2.imread(img_path)
        return frame

    def get_pressure(self):
        """
        Get the current pressure values.
        """
        return self.cur_pressure_1, self.cur_pressure_2, self.cur_pressure_3
    
    def get_track_signal(self):
        """
        Get the current track signal.
        """
        return self.track

    def init_rt_threads(self):
        # Listen to the UDP connection in a separate thread for pressure data
        self.udp_thread_pressure = threading.Thread(target=self.listen_pressure_udp)
        self.udp_thread_pressure.start()
        time.sleep(0.1)

        # Listen to the UDP connection in a separate thread for track signal
        self.udp_thread_track_signal = threading.Thread(target=self.listen_track_signal_udp)
        self.udp_thread_track_signal.start()
        time.sleep(0.1)

        # Listen to the UDP connection in a separate thread for quit signal
        self.udp_thread_quit = threading.Thread(target=self.listen_quit_signal)
        self.udp_thread_quit.start()
        time.sleep(0.1)

        # Give some time for the threads to start
        time.sleep(1)  
        print("Real-time threads initialized.")

    def listen_pressure_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_PRESSURE_PORT))
        sock.settimeout(1.0)
        print(f"Tracker listening on {config.UDP_IP}:{config.UDP_PRESSURE_PORT}")
        try:
            while not self.quit:
                try:
                    data, addr = sock.recvfrom(1024)
                    try:
                        values_double = struct.unpack('ddd', data)
                        self.cur_pressure_1 = values_double[0]
                        self.cur_pressure_2 = values_double[1]
                        self.cur_pressure_3 = values_double[2]
                    except:
                        print(f"Received unhandled data type: {data} from {addr}")
                        pass
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("Stopping receiver.")
        finally:
            sock.close()

    def listen_quit_signal(self):
        """
        Listen to the quit signal from the explorer via UDP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_QUIT_TRACK_PORT))
        sock.settimeout(1.0)
        print(f"Tracker listening for quit signal on {config.UDP_IP}:{config.UDP_QUIT_TRACK_PORT}")
        try:
            while not self.quit:
                try:
                    data, addr = sock.recvfrom(1024)
                    try:
                        quit_signal = struct.unpack('?', data)[0]
                        if quit_signal:
                            self.quit = True
                            print(f"Received quit signal: {quit_signal} from {addr}")
                            return
                    except:
                        print(f"Received unhandled data type: {data} from {addr}")
                        pass
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("Stopping receiver.")
        finally:
            sock.close()

    def listen_track_signal_udp(self):
        """
        Listen to the track signal from the explorer via UDP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_E2T_TRACK_SIGNAL_PORT))
        sock.settimeout(1.0)
        print(f"Tracker listening for track signal on {config.UDP_IP}:{config.UDP_E2T_TRACK_SIGNAL_PORT}")
        try:
            while not self.quit:
                try:
                    data, addr = sock.recvfrom(1024)
                    try:
                        track_signal = struct.unpack('?', data)[0]
                        self.track = track_signal
                        if self.quit:
                            return
                    except:
                        print(f"Received unhandled data type: {data} from {addr}")
                        pass
                except socket.timeout:
                    continue
        except KeyboardInterrupt:
            print("Stopping receiver.")
        finally:
            sock.close()

    def load_projection_matrix(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        P = np.array(data["projection_matrix"], dtype=np.float64)
        return P
    
    # For data collection
    def run_realtime_tracking(self):
        """
        Function to track the robot in real-time, buffering data in memory and writing to file at the end.
        """
        debug = False
        # Use threaded camera streams
        cam_left = CameraStream(self.cam_left_index)
        cam_right = CameraStream(self.cam_right_index)
        time.sleep(0.5)  # Let threads warm up
        # Check if cameras opened
        if not cam_left.cap.isOpened() or not cam_right.cap.isOpened():
            print("Error: Couldn't open the cameras.")
            cam_left.release()
            cam_right.release()
            return

        k = 0
        while self.quit is False:
            if self.track:
                
                # Wait according to the sampling frequency
                time.sleep(self.dt/1000)

                # Start the timer
                start_track = time.time()

                # Read the frames from the camera threads
                start_read = time.time()
                ret_left, frame_left = cam_left.read()
                ret_right, frame_right = cam_right.read()
                end_read = time.time()
                read_time = end_read - start_read
                L_w = cam_left.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                L_h = cam_left.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                R_w = cam_right.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                R_h = cam_right.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if debug: print("\r\nReading frames at resolution L:{}x{} and R:{}x{}".format(L_w, L_h, R_w, R_h), end="", flush=True)
                if debug: print("\r\nRead time: {} ms".format(round(read_time*1000, 3)))

                # Read pressure as close as possible to the frame reading
                pressure_1, pressure_2, pressure_3 = self.get_pressure()
                p = [pressure_1, pressure_2, pressure_3]

                if not ret_left or not ret_right:
                    print("Error: Couldn't read the frames.")
                    break

                # Triangulate the points
                start_triangulate = time.time()
                tip_3d, base_3d, _ = self.triangulate(frame_left, frame_right, with_body=False)
                if tip_3d is None or base_3d is None:
                    print("\nBad triangulation. Skipping this iteration.")
                    continue
                end_triangulate = time.time()
                triangulation_time = end_triangulate - start_triangulate
                if debug: print("\rTriangulation time: {} ms".format(round(triangulation_time*1000, 3)))

                # Set the current tip and base positions
                start_buffer = time.time()
                self.cur_tip_3d = tip_3d - base_3d # NOTE: tip position is relative to the base position
                self.cur_base_3d = base_3d

                # If there are at least two previous positions calculate velocity
                if self.prev_tip_3d is not None and self.prev_base_3d is not None:
                    # Calculate the velocity of the tip and base
                    self.cur_tip_vel_3d = (self.cur_tip_3d - self.prev_tip_3d) / self.dt
                    self.cur_base_vel_3d = (self.cur_base_3d - self.prev_base_3d) / self.dt

                    # Calculate the acceleration of the tip
                    if self.prev_tip_vel_3d is not None:
                        self.cur_tip_acc_3d = (self.cur_tip_vel_3d - self.prev_tip_vel_3d) / self.dt

                # Buffer the real-time tracking data in memory
                row = {
                    'k': k,
                    'pressure_1': p[0] if p[0] is not None else None,
                    'pressure_2': p[1] if p[1] is not None else None,
                    'pressure_3': p[2] if p[2] is not None else None,
                    'base_x': self.cur_base_3d[0] if self.cur_base_3d is not None else None,
                    'base_y': self.cur_base_3d[1] if self.cur_base_3d is not None else None,
                    'base_z': self.cur_base_3d[2] if self.cur_base_3d is not None else None,
                    'tip_x': self.cur_tip_3d[0] if self.cur_tip_3d is not None else None,
                    'tip_y': self.cur_tip_3d[1] if self.cur_tip_3d is not None else None,
                    'tip_z': self.cur_tip_3d[2] if self.cur_tip_3d is not None else None,
                    'tip_velocity_x': self.cur_tip_vel_3d[0] if self.cur_tip_vel_3d is not None else None,
                    'tip_velocity_y': self.cur_tip_vel_3d[1] if self.cur_tip_vel_3d is not None else None,
                    'tip_velocity_z': self.cur_tip_vel_3d[2] if self.cur_tip_vel_3d is not None else None,
                    'tip_acceleration_x': self.cur_tip_acc_3d[0] if self.cur_tip_acc_3d is not None else None,
                    'tip_acceleration_y': self.cur_tip_acc_3d[1] if self.cur_tip_acc_3d is not None else None,
                    'tip_acceleration_z': self.cur_tip_acc_3d[2] if self.cur_tip_acc_3d is not None else None
                }
                self.data_buffer.append(row)
                end_buffer = time.time()
                buffer_time = end_buffer - start_buffer
                if debug: print("\rBuffering time: {} ms".format(round(buffer_time*1000, 3)))

                # Measure tracking time
                end_track = time.time()
                tracking_time = end_track - start_track
                if debug: print("\rTracking time: {} ms".format(round(tracking_time*1000, 3)))

                # Update the previous positions and velocities
                self.prev_tip_3d = self.cur_tip_3d.copy()
                self.prev_base_3d = self.cur_base_3d.copy()
                if self.cur_tip_vel_3d is not None:
                    self.prev_tip_vel_3d = self.cur_tip_vel_3d.copy() 

                # Update counter
                k += 1
            else:
                # If data buffer is not empty write to file
                if self.data_buffer != []:
                    self.write_data_buffer_to_csv()

        # Wait for the threads to finish
        self.udp_thread_pressure.join()
        print("Pressure thread finished.")
        time.sleep(1)  
        self.udp_thread_track_signal.join()
        print("Track signal thread finished.")
        time.sleep(1)
        self.udp_thread_quit.join()
        print("Quit signal thread finished.")
        time.sleep(1)

        # Release the camera threads
        cam_left.release()
        cam_right.release()
        return

    # For animations
    def real_time_tracking(self):
        """
        Function to track the robot in real-time
        """
        # Initialize the camera
        cap_left = cv2.VideoCapture(self.cam_left_index, cv2.CAP_DSHOW)
        cap_right = cv2.VideoCapture(self.cam_right_index, cv2.CAP_DSHOW)

        if not cap_left.isOpened() or not cap_right.isOpened():
            print("Error: Couldn't open the cameras.")
            return

        while True:
            # Start the timer
            start = time.time()

            # Read the frames from the cameras
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                print("Error: Couldn't read the frames.")
                break

            # Triangulate the points
            tip_3d, base_3d, body_3d = self.triangulate(frame_left, frame_right)
            if tip_3d is None or base_3d is None:
                continue
            end = time.time()
            tracking_time = end - start
            
            # print("\rTracking time: {}".format(tracking_time), end="", flush=True)

            # Set the current tip and base positions
            self.cur_tip_3d = tip_3d
            self.cur_base_3d = base_3d

            if body_3d is not None:
                # Apply filtering to body coordinates
                self.cur_body_3d = self.filter_coordinates(body_3d)

            # # Display the frames
            # cv2.imshow("Left Camera", frame_left)
            # cv2.imshow("Right Camera", frame_right)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the cameras
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    def run(self):
        """
        main function
        """
        all_rows = []
        # Define the expected fieldnames including the new 3D coordinate columns
        fieldnames = ['timestamp', 'volume_1', 'volume_2', 'volume_3', 'pressure_1', 'pressure_2', 'pressure_3', 'img_left', 'img_right',
                    'tip_x', 'tip_y', 'tip_z', 'base_x', 'base_y', 'base_z']  # Add your existing columns plus the new ones

        # Get images from the csv file
        # If the CSV file does not exist, create it with the correct header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(self.csv_path, mode="r") as csvfile:
            reader = csv.DictReader(csvfile)  # Use DictReader to read the CSV

            # Ensure fieldnames match between the CSV file and the updated fieldnames
            if set(reader.fieldnames) != set(fieldnames):
                print("Warning: CSV columns do not match the expected columns!")

            # Read all rows into a list
            all_rows = list(reader)

        # Loop through all the rows
        for row in all_rows:
            # Read 4th and 5th columns of the csv file
            img_left_rel_path = row['img_left']
            img_right_rel_path = row['img_right']

            # take everything between experiment_name and .png
            img_left_name = img_left_rel_path[img_left_rel_path.find(self.experiment_name) + len(self.experiment_name) + 1: img_left_rel_path.find(".png") + 4]
            img_right_name = img_right_rel_path[img_right_rel_path.find(self.experiment_name) + len(self.experiment_name) + 1: img_right_rel_path.find(".png") + 4]

            # Get the full path of the images
            img_left_path = os.path.abspath(os.path.join(self.save_dir, img_left_name))
            img_right_path = os.path.abspath(os.path.join(self.save_dir, img_right_name))

            # Check if the files exist.
            if not os.path.exists(img_left_path):
                print("File does not exist:", img_left_path)
                continue
            if not os.path.exists(img_right_path):
                print("File does not exist:", img_right_path)
                continue

            # Load images using the helper function.
            img_left = self.get_image_from_csv(img_left_path)
            img_right = self.get_image_from_csv(img_right_path)

            if img_left is None or img_right is None:
                print("Error reading one of the images.")
                continue

            # Triangulate the points
            tip_3d, base_3d, _ = self.triangulate(img_left, img_right)
            if tip_3d is None or base_3d is None:
                print("Bad image paths:", img_left_path, img_right_path)
                continue
            print("\rTip coordinates: {}   Base coordinates: {}".format(tip_3d.flatten(), base_3d.flatten()), end="", flush=True)

            # Update the row with the new 3D coordinates
            row['tip_x'] = tip_3d[0][0]
            row['tip_y'] = tip_3d[1][0]
            row['tip_z'] = tip_3d[2][0]
            row['base_x'] = base_3d[0][0]
            row['base_y'] = base_3d[1][0]
            row['base_z'] = base_3d[2][0]

        # Write the updated rows back to the CSV file (overwrite the file)
        with open(self.csv_path, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()  # Write the header
            writer.writerows(all_rows)  # Write the updated rows

    def send_save_signal(self, save_signal):
        """
        Send the track signal to the explorer via UDP
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send the track signal to the explorer
        data = struct.pack('?', bool(save_signal))
        sock.sendto(data, (config.UDP_IP, config.UDP_T2E_TRACK_SIGNAL_PORT))
        sock.close()

    def triangulate(self, img_left, img_right, with_body=False):
        """
        Input: img_left - Image from left camera
                img_right - Image from right camera
        Output: x,y,z - Coordinates of the robot tip in 3D space
                x,y,z - Coordinates of the robot base in 3D space
        """

        # Get tip and base points from the images
        tip_left = self.detect_tip(img_left)
        base_left = self.detect_base(img_left)
        tip_right = self.detect_tip(img_right)
        base_right = self.detect_base(img_right)

        # Backup the base positions
        if base_left is not None and base_right is not None and self.first_base is False:
            self.base_left_bck = base_left
            self.base_right_bck = base_right
            self.first_base = True

        # Always use the very first detection of the base since the position is fixed
        if self.base_left_bck is not None:
            base_left = self.base_left_bck
        if self.base_right_bck is not None:
            base_right = self.base_right_bck
        
        if tip_left is None or base_left is None or tip_right is None or base_right is None:
            print("Couldn't detect all points in both images.")
            if base_left is None and self.base_left_bck is not None:
                print("Using the backup base position for the left camera.")
                base_left = self.base_left_bck
            if base_right is None and self.base_right_bck is not None:
                print("Using the backup base position for the right camera.")
                base_right = self.base_right_bck
            
            return None, None

        # Convert the points to the format required by triangulatePoints (2xN array)
        tip_left = np.array([[tip_left[0]], [tip_left[1]]], dtype=np.float32)  # (2, 1)
        base_left = np.array([[base_left[0]], [base_left[1]]], dtype=np.float32)  # (2, 1)
        tip_right = np.array([[tip_right[0]], [tip_right[1]]], dtype=np.float32)  # (2, 1)
        base_right = np.array([[base_right[0]], [base_right[1]]], dtype=np.float32)  # (2, 1)

        # Triangulate the points
        tip_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, tip_left, tip_right)
        base_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, base_left, base_right)
                                        
        # Convert from homogeneous coordinates to 3D.
        tip_3d = tip_4d[:3] / tip_4d[3]
        base_3d = base_4d[:3] / base_4d[3]

        # If with body
        body_3d = None
        if with_body:
            body_left = self.detect_body(img_left)
            body_right = self.detect_body(img_right)
            if body_left is None or body_right is None:
                print("Couldn't detect body points in both images.")
                return tip_3d, base_3d, None
            body_left = np.array([[body_left[0]], [body_left[1]]], dtype=np.float32)
            body_right = np.array([[body_right[0]], [body_right[1]]], dtype=np.float32)
            body_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, body_left, body_right)
            body_3d = body_4d[:3] / body_4d[3]

        return tip_3d, base_3d, body_3d

    def write_data_buffer_to_csv(self):
        """
        Write the buffered data to the CSV file.
        """

        # Prepare to buffer data in memory
        fieldnames = [
            'k', 'pressure_1', 'pressure_2', 'pressure_3',
            'base_x', 'base_y', 'base_z', 'tip_x', 'tip_y', 'tip_z',
            'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z',
            'tip_acceleration_x', 'tip_acceleration_y', 'tip_acceleration_z'
        ]

        # Ensure the CSV file exists and has the correct header
        if not os.path.exists(self.temp_csv_path):
            with open(self.temp_csv_path, mode="w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        with open(self.temp_csv_path, mode="a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.data_buffer[0].keys())
            for row in self.data_buffer:
                writer.writerow(row)
        print(f"Wrote {len(self.data_buffer)} rows to {self.temp_csv_path}")
        self.data_buffer = []  # Clear the buffer after writing

        # Send the save signal to the explorer
        self.send_save_signal(True)

if __name__ == "__main__":

    # Experiment name
    experiment_name = "exp_2025-07-01_15-18-16"
    save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
    output_file = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))

    tracker = Tracker(experiment_name, save_dir, output_file)
    tracker.run()