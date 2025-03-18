import csv
import os
import socket
import struct
import threading
import time
import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Connection
import cv2
import src.config as config
# import config as config

class Explorer:
    def __init__(self, save_dir, csv_path):
        self.cam_left_index = config.cam_left_index
        self.cam_right_index = config.cam_right_index
        self.save_dir = save_dir
        self.output_file = csv_path
        self.pressure_values = []
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

    def listen_pressure_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_PORT))

        print(f"Listening on {config.UDP_IP}:{config.UDP_PORT}")
        try:
            while True:
                data, addr = sock.recvfrom(1024)  # Adjust buffer size if necessary
                try:
                    # Attempt to decode as a double precision float 
                    values_double = struct.unpack('ddd', data)
                    # print(f"Received double: {values_double[0]} from {addr}")
                    # print(f"Received double: {values_double[1]} from {addr}")
                    # print(f"Received double: {values_double[2]} from {addr}")
                    cur_pressure_1 = values_double[0]
                    cur_pressure_2 = values_double[1]
                    cur_pressure_3 = values_double[2]
                    cur_pressure_values = [cur_pressure_1, cur_pressure_2, cur_pressure_3]
                    self.set_pressure_values(cur_pressure_values)
                except:
                    print(f"Received unhandled data type: {data} from {addr}")
                    pass

        except KeyboardInterrupt:
            print("Stopping receiver.")
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

            # Move each axis to the minimum position
            self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            
            userInput = input("Enter 2 to continue:\n")
            
            if userInput == '2':
                i = 1
                j = 0
                k = 0
                j_flipFlag = 1
                k_flipFlag = 1
                stepCounter = 0  
                
                position_i = config.initial_pos
                position_j = config.initial_pos
                position_k = config.initial_pos
                
                while i <= config.steps + 1:
                    while j <= config.steps:
                        while k <= config.steps:
                            if k == 0 and k_flipFlag == -1:
                                k_flipFlag = -k_flipFlag
                                print("\r", i, j, k, " ", end="", flush=True)
                                self.step_and_save(position_i, position_j, position_k)
                                position_k = config.initial_pos + k * config.stepSize
                                stepCounter += 1
                                break
                            if k == config.steps and k_flipFlag == 1:
                                k_flipFlag = -k_flipFlag
                                print("\r", i, j, k, " ", end="", flush=True)
                                self.step_and_save(position_i, position_j, position_k)
                                position_k = config.initial_pos + k * config.stepSize
                                stepCounter += 1
                                break
                            print("\r", i, j, k, " ", end="", flush=True)
                            self.step_and_save(position_i, position_j, position_k)  
                            position_k = config.initial_pos + k * config.stepSize
                            k = k + k_flipFlag
                            stepCounter += 1
                        if j == 0 and j_flipFlag == -1:
                            j_flipFlag = -j_flipFlag
                            position_j = config.initial_pos + j * config.stepSize
                            break
                        if j == config.steps and j_flipFlag == 1:
                            j_flipFlag = -j_flipFlag
                            position_j = config.initial_pos + j * config.stepSize
                            break
                        j = j + j_flipFlag
                        position_j = config.initial_pos + j * config.stepSize
                    position_i = config.initial_pos + i * config.stepSize
                    i = i + 1
            
            self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.2)
            print("Finished explorer")
            
        except Exception as exception:
            connection.close()
            raise exception
            
        connection.close()

    def move_from_csv(self):
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

            # Move each axis to the minimum position
            self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            
            userInput = input("Enter 2 to continue:\n")
            
            if userInput == '2':
                with open(config.input_volume_path, mode="r") as csvfile:
                    reader = csv.DictReader(csvfile)
                    next(reader)  # Skip the header
                    for row in reader:
                        position_i = float(row['volume_1']) + config.initial_pos
                        position_j = float(row['volume_2']) + config.initial_pos
                        position_k = float(row['volume_3']) + config.initial_pos
                        self.step_and_save(position_i, position_j, position_k)
            
            self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
            self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
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
            # self.move()
            self.move_from_csv()
        except Exception as exception:
            print("An error occurred, stopping the motors")
            print(exception)
            

        # Probably I should join the thread here

    def save_data(self, volume_values, pressure_values, frame_1_name, frame_2_name, timestamp):
        """
        Save data in a csv with columns:
        timestamp - volume_1 - volume_2 - volume_3 - frame_1 - frame_2 - tip_x - tip_y - tip_z - base_x - base_y - base_z
        """
        # Not checking if file exists since it should be created in config.py
        with open(self.output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp] + volume_values + pressure_values + [frame_1_name, frame_2_name])

    def set_pressure_values(self, pressure_values):
        self.pressure_values = pressure_values

    def step_and_save(self, position_i, position_j, position_k):
        """
        Move the motors to the position and save the data
        """

        # Skip if positions are above the maximum volume
        if position_i > config.max_vol_1 or position_j > config.max_vol_2 or position_k > config.max_vol_3:
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
    save_dir = config.save_dir
    csv_path = config.csv_path
    explorer = Explorer(save_dir, csv_path)
    explorer.run()