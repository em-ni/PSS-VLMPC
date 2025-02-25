import csv
import os
import time
import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Connection
import cv2

cam_1_index = 1
cam_2_index = 0
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.join(".", "data", experiment_name)
output_file = os.path.join(save_dir, f"output_{experiment_name}.csv")

class Durability:
    def __init__(self):
        pass

    def get_image(self, cam_index, timestamp):
        """
        Input: cam_index - Index of the camera
        Output: img - Image from the camera
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("Error: Cannot access the camera")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            cap.release()
            return

        photo_name = f"cam_{cam_index}_{timestamp}.png"
        photo_path = os.path.join(save_dir, photo_name)
        cv2.imwrite(photo_path, frame)
        print(f"Photo saved at {photo_path}")

        cap.release()
        cv2.destroyAllWindows()

        return frame, photo_path

    def move(self):
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()
        
        # demonstration
        offset = 0  # 21 mm in comment, but offset is set to 0
        initial_pos = 110 + offset  # mm

        # PositionSweep Settings
        steps = 5
        stroke = 3  # mm
        stepSize = stroke / steps

        try:
            # connection.enableAlerts()  # (commented out as in MATLAB)
            device_list = connection.detect_devices()
            print("Found {} devices.".format(len(device_list)))
            print(device_list)
            
            # Home each axis
            axis_1 = device_list[0].get_axis(1)
            axis_1.home()
            axis_2 = device_list[1].get_axis(1)
            axis_2.home()
            axis_3 = device_list[2].get_axis(1)
            axis_3.home()

            # Move each axis to the minimum position
            axis_1.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_2.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_3.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            
            userInput = input("Enter 2 to continue:\n")
            
            if userInput == '2':
                position_matrix = np.zeros((3, 216))
                
                i = 1
                j = 0
                k = 0
                j_flipFlag = 1
                k_flipFlag = 1
                stepCounter = 0  # Python index starts at 0
                
                position_i = initial_pos
                position_j = initial_pos
                position_k = initial_pos
                
                while i <= steps + 1:
                    while j <= steps:
                        while k <= steps:
                            if k == 0 and k_flipFlag == -1:
                                k_flipFlag = -k_flipFlag
                                position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                                print(i, j, k)
                                axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                                axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                                axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                                time.sleep(0.2)

                                # Volume values
                                volume_values = [position_i, position_j, position_k]

                                # Take images from the cameras
                                timestamp = time.time()
                                _, img1_path = self.get_image(cam_1_index, timestamp)
                                _, img2_path = self.get_image(cam_2_index, timestamp)

                                # Save data
                                self.save_data(volume_values, img1_path, img2_path, timestamp)
                                
                                position_k = initial_pos + k * stepSize
                                stepCounter += 1
                                break
                            if k == steps and k_flipFlag == 1:
                                k_flipFlag = -k_flipFlag
                                position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                                print(i, j, k)
                                axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                                axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                                axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                                time.sleep(0.2)

                                # Volume values
                                volume_values = [position_i, position_j, position_k]

                                # Take images from the cameras
                                timestamp = time.time()
                                _, img1_path = self.get_image(cam_1_index, timestamp)
                                _, img2_path = self.get_image(cam_2_index, timestamp)

                                # Save data
                                self.save_data(volume_values, img1_path, img2_path, timestamp)

                                position_k = initial_pos + k * stepSize
                                stepCounter += 1
                                break

                            position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                            print(i, j, k)
                            axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                            axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                            axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                            time.sleep(0.2)

                            # Volume values
                            volume_values = [position_i, position_j, position_k]

                            # Take images from the cameras
                            timestamp = time.time()
                            _, img1_path = self.get_image(cam_1_index, timestamp)
                            _, img2_path = self.get_image(cam_2_index, timestamp)

                            # Save data
                            self.save_data(volume_values, img1_path, img2_path, timestamp)
                            
                            position_k = initial_pos + k * stepSize
                            k = k + k_flipFlag
                            stepCounter += 1
                        if j == 0 and j_flipFlag == -1:
                            j_flipFlag = -j_flipFlag
                            position_j = initial_pos + j * stepSize
                            break
                        if j == steps and j_flipFlag == 1:
                            j_flipFlag = -j_flipFlag
                            position_j = initial_pos + j * stepSize
                            break
                        j = j + j_flipFlag
                        position_j = initial_pos + j * stepSize
                    position_i = initial_pos + i * stepSize
                    i = i + 1
            
            axis_1.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_2.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_3.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.2)
            print("Finished")
            
        except Exception as exception:
            connection.close()
            raise exception
            
        connection.close()

    def run(self):
        # Execute the movement (which updates the message)
        self.move()


    def save_data(self, volume_values, frame_1_name, frame_2_name, timestamp):
        """
        Save data in a csv with columns:
        timestamp - pressure_1 - pressure_2 - ... - tip_x - tip_y - tip_z - base_x - base_y - base_z
        """
        header = (
            ["timestamp"]
            + [f"volume_{i+1}" for i in range(len(volume_values))]
            + ["frame_1", "frame_2"]
        )
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)

            writer.writerow([timestamp] + volume_values + [frame_1_name, frame_2_name])


    
if __name__ == "__main__":
    durability = Durability()
    durability.run()