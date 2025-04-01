import os
import socket
import threading
import time
import cv2
import numpy as np
import yaml
import csv
import src.config as config
# import config as config

class Tracker:
    def __init__(self, experiment_name, save_dir, csv_path):
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

    def load_projection_matrix(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        P = np.array(data["projection_matrix"], dtype=np.float64)
        return P
    
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
            tip_3d, base_3d = self.triangulate(frame_left, frame_right)
            if tip_3d is None or base_3d is None:
                continue
            end = time.time()
            tracking_time = end - start
            
            # print("\rTracking time: {}".format(tracking_time), end="", flush=True)

            # Set the current tip and base positions
            self.cur_tip_3d = tip_3d
            self.cur_base_3d = base_3d

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
            tip_3d, base_3d = self.triangulate(img_left, img_right)
            if tip_3d is None or base_3d is None:
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

    def triangulate(self, img_left, img_right):
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
        body_left = self.detect_body(img_left)
        body_right = self.detect_body(img_right)

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
        
        if tip_left is None or base_left is None or tip_right is None or base_right is None or body_left is None or body_right is None:
            print("Couldn't detect all points in both images.")
            if base_left is None and self.base_left_bck is not None:
                print("Using the backup base position for the left camera.")
                base_left = self.base_left_bck
            if base_right is None and self.base_right_bck is not None:
                print("Using the backup base position for the right camera.")
                base_right = self.base_right_bck

        # Convert the points to the format required by triangulatePoints (2xN array)
        tip_left = np.array([[tip_left[0]], [tip_left[1]]], dtype=np.float32)  # (2, 1)
        base_left = np.array([[base_left[0]], [base_left[1]]], dtype=np.float32)  # (2, 1)
        tip_right = np.array([[tip_right[0]], [tip_right[1]]], dtype=np.float32)  # (2, 1)
        base_right = np.array([[base_right[0]], [base_right[1]]], dtype=np.float32)  # (2, 1)
        body_left = np.array([[body_left[0]], [body_left[1]]], dtype=np.float32)
        body_right = np.array([[body_right[0]], [body_right[1]]], dtype=np.float32)

        # Triangulate the points
        tip_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, tip_left, tip_right)
        base_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, base_left, base_right)
        body_4d = cv2.triangulatePoints(self.P_left_matrix, self.P_right_matrix, body_left, body_right)
                                        
        # Convert from homogeneous coordinates to 3D.
        tip_3d = tip_4d[:3] / tip_4d[3]
        base_3d = base_4d[:3] / base_4d[3]
        body_3d = body_4d[:3] / body_4d[3]
        
        if body_3d is not None:
            # Apply filtering to body coordinates
            self.cur_body_3d = self.filter_coordinates(body_3d)
        else:
            print("Couldn't detect the body in both images.")


        return tip_3d, base_3d

if __name__ == "__main__":

    # Experiment name
    experiment_name = "exp_2025-02-27_12-16-06"
    save_dir = os.path.join(".", "data", experiment_name)
    output_file = os.path.join(save_dir, f"output_{experiment_name}.csv")

    tracker = Tracker(experiment_name, save_dir, output_file)
    tracker.run()