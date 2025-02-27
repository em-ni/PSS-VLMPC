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
        print("Projection Matrix for right camera:\n", self.P_right_matrix)
        self.P_left_matrix = self.load_projection_matrix(config.P_left_yaml)
        print("Projection Matrix for left camera:\n", self.P_left_matrix)

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

        # # Define the size of the circular structuring element
        # kernel_size = 7  # You can adjust the size as needed
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # # Perform erosion (shrinking the white regions in the mask)
        # mask_red = cv2.erode(mask_red, kernel, iterations=1)

        # # Perform dilation (expanding the white regions in the mask)
        # mask_red = cv2.dilate(mask_red, kernel, iterations=1)

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

        # # Get the red tip points
        # red_tip = cv2.bitwise_and(frame, frame, mask=mask_red)
        # red_tip_points = np.where(red_tip > 0)
        # red_tip_points = np.column_stack((red_tip_points[1], red_tip_points[0]))

        # if red_tip_points.size == 0:
        #     return np.nan, np.nan

        # # Get average of coorinates of point in red tip
        # red_tip_x = red_tip_points[:, 0]
        # red_tip_y = red_tip_points[:, 1]
        # x_tip = np.mean(red_tip_x)
        # y_tip = np.mean(red_tip_y)

        # return x_tip, y_tip

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
        # # Get the yellow base points
        # yellow_base = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        # yellow_base_gray = cv2.cvtColor(yellow_base, cv2.COLOR_BGR2GRAY)
        # _, yellow_base_thresh = cv2.threshold(
        #     yellow_base_gray, 127, 255, cv2.THRESH_BINARY
        # )
        # yellow_base_edges = cv2.Canny(yellow_base_thresh, 50, 150)
        # yellow_base_points = np.where((yellow_base_edges > 0))
        # yellow_base_points = np.column_stack(
        #     (yellow_base_points[1], yellow_base_points[0])
        # )

        # # Get average coordinates of yellow base
        # yellow_base_x = yellow_base_points[:, 0]
        # yellow_base_y = yellow_base_points[:, 1]
        # x_base = np.mean(yellow_base_x)
        # y_base = np.mean(yellow_base_y)

        # return x_base, y_base

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

        if tip_left is None or base_left is None or tip_right is None or base_right is None:
            print("Couldn't detect all points in both images.")
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

        return tip_3d, base_3d

if __name__ == "__main__":

    # Experiment name
    experiment_name = "exp_2025-02-27_12-16-06"
    save_dir = os.path.join(".", "data", experiment_name)
    output_file = os.path.join(save_dir, f"output_{experiment_name}.csv")

    tracker = Tracker(experiment_name, save_dir, output_file)
    tracker.run()