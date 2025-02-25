import os
import socket
import threading
import time
import cv2
import numpy as np
import yaml
import csv

cam_1_index = 1
cam_2_index = 0

P1_yaml = os.path.join("calibration_images_cam4_640x480p", "projection_matrix.yaml")
P2_yaml = os.path.join("calibration_images_cam2_640x480p", "projection_matrix.yaml")

# Experiment name
experiment_name = "exp_2025-02-25_16-13-52"
save_dir = os.path.join(".", "data", experiment_name)
output_file = os.path.join(save_dir, f"output_{experiment_name}.csv")

# Colors range for detection
# Define a color range for yellow
lower_yellow = np.array([23, 88, 0])
upper_yellow = np.array([36, 254, 255])

# Define the color ranges for red
lower_red1 = np.array([0, 55, 0])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([171, 55, 0])
upper_red2 = np.array([180, 255, 255])


class Tracker:
    def __init__(self):

        self.csv_path = output_file

        # Initialize the projection matrices
        self.P1_matrix = None
        self.P2_matrix = None

        # Load the projection matrices
        self.P1_matrix = self.load_projection_matrix(P1_yaml)
        print("Projection Matrix for Camera 1 (P1):\n", self.P1_matrix)
        self.P2_matrix = self.load_projection_matrix(P2_yaml)
        print("Projection Matrix for Camera 2 (P2):\n", self.P2_matrix)

    def detect_tip(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the red tip of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make masks for red color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Get the red tip points
        red_tip = cv2.bitwise_and(frame, frame, mask=mask_red)
        red_tip_points = np.where(red_tip > 0)
        red_tip_points = np.column_stack((red_tip_points[1], red_tip_points[0]))

        if red_tip_points.size == 0:
            return np.nan, np.nan

        # Get average of coorinates of point in red tip
        red_tip_x = red_tip_points[:, 0]
        red_tip_y = red_tip_points[:, 1]
        x_tip = np.mean(red_tip_x)
        y_tip = np.mean(red_tip_y)

        return x_tip, y_tip

    def detect_base(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the base of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make mask for yellow color
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Get the yellow base points
        yellow_base = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        yellow_base_gray = cv2.cvtColor(yellow_base, cv2.COLOR_BGR2GRAY)
        _, yellow_base_thresh = cv2.threshold(
            yellow_base_gray, 127, 255, cv2.THRESH_BINARY
        )
        yellow_base_edges = cv2.Canny(yellow_base_thresh, 50, 150)
        yellow_base_points = np.where((yellow_base_edges > 0))
        yellow_base_points = np.column_stack(
            (yellow_base_points[1], yellow_base_points[0])
        )

        # Get average coordinates of yellow base
        yellow_base_x = yellow_base_points[:, 0]
        yellow_base_y = yellow_base_points[:, 1]
        x_base = np.mean(yellow_base_x)
        y_base = np.mean(yellow_base_y)

        return x_base, y_base

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
        rows = []
        # Get images from the csv file
        with open(self.csv_path, mode="r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                
                # Read 4th and 5th columns of the csv file
                img1_rel_path = row[4]
                img2_rel_path = row[5]

                # take everything between experiment_name and .png
                img1_name = img1_rel_path[img1_rel_path.find(experiment_name)+len(experiment_name)+1:img1_rel_path.find(".png")+4]
                img2_name = img2_rel_path[img2_rel_path.find(experiment_name)+len(experiment_name)+1:img2_rel_path.find(".png")+4]

                # Get the full path of the images
                img1_path = os.path.abspath(os.path.join(save_dir, img1_name))
                img2_path = os.path.abspath(os.path.join(save_dir, img2_name))

                print("Image 1 path:", img1_path)
                print("Image 2 path:", img2_path)

                # Check if the files exist.
                if not os.path.exists(img1_path):
                    print("File does not exist:", img1_path)
                    continue
                if not os.path.exists(img2_path):
                    print("File does not exist:", img2_path)
                    continue

                # Load images using the helper function.
                img1 = self.get_image_from_csv(img1_path)
                img2 = self.get_image_from_csv(img2_path)
                
                if img1 is None or img2 is None:
                    print("Error reading one of the images.")
                    continue

                # Triangulate the points
                tip_3d, base_3d = self.triangulate(
                    img1, img2
                )
                print("\rTip coordinates:", tip_3d, end="", flush=True)
                print("\rBase coordinates:", base_3d, end="", flush=True)

                # Append the 3D coordinates to the csv file
                # csv file columns: timestamp - volume_1 - volume_2 - volume_3 - tip_x - tip_y - tip_z - base_x - base_y - base_z
                row.append(tip_3d[0][0])
                row.append(tip_3d[1][0])
                row.append(tip_3d[2][0])
                row.append(base_3d[0][0])
                row.append(base_3d[1][0])
                row.append(base_3d[2][0])
                rows.append(row)

        # Write the updated rows back to the CSV file (or to a new file)
        with open(self.csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)

    def triangulate(self, img1, img2):
        """
        Input: img1 - Image from camera 1
                img2 - Image from camera 2
        Output: x,y,z - Coordinates of the robot tip in 3D space
                x,y,z - Coordinates of the robot base in 3D space
        """

        # Get tip and base points from the images
        tip1 = self.detect_tip(img1)
        base1 = self.detect_base(img1)
        tip2 = self.detect_tip(img2)
        base2 = self.detect_base(img2)

        # Triangulate the points
        tip_4d = cv2.triangulatePoints(self.P1_matrix, self.P2_matrix, tip1, tip2)
        tip_3d = tip_4d[:3] / tip_4d[3]
        base_4d = cv2.triangulatePoints(self.P1_matrix, self.P2_matrix, base1, base2)
        base_3d = base_4d[:3] / base_4d[3]

        return tip_3d, base_3d

if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()