import csv
import os
import time

import numpy as np

# Cameras
cam_1_index = 1
cam_2_index = 0
P1_yaml = os.path.abspath(os.path.join("calibration_images_cam4_640x480p", "projection_matrix.yaml"))
P2_yaml = os.path.abspath(os.path.join("calibration_images_cam2_640x480p", "projection_matrix.yaml"))

# Set experiment name and save directory
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
csv_path = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))

# If they dont exist, create the directories and the csv file
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the csv file columns
csv_columns = ["timestamp", "volume_1", "volume_2", "volume_3", "pressure_1", "pressure_2", "pressure_3", "img_1", "img_2", "tip_x", "tip_y", "tip_z", "base_x", "base_y", "base_z"]
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)

# Colors range for detection
# Yellow
lower_yellow = np.array([23, 88, 0])
upper_yellow = np.array([36, 254, 255])

# Red
lower_red1 = np.array([0, 55, 0])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([171, 55, 0])
upper_red2 = np.array([180, 255, 255])

# Move settings
home_first = True
offset = 0 
initial_pos = 110 + offset
steps = 12
stroke = 3  # mm
stepSize = stroke / steps

# Configuration (UDP receiver) (data: pressure sensors -> quanser -> simulink -> python)
UDP_IP = "127.0.0.1"
UDP_PORT = 25000

