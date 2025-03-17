import csv
import os
import time

import numpy as np

# Cameras
print("\n\nIMPORTANT: Check if the camera indexes are correct every time you run the code.\n\n")
cam_left_index = 0
cam_right_index = 2
P_left_yaml = os.path.abspath(os.path.join("calibration", "calibration_images_camleft_640x480p", "projection_matrix.yaml"))
P_right_yaml = os.path.abspath(os.path.join("calibration", "calibration_images_camright_640x480p", "projection_matrix.yaml"))

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
csv_columns = ["timestamp", "volume_1", "volume_2", "volume_3", "pressure_1", "pressure_2", "pressure_3", "img_left", "img_right", "tip_x", "tip_y", "tip_z", "base_x", "base_y", "base_z"]
with open(csv_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_columns)

# Colors range for detection
# Yellow
lower_yellow = np.array([23, 88, 0])
upper_yellow = np.array([36, 254, 255])

# Red
lower_red1 = np.array([0, 80, 0])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([172, 80, 0])
upper_red2 = np.array([180, 255, 255])

# Green
lower_green = np.array([36, 50, 70])
upper_green = np.array([86, 255, 255])

# Move settings
home_first = False
offset = 5
initial_pos = 110 + offset
steps = 10
stroke = 4  # mm
stepSize = stroke / steps
max_vol_1 = initial_pos + 10
max_vol_2 = initial_pos + 10
max_vol_3 = initial_pos + 10

# Configuration (UDP receiver) (data: pressure sensors -> quanser -> simulink -> python)
UDP_IP = "127.0.0.1"
UDP_PORT = 25000

# NN
# If we are focusing on pressures only, output dimension = 3.
pressure_only = False
if pressure_only:
    output_dim = 3
else:
    output_dim = 6

# Path to volume inputs (to be used in durability.move_from_csv)
input_volume_path = os.path.abspath(os.path.join("data", "volume_inputs", "inputs_2.csv"))

