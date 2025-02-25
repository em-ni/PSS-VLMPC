import os
import time

import numpy as np

# Cameras
cam_1_index = 1
cam_2_index = 0
P1_yaml = os.path.join("calibration_images_cam4_640x480p", "projection_matrix.yaml")
P2_yaml = os.path.join("calibration_images_cam2_640x480p", "projection_matrix.yaml")

# Set experiment name and save directory
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.join(".", "data", experiment_name)
csv_path = os.path.join(save_dir, f"output_{experiment_name}.csv")

