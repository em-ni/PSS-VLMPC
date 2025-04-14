import csv
import os
import time
import numpy as np

new_experiment = False

# Cameras
print("\n\nIMPORTANT: Check if the camera indexes are correct every time you run the code.\n\n")
cam_left_index = 0
cam_right_index = 3
P_left_yaml = os.path.abspath(os.path.join("calibration", "calibration_images_camleft_640x480p", "projection_matrix.yaml"))
P_right_yaml = os.path.abspath(os.path.join("calibration", "calibration_images_camright_640x480p", "projection_matrix.yaml"))

# Set experiment name and save directory
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
csv_path = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))
data_dir = os.path.abspath(os.path.join(".", "data"))

if new_experiment:
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

# Blue
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Move settings
home_first = False
initial_pos = 116
steps = 21
max_stroke = 3  # mm
stepSize = max_stroke / steps
max_vol_1 = initial_pos + max_stroke
max_vol_2 = initial_pos + max_stroke
max_vol_3 = initial_pos + max_stroke
init_pressure = 1.0

# Map quanser index to axis index
axis_mapping = {
    0: 2,  
    1: 1,  
    2: 3   
}

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

# LSTM
sequence_length = 1  # T=3 -> sequence length 4 (t, t-1, t-2, t-3)
n_features_tau = 3   # volume_1, volume_2, volume_3
n_features_x = 3     # delta_x, delta_y, delta_z = tip_x, tip_y, tip_z - base_x, base_y, base_z
total_features = n_features_tau + n_features_x
output_dim = n_features_x
lstm_hidden_units = 64
lstm_num_layers = 2    

# Path to volume inputs (to be used in explorer.move_from_csv)
input_volume_path = os.path.abspath(os.path.join("data", "volume_inputs", "inputs_2.csv"))

# RL goal
pick_random_goal = False
use_trajectory = False
N_points = 10
rl_goal = np.array([2.5, 1.7, 1.0], dtype=np.float32)

