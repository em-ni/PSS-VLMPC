import csv
from datetime import datetime
import os
import time
import numpy as np

# TODO: 
# -fix pressure sensor and start data colelction with each muscle under the same pressure
# -find a better way to scan the workspace uniformly including elongation

new_experiment = False

# --- Data collection settings ---
# Cameras
if new_experiment: print("\n\nIMPORTANT: Check if the camera indexes are correct every time you run the code.\n\n")
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
initial_pos = 115
steps = 50 # Suggestion: use utils/workspace_preview.py 
window_steps = 25 # Windows length in steps
elongationstepSize = window_steps # To regulate overlap between windows (how much a window is shifted)
max_stroke = 5  # distance in mm from the initial position to final position
stepSize = max_stroke / steps
max_vol_1 = initial_pos + max_stroke
max_vol_2 = initial_pos + max_stroke
max_vol_3 = initial_pos + max_stroke
init_pressure = 2.0

# Map quanser index to axis index
axis_mapping = {
    0: 2,  
    1: 1,  
    2: 3   
}

# Configuration (UDP receiver) (data: pressure sensors -> quanser -> simulink -> python)
UDP_IP = "127.0.0.1"
UDP_PORT = 25000

# Path to volume inputs (to be used in explorer.move_from_csv)
input_volume_path = os.path.abspath(os.path.join("data", "volume_inputs", "inputs_2.csv"))
# ---------------------------------

# --- Kinematic Neural Network Model settings ---
# If we are focusing on pressures only, output dimension = 3.
pressure_only = False
if pressure_only:
    output_dim = 3
else:
    output_dim = 6
MODEL_PATH = "data/exp_2025-04-28_15-58-15/volume_net.pth"
SCALERS_PATH = "data/exp_2025-04-28_15-58-15/volume_net_scalers.npz"
POINT_CLOUD_PATH = "data/exp_2025-04-28_15-58-15/dataset.csv"

# LSTM
sequence_length = 1  # T=3 -> sequence length 4 (t, t-1, t-2, t-3)
n_features_tau = 3   # volume_1, volume_2, volume_3
n_features_x = 3     # delta_x, delta_y, delta_z = tip_x, tip_y, tip_z - base_x, base_y, base_z
total_features = n_features_tau + n_features_x
output_dim = n_features_x
lstm_hidden_units = 64
lstm_num_layers = 2    
# ---------------------------------

# --- RL settings ---
N_POINTS = 3      # Number of waypoints in the trajectory
N_ENVS = 24       # Number of environments to run in parallel
ALGORITHM = "TRPO"

CHECKPOINT_STEPS = 1000000        # Number of steps to save checkpoint
TOTAL_TRAINING_STEPS = 1000000  # Total training steps
TOTAL_N_STEPS = 2048              # Total steps before updating the policy

CHECKPOINTS_DIR = os.path.join(data_dir, "rl", "checkpoints")
POLICY_DIR = os.path.join(data_dir, "rl", "policy", "trained_policy.zip")
METRICS_DIR = os.path.join(data_dir, "rl", "training_metrics", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

EVAL_EPISODES = 10
# -------------------------------

# --- MPC settings ---
STATE_DIM = 3
CONTROL_DIM = 3
VOLUME_DIM = 3
DT = 0.01
T_SIM = 3.0
N_sim_steps = int(T_SIM / DT)
N_WAYPOINTS = 3

U_MAX_CMD = float(max_stroke)
U_MIN_CMD = 0

INITIAL_POS_VAL = float(initial_pos)
V_MIN_PHYSICAL = INITIAL_POS_VAL + U_MIN_CMD
V_MAX_PHYSICAL = INITIAL_POS_VAL + U_MAX_CMD
VOLUME_BOUNDS_LIST = [(V_MIN_PHYSICAL, V_MAX_PHYSICAL)] * VOLUME_DIM
V_REST = np.array([INITIAL_POS_VAL] * VOLUME_DIM)

Q_WEIGHT = 1e6 
R_WEIGHT = 0
Q_matrix = np.diag([Q_WEIGHT] * STATE_DIM)
R_matrix = np.diag([R_WEIGHT] * VOLUME_DIM)
Q_terminal_matrix = np.diag([Q_WEIGHT] * STATE_DIM)
R_DELTA_V_WEIGHT = 0
R_delta_matrix = np.diag([R_DELTA_V_WEIGHT] * VOLUME_DIM)
N_HORIZON = 1

OPTIMIZER_METHOD = 'trust-constr' # 'SLSQP', 'L-BFGS-B', 'TNC' are also options but not good
PERTURBATION_SCALE = 0.003

TRAJ_DIR = os.path.join(data_dir, "mpc", "planned_trajectory.csv")
# ---------------------------------

