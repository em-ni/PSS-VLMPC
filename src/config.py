import csv
from datetime import datetime
import os
import time
import numpy as np

# TODO:
# -Check if rl still works after nn refactoring for prediction made as absolute_volume - initial_pos - offsets

new_experiment = False

# --- Data collection settings ---
# Cameras
if new_experiment:
    print(
        "\n\nIMPORTANT: Check if the camera indexes are correct every time you run the code.\n\n"
    )
cam_left_index = 0
cam_right_index = 3
P_left_yaml = os.path.abspath(
    os.path.join(
        "calibration", "calibration_images_camleft_640x480p", "projection_matrix.yaml"
    )
)
P_right_yaml = os.path.abspath(
    os.path.join(
        "calibration", "calibration_images_camright_640x480p", "projection_matrix.yaml"
    )
)

# Set experiment name and save directory
today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.abspath(os.path.join(".", "data", experiment_name))
csv_path = os.path.abspath(os.path.join(save_dir, f"output_{experiment_name}.csv"))
data_dir = os.path.abspath(os.path.join(".", "data"))
offsets_path = os.path.abspath(
    os.path.join(data_dir, "nn_train_offsets")
)  # Offsets are saved only after data collection for training

if new_experiment:
    # If they dont exist, create the directories and the csv file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set the csv file columns
    csv_columns = [
        "timestamp",
        "volume_1",
        "volume_2",
        "volume_3",
        "pressure_1",
        "pressure_2",
        "pressure_3",
        "img_left",
        "img_right",
        "tip_x",
        "tip_y",
        "tip_z",
        "base_x",
        "base_y",
        "base_z",
    ]
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
init_pressure = 0.5
initial_pos = 115
window_steps = 20  # Windows length in steps
max_stroke = 5  # distance in mm from the initial position to final position
steps = 40  # Suggestion: use utils/workspace_preview.py

elongationstepSize = (
    window_steps  # To regulate overlap between windows (how much a window is shifted)
)
stepSize = max_stroke / steps
max_vol_1 = initial_pos + max_stroke
max_vol_2 = initial_pos + max_stroke
max_vol_3 = initial_pos + max_stroke

# Map quanser index to axis index
axis_mapping = {0: 2, 1: 1, 2: 3}

# Configuration (UDP receiver) (data: pressure sensors -> quanser -> simulink -> python)
UDP_IP = "127.0.0.1"
UDP_PORT = 25000

# Path to volume inputs (to be used in explorer.move_from_csv)
input_volume_path = os.path.abspath(
    os.path.join("data", "volume_inputs", "inputs_2.csv")
)
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
n_features_tau = 3  # volume_1, volume_2, volume_3
n_features_x = (
    3  # delta_x, delta_y, delta_z = tip_x, tip_y, tip_z - base_x, base_y, base_z
)
total_features = n_features_tau + n_features_x
output_dim = n_features_x
lstm_hidden_units = 64
lstm_num_layers = 2
# ---------------------------------

# --- RL settings ---
N_POINTS = 3  # Number of waypoints in the trajectory
N_ENVS = 24  # Number of environments to run in parallel
ALGORITHM = "TRPO"

CHECKPOINT_STEPS = 1000000  # Number of steps to save checkpoint
TOTAL_TRAINING_STEPS = 1000000  # Total training steps
TOTAL_N_STEPS = 2048  # Total steps before updating the policy

CHECKPOINTS_DIR = os.path.join(data_dir, "rl", "checkpoints")
POLICY_DIR = os.path.join(data_dir, "rl", "policy", "trained_policy.zip")
METRICS_DIR = os.path.join(
    data_dir,
    "rl",
    "training_metrics",
    f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)

EVAL_EPISODES = 10
# -------------------------------

# --- MPC settings ---
STATE_DIM = 3
CONTROL_DIM = 3
VOLUME_DIM = 3
DT = 0.001
T_SIM = 3.0
N_sim_steps = int(T_SIM / DT)
N_WAYPOINTS = 3

U_MAX_CMD = float(max_stroke)
U_MIN_CMD = 0

# INITIAL_POS_VAL = float(initial_pos)
INITIAL_POS_VAL = float(0.0)
V_MIN_PHYSICAL = INITIAL_POS_VAL + U_MIN_CMD
V_MAX_PHYSICAL = INITIAL_POS_VAL + U_MAX_CMD
VOLUME_BOUNDS_LIST = [(V_MIN_PHYSICAL, V_MAX_PHYSICAL)] * VOLUME_DIM
# V_REST = np.array([INITIAL_POS_VAL] * VOLUME_DIM)
V_REST = np.array([0.0] * VOLUME_DIM)

Q_WEIGHT = 1e10
R_WEIGHT = 0
Q_matrix = np.diag([Q_WEIGHT] * STATE_DIM)
R_matrix = np.diag([R_WEIGHT] * VOLUME_DIM)
Q_terminal_matrix = np.diag([Q_WEIGHT] * STATE_DIM)
R_DELTA_V_WEIGHT = 0
R_delta_matrix = np.diag([R_DELTA_V_WEIGHT] * VOLUME_DIM)
N_HORIZON = 1

OPTIMIZER_METHOD = "COBYQA"  # 'trust-constr' # 'SLSQP', 'L-BFGS-B', 'TNC' are also options but not good
# PERTURBATION_SCALE = 0.0065 # for trust-constr
PERTURBATION_SCALE = 0.1  # for COBYQA

TRAJ_DIR = os.path.join(data_dir, "mpc", "planned_trajectory.csv")
SMOOTH_CONTORL = False
# ---------------------------------

"""
Notes on solvers:
This function `minimize` is a generic interface to various optimization algorithms, likely from `scipy.optimize.minimize`. Here's a breakdown of the different solver methods:

**Derivative-Free Methods:**
These methods do not require the gradient (Jacobian) or Hessian of the objective function. They are useful when derivatives are difficult or impossible to compute.

*   **'Nelder-Mead'**:
    *   **Algorithm**: Uses a simplex (a geometric figure in n-dimensions consisting of n+1 vertices) that adapts to the local landscape.
    *   **Pros**: Simple, can work on non-smooth and noisy functions.
    *   **Cons**: Can be slow, especially for higher-dimensional problems. May get stuck in local minima. Not guaranteed to converge to a true minimum.
    *   **Constraints**: Only supports bound constraints if implemented via a wrapper or if the specific version supports it (historically, it was unconstrained).

*   **'Powell'**:
    *   **Algorithm**: A conjugate direction method. It iteratively minimizes the function along a set of directions, which are updated to be (approximately) conjugate.
    *   **Pros**: Can be more efficient than Nelder-Mead for smoother, unimodal functions.
    *   **Cons**: Can be slow for many variables. Does not use gradient information.
    *   **Constraints**: Primarily for unconstrained problems.

*   **'COBYLA' (Constrained Optimization BY Linear Approximation)**:
    *   **Algorithm**: Uses linear approximations to the objective function and constraints. It works by iteratively refining a simplex.
    *   **Pros**: Handles general inequality constraints.
    *   **Cons**: Can be slow, especially as the number of variables or constraints increases. May not be very accurate.
    *   **Constraints**: Supports inequality constraints.

*   **'COBYQA' (Constrained Optimization BY Quadratic Approximation)**:
    *   **Algorithm**: An improvement over COBYLA, using quadratic approximations within a trust-region framework.
    *   **Pros**: Aims for better performance and accuracy than COBYLA for derivative-free constrained optimization.
    *   **Cons**: Newer method, might be less battle-tested than COBYLA in some scenarios.
    *   **Constraints**: Supports bound and linear inequality constraints.

**Gradient-Based Methods:**
These methods require the gradient (Jacobian) of the objective function. Some also use the Hessian (second derivatives) or approximations of it.

*   **'CG' (Conjugate Gradient)**:
    *   **Algorithm**: Iteratively finds search directions that are conjugate with respect to the Hessian (for quadratic functions).
    *   **Pros**: Good for large-scale unconstrained problems as it doesn't require storing a Hessian matrix.
    *   **Cons**: Can be slower than BFGS for smaller problems. Convergence can be sensitive to the line search.
    *   **Constraints**: Primarily for unconstrained problems.

*   **'BFGS' (Broyden-Fletcher-Goldfarb-Shanno)**:
    *   **Algorithm**: A quasi-Newton method. It approximates the inverse Hessian matrix using gradient information from previous iterations.
    *   **Pros**: Very popular and generally efficient for smooth, unconstrained problems. Good convergence properties.
    *   **Cons**: Requires storing an approximation of the Hessian (n x n matrix), which can be memory-intensive for very large `n`.
    *   **Constraints**: Primarily for unconstrained problems.

*   **'Newton-CG' (Newton-Conjugate Gradient)**:
    *   **Algorithm**: A modified Newton's method where the search direction (Newton step) is found by solving `H*p = -g` (where H is Hessian, p is step, g is gradient) using a conjugate gradient algorithm.
    *   **Pros**: Can be very fast if the Hessian is available (or Hessian-vector products can be computed efficiently). Good for large problems if Hessian-vector products are used.
    *   **Cons**: Requires the Hessian (or Hessian-vector products).
    *   **Constraints**: Primarily for unconstrained problems.

*   **'L-BFGS-B' (Limited-memory BFGS with Bounds)**:
    *   **Algorithm**: A version of BFGS that approximates the inverse Hessian using only a limited number of past gradients, making it suitable for problems with many variables.
    *   **Pros**: Memory efficient for large-scale problems. Handles bound constraints.
    *   **Cons**: May be less accurate or converge slower than BFGS on smaller problems.
    *   **Constraints**: Supports bound (box) constraints.

*   **'TNC' (Truncated Newton Constrained)**:
    *   **Algorithm**: A Newton-type algorithm that uses a truncated Newton approach (solving the Newton system approximately) to handle bound constraints.
    *   **Pros**: Efficient for problems with many variables and simple bound constraints.
    *   **Cons**: Requires gradients.
    *   **Constraints**: Supports bound (box) constraints.

*   **'SLSQP' (Sequential Least SQuares Programming)**:
    *   **Algorithm**: Solves a sequence of quadratic programming subproblems. Uses gradients.
    *   **Pros**: Versatile; handles both equality and inequality constraints, as well as bounds. Often a good general-purpose constrained optimizer.
    *   **Cons**: Performance can depend on the problem structure.
    *   **Constraints**: Supports bound, equality, and inequality constraints.

**Trust-Region Methods:**
These methods define a "trust region" around the current point where a model of the objective function (often quadratic) is trusted to be accurate. The algorithm then solves a subproblem to find the next step within this region.

*   **'trust-constr'**:
    *   **Algorithm**: A trust-region algorithm for constrained optimization. Can use various methods to solve the trust-region subproblem.
    *   **Pros**: Handles general nonlinear constraints (equality and inequality) and bounds. Often robust. Allows for different ways to approximate the Hessian.
    *   **Cons**: Can be more computationally intensive per iteration than some other methods. Requires gradients and often Hessians (or approximations).
    *   **Constraints**: Supports bound, equality, and inequality constraints.

*   **'dogleg'**:
    *   **Algorithm**: A specific trust-region method for unconstrained optimization. It computes the step by choosing between the steepest descent direction and a Newton-like step, forming a "dogleg" path.
    *   **Pros**: Robust, good for medium-sized unconstrained problems.
    *   **Cons**: Requires gradients and Hessians.
    *   **Constraints**: Unconstrained problems.

*   **'trust-ncg' (Trust-Region Newton Conjugate Gradient)**:
    *   **Algorithm**: Similar to Newton-CG but within a trust-region framework. The Newton step is computed using CG, but constrained to the trust region.
    *   **Pros**: Good for large-scale unconstrained problems where Hessian-vector products are available.
    *   **Cons**: Requires gradients and Hessian-vector products.
    *   **Constraints**: Unconstrained problems.

*   **'trust-exact'**:
    *   **Algorithm**: A trust-region method that attempts to solve the trust-region subproblem (minimizing a quadratic model within a ball) nearly exactly.
    *   **Pros**: Can be very accurate.
    *   **Cons**: Computationally expensive, especially for large problems, as solving the subproblem exactly can be hard. Requires gradients and Hessians.
    *   **Constraints**: Unconstrained problems.

*   **'trust-krylov'**:
    *   **Algorithm**: A trust-region method that uses Krylov subspace methods (like Lanczos or CG) to approximately solve the trust-region subproblem.
    *   **Pros**: Suitable for large-scale unconstrained problems, especially when only Hessian-vector products are available. Balances efficiency and accuracy.
    *   **Cons**: Requires gradients and Hessian-vector products.
    *   **Constraints**: Unconstrained problems.

**Key Differences Summarized:**

*   **Derivatives**: Some (Nelder-Mead, Powell, COBYLA, COBYQA) are derivative-free; others require gradients, and some also benefit from Hessians (or Hessian-vector products).
*   **Constraints**:
    *   Unconstrained: CG, BFGS, Newton-CG, dogleg, trust-ncg, trust-exact, trust-krylov.
    *   Bound Constraints: L-BFGS-B, TNC.
    *   General Constraints: COBYLA (inequality), COBYQA (bound, linear inequality), SLSQP (equality, inequality), trust-constr (equality, inequality).
*   **Problem Size**:
    *   Small to Medium: Nelder-Mead, Powell, BFGS, dogleg.
    *   Large: CG, L-BFGS-B, Newton-CG (with Hessian-vector products), trust-ncg, trust-krylov, trust-constr (can be).
*   **Robustness vs. Speed**: Derivative-free methods can be more robust to noise but are often slower. Gradient-based methods are faster if derivatives are accurate and cheap to compute. Trust-region methods are often robust.
*   **Memory**: L-BFGS-B is designed for low memory usage. Methods requiring full Hessian storage (like standard BFGS or some Newton methods if Hessian is dense) can be memory-intensive.

The choice of solver depends heavily on the characteristics of your objective function, the availability of derivatives, the nature of constraints, and the size of the problem.
"""
