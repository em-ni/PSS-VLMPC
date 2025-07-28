# config.py (Tuned for Stability)

import numpy as np

# --- System Configuration ---
class SystemConfig:
    STATE_DIM = 6
    CONTROL_DIM = 3
    ACCEL_DIM = 3  # Assuming acceleration is part of the control input
    FULL_INPUT_DIM = STATE_DIM + CONTROL_DIM + ACCEL_DIM  # Full input includes

# --- MPC Configuration ---
class MPCConfig:
    T = 10.0  # Total simulation time (seconds)
    DT = 0.05  # Time step for simulation
    N_HORIZON = 20  # Number of steps in the prediction horizon
    T_HORIZON = 1.0  # Total time for the prediction horizon (seconds)
    
    # --- MPC TUNING: COST MATRICES ---

    # Q: State Cost Matrix. How much do we care about tracking error?
    # Penalize position error more than velocity error.
    Q_pos = 100.0  # Penalty on position error (x0, x1, x2)
    Q_vel = 1.0    # Lower penalty on velocity error (x3, x4, x5)
    Q = np.diag([Q_pos, Q_pos, Q_pos, Q_vel, Q_vel, Q_vel])

    # R: Control Cost Matrix. How much do we care about control effort?
    # This is the most important parameter for stability.
    # Start with a significantly higher value to prevent chattering.
    R_val = 0.5  # INCREASED SIGNIFICANTLY (was 0.01 or 0.1)
    R = np.diag([R_val] * SystemConfig.CONTROL_DIM)
    
    # Control constraints
    U_MIN = [-1.0] * SystemConfig.CONTROL_DIM
    U_MAX = [1.0] * SystemConfig.CONTROL_DIM

# --- Neural Network Configuration ---
class NeuralNetConfig:
    INPUT_DIM = SystemConfig.STATE_DIM + SystemConfig.CONTROL_DIM
    # OUTPUT_DIM = SystemConfig.STATE_DIM // 2
    OUTPUT_DIM = SystemConfig.STATE_DIM # For example training prediction is full x_dot
    HIDDEN_LAYERS = 2
    HIDDEN_SIZE = 128
    ACTIVATION = 'Tanh'

# --- Training Configuration ---
class TrainingConfig:
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    REAL_DATASET_PATH = "model/data/output_exp_2025-07-22_12-23-07.csv"
    MODEL_PATH = "model/data/trained_model.pth"
    INPUT_SCALER_PATH = "model/data/input_scaler.joblib"
    OUTPUT_SCALER_PATH = "model/data/output_scaler.joblib"
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    PLOT_OUTPUT_PATH = "model/data/prediction_performance_plot.png"
    
    # Example variables
    EX_MODEL_SAVE_PATH = "model/data/example_model.pth"
    NUM_TRAIN_SAMPLES = 10000
    NUM_VAL_SAMPLES = 2000