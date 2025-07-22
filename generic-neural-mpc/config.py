import numpy as np

# --- System Configuration ---
class SystemConfig:
    STATE_DIM = 6
    CONTROL_DIM = 3

# --- MPC Configuration ---
class MPCConfig:
    T = 10.0  # Total simulation time (seconds)
    DT = 0.05  # Time step for simulation
    N_HORIZON = 20  # Number of steps in the prediction horizon
    T_HORIZON = 1.0  # Total time for the prediction horizon (seconds)
    
    # --- GENERIC COST MATRICES AND CONSTRAINTS ---
    # State cost: Penalize the first state more, the rest less.
    Q_first_state = 1000.0
    Q_other_states = 100.0
    Q = np.diag([Q_first_state] + [Q_other_states] * (SystemConfig.STATE_DIM - 1))

    # Control cost: Penalize all control inputs equally.
    R_val = 0.01
    R = np.diag([R_val] * SystemConfig.CONTROL_DIM)
    
    # Control constraints
    U_MIN = [-1.0] * SystemConfig.CONTROL_DIM
    U_MAX = [1.0] * SystemConfig.CONTROL_DIM

# --- Neural Network Configuration ---
class NeuralNetConfig:
    INPUT_DIM = SystemConfig.STATE_DIM + SystemConfig.CONTROL_DIM
    OUTPUT_DIM = SystemConfig.STATE_DIM
    
    HIDDEN_LAYERS = 3
    HIDDEN_SIZE = 256
    ACTIVATION = 'Tanh' # Options: 'ReLU', 'Tanh', 'Sigmoid'

# --- Training Configuration ---
class TrainingConfig:
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    
    NUM_TRAIN_SAMPLES = 20000
    NUM_VAL_SAMPLES = 2000

    REAL_DATASET_PATH = "model/data/output_exp_2025-07-22_12-23-07.csv"
    MODEL_SAVE_PATH = "model/data/trained_model.pth"