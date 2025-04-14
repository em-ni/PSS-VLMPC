import os
import sys
import numpy as np
import torch
import casadi as ca
import do_mpc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Environment Setup & Imports ---

# Add project root to sys.path if necessary (adjust path as needed)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
     sys.path.insert(0, project_root)
# Make sure these modules can be imported
try:
    import src.config as config
    from src.nn_model import VolumeNet
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure src/config.py and src/nn_model.py exist and paths are correct.")
    sys.exit(1)

# --- 2. NN Model Loading & Prediction Function ---

# Global variables to hold the model and scalers (load once)
nn_model = None
scaler_volumes = None
scaler_deltas = None
nn_device = None

def load_nn_model_and_scalers(model_path, scalers_path):
    """Loads the PyTorch model and scikit-learn scalers."""
    global nn_model, scaler_volumes, scaler_deltas, nn_device

    if nn_model is not None: # Already loaded
        return

    print("Loading NN model and scalers...")
    try:
        # Load scalers
        scalers = np.load(scalers_path)
        scaler_volumes = MinMaxScaler()
        scaler_volumes.min_ = scalers['volumes_min']
        scaler_volumes.scale_ = scalers['volumes_scale']

        scaler_deltas = MinMaxScaler()
        scaler_deltas.min_ = scalers['deltas_min']
        scaler_deltas.scale_ = scalers['deltas_scale']
        print("Scalers loaded.")

        # Set device
        nn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {nn_device}")

        # Load model
        nn_model = VolumeNet(input_dim=3, output_dim=3)
        nn_model.load_state_dict(torch.load(model_path, map_location=nn_device))
        nn_model = nn_model.to(nn_device)
        nn_model.eval()
        print("NN model loaded.")

    except FileNotFoundError as e:
        print(f"Error loading NN files: {e}")
        print(f"Ensure model path '{model_path}' and scalers path '{scalers_path}' are correct.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during NN loading: {e}")
        sys.exit(1)

def predict_tip_standalone(command: np.ndarray) -> np.ndarray:
    """
    Predicts the tip position based on the command using the loaded NN model.
    Input: command (np.ndarray, shape (3,)) - motor strokes/commands
    Output: tip_position (np.ndarray, shape (3,))
    """
    global nn_model, scaler_volumes, scaler_deltas, nn_device

    if nn_model is None:
        raise RuntimeError("NN Model not loaded. Call load_nn_model_and_scalers first.")

    # Ensure command is float32
    command = command.astype(np.float32)

    # Calculate volumes based on command and initial position
    volumes = np.zeros(3, dtype=np.float32)
    volumes[0] = config.initial_pos + command[0]
    volumes[1] = config.initial_pos + command[1]
    volumes[2] = config.initial_pos + command[2]

    # Reshape for scaler
    volumes_2d = volumes.reshape(1, -1)

    # Scale inputs
    volumes_scaled = scaler_volumes.transform(volumes_2d)

    # Convert to tensor and move to device
    volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32).to(nn_device)

    # Make predictions
    with torch.no_grad():
        predictions_tensor = nn_model(volumes_tensor)
        predictions_scaled = predictions_tensor.cpu().numpy()

    # Inverse transform predictions
    predictions = scaler_deltas.inverse_transform(predictions_scaled)

    # Return the 3D tip position
    return predictions[0]


# --- 3. CasADi Wrapper for NN ---

# This function will be called by CasADi during optimization.
# It needs to take CasADi symbolic types (like DM or SX) and return them.
# Internally, it converts to numpy, calls the PyTorch function, and converts back.
def nn_kinematics_casadi_wrapper(u_ca):
    """
    CasADi wrapper for the predict_tip_standalone function.
    Input: u_ca (CasADi symbolic type, shape (3,1)) - motor commands
    Output: x_next_ca (CasADi symbolic type, shape (3,1)) - predicted tip position
    """
    # Convert CasADi input to numpy
    # .full() converts DM to numpy array
    u_np = u_ca.full().flatten()

    # Call the standalone prediction function
    x_next_np = predict_tip_standalone(u_np)

    # Convert numpy output back to CasADi DM
    x_next_ca = ca.DM(x_next_np).reshape((3, 1))
    return x_next_ca

# Create the CasADi external function
# 'nn_func' is the name CasADi uses.
# The second argument is the Python function to wrap.
# Input/output definitions tell CasADi about the shapes.
input_sx = ca.SX.sym('u_sx', 3, 1) # Define symbolic input for shape deduction
output_sx = ca.SX.sym('x_next_sx', 3, 1) # Define symbolic output for shape deduction

try:
    # Attempt to load the model now to ensure paths are valid before defining the external function
    nn_model_path = r"data/exp_2025-04-04_19-17-42/volume_net.pth" # ADJUST PATH
    nn_scalers_path = r"data/exp_2025-04-04_19-17-42/volume_net_scalers.npz" # ADJUST PATH
    load_nn_model_and_scalers(nn_model_path, nn_scalers_path)

    # Define the external function using the wrapper
    nn_casadi_func = ca.external('nn_kinematics', nn_kinematics_casadi_wrapper)
    # Test call (optional, for debugging)
    # test_input = ca.DM([0.1, 0.2, 0.3])
    # test_output = nn_casadi_func(test_input)
    # print("CasADi external function test output:", test_output)

except Exception as e:
    print(f"Error creating CasADi external function: {e}")
    print("Ensure NN model/scalers loaded correctly and CasADi is working.")
    sys.exit(1)


# --- 4. do-mpc Model Setup ---

model_type = 'discrete' # Using discrete model x_k+1 = f(u_k)
model = do_mpc.model.Model(model_type)

# States struct (tip position)
pos = model.set_variable(var_type='_x', var_name='pos', shape=(3,1))
tip_x = pos[0]
tip_y = pos[1]
tip_z = pos[2]

# Input struct (commands)
u = model.set_variable(var_type='_u', var_name='cmd', shape=(3,1))

# Time-varying parameters struct (reference trajectory)
ref = model.set_variable(var_type='_tvp', var_name='reference', shape=(3,1))

# System dynamics using the NN external function
# x_{k+1} = nn_casadi_func(u_k)
x_next = nn_casadi_func(u)
model.set_rhs('pos', x_next)

# Setup model
try:
    model.setup()
    print("do-mpc model setup complete.")
except Exception as e:
    print(f"Error during do-mpc model setup: {e}")
    sys.exit(1)

# --- 5. Trajectory Definition ---

# Example: Circular trajectory in the xy-plane
sim_steps = 100
t_step_traj = 0.1 # Time step used for generating the trajectory points
total_time = sim_steps * t_step_traj
radius = 5.0
center_x = 5.0
center_y = 0.0
fixed_z = 10.0 # Assuming a fixed height for simplicity

time_vec = np.linspace(0, total_time, sim_steps + 1)
ref_trajectory = np.zeros((sim_steps + 1, 3))
ref_trajectory[:, 0] = center_x + radius * np.cos(2 * np.pi * time_vec / total_time)
ref_trajectory[:, 1] = center_y + radius * np.sin(2 * np.pi * time_vec / total_time)
ref_trajectory[:, 2] = fixed_z

# --- 6. do-mpc MPC Setup ---

mpc = do_mpc.controller.MPC(model)

# MPC parameters
setup_mpc = {
    'n_horizon': 20,      # Prediction horizon
    't_step': 0.1,        # Control interval length (should match trajectory time step ideally)
    'n_robust': 0,        # No robustness needed for now
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)
mpc.settings.supress_ipopt_output() # Suppress solver output

# Objective function: ||x_k - x_ref,k||^2_Q + ||delta_u_k||^2_R
# Weighting matrices
Q = np.diag([1.0, 1.0, 1.0]) # State tracking weight
R = np.diag([0.1, 0.1, 0.1]) # Input rate penalty weight

# Mayer term (at the end of horizon) - identical to Lagrange term here
mterm = ca.sumsqr(Q @ (model.x['pos'] - model.tvp['reference']))
# Lagrange term (at each step)
lterm = ca.sumsqr(Q @ (model.x['pos'] - model.tvp['reference']))

mpc.set_objective(mterm=mterm, lterm=lterm)

# Input rate penalty (delta_u = u_k - u_{k-1})
mpc.set_rterm(cmd=R[0,0]) # Assuming diagonal R, same weight for all inputs

# Constraints
# Input bounds (adjust based on config.max_stroke)
max_command = config.max_stroke # Example value, use your actual config
mpc.bounds['lower','_u','cmd'] = 0.0
mpc.bounds['upper','_u','cmd'] = max_command

# Workspace bounds (optional, example) - Be careful with feasibility
# mpc.bounds['lower','_x','pos'] = np.array([-10, -10, 0])
# mpc.bounds['upper','_x','pos'] = np.array([20, 20, 20])

# Time-varying parameter function (provides reference for the horizon)
tvp_template = mpc.get_tvp_template()

def tvp_fun(t_now):
    """
    Provides the reference trajectory points for the entire prediction horizon.
    """
    current_time_index = int(round(t_now / setup_mpc['t_step']))

    for k in range(setup_mpc['n_horizon'] + 1):
        # Calculate the index in the precomputed trajectory
        traj_index = min(current_time_index + k, sim_steps) # Don't go beyond trajectory length
        tvp_template['_tvp', k, 'reference'] = ref_trajectory[traj_index, :]

    return tvp_template

mpc.set_tvp_fun(tvp_fun)

# Setup MPC
try:
    mpc.setup()
    print("do-mpc MPC setup complete.")
except Exception as e:
    print(f"Error during do-mpc MPC setup: {e}")
    sys.exit(1)


# --- 7. Simulation Loop ---

# Initialize state and guess
# Use the prediction for zero command as the initial state
x0 = predict_tip_standalone(np.zeros(3))
mpc.x0['pos'] = x0
mpc.set_initial_guess()

# Data storage
x_history = [x0]
u_history = []
ref_history = [ref_trajectory[0, :]]
time_history = [0.0]

print("Starting simulation loop...")
for k in range(sim_steps):
    current_time = k * setup_mpc['t_step']
    print(f"Step {k+1}/{sim_steps}, Time: {current_time:.2f}")

    # 1. Get MPC control command
    try:
        u_k = mpc.make_step(x0)
    except Exception as e:
        print(f"\n!!! Error during MPC step {k+1}: {e} !!!")
        print("Solver likely failed. Check model, constraints, objective, and initial guess.")
        print(f"Current state x0: {x0.flatten()}")
        # You might want to stop or try a fallback control action
        break # Stop simulation on error

    # 2. Simulate the system (using the original Python NN function)
    x_next = predict_tip_standalone(u_k)

    # 3. Store results
    x_history.append(x_next)
    u_history.append(u_k.flatten()) # Store as flat numpy array
    ref_history.append(ref_trajectory[k+1, :])
    time_history.append(current_time + setup_mpc['t_step'])

    # 4. Update state for next iteration
    x0 = x_next
    mpc.x0['pos'] = x0 # Update MPC's initial state understanding

print("Simulation loop finished.")

# Convert history to numpy arrays
x_history = np.array(x_history)
u_history = np.array(u_history)
ref_history = np.array(ref_history)
time_history = np.array(time_history)

# --- 8. Plotting ---

print("Plotting results...")

# 3D Trajectory Plot
fig3d = plt.figure(figsize=(10, 8))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], label='Actual Trajectory', marker='.', linestyle='-')
ax3d.plot(ref_history[:, 0], ref_history[:, 1], ref_history[:, 2], label='Reference Trajectory', marker='x', linestyle='--')
ax3d.scatter(x_history[0, 0], x_history[0, 1], x_history[0, 2], color='green', s=100, label='Start')
ax3d.scatter(ref_history[-1, 0], ref_history[-1, 1], ref_history[-1, 2], color='red', s=100, label='Goal')
ax3d.set_xlabel('X position')
ax3d.set_ylabel('Y position')
ax3d.set_zlabel('Z position')
ax3d.set_title('3D Robot Trajectory Tracking')
ax3d.legend()
ax3d.grid(True)
ax3d.axis('equal') # Ensure aspect ratio is equal

# State Tracking Plots (X, Y, Z vs Time)
fig_states, axs_states = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs_states[0].plot(time_history, x_history[:, 0], label='Actual X')
axs_states[0].plot(time_history, ref_history[:, 0], label='Reference X', linestyle='--')
axs_states[0].set_ylabel('X position')
axs_states[0].legend()
axs_states[0].grid(True)

axs_states[1].plot(time_history, x_history[:, 1], label='Actual Y')
axs_states[1].plot(time_history, ref_history[:, 1], label='Reference Y', linestyle='--')
axs_states[1].set_ylabel('Y position')
axs_states[1].legend()
axs_states[1].grid(True)

axs_states[2].plot(time_history, x_history[:, 2], label='Actual Z')
axs_states[2].plot(time_history, ref_history[:, 2], label='Reference Z', linestyle='--')
axs_states[2].set_ylabel('Z position')
axs_states[2].set_xlabel('Time (s)')
axs_states[2].legend()
axs_states[2].grid(True)

fig_states.suptitle('State Tracking vs Time')
fig_states.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle

# Control Input Plots (Commands vs Time)
if u_history.size > 0: # Check if simulation completed at least one step
    fig_inputs, axs_inputs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    time_inputs = time_history[:-1] # Input corresponds to the start of the interval

    axs_inputs[0].plot(time_inputs, u_history[:, 0], label='Command 1')
    axs_inputs[0].set_ylabel('Cmd 1')
    axs_inputs[0].legend()
    axs_inputs[0].grid(True)

    axs_inputs[1].plot(time_inputs, u_history[:, 1], label='Command 2')
    axs_inputs[1].set_ylabel('Cmd 2')
    axs_inputs[1].legend()
    axs_inputs[1].grid(True)

    axs_inputs[2].plot(time_inputs, u_history[:, 2], label='Command 3')
    axs_inputs[2].set_ylabel('Cmd 3')
    axs_inputs[2].set_xlabel('Time (s)')
    axs_inputs[2].legend()
    axs_inputs[2].grid(True)

    # Add bounds lines
    for ax in axs_inputs:
        ax.axhline(0.0, color='r', linestyle='--', linewidth=0.8, label='Lower Bound')
        ax.axhline(max_command, color='r', linestyle='--', linewidth=0.8, label='Upper Bound')
    # Avoid duplicate labels in legend
    handles, labels = axs_inputs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs_inputs[0].legend(by_label.values(), by_label.keys())


    fig_inputs.suptitle('Control Inputs vs Time')
    fig_inputs.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout for suptitle

plt.show()