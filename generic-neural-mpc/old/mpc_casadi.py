# mpc_casadi.py
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.func import vmap, jacrev
import casadi as ca
import joblib
import matplotlib.pyplot as plt

# Import exactly as requested
from model.train_robot import StatePredictor, TrainingConfig as TrainConfig

# --- Configuration ---
class MPCConfig:
    MODEL_PATH = TrainConfig.MODEL_PATH
    INPUT_SCALER_PATH = TrainConfig.INPUT_SCALER_PATH
    OUTPUT_SCALER_PATH = TrainConfig.OUTPUT_SCALER_PATH
    REAL_DATASET_PATH = TrainConfig.REAL_DATASET_PATH
    N = 20
    DT = 0.022
    SIM_TIME = 10.0
    q_pos = 100.0
    q_vel = 0.0
    Q_diag = [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
    r_diag = 0.0
    R_diag = [r_diag, r_diag, r_diag]
    r_rate_diag = 10.0
    R_rate_diag = [r_rate_diag, r_rate_diag, r_rate_diag]
    U_MIN = [115.0, 115.0, 115.0]
    U_MAX = [120.0, 120.0, 120.0]

# --- MODIFIED: BATCHED PyTorch function for evaluation and jacobian ---
def get_batch_predictions_and_jacobians(model, input_scaler, output_scaler, x_traj_np, u_traj_np):
    """
    Takes full numpy trajectories, returns batches of predictions and Jacobians.
    This function uses vmap for parallel computation.
    """
    model.to('cpu').eval()
    
    input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32)
    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
    output_scale = torch.tensor(output_scaler.scale_, dtype=torch.float32)
    output_mean = torch.tensor(output_scaler.mean_, dtype=torch.float32)

    def full_pytorch_model(x_and_u_torch):
        scaled_input = (x_and_u_torch - input_mean) / input_scale
        scaled_output = model(scaled_input)
        return scaled_output * output_scale + output_mean

    # Create a function that computes the Jacobian for a single sample
    def get_jac_for_sample(x_and_u_sample):
        return jacrev(full_pytorch_model)(x_and_u_sample)

    # Combine state and control trajectories into a single input batch
    dt_col = np.full((u_traj_np.shape[0], 1), MPCConfig.DT)
    x_and_u_traj_np = np.hstack([u_traj_np, x_traj_np, dt_col])
    x_and_u_torch = torch.tensor(x_and_u_traj_np, dtype=torch.float32)

    # --- KEY CHANGE: Use vmap to parallelize the Jacobian calculation ---
    # This maps the `get_jac_for_sample` function over the batch dimension (dim=0) of the input tensor.
    J_batch = vmap(get_jac_for_sample)(x_and_u_torch)

    # Also get the regular forward-pass predictions in a single batch
    with torch.no_grad():
        y_pred_batch = full_pytorch_model(x_and_u_torch)

    return y_pred_batch.detach().numpy(), J_batch.detach().numpy()


def run_mpc_simulation():
    # --- Phase 1: Initialization ---
    print("--- Loading 'no acceleration' assets ---")
    try:
        df = pd.read_csv(MPCConfig.REAL_DATASET_PATH); 
        df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)
        model = StatePredictor(input_dim=10, output_dim=6)
        model.load_state_dict(torch.load(MPCConfig.MODEL_PATH))
        # --- JIT Compile the model for extra speed ---
        dummy_input = torch.randn(1, 10) # 1 sample, 10 features
        model = torch.jit.trace(model, dummy_input)
        print("Model JIT-compiled with TorchScript.")
        input_scaler = joblib.load(MPCConfig.INPUT_SCALER_PATH)
        output_scaler = joblib.load(MPCConfig.OUTPUT_SCALER_PATH)
    except FileNotFoundError as e: print(f"Error: A required file was not found: {e.filename}"); sys.exit()

    # --- Phase 2: Define the Optimization Problem (OCP) ---
    print("--- Setting up OCP with Balanced Tuning ---")
    opti = ca.Opti()
    n_states, n_controls = 6, 3
    
    X = opti.variable(n_states, MPCConfig.N + 1); 
    U = opti.variable(n_controls, MPCConfig.N)
    x0 = opti.parameter(n_states, 1)
    x_ref = opti.parameter(n_states, 1); u_prev = opti.parameter(n_controls, 1)
    
    A_params = [opti.parameter(n_states, n_states) for _ in range(MPCConfig.N)]
    B_params = [opti.parameter(n_states, n_controls) for _ in range(MPCConfig.N)]
    C_params = [opti.parameter(n_states, 1) for _ in range(MPCConfig.N)]

    cost = 0
    Q = ca.diag(MPCConfig.Q_diag)
    R = ca.diag(MPCConfig.R_diag)
    R_rate = ca.diag(MPCConfig.R_rate_diag)
    
    for k in range(MPCConfig.N):
        cost += (X[:, k] - x_ref).T @ Q @ (X[:, k] - x_ref)
        cost += U[:, k].T @ R @ U[:, k]
        if k == 0: cost += (U[:, k] - u_prev).T @ R_rate @ (U[:, k] - u_prev)
        else: cost += (U[:, k] - U[:, k-1]).T @ R_rate @ (U[:, k] - U[:, k-1])

    cost += (X[:, MPCConfig.N] - x_ref).T @ Q @ (X[:, MPCConfig.N] - x_ref)
    opti.minimize(cost)
    
    opti.subject_to(X[:, 0] == x0)
    for k in range(MPCConfig.N):
        opti.subject_to(X[:, k+1] == A_params[k] @ X[:, k] + B_params[k] @ U[:, k] + C_params[k])
    for k in range(MPCConfig.N):
        opti.subject_to(opti.bounded(MPCConfig.U_MIN, U[:, k], MPCConfig.U_MAX))
    
    solver_opts = {
        'ipopt.print_level': 0, 
        'print_time': 0, 
        'ipopt.sb': 'yes',
        # 'ipopt.tol': 1e-6,           # Relax the main tolerance (this fucks up things)
        'ipopt.acceptable_tol': 1e-3   # Allow it to stop even sooner
    }
    opti.solver('ipopt', solver_opts)

    # --- Phase 3: The Simulation Loop ---
    print("--- Starting MPC simulation ---")
    state_cols = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    sample = df[state_cols].dropna().sample(2, random_state=42)
    x_current = sample.iloc[0].values
    x_target = sample.iloc[1].values
    
    history_x, history_u = [x_current], []
    n_steps = int(MPCConfig.SIM_TIME / MPCConfig.DT)
    
    u_guess = np.zeros((n_controls, MPCConfig.N))
    last_u_optimal = np.zeros(n_controls)
    
    sim_times = []
    start = time.time()
    for i in range(n_steps):
        # --- Step 3a: Prediction and Linearization (BATCHED) ---
        x_guess_np = np.zeros((MPCConfig.N, n_states))
        x_guess_np[0, :] = x_current
        
        # Sequentially roll out the nominal trajectory (this is usually fast)
        for k in range(MPCConfig.N - 1):
             model_input_k = np.concatenate([u_guess[:, k], x_guess_np[k, :], [MPCConfig.DT]])
             # We need a single prediction function for this rollout
             with torch.no_grad():
                model_input_torch = torch.from_numpy(model_input_k).float()
                x_guess_np[k+1, :] = full_pytorch_model(model_input_torch).numpy()

        # Get all predictions and Jacobians for the horizon in one parallel call
        y_pred_batch, J_batch = get_batch_predictions_and_jacobians(model, input_scaler, output_scaler, x_guess_np, u_guess.T)
        
        # Set the parameters for the optimizer in a simple loop
        for k in range(MPCConfig.N):
            J_k = J_batch[k]
            B_k = J_k[:, :n_controls]
            A_k = J_k[:, n_controls:n_controls+n_states]
            C_k = y_pred_batch[k] - A_k @ x_guess_np[k] - B_k @ u_guess[:, k]
            opti.set_value(A_params[k], A_k)
            opti.set_value(B_params[k], B_k)
            opti.set_value(C_params[k], C_k.reshape(-1, 1))
        
        # --- Step 3b: Set Current Values and Solve ---
        opti.set_value(x0, x_current)
        opti.set_value(x_ref, x_target)
        opti.set_value(u_prev, last_u_optimal)
        opti.set_initial(U, u_guess)
        opti.set_initial(X, np.hstack([x_current.reshape(-1,1), x_guess_np.T]))

        try:
            sol = opti.solve()
            u_optimal_all = sol.value(U)
            last_u_optimal = u_optimal_all[:, 0]
            u_guess = np.roll(u_optimal_all, -1, axis=1)
            u_guess[:, -1] = last_u_optimal
        except Exception as e:
            print(f"\nSolver failed at step {i}: {e}"); break
        
        # --- Step 3c: Simulate the System ---
        start_sim = time.time()
        model_input_sim = np.concatenate([last_u_optimal, x_current, [MPCConfig.DT]])
        with torch.no_grad():
             x_current = full_pytorch_model(torch.from_numpy(model_input_sim).float()).numpy()
        history_x.append(x_current)
        history_u.append(last_u_optimal)
        end_sim = time.time()
        sim_times.append(end_sim - start_sim)
        
        if i % 10 == 0 or i == 0:
            dist_to_target = np.linalg.norm(x_current[:3] - x_target[:3])
            print(f"Step {i+1}/{n_steps}, Pos. Distance to target: {dist_to_target:.4f}")
    
    end = time.time()
    total_sim_time = sum(sim_times)
    mpc_time = end - start - total_sim_time
    print(f"\nSimulated {MPCConfig.SIM_TIME:.1f}s in {end - start:.2f} seconds")
    print(f"Total simulation time: {total_sim_time:.2f} seconds.")
    print(f"Avg simulation time per step: {1000 * total_sim_time / n_steps:.2f} ms.")
    print(f"Total MPC time: {mpc_time:.2f} seconds")
    print(f"Avg MPC time per step: {1000 * mpc_time / n_steps:.2f} ms.")

    # --- Phase 4: Plotting ---
    history_x = np.array(history_x)
    history_u = np.array(history_u)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    time_axis = np.arange(history_x.shape[0]) * MPCConfig.DT
    axs[0].plot(time_axis, history_x[:, 0], label='Tip X')
    axs[0].plot(time_axis, history_x[:, 1], label='Tip Y')
    axs[0].plot(time_axis, history_x[:, 2], label='Tip Z')
    axs[0].axhline(y=x_target[0], color='r', linestyle='--', label='Target X')
    axs[0].axhline(y=x_target[1], color='g', linestyle='--', label='Target Y')
    axs[0].axhline(y=x_target[2], color='b', linestyle='--', label='Target Z')
    axs[0].set_ylabel('Position (cm)')
    axs[0].set_title('MPC Trajectory (Batched Linearization)')
    axs[0].legend(); axs[0].grid(True)
    axs[1].plot(time_axis, history_x[:, 3:6])
    axs[1].axhline(y=0, color='k', linestyle='--')
    axs[1].set_ylabel('Velocity (cm/s)'); axs[1].grid(True)
    if history_u.size > 0:
        time_axis_u = np.arange(history_u.shape[0]) * MPCConfig.DT
        axs[2].step(time_axis_u, history_u[:, 0], where='post', label='Volume 1')
        axs[2].step(time_axis_u, history_u[:, 1], where='post', label='Volume 2')
        axs[2].step(time_axis_u, history_u[:, 2], where='post', label='Volume 3')
    axs[2].set_ylabel('Control Input')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('MPC Control Inputs'); axs[2].legend(); axs[2].grid(True)
    plt.tight_layout()
    plot_path = 'results/mpc_casadi.png'
    plt.savefig(plot_path)
    print(f"\nMPC trajectory plot saved as '{plot_path}'.")

if __name__ == "__main__":
    # Define the full_pytorch_model in the global scope for the sequential rollout
    # This is a bit of a hack but avoids passing all scaler objects around
    payload = torch.load(TrainConfig.MODEL_PATH)
    model_g = StatePredictor(input_dim=10, output_dim=6)
    model_g.load_state_dict(payload)
    input_scaler_g = joblib.load(TrainConfig.INPUT_SCALER_PATH)
    output_scaler_g = joblib.load(TrainConfig.OUTPUT_SCALER_PATH)
    
    input_scale_g = torch.tensor(input_scaler_g.scale_, dtype=torch.float32)
    input_mean_g = torch.tensor(input_scaler_g.mean_, dtype=torch.float32)
    output_scale_g = torch.tensor(output_scaler_g.scale_, dtype=torch.float32)
    output_mean_g = torch.tensor(output_scaler_g.mean_, dtype=torch.float32)

    def full_pytorch_model(x_and_u_torch):
        scaled_input = (x_and_u_torch - input_mean_g) / input_scale_g
        scaled_output = model_g(scaled_input)
        return scaled_output * output_scale_g + output_mean_g
    
    run_mpc_simulation()