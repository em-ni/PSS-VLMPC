import sys
import time
import pandas as pd
import numpy as np
import torch
import casadi as ca
import l4casadi as l4c
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

# --- NEW: PyTorch model wrapper with integrated scaling ---
class ScaledStatePredictor(torch.nn.Module):
    def __init__(self, model, input_scaler, output_scaler):
        super().__init__()
        self.model = model
        self.register_buffer('input_mean', torch.tensor(input_scaler.mean_, dtype=torch.float32))
        self.register_buffer('input_scale', torch.tensor(input_scaler.scale_, dtype=torch.float32))
        self.register_buffer('output_mean', torch.tensor(output_scaler.mean_, dtype=torch.float32))
        self.register_buffer('output_scale', torch.tensor(output_scaler.scale_, dtype=torch.float32))

    def forward(self, x_and_u_with_dt):
        scaled_input = (x_and_u_with_dt - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        output = scaled_output * self.output_scale + self.output_mean
        return output

def run_mpc_simulation():
    # --- Phase 1: Initialization ---
    print("--- Loading assets and preparing l4casadi model ---")
    try:
        df = pd.read_csv(MPCConfig.REAL_DATASET_PATH); df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)
        base_model = StatePredictor(input_dim=10, output_dim=6); base_model.load_state_dict(torch.load(MPCConfig.MODEL_PATH))
        input_scaler = joblib.load(MPCConfig.INPUT_SCALER_PATH); output_scaler = joblib.load(MPCConfig.OUTPUT_SCALER_PATH)
        
        dummy_input = torch.randn(1, 10)
        base_model = torch.jit.trace(base_model, dummy_input)
        print("Base model JIT-compiled with TorchScript.")

        full_model = ScaledStatePredictor(base_model, input_scaler, output_scaler)
        full_model.to('cpu').eval()

        l4c_model = l4c.realtime.RealTimeL4CasADi(full_model, approximation_order=1, name='l4c_dynamics')
        
        n_states, n_controls = 6, 3
        n_model_inputs = n_controls + n_states + 1
        
        symbolic_input = ca.MX.sym('inp', n_model_inputs, 1)
        symbolic_output = l4c_model(symbolic_input)
        symbolic_params = l4c_model.get_sym_params()
        
        dynamics_func = ca.Function(
            'dynamics_func',
            [symbolic_input, symbolic_params],
            [symbolic_output]
        )
        print("l4casadi CasADi function created.")

    except FileNotFoundError as e: print(f"Error: A required file was not found: {e.filename}"); sys.exit()

    # --- Phase 2: Define the Optimization Problem (OCP) ---
    print("--- Setting up OCP with l4casadi model ---")
    opti = ca.Opti()
    
    X = opti.variable(n_states, MPCConfig.N + 1)
    U = opti.variable(n_controls, MPCConfig.N)
    x0 = opti.parameter(n_states, 1)
    x_ref = opti.parameter(n_states, 1)
    u_prev = opti.parameter(n_controls, 1)
    dt_param = opti.parameter(1)

    num_l4c_params = symbolic_params.shape[0]
    l4c_params = [opti.parameter(num_l4c_params) for _ in range(MPCConfig.N)]
    print(f"l4casadi model requires {num_l4c_params} parameters per time step.")
    
    cost = 0
    Q = ca.diag(MPCConfig.Q_diag); R = ca.diag(MPCConfig.R_diag); R_rate = ca.diag(MPCConfig.R_rate_diag)
    
    for k in range(MPCConfig.N):
        cost += (X[:, k] - x_ref).T @ Q @ (X[:, k] - x_ref)
        cost += U[:, k].T @ R @ U[:, k]
        if k == 0: cost += (U[:, k] - u_prev).T @ R_rate @ (U[:, k] - u_prev)
        else: cost += (U[:, k] - U[:, k-1]).T @ R_rate @ (U[:, k] - U[:, k-1])

    cost += (X[:, MPCConfig.N] - x_ref).T @ Q @ (X[:, MPCConfig.N] - x_ref)
    opti.minimize(cost)
    
    opti.subject_to(X[:, 0] == x0)
    for k in range(MPCConfig.N):
        model_input_sym = ca.vertcat(U[:, k], X[:, k], dt_param)
        x_next_pred = dynamics_func(model_input_sym, l4c_params[k])
        opti.subject_to(X[:, k+1] == x_next_pred)

    for k in range(MPCConfig.N):
        opti.subject_to(opti.bounded(MPCConfig.U_MIN, U[:, k], MPCConfig.U_MAX))
    
    solver_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opti.solver('ipopt', solver_opts)

    # --- Phase 3: The Simulation Loop ---
    print("--- Starting MPC simulation ---")
    state_cols = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    sample = df[state_cols].dropna().sample(2, random_state=42)
    x_current = sample.iloc[0].values; x_target = sample.iloc[1].values
    
    history_x, history_u = [x_current], []
    n_steps = int(MPCConfig.SIM_TIME / MPCConfig.DT)
    
    # *** FIXED: Initialize guess to the midpoint of the control bounds ***
    mid_point_u = [(u_min + u_max) / 2.0 for u_min, u_max in zip(MPCConfig.U_MIN, MPCConfig.U_MAX)]
    u_guess = np.array([mid_point_u] * MPCConfig.N).T
    last_u_optimal = np.array(mid_point_u) # Also start u_prev from a feasible point
    
    sim_times = []
    start = time.time()
    for i in range(n_steps):
        x_guess_np = np.zeros((MPCConfig.N + 1, n_states))
        x_guess_np[0, :] = x_current
        with torch.no_grad():
            for k in range(MPCConfig.N):
                model_input_k = np.concatenate([u_guess[:, k], x_guess_np[k, :], [MPCConfig.DT]])
                model_input_torch = torch.from_numpy(model_input_k).float().unsqueeze(0)
                x_guess_np[k+1, :] = full_model(model_input_torch).squeeze(0).numpy()

        linearization_points_u = u_guess.T
        linearization_points_x = x_guess_np[:-1, :]
        dt_col = np.full((MPCConfig.N, 1), MPCConfig.DT)
        linearization_points_np = np.hstack([linearization_points_u, linearization_points_x, dt_col])

        casadi_params_batch = l4c_model.get_params(linearization_points_np)
        
        opti.set_value(dt_param, MPCConfig.DT)
        for k in range(MPCConfig.N):
            opti.set_value(l4c_params[k], casadi_params_batch[k, :])
        
        opti.set_value(x0, x_current); opti.set_value(x_ref, x_target); opti.set_value(u_prev, last_u_optimal)
        opti.set_initial(U, u_guess); opti.set_initial(X, x_guess_np.T)

        try:
            sol = opti.solve()
            u_optimal_all = sol.value(U)
            last_u_optimal = u_optimal_all[:, 0]
            u_guess = np.roll(u_optimal_all, -1, axis=1); u_guess[:, -1] = last_u_optimal
        except Exception as e:
            print(f"\nSolver failed at step {i}: {e}"); break
        
        start_sim = time.time()
        model_input_sim = np.concatenate([last_u_optimal, x_current, [MPCConfig.DT]])
        with torch.no_grad():
            model_input_torch = torch.from_numpy(model_input_sim).float().unsqueeze(0)
            x_current = full_model(model_input_torch).squeeze(0).numpy()
        history_x.append(x_current); history_u.append(last_u_optimal)
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
    history_x = np.array(history_x); history_u = np.array(history_u)
    import os
    os.makedirs('results', exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    time_axis = np.arange(history_x.shape[0]) * MPCConfig.DT
    axs[0].plot(time_axis, history_x[:, 0], label='Tip X'); axs[0].plot(time_axis, history_x[:, 1], label='Tip Y'); axs[0].plot(time_axis, history_x[:, 2], label='Tip Z')
    axs[0].axhline(y=x_target[0], color='r', linestyle='--', label='Target X'); axs[0].axhline(y=x_target[1], color='g', linestyle='--', label='Target Y'); axs[0].axhline(y=x_target[2], color='b', linestyle='--', label='Target Z')
    axs[0].set_ylabel('Position (cm)'); axs[0].set_title('MPC Trajectory (l4casadi)'); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(time_axis, history_x[:, 3:6]); axs[1].axhline(y=0, color='k', linestyle='--'); axs[1].set_ylabel('Velocity (cm/s)'); axs[1].grid(True)
    if history_u.size > 0:
        time_axis_u = np.arange(history_u.shape[0]) * MPCConfig.DT
        axs[2].step(time_axis_u, history_u[:, 0], where='post', label='Volume 1'); axs[2].step(time_axis_u, history_u[:, 1], where='post', label='Volume 2'); axs[2].step(time_axis_u, history_u[:, 2], where='post', label='Volume 3')
        axs[2].legend() # Only add legend if there are plots
    axs[2].set_ylabel('Control Input'); axs[2].set_xlabel('Time (s)'); axs[2].set_title('MPC Control Inputs'); axs[2].grid(True)
    plt.tight_layout()
    plot_path = 'results/mpc_l4casadi.png'
    plt.savefig(plot_path)
    print(f"\nMPC trajectory plot saved as '{plot_path}'.")

if __name__ == "__main__":
    run_mpc_simulation()