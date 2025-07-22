import time
import numpy as np
import torch
from tqdm import tqdm
import os
import shutil

from config import SystemConfig, MPCConfig, TrainingConfig
from system.system import GenericSystem
from system.system_mpc import SystemMPC
from utils.plotter import plot_results
from model.train_soft_robot import NeuralNetwork

def nn_plant_dynamics(model, input_mean, input_std, output_mean, output_std, x, u, device):
    """Predict x_dot using the trained neural network."""
    x_torch = torch.from_numpy(x).unsqueeze(0).float().to(device)
    u_torch = torch.from_numpy(u).unsqueeze(0).float().to(device)
    xu = torch.cat((x_torch, u_torch), dim=1)
    norm_input = (xu - input_mean) / input_std
    with torch.no_grad():
        norm_pred_x_dot = model(norm_input)
        pred_x_dot = (norm_pred_x_dot * output_std) + output_mean
    return pred_x_dot.squeeze(0).cpu().numpy()

def generate_reference_trajectory(sim_time, dt, N_horizon):
    num_steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, num_steps + N_horizon + 1)
    x_ref = np.zeros((len(t), SystemConfig.STATE_DIM))
    if SystemConfig.STATE_DIM >= 2:
        x_ref[:, 0] = np.sin(t)
        x_ref[:, 1] = np.cos(t)
    return t, x_ref

def cleanup_acados_files():
    if os.path.exists('acados_ocp.json'):
        os.remove('acados_ocp.json')
    if os.path.exists('libacados_ocp_solver_generic_system_model.so'):
        os.remove('libacados_ocp_solver_generic_system_model.so')
    if os.path.exists('c_generated_code'):
        shutil.rmtree('c_generated_code')
    print("Cleaned up acados build files.")

def rk4_step(model, input_mean, input_std, output_mean, output_std, x, u, dt, device):
    """Runge-Kutta 4th order integration using NN plant dynamics."""
    k1 = nn_plant_dynamics(model, input_mean, input_std, output_mean, output_std, x, u, device)
    k2 = nn_plant_dynamics(model, input_mean, input_std, output_mean, output_std, x + 0.5 * dt * k1, u, device)
    k3 = nn_plant_dynamics(model, input_mean, input_std, output_mean, output_std, x + 0.5 * dt * k2, u, device)
    k4 = nn_plant_dynamics(model, input_mean, input_std, output_mean, output_std, x + dt * k3, u, device)
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x_next

def main():
    cleanup_acados_files()
    print("Initializing Neural MPC for soft robot system...")

    # Load trained neural network model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    payload = torch.load(TrainingConfig.MODEL_SAVE_PATH, map_location=device)
    model = NeuralNetwork().to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    input_mean = payload['input_mean'].to(device)
    input_std = payload['input_std'].to(device)
    output_mean = payload['output_mean'].to(device)
    output_std = payload['output_std'].to(device)

    # --- 1. Setup ---
    sim_time = MPCConfig.T
    dt = MPCConfig.DT

    initial_state = np.zeros(SystemConfig.STATE_DIM)
    mpc = SystemMPC()

    t_series, x_ref_full = generate_reference_trajectory(sim_time, dt, MPCConfig.N_HORIZON)
    num_sim_steps = int(sim_time / dt)
    x_history = np.zeros((num_sim_steps + 1, SystemConfig.STATE_DIM))
    u_history = np.zeros((num_sim_steps, SystemConfig.CONTROL_DIM))
    x_ref_history = np.zeros_like(x_history)

    x_history[0, :] = initial_state
    x_ref_history[0, :] = x_ref_full[0, :]

    # --- 2. Simulation Loop ---
    print("Starting simulation...")
    start = time.time()
    x = initial_state.copy()
    for i in tqdm(range(num_sim_steps)):
        x_ref_horizon = x_ref_full[i : i + MPCConfig.N_HORIZON + 1, :]
        u_optimal = mpc.solve(x, x_ref_horizon)
        # Integrate using NN plant dynamics
        x = rk4_step(model, input_mean, input_std, output_mean, output_std, x, u_optimal, dt, device)
        x_history[i+1, :] = x
        u_history[i, :] = u_optimal
        x_ref_history[i+1, :] = x_ref_full[i+1, :]

    end = time.time()
    print(f"Simulated {sim_time} seconds in {end - start:.2f} seconds.")
    print("Simulation finished.")

    # --- 3. Plot Results ---
    plot_results(t_series[:num_sim_steps+1], x_history, u_history, x_ref_history)

if __name__ == '__main__':
    main()