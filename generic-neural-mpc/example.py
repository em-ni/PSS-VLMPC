import time
import numpy as np
import torch
from tqdm import tqdm
import os
import shutil

from config import SystemConfig, MPCConfig
from system_x_dot.system import GenericSystem
from system_x_dot.system_mpc import SystemMPC
from utils.plotter import plot_results
from model.train_example import true_system_dynamics_dt

def plant_dynamics(x, u):
    """Wrapper to use the torch-based true dynamics with numpy."""
    x_torch = torch.from_numpy(x).unsqueeze(0).float()
    u_torch = torch.from_numpy(u).unsqueeze(0).float()
    x_dot = true_system_dynamics_dt(x_torch, u_torch)
    return x_dot.squeeze(0).numpy()

def generate_reference_trajectory(sim_time, dt, N_horizon):
    """Generates a generic reference trajectory."""
    num_steps = int(sim_time / dt)
    t = np.linspace(0, sim_time, num_steps + N_horizon + 1)
    
    x_ref = np.zeros((len(t), SystemConfig.STATE_DIM))
    
    # Generate a sine wave for the first state, its derivative for the second,
    # and zeros for all other states. This provides a simple, non-trivial
    # target for any system with at least 2 states.
    if SystemConfig.STATE_DIM >= 2:
        x_ref[:, 0] = np.sin(t)
        x_ref[:, 1] = np.cos(t)
    
    return t, x_ref

def cleanup_acados_files():
    """Programmatically cleans up acados build artifacts."""
    if os.path.exists('acados_ocp.json'):
        os.remove('acados_ocp.json')
    if os.path.exists('libacados_ocp_solver_generic_system_model.so'):
        os.remove('libacados_ocp_solver_generic_system_model.so')
    if os.path.exists('c_generated_code'):
        shutil.rmtree('c_generated_code')
    print("Cleaned up acados build files.")

def main():
    cleanup_acados_files()
    print("Initializing Neural MPC for a generic system...")
    
    # --- 1. Setup ---
    sim_time = MPCConfig.T
    dt = MPCConfig.DT
    
    # Generic initial state
    initial_state = np.zeros(SystemConfig.STATE_DIM)
    plant = GenericSystem(initial_state)
    mpc = SystemMPC()

    t_series, x_ref_full = generate_reference_trajectory(sim_time, dt, MPCConfig.N_HORIZON)
    
    num_sim_steps = int(sim_time / dt)
    x_history = np.zeros((num_sim_steps + 1, SystemConfig.STATE_DIM))
    u_history = np.zeros((num_sim_steps, SystemConfig.CONTROL_DIM))
    x_ref_history = np.zeros_like(x_history)

    x_history[0, :] = plant.get_state()
    x_ref_history[0, :] = x_ref_full[0, :]
    
    # --- 2. Simulation Loop ---
    print("Starting simulation...")
    start = time.time()
    for i in tqdm(range(num_sim_steps)):
        current_x = plant.get_state()
        
        x_ref_horizon = x_ref_full[i : i + MPCConfig.N_HORIZON + 1, :]
        
        u_optimal = mpc.solve(current_x, x_ref_horizon)
        
        plant.rk4_step(lambda x, u: plant_dynamics(x, u), u_optimal, dt)
        
        x_history[i+1, :] = plant.get_state()
        u_history[i, :] = u_optimal
        x_ref_history[i+1, :] = x_ref_full[i+1, :]

    end = time.time()
    print(f"Simulated {sim_time} seconds in {end - start:.2f} seconds.")
    print("Simulation finished.")
    
    # --- 3. Plot Results ---
    plot_results(t_series[:num_sim_steps+1], x_history, u_history, x_ref_history)

if __name__ == '__main__':
    main()