import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(t_series, x_history, u_history, x_ref_history, title="MPC Trajectory Tracking"):
    """Plots the state and control trajectories and saves the figure."""
    
    num_states = x_history.shape[1]
    num_controls = u_history.shape[1]
    
    fig, axes = plt.subplots(num_states + num_controls, 1, figsize=(12, 2.5 * (num_states + num_controls)), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # Plot states
    for i in range(num_states):
        axes[i].plot(t_series, x_history[:, i], label=f'State x{i} (Actual)')
        axes[i].plot(t_series, x_ref_history[:, i], 'r--', label=f'State x{i} (Reference)')
        axes[i].set_ylabel(f'x{i}')
        axes[i].legend()
        axes[i].grid(True)
        
    # Plot controls
    for i in range(num_controls):
        ax_idx = num_states + i
        # Note: u_history has one less element than t_series
        axes[ax_idx].plot(t_series[:-1], u_history[:, i], label=f'Control u{i}')
        axes[ax_idx].set_ylabel(f'u{i}')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- Save the figure instead of showing it ---
    save_path = "results.png"
    plt.savefig(save_path)
    print(f"Plot saved to {os.path.abspath(save_path)}")
    plt.close(fig) # Close the figure to free up memory
    # plt.show() # Comment out or remove this line