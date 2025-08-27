import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
import numpy as np
from utils.nn_functions import load_model_and_scalers, predict_tip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors

# --- Configuration ---
N_SAMPLES = 10000  # Number of base u_values to sample
NUM_PERTURBATIONS = 15 # Number of perturbations for each base u_value
PERTURBATION_MAGNITUDE = 1 # Magnitude for random noise (e.g., noise in [-magnitude, +magnitude])
# --- End Configuration ---

min_u = config.U_MIN_CMD
max_u = config.U_MAX_CMD

# Load the model and scalers
model_path = config.MODEL_PATH
scalers_path = config.SCALERS_PATH
model, scaler_volumes, scaler_deltas, device = load_model_and_scalers(model_path, scalers_path)
if model is None or scaler_volumes is None or scaler_deltas is None:
    print("Error loading model or scalers. Cannot proceed.")
    exit(1)

results_data = []

# Randomly sample N base 3D u_sample_val values from the range
for i in range(N_SAMPLES):
    # Each original command is a 3D vector with independently sampled components
    u_sample_val_vector = np.random.uniform(min_u, max_u, 3).astype(np.float32)

    # Original command and prediction
    original_command = u_sample_val_vector # This is now the 3D random vector
    original_prediction = predict_tip(original_command, scaler_volumes, scaler_deltas, model, device)

    current_perturbed_commands = []
    current_perturbed_predictions = []

    # Perturbate the command and predict for each perturbation
    for _ in range(NUM_PERTURBATIONS):
        perturbation_noise = np.random.uniform(-PERTURBATION_MAGNITUDE, PERTURBATION_MAGNITUDE, 3)
        
        perturbed_command_channels = np.zeros(3, dtype=np.float32)
        # Apply perturbation to each component of the original_command
        perturbed_command_channels[0] = original_command[0] + perturbation_noise[0]
        perturbed_command_channels[1] = original_command[1] + perturbation_noise[1]
        perturbed_command_channels[2] = original_command[2] + perturbation_noise[2]
        
        # Clip the perturbed command to be within [min_u, max_u]
        perturbed_command_channels = np.clip(perturbed_command_channels, min_u, max_u)

        perturbed_prediction = predict_tip(perturbed_command_channels, scaler_volumes, scaler_deltas, model, device)
        
        current_perturbed_commands.append(perturbed_command_channels)
        current_perturbed_predictions.append(perturbed_prediction)

    # Calculate the max distance between the original prediction and its perturbed predictions
    max_distance_for_current_sample = 0.0
    if current_perturbed_predictions: # Ensure list is not empty
        for pert_pred in current_perturbed_predictions:
            distance = np.linalg.norm(original_prediction - pert_pred)
            if distance > max_distance_for_current_sample:
                max_distance_for_current_sample = distance
    
    results_data.append({
        "original_command": original_command, # Storing the 3D command
        "original_prediction": original_prediction,
        "perturbed_commands": current_perturbed_commands,
        "perturbed_predictions": current_perturbed_predictions,
        "max_distance_to_perturbations": max_distance_for_current_sample
    })

# --- Plotting ---
if results_data:
    # Plot 1: Original Predictions Colored by Sensitivity
    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')

    original_preds_x = [res['original_prediction'][0] for res in results_data]
    original_preds_y = [res['original_prediction'][1] for res in results_data]
    original_preds_z = [res['original_prediction'][2] for res in results_data]
    max_distances = [res['max_distance_to_perturbations'] for res in results_data]

    min_dist = min(max_distances) if max_distances else 0
    max_dist = max(max_distances) if max_distances else 1
    if min_dist == max_dist:
        norm = colors.Normalize(vmin=min_dist - 0.1 if min_dist > 0 else 0, vmax=max_dist + 0.1 if max_dist <1 else 1)
        if min_dist == 0 and max_dist == 0:
             norm = colors.Normalize(vmin=0, vmax=0.1)
    else:
        norm = colors.Normalize(vmin=min_dist, vmax=max_dist)
        
    cmap = cm.viridis

    scatter1 = ax1.scatter(original_preds_x, original_preds_y, original_preds_z, 
                           c=max_distances, cmap=cmap, norm=norm, s=60, alpha=0.8)

    cbar = fig1.colorbar(scatter1, ax=ax1, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Max Distance to Perturbed Predictions', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax1.set_xlabel('X Tip Position (m)', fontsize=12)
    ax1.set_ylabel('Y Tip Position (m)', fontsize=12)
    ax1.set_zlabel('Z Tip Position (m)', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.set_title(f'Original Predictions Colored by Sensitivity to Perturbations\n(N={N_SAMPLES}, Perturbation Mag={PERTURBATION_MAGNITUDE})', fontsize=14)
    fig1.tight_layout()

    # Plot 2: Original Inputs (now randomly distributed in 3D command space)
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111, projection='3d')

    original_cmds_x = [res['original_command'][0] for res in results_data]
    original_cmds_y = [res['original_command'][1] for res in results_data]
    original_cmds_z = [res['original_command'][2] for res in results_data]

    ax2.scatter(original_cmds_x, original_cmds_y, original_cmds_z, 
                c='blue', s=50, alpha=0.7) # Single color for all points

    ax2.set_xlabel('Command Channel 1 (u)', fontsize=12)
    ax2.set_ylabel('Command Channel 2 (u)', fontsize=12)
    ax2.set_zlabel('Command Channel 3 (u)', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(f'Original Input Commands (N={N_SAMPLES})', fontsize=14)
    
    # Set axis limits to be the same for better visualization of the command space
    # This might not be as critical now that points are not on x=y=z, but can still be useful
    all_cmd_values = original_cmds_x + original_cmds_y + original_cmds_z
    cmd_min_val = min(all_cmd_values) if all_cmd_values else min_u
    cmd_max_val = max(all_cmd_values) if all_cmd_values else max_u
    ax2.set_xlim([cmd_min_val, cmd_max_val])
    ax2.set_ylim([cmd_min_val, cmd_max_val])
    ax2.set_zlim([cmd_min_val, cmd_max_val])
    
    fig2.tight_layout()

    plt.show()
else:
    print("\nNo data to plot.")