# train_sim.py - Improved version to prevent overfitting
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from functorch import vmap, jacrev, hessian # deprecated
from torch.func import vmap, jacrev, hessian
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Training Configuration ---
class TrainingConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    SIM_DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../sim/results/sim_dataset.csv"))
    # SIM_DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../sim/results/sim_dataset_old.csv"))
    MODEL_PATH = os.path.join(BASE_DIR, "data", "sim_rob_f.pth")
    INPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "sim_rob_i_scaler.joblib")
    OUTPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "sim_rob_o_scaler.joblib")
    PLOT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "sim_rob_perf.png")
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
class StatePredictor(nn.Module):
    """
    A right-sized neural network for state prediction.
    Notes: the model needs to be at least C1 continuous because:
        - uniform continuity assumption 4 (Seel et al., "Neural Network-Based...")
        - differentiability requirement for jacobian computation ("Salzmann et al., "Real-time Neural MPC")
        - eventually C2 for hessian computation (if using second-order approximation)
    
    Derivatives:
        - Tanh: tanh(x) -> sech^2(x) -> -2sech^2(x) * tanh(x)
        - SiLU: x * sigmoid(x) -> sigmoid(x) + x * sigmoid'(x) -> sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        - GELU: ?
    """
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
class RobotStateDataset(Dataset):
    """Custom PyTorch Dataset."""
    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_prepare_data(filepath):
    """
    Loads data from a single continuous trajectory CSV.
    Creates pairs where:
      X = [torques_k, state_k]
      y = [state_k+1]
    
    The input CSV format is assumed to be:
    T,rod1_torque_x,rod1_torque_y,rod2_torque_x,rod2_torque_y,tip_position_x,tip_position_y,tip_position_z,tip_velocity_x,tip_velocity_y,tip_velocity_z
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Clean up column names (remove whitespace)
    df.columns = df.columns.str.strip()

    # Define the columns for state (what we predict) and control inputs (what we command)
    STATE_COLS = [
        'tip_position_x', 'tip_position_y', 'tip_position_z',
        'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z'
    ]
    INPUT_COLS = [
        'rod1_torque_x', 'rod1_torque_y',
        'rod2_torque_x', 'rod2_torque_y'
    ]
    
    # These are all the features that describe the system at step 'k'
    CURRENT_FEATURES = INPUT_COLS + STATE_COLS
    
    # Ensure all required columns exist and drop rows with any missing values
    all_required_cols = ['T'] + CURRENT_FEATURES
    for col in all_required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file.")
    
    df.dropna(subset=all_required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 2:
        raise ValueError("Not enough data to create training pairs (must have at least 2 rows).")

    print("Processing data as a single continuous trajectory...")
    
    # Get features for the current time step 'k' (all rows except the last one)
    current_features = df[CURRENT_FEATURES].iloc[:-1].values
    
    # Get the target state for the next time step 'k+1' (all rows except the first one)
    next_state = df[STATE_COLS].iloc[1:].values
    
    # Assemble the final input matrix # X = [u_k, x_k]
    X = current_features
    
    # The output matrix y is simply the next state
    y = next_state
    
    print("Finished processing data.")
    return X, y


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """The main training loop."""
    criterion = nn.MSELoss()  # Mean Squared Error is good for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    model.to(device)
    
    history = {'train_loss': [], 'val_loss': []}

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        
        train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print("Training finished.")
    return history


def plot_predictions(y_true, y_pred, save_path):
    """Generates Predicted vs. Actual plots and saves the figure to a file."""
    # These labels match the output of our new model, so they don't need to change.
    state_labels = [
        'Tip Position X', 'Tip Position Y', 'Tip Position Z',
        'Tip Velocity X', 'Tip Velocity Y', 'Tip Velocity Z'
    ]
    
    num_states = y_true.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(num_states):
        ax = axes[i]
        # Use a smaller subset of points for plotting if the test set is very large
        sample_size = min(len(y_true), 2000)
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        
        ax.scatter(y_true[indices, i], y_pred[indices, i], alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(state_labels[i], fontsize=14)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(pad=3.0)
    plt.suptitle("Predicted vs. Actual State Values on Test Set", fontsize=20, y=1.02)
    
    # Save the plot to a file.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")
    plt.close(fig) # Close the figure to free up memory


def plot_rollout_performance(results, save_path):
    """
    Plots the model's rollout performance (MSE) as a function of the prediction horizon.
    """
    horizons = results['horizons']
    avg_mse = results['avg_mse']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(horizons, avg_mse, 'bo-', label='Average MSE')
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=12)
    ax.set_ylabel("Average Mean Squared Error (MSE)", fontsize=12)
    ax.set_title("Model Prediction Error vs. Rollout Horizon", fontsize=16)
    ax.set_xticks(horizons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which='both', linestyle='--')
    ax.legend()
    
    # Use a logarithmic scale if the error grows very fast
    if max(avg_mse) / min(avg_mse) > 50:
        ax.set_yscale('log')
        ax.set_ylabel("Average Mean Squared Error (MSE) - Log Scale", fontsize=12)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nRollout performance plot saved to: {save_path}")
    plt.close(fig)


def plot_approximation_comparison(results_dict, save_path):
    """
    Plots comparison of different approximation orders in rollout performance.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    
    for i, (order_name, results) in enumerate(results_dict.items()):
        horizons = results['horizons']
        avg_mse = results['avg_mse']
        
        ax.plot(horizons, avg_mse, color=colors[i], marker=markers[i], 
                linewidth=2.5, markersize=8, linestyle=linestyles[i], 
                markerfacecolor='white', markeredgecolor=colors[i], 
                markeredgewidth=2, label=order_name)
    
    ax.set_xlabel("Prediction Horizon (steps)", fontsize=14)
    ax.set_ylabel("Average Mean Squared Error (MSE)", fontsize=14)
    ax.set_title("Model Prediction Error vs. Rollout Horizon\n(Neural Network vs Approximations)", fontsize=16)
    ax.set_xticks(horizons)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, loc='best')
    
    # Use logarithmic scale if needed
    max_mse = max([max(results['avg_mse']) for results in results_dict.values()])
    min_mse = min([min(results['avg_mse']) for results in results_dict.values()])
    if max_mse / min_mse > 50:
        ax.set_yscale('log')
        ax.set_ylabel("Average Mean Squared Error (MSE) - Log Scale", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRollout performance comparison plot saved to: {save_path}")
    plt.close(fig)


class NeuralNetworkApproximator:
    """Class to handle neural network approximations for rollout evaluation"""
    
    def __init__(self, model, input_scaler, output_scaler, device):
        self.model = model.to(device).eval()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.device = device
        
        # Setup scaling tensors for PyTorch operations
        self.input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
        self.input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
        self.output_scale = torch.tensor(output_scaler.scale_, dtype=torch.float32, device=device)
        self.output_mean = torch.tensor(output_scaler.mean_, dtype=torch.float32, device=device)
    
    def full_pytorch_model(self, x_and_u_torch):
        """Full PyTorch model with scaling"""
        scaled_input = (x_and_u_torch - self.input_mean) / self.input_scale
        scaled_output = self.model(scaled_input)
        return scaled_output * self.output_scale + self.output_mean
    
    def first_order_approximation(self, x_and_u_torch, linearization_point):
        """First-order (linear) approximation of the neural network"""
        # Get the jacobian at the linearization point
        jac = jacrev(self.full_pytorch_model)(linearization_point)
        
        # Get the function value at linearization point
        f0 = self.full_pytorch_model(linearization_point)
        
        # Linear approximation: f(x) ≈ f(x0) + J(x0) * (x - x0)
        delta_x = x_and_u_torch - linearization_point
        return f0 + torch.matmul(jac, delta_x)
    
    def second_order_approximation(self, x_and_u_torch, linearization_point):
        """Second-order (quadratic) approximation of the neural network with numerical stability"""
        try:
            # Get function value, jacobian, and hessian at linearization point
            f0 = self.full_pytorch_model(linearization_point)
            jac = jacrev(self.full_pytorch_model)(linearization_point)
            hess = hessian(self.full_pytorch_model)(linearization_point)
            
            # Check for NaN/Inf in derivatives
            if torch.isnan(jac).any() or torch.isinf(jac).any():
                print("Warning: NaN/Inf detected in Jacobian, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            if torch.isnan(hess).any() or torch.isinf(hess).any():
                print("Warning: NaN/Inf detected in Hessian, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            delta_x = x_and_u_torch - linearization_point
            
            # Limit the step size to prevent explosion
            max_step_size = 0.1  # Limit how far we extrapolate
            delta_x_norm = torch.norm(delta_x)
            if delta_x_norm > max_step_size:
                delta_x = delta_x * (max_step_size / delta_x_norm)
            
            linear_term = torch.matmul(jac, delta_x)
            
            # For the quadratic term with regularization
            quadratic_term = torch.zeros_like(f0)
            hessian_regularization = 1e-6  # Small regularization term
            
            for i in range(f0.shape[0]):  # For each output dimension
                hess_i = hess[i]
                
                # Add regularization to make Hessian more stable
                hess_i_reg = hess_i + hessian_regularization * torch.eye(hess_i.shape[0], device=hess_i.device)
                
                # Check condition number - if too high, skip quadratic term
                try:
                    eigenvals = torch.linalg.eigvals(hess_i_reg)
                    max_eigval = torch.max(torch.real(eigenvals))
                    min_eigval = torch.min(torch.real(eigenvals))
                    
                    if min_eigval > 0:  # Only compute if positive definite
                        condition_number = max_eigval / min_eigval
                        if condition_number > 1e6:  # Skip if ill-conditioned
                            continue
                except:
                    continue  # Skip if eigenvalue computation fails
                
                quad_i = 0.5 * torch.matmul(torch.matmul(delta_x, hess_i_reg), delta_x)
                
                # Clip the quadratic term to prevent explosion
                quad_i = torch.clamp(quad_i, -1e3, 1e3)
                
                # Check for numerical issues in the quadratic term
                if torch.isnan(quad_i) or torch.isinf(quad_i):
                    continue  # Skip this quadratic term
                
                quadratic_term[i] = quad_i
            
            result = f0 + linear_term + quadratic_term
            
            # Final clipping to prevent explosion
            result = torch.clamp(result, -1e4, 1e4)
            
            # Final check for NaN/Inf in result
            if torch.isnan(result).any() or torch.isinf(result).any():
                print("Warning: NaN/Inf detected in second-order result, falling back to first-order approximation")
                return self.first_order_approximation(x_and_u_torch, linearization_point)
            
            return result
            
        except Exception as e:
            print(f"Warning: Exception in second-order approximation: {e}, falling back to first-order")
            return self.first_order_approximation(x_and_u_torch, linearization_point)


def evaluate_approximation_rollouts(model, X_test_orig, y_test_orig, input_scaler, output_scaler, device,
                                   approximation_order=1, horizons=[1, 5, 10, 20, 40, 80, 160], num_rollouts=100):
    """
    Evaluates neural network approximations on multi-horizon rollouts.
    
    Args:
        approximation_order (int): 1 for first-order, 2 for second-order approximation
        Other args same as evaluate_multi_horizon_rollouts
    """
    print(f"\n--- Evaluating {approximation_order}-order NN Approximation (Multi-Horizon Rollouts) ---")
    
    # Setup approximator
    approximator = NeuralNetworkApproximator(model, input_scaler, output_scaler, device)
    
    # X = [torques_k (4), state_k (6)] -> state is columns 4 to 10
    num_state_dims = y_test_orig.shape[1]
    state_start_col = 4  # Assumes 4 torque inputs
    state_end_col = state_start_col + num_state_dims
    
    # Store results
    results = {'horizons': [], 'avg_mse': [], 'avg_mae': []}
    
    for horizon in horizons:
        # Skip if the horizon is longer than the available test data
        if horizon >= len(X_test_orig):
            print(f"Horizon {horizon} is too long for the test set ({len(X_test_orig)} samples). Skipping.")
            continue
        
        horizon_errors_mse = []
        horizon_errors_mae = []
        
        # Determine the possible starting points for a rollout of this length
        max_start_idx = len(X_test_orig) - horizon
        start_indices = np.random.choice(max_start_idx, size=min(num_rollouts, max_start_idx), replace=False)
        
        for start_idx in start_indices:
            # --- Perform one short rollout using approximation ---
            
            # Get the real starting state from the test set
            current_state = X_test_orig[start_idx, state_start_col:state_end_col]
            
            # For second-order approximation, recompute linearization point more frequently
            linearization_frequency = 5 if approximation_order == 2 else horizon  # Relinearize every 5 steps for 2nd order
            
            predicted_trajectory = []
            rollout_failed = False
            
            with torch.no_grad():
                for i in range(horizon):
                    # Index for the current step in the original test set
                    step_idx = start_idx + i
                    
                    # Get the ground-truth control inputs for this step
                    control_inputs = X_test_orig[step_idx, :state_start_col]
                    
                    # Assemble the input vector for the model using the *predicted* state
                    input_vec = np.hstack([control_inputs, current_state])
                    
                    # Convert to tensor
                    input_tensor = torch.tensor(input_vec, dtype=torch.float32, device=device)
                    
                    # Update linearization point for second-order approximation
                    if i % linearization_frequency == 0:
                        linearization_input = input_tensor.clone()
                    
                    # Apply the approximation
                    try:
                        if approximation_order == 1:
                            # Use original linearization point for first-order
                            if i == 0:
                                linearization_input = torch.tensor(X_test_orig[start_idx], dtype=torch.float32, device=device)
                            pred_tensor = approximator.first_order_approximation(input_tensor, linearization_input)
                        elif approximation_order == 2:
                            pred_tensor = approximator.second_order_approximation(input_tensor, linearization_input)
                        else:
                            # Fallback to full model
                            scaled_input = (input_tensor - approximator.input_mean) / approximator.input_scale
                            scaled_output = approximator.model(scaled_input)
                            pred_tensor = scaled_output * approximator.output_scale + approximator.output_mean
                        
                        current_state = pred_tensor.cpu().numpy()
                        
                        # Check for NaN/Inf in the predicted state
                        if np.isnan(current_state).any() or np.isinf(current_state).any():
                            print(f"Warning: NaN/Inf detected in predicted state at horizon {horizon}, step {i}")
                            rollout_failed = True
                            break
                        
                        # Additional check for explosive values
                        if np.abs(current_state).max() > 1e6:
                            print(f"Warning: Explosive values detected in predicted state at horizon {horizon}, step {i}")
                            rollout_failed = True
                            break
                        
                        predicted_trajectory.append(current_state)
                        
                    except Exception as e:
                        print(f"Warning: Exception during approximation at horizon {horizon}, step {i}: {e}")
                        rollout_failed = True
                        break
            
            # Only compute metrics if the rollout was successful
            if not rollout_failed and len(predicted_trajectory) == horizon:
                # Compare the predicted trajectory to the ground truth
                predicted_trajectory = np.array(predicted_trajectory)
                ground_truth_trajectory = y_test_orig[start_idx : start_idx + horizon]
                
                try:
                    mse = mean_squared_error(ground_truth_trajectory, predicted_trajectory)
                    mae = mean_absolute_error(ground_truth_trajectory, predicted_trajectory)
                    
                    # Additional check for explosive metrics
                    if np.isnan(mse) or np.isnan(mae) or np.isinf(mse) or np.isinf(mae) or mse > 1e6 or mae > 1e6:
                        print(f"Warning: Explosive metrics at horizon {horizon}: MSE={mse}, MAE={mae}")
                        continue
                    
                    horizon_errors_mse.append(mse)
                    horizon_errors_mae.append(mae)
                except Exception as e:
                    print(f"Warning: Exception computing metrics at horizon {horizon}: {e}")
                    continue
        
        # Only compute averages if we have valid results
        if len(horizon_errors_mse) > 0:
            avg_mse = np.mean(horizon_errors_mse)
            avg_mae = np.mean(horizon_errors_mae)
            
            results['horizons'].append(horizon)
            results['avg_mse'].append(avg_mse)
            results['avg_mae'].append(avg_mae)
            
            print(f"Horizon: {horizon:3d} steps -> Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f} ({len(horizon_errors_mse)} successful rollouts)")
        else:
            print(f"Warning: No successful rollouts for horizon {horizon} steps")
    
    return results


def evaluate_multi_horizon_rollouts(model, X_test_orig, y_test_orig, input_scaler, output_scaler, device, 
                                    horizons=[1, 5, 10, 20, 40, 80, 160], num_rollouts=100):
    """
    Performs multiple short rollouts for various horizon lengths to evaluate
    prediction error accumulation. This is much more representative of MPC usage.
    """
    print("\n--- Evaluating on Test Set (Multi-Horizon Rollouts) ---")
    model.eval()

    # X = [torques_k (4), state_k (6)] -> state is columns 4 to 10
    num_state_dims = y_test_orig.shape[1]
    state_start_col = 4 # Assumes 4 torque inputs
    state_end_col = state_start_col + num_state_dims
    
    # Store results
    results = {'horizons': [], 'avg_mse': [], 'avg_mae': []}

    for horizon in horizons:
        # Skip if the horizon is longer than the available test data
        if horizon >= len(X_test_orig):
            print(f"Horizon {horizon} is too long for the test set ({len(X_test_orig)} samples). Skipping.")
            continue
            
        horizon_errors_mse = []
        horizon_errors_mae = []
        
        # Determine the possible starting points for a rollout of this length
        max_start_idx = len(X_test_orig) - horizon
        # Randomly select starting points for our rollouts
        start_indices = np.random.choice(max_start_idx, size=min(num_rollouts, max_start_idx), replace=False)
        
        for start_idx in start_indices:
            # --- Perform one short rollout ---
            
            # Get the real starting state from the test set
            current_state = X_test_orig[start_idx, state_start_col:state_end_col]
            
            predicted_trajectory = []
            
            with torch.no_grad():
                for i in range(horizon):
                    # Index for the current step in the original test set
                    step_idx = start_idx + i
                    
                    # Get the ground-truth control inputs for this step
                    control_inputs = X_test_orig[step_idx, :state_start_col]
                    
                    # Assemble the input vector for the model using the *predicted* state
                    input_vec = np.hstack([control_inputs, current_state])

                    # Scale -> Predict -> Unscale
                    input_tensor = torch.tensor(input_scaler.transform(input_vec.reshape(1, -1)), dtype=torch.float32).to(device)
                    pred_scaled = model(input_tensor)
                    current_state = output_scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()
                    
                    predicted_trajectory.append(current_state)

            # Compare the predicted trajectory to the ground truth
            predicted_trajectory = np.array(predicted_trajectory)
            ground_truth_trajectory = y_test_orig[start_idx : start_idx + horizon]
            
            horizon_errors_mse.append(mean_squared_error(ground_truth_trajectory, predicted_trajectory))
            horizon_errors_mae.append(mean_absolute_error(ground_truth_trajectory, predicted_trajectory))
            
        # Average the errors over all the short rollouts for this horizon
        avg_mse = np.mean(horizon_errors_mse)
        avg_mae = np.mean(horizon_errors_mae)
        
        results['horizons'].append(horizon)
        results['avg_mse'].append(avg_mse)
        results['avg_mae'].append(avg_mae)
        
        print(f"Horizon: {horizon:3d} steps -> Avg MSE: {avg_mse:.6f}, Avg MAE: {avg_mae:.6f}")
    
    # Generate the performance plot
    rollout_plot_path = TrainingConfig.PLOT_OUTPUT_PATH.replace('.png', '_rollout.png')
    plot_rollout_performance(results, rollout_plot_path)

    return results

# --- Main Execution ---
if __name__ == "__main__":
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train a neural network model for robot state prediction.")
    parser.add_argument('--rollouts-eval', action='store_true', help="Evaluate multi-horizon rollouts after training.")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(TrainingConfig.MODEL_PATH), exist_ok=True)

    # --- Load Data ---
    try:
        X, y = load_and_prepare_data(TrainingConfig.SIM_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {TrainingConfig.SIM_DATASET_PATH}")
        sys.exit()
    except (KeyError, ValueError) as e:
        print(f"Error during data preparation: {e}")
        sys.exit()

    print(f"Total samples loaded: {len(X)}")
    print(f"Shape of input features (X): {X.shape}")
    print(f"Shape of target features (y): {y.shape}")
    
    # --- Chronological Data Splitting ---
    print("\n--- Performing Chronological Split ---")
    
    total_samples = len(X)
    train_end_idx = int(total_samples * (1 - TrainingConfig.TEST_SIZE - TrainingConfig.VAL_SIZE))
    val_end_idx = int(total_samples * (1 - TrainingConfig.TEST_SIZE))

    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_val, y_val = X[train_end_idx:val_end_idx], y[train_end_idx:val_end_idx]
    X_test, y_test = X[val_end_idx:], y[val_end_idx:]

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("\nError: Dataset is too small. Please provide more data or adjust split sizes.")
        sys.exit()

    # --- Scale Data ---
    print("\nScaling data...")
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)
    
    X_val_scaled = input_scaler.transform(X_val)
    y_val_scaled = output_scaler.transform(y_val)
    print("Data scaled successfully.")
    
    # --- Create DataLoaders ---
    train_dataset = RobotStateDataset(X_train_scaled, y_train_scaled)
    val_dataset = RobotStateDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    # --- Initialize and Train Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = StatePredictor(input_dim, output_dim)
    print(f"Model initialized with input_dim={input_dim} and output_dim={output_dim}")

    history = train_model(model, train_loader, val_loader, TrainingConfig.NUM_EPOCHS, TrainingConfig.LEARNING_RATE, device)

    # --- Save the trained model and the scalers ---
    print("\n--- Saving model and scalers ---")
    torch.save(model.state_dict(), TrainingConfig.MODEL_PATH)
    joblib.dump(input_scaler, TrainingConfig.INPUT_SCALER_PATH)
    joblib.dump(output_scaler, TrainingConfig.OUTPUT_SCALER_PATH)
    print(f"Model saved to {TrainingConfig.MODEL_PATH}")
    print(f"Input scaler saved to {TrainingConfig.INPUT_SCALER_PATH}")
    print(f"Output scaler saved to {TrainingConfig.OUTPUT_SCALER_PATH}")
    
    # --- Evaluation on Test Set (One-Step-Ahead) ---
    print("\n--- Evaluating on Test Set (Single Step Predictions) ---")
    model.eval()
    X_test_scaled = input_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).cpu().numpy()

    predictions = output_scaler.inverse_transform(predictions_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Performance Metrics (One-Step-Ahead) ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"R-squared (R²):           {r2:.4f}")

    plot_predictions(y_test, predictions, TrainingConfig.PLOT_OUTPUT_PATH)
    
    if args.rollouts_eval:
        print("\n--- Evaluating Multi-Horizon Rollouts ---")
    
        # --- Full NN Multi-Horizon Rollout Evaluation ---
        results_full = evaluate_multi_horizon_rollouts(
            model, 
            X_test, 
            y_test, 
            input_scaler, 
            output_scaler, 
            device
        )
        
        # --- First-Order Approximation Rollout Evaluation ---
        results_first_order = evaluate_approximation_rollouts(
            model,
            X_test,
            y_test,
            input_scaler,
            output_scaler,
            device,
            approximation_order=1,
            horizons=[1, 5, 10, 20, 40, 80, 160],
            num_rollouts=1000
        )
        
        # --- Second-Order Approximation Rollout Evaluation ---
        results_second_order = evaluate_approximation_rollouts(
            model,
            X_test,
            y_test,
            input_scaler,
            output_scaler,
            device,
            approximation_order=2,
            horizons=[1, 5, 10, 20, 40, 80, 160],
            num_rollouts=1000
        )
        
        # --- Generate Combined Rollout Performance Plot ---
        results_comparison = {
            'Full Neural Network': results_full,
            'First-Order Approximation': results_first_order,
            'Second-Order Approximation': results_second_order
        }
        
        # Replace the individual rollout plot with the comparison plot
        rollout_plot_path = TrainingConfig.PLOT_OUTPUT_PATH.replace('.png', '_rollout.png')
        plot_approximation_comparison(results_comparison, rollout_plot_path)
        
        # Compare approximations at different horizons
        for i, horizon in enumerate(results_full['horizons']):
            if horizon in results_first_order['horizons'] and horizon in results_second_order['horizons']:
                full_mse = results_full['avg_mse'][i]
                first_mse = results_first_order['avg_mse'][results_first_order['horizons'].index(horizon)]
                second_mse = results_second_order['avg_mse'][results_second_order['horizons'].index(horizon)]
                
                first_error_ratio = first_mse / full_mse if full_mse > 0 else float('inf')
                second_error_ratio = second_mse / full_mse if full_mse > 0 else float('inf')
                
                print(f"\nHorizon {horizon:3d} steps:")
                print(f"  Full NN MSE:           {full_mse:.6f}")
                print(f"  1st-order MSE:         {first_mse:.6f} ({first_error_ratio:.2f}x)")
                print(f"  2nd-order MSE:         {second_mse:.6f} ({second_error_ratio:.2f}x)")
                
    
    print(f"\nAll evaluation plots saved to: {os.path.dirname(TrainingConfig.PLOT_OUTPUT_PATH)}")
    print("Training and evaluation completed successfully!")