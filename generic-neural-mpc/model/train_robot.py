# train_robot.py
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Training Configuration ---
class TrainingConfig:
    
    REAL_DATASET_PATH = "model/data/output_exp_2025-07-22_12-23-07.csv"
    MODEL_PATH = "model/data/real_rob_f.pth"
    INPUT_SCALER_PATH = "model/data/real_rob_i_scaler.joblib"
    OUTPUT_SCALER_PATH = "model/data/real_rob_o_scaler.joblib"
    PLOT_OUTPUT_PATH = "model/data/real_rob_perf.png"
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001


def load_and_prepare_data(filepath):
    """
    Loads data, calculates dt, and creates pairs where X = [u_k, x_k, dt]
    x_k = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    y = [x_k+1].
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

    STATE_COLS = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    INPUT_COLS = ['volume_1', 'volume_2', 'volume_3']
    CURRENT_FEATURES = INPUT_COLS + STATE_COLS
    
    df.dropna(subset=CURRENT_FEATURES + ['T'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    X_list, y_list = [], []
    
    print("Processing trajectories (without acceleration)...")
    for traj_id, group in df.groupby('trajectory'):
        if len(group) < 2:
            continue
            
        current_features = group[CURRENT_FEATURES].iloc[:-1].values
        next_state = group[STATE_COLS].iloc[1:].values
        
        times_ms = group['T'].values
        dt_seconds = (times_ms[1:] - times_ms[:-1]) / 1000.0
        dt_seconds_col = dt_seconds.reshape(-1, 1)
        
        # X = [u_k, x_k, dt]
        X_with_dt = np.hstack([current_features, dt_seconds_col])
        
        X_list.append(X_with_dt)
        y_list.append(next_state)

    if not X_list:
        raise ValueError("Not enough data to create training pairs.")

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    print("Finished processing data.")
    return X, y
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

class StatePredictor(nn.Module):
    """A simple feed-forward neural network for state prediction."""
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
# This bigger model is more precise but the mpc has slower convergence
"""A bigger, more precise neural network can paradoxically lead to slower MPC convergence because its complexity creates a "jagged" and non-smooth function landscape. The MPC relies on linear approximations (the tangent slope) at each step to find the next best move. On a smooth landscape, these approximations are accurate over a large area, allowing the optimizer to take confident, large steps and converge quickly. On the jagged landscape of the bigger model, the linear approximation is only valid for a tiny, immediate area. This forces the optimizer to take very small, cautious steps and run many more iterations, dramatically slowing down the process. Essentially, for this type of optimization, the smoothness of the model is more important than its absolute precision, and the simpler model provides a much smoother, more navigable landscape for the controller to work with.
"""
# class StatePredictor(nn.Module):
#     """A simple feed-forward neural network for state prediction."""
#     def __init__(self, input_dim, output_dim):
#         super(StatePredictor, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim)
#         )
    
#     def forward(self, x):
#         return self.network(x)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """The main training loop."""
    criterion = nn.MSELoss()  # Mean Squared Error is good for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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

# --- Main Execution ---
if __name__ == "__main__":
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(TrainingConfig.MODEL_PATH), exist_ok=True)

    # --- Create Data and Load ---
    try:
        X, y = load_and_prepare_data(TrainingConfig.REAL_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {TrainingConfig.REAL_DATASET_PATH}")
        print("Please create the file or update the path in the TrainingConfig class.")
        sys.exit()
    except (KeyError, ValueError) as e:
        print(f"Error during data preparation: {e}")
        sys.exit()

    print(f"Shape of input features (X): {X.shape}")
    print(f"Shape of target features (y): {y.shape}")
    
    # --- Split Data (using a fixed random_state is crucial for reproducibility) ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=TrainingConfig.VAL_SIZE / (1 - TrainingConfig.TEST_SIZE), random_state=42
    )

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
    
    # The input dimension is now automatically inferred from the data, including dt
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