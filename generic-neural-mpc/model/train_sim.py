# train_sim.py
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
    SIM_DATASET_PATH = "../sim/results/sim_dataset.csv" 
    MODEL_PATH = "model/data/sim_rob_f.pth"
    INPUT_SCALER_PATH = "model/data/sim_rob_i_scaler.joblib"
    OUTPUT_SCALER_PATH = "model/data/sim_rob_o_scaler.joblib"
    PLOT_OUTPUT_PATH = "model/data/sim_rob_perf.png"
    
    NUM_EPOCHS = 200
    BATCH_SIZE = 256
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-4


def load_and_prepare_data(filepath):
    """
    Loads data from a single continuous trajectory CSV.
    Calculates dt and creates pairs where:
      X = [torques_k, state_k, dt]
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
    
    # Calculate dt
    times_ms = df['T'].values
    dt_seconds = (times_ms[1:] - times_ms[:-1])
    dt_seconds_col = dt_seconds.reshape(-1, 1) # Reshape for horizontal stacking
    
    # Assemble the final input matrix # X = [u_k, x_k, dt]
    X = np.hstack([current_features, dt_seconds_col])
    
    # The output matrix y is simply the next state
    y = next_state
    
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
    """A larger feed-forward neural network for state prediction on large simulation datasets."""
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

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
        X, y = load_and_prepare_data(TrainingConfig.SIM_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {TrainingConfig.SIM_DATASET_PATH}")
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
    # Adjust validation split size based on the remaining data
    val_split_ratio = TrainingConfig.VAL_SIZE / (1 - TrainingConfig.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split_ratio, random_state=42
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
    
    # The input and output dimensions are automatically inferred from the data shape
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