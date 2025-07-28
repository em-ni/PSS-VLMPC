import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig

def load_and_prepare_data(filepath):
    """Loads data, selects features, and creates (current_state, next_state) pairs."""
    df = pd.read_csv(filepath)
    # Clean up original column names by removing spaces and units
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

    # Define the lists of CLEANED column names
    STATE_COLS_CLEAN = ['tip_x', 'tip_y', 'tip_z', 'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z']
    INPUT_COLS_CLEAN = ['volume_1', 'volume_2', 'volume_3']
    ACCEL_COLS_CLEAN = ['tip_acceleration_x', 'tip_acceleration_y', 'tip_acceleration_z']
    
    # Use the cleaned column names for all subsequent operations
    all_feature_cols = INPUT_COLS_CLEAN + STATE_COLS_CLEAN + ACCEL_COLS_CLEAN
    df.dropna(subset=all_feature_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    X_list, y_list = [], []
    
    for traj_id, group in df.groupby('trajectory'):
        if len(group) < 2:
            continue
            
        # Use the cleaned names to select data
        current_data = group[INPUT_COLS_CLEAN + STATE_COLS_CLEAN + ACCEL_COLS_CLEAN].iloc[:-1]
        next_state = group[STATE_COLS_CLEAN].iloc[1:]
        
        X_list.append(current_data)
        y_list.append(next_state)

    if not X_list:
        raise ValueError("Not enough data to create training pairs after cleaning. Check your CSV.")

    X = pd.concat(X_list).values
    y = pd.concat(y_list).values
    
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
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
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
    
    # --- Create Data and Load ---
    try:
        X, y = load_and_prepare_data(TrainingConfig.REAL_DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {TrainingConfig.REAL_DATASET_PATH}")
        exit()
    except (KeyError, ValueError) as e:
        print(f"Error during data preparation: {e}")
        exit()

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
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)
    
    X_val_scaled = input_scaler.transform(X_val)
    y_val_scaled = output_scaler.transform(y_val)
    
    # --- Create DataLoaders ---
    train_dataset = RobotStateDataset(X_train_scaled, y_train_scaled)
    val_dataset = RobotStateDataset(X_val_scaled, y_val_scaled)

    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False)

    # --- Initialize and Train Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = StatePredictor(input_dim, output_dim)

    history = train_model(model, train_loader, val_loader, TrainingConfig.NUM_EPOCHS, TrainingConfig.LEARNING_RATE, device)

    # --- NEW: Save the trained model and the scalers ---
    print("\n--- Saving model and scalers ---")
    torch.save(model.state_dict(), TrainingConfig.MODEL_PATH)
    joblib.dump(input_scaler, TrainingConfig.INPUT_SCALER_PATH)
    joblib.dump(output_scaler, TrainingConfig.OUTPUT_SCALER_PATH)
    print(f"Model saved to {TrainingConfig.MODEL_PATH}")
    print(f"Input scaler saved to {TrainingConfig.INPUT_SCALER_PATH}")
    print(f"Output scaler saved to {TrainingConfig.OUTPUT_SCALER_PATH}")