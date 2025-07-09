import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import copy 
import os
import argparse
from src.lstm_model import LSTMModel
import src.config as config

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM model training')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train',
                       help='Mode to run the script (only "train" supported for now)')
    parser.add_argument('--data', type=str, required=False,
                       help='Path to the CSV data file for training')
    parser.add_argument('--model', type=str, required=False,
                       help='Path to the model file for testing')
    parser.add_argument('--test_input', type=str, required=False,
                       help='Path to the input file for testing')
    return parser.parse_args()

def train_model(args):
    # Load data
    df = pd.read_csv(args.data)
    
    # Compute x as the difference between tip and base coordinates
    # (this gives the 3d position error vector for the end-effector)
    df['x_tip'] = df['tip_x'] - df['base_x']
    df['y_tip'] = df['tip_y'] - df['base_y']
    df['z_tip'] = df['tip_z'] - df['base_z']
    
    # Extract x (3d position) and tau (3d actuation)
    # Note: according to the dataset the same row's volumes correspond to τ_t,
    # and the coordinates are x_t+1.
    x = df[['x_tip', 'y_tip', 'z_tip']].values  # shape (N, 3)
    tau = df[['volume_1', 'volume_2', 'volume_3']].values  # shape (N, 3)
    
    # We will build sequences with horizon T = 3.
    # The mapping is: (τ_t, τ_{t-1}, τ_{t-2}, τ_{t-3}, x_t, x_{t-1}, x_{t-2}, x_{t-3}) -> x_{t+1}
    # To do so, we will construct sequences of 4 consecutive x's and tau's.
    T = 3
    X_seq = []
    tau_seq = []
    y_target = []
    
    # We start at index T (so that we have 4 x values: x[t-3] ... x[t]) and use x[t+1] as target.
    for i in range(T, len(df) - 1):
        # Sequence of x from t-3 to t (length T+1 = 4)
        seq_x = x[i - T:i + 1]  # shape (4, 3)
        # Use tau sequence from t-3 to t (same historical window as x)
        seq_tau = tau[i - T:i + 1]  # shape (4, 3)
        target = x[i + 1]  # this is x_{t+1}
        
        X_seq.append(seq_x)
        tau_seq.append(seq_tau)
        y_target.append(target)
    
    X_seq = np.array(X_seq)   # shape (samples, 4, 3)
    tau_seq = np.array(tau_seq)  # shape (samples, 4, 3)
    y_target = np.array(y_target)  # shape (samples, 3)
    
    # Scale the x values to the range [-1, 1]
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_seq_reshaped = X_seq.reshape(-1, 3)
    scaler_x.fit(X_seq_reshaped)
    X_seq_scaled = scaler_x.transform(X_seq_reshaped).reshape(X_seq.shape)
    y_target_scaled = scaler_x.transform(y_target)
    
    # Scale tau as well to [-1, 1]
    scaler_tau = MinMaxScaler(feature_range=(-1, 1))
    tau_seq_reshaped = tau_seq.reshape(-1, 3)
    scaler_tau.fit(tau_seq_reshaped)
    tau_seq_scaled = scaler_tau.transform(tau_seq_reshaped).reshape(tau_seq.shape)
    
    # Incorporate tau into the input sequence by concatenating with x at each timestep
    X_input = np.concatenate([X_seq_scaled, tau_seq_scaled], axis=2)  # shape (samples, 4, 6)
    
    # Model parameters
    input_dim = 6        # 3 for x and 3 for τ at each timestep
    hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 64
    output_dim = 3       # predicting 3d x_{t+1}
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 2
    
    # Initialize model and move to device
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create training and validation datasets
    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    y_tensor = torch.tensor(y_target_scaled, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_data, val_data = train_test_split(dataset, test_size=0.15, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    
    # Define loss, optimizer, and LR scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=50)
    
    # Training loop with early stopping (patience = 100 epochs)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 100:
            print("Early stopping triggered")
            break

    # Load best model weights and save the model
    model.load_state_dict(best_model_wts)
    model_path = os.path.join("models", "lstm_model.pth")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

def test_model(args):
    # Check that the test input and model file are provided
    if not args.test_input:
        print("Error: Test input file is required. Use --test_input to specify.")
        return
    
    if not args.model and not os.path.exists(os.path.join("models", "lstm_model.pth")):
        print("Error: Model file is required. Use --model to specify or train a model first.")
        return
    
    # Read the test input file (each line has volumes and coordinates)
    with open(args.test_input, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 5:  # Need at least 4 historical points + 1 target
        print("Error: Test file must contain at least 5 lines (4 for history and 1 for target)")
        return
    
    # Parse the input data
    input_data = []
    for line in lines:
        values = [float(val) for val in line.strip().split(',')]
        if len(values) != 6:  # 3 volumes + 3 coordinates
            print(f"Error: Each line must contain 6 values (vol1,vol2,vol3,x,y,z), got {len(values)}")
            return
        input_data.append(values)
    
    # Convert to numpy array and extract volumes and positions
    input_data = np.array(input_data)
    
    # Last line is the target (for validation if available)
    target = input_data[-1, 3:] if len(input_data) > 4 else None
    
    # Use only the first 4 lines for prediction
    history_data = input_data[:4]
    volumes = history_data[:, :3]  # First 3 columns are volumes
    positions = history_data[:, 3:]  # Last 3 columns are positions
    
    # Create the sequence similar to training
    X_seq = positions.reshape(1, 4, 3)  # shape (1, 4, 3)
    tau_seq = volumes.reshape(1, 4, 3)  # shape (1, 4, 3)
    
    # Scale the inputs using similar scalers to training
    # For positions
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    X_seq_reshaped = X_seq.reshape(-1, 3)
    scaler_x.fit(X_seq_reshaped)
    X_seq_scaled = scaler_x.transform(X_seq_reshaped).reshape(X_seq.shape)
    
    # For volumes
    scaler_tau = MinMaxScaler(feature_range=(-1, 1))
    tau_seq_reshaped = tau_seq.reshape(-1, 3)
    scaler_tau.fit(tau_seq_reshaped)
    tau_seq_scaled = scaler_tau.transform(tau_seq_reshaped).reshape(tau_seq.shape)
    
    # Combine inputs
    X_input = np.concatenate([X_seq_scaled, tau_seq_scaled], axis=2)  # shape (1, 4, 6)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_input, dtype=torch.float32)
    
    # Load the model
    model_path = args.model if args.model else os.path.join("models", "lstm_model.pth")
    input_dim = 6
    hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else 64
    output_dim = 3
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 2
    
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(X_tensor).numpy()
    
    # Convert prediction back to original scale
    prediction = scaler_x.inverse_transform(prediction_scaled)[0]
    
    print("\nModel Prediction:")
    print(f"Predicted next position: x={prediction[0]:.4f}, y={prediction[1]:.4f}, z={prediction[2]:.4f}")
    
    # Compare with target if available
    if target is not None:
        print("\nActual vs Predicted:")
        print(f"Actual: x={target[0]:.4f}, y={target[1]:.4f}, z={target[2]:.4f}")
        print(f"Predicted: x={prediction[0]:.4f}, y={prediction[1]:.4f}, z={prediction[2]:.4f}")
        
        # Calculate error
        error = np.sqrt(np.sum((prediction - target)**2))
        print(f"Euclidean error: {error:.4f}")

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
