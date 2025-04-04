import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
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
    parser.add_argument('--mode', type=str, choices=['train'], default='train',
                       help='Mode to run the script (only "train" supported for now)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to the data file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Use the provided data path
    FILE_PATH = args.data
    
    # --- 0. Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Configuration (same as before) ---
    # FILE_PATH is now defined from command line arguments
    
    # ACTIVATION 'tanh' is the default for PyTorch LSTM
    LEARNING_RATE = 0.001
    EPOCHS = 500 # Max epochs, EarlyStopping will control
    BATCH_SIZE = 64
    VALIDATION_SPLIT_RATIO = 0.10 / (1.0 - 0.15) # 10% validation from the non-test set
    TEST_SPLIT_RATIO = 0.15
    
    # Callbacks parameters from paper
    EARLY_STOPPING_PATIENCE = 100
    LR_REDUCE_PATIENCE = 50
    LR_REDUCE_FACTOR = 0.7
    
    # --- 2. Data Loading and Preprocessing (same as before) ---
    print("Loading data...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
        print("Please provide a valid file path with -data argument.")
        exit()
    
    print("Calculating relative position 'x'...")
    df['delta_x'] = df['tip_x'] - df['base_x']
    df['delta_y'] = df['tip_y'] - df['base_y']
    df['delta_z'] = df['tip_z'] - df['base_z']

    feature_cols = ['volume_1', 'volume_2', 'volume_3', 'delta_x', 'delta_y', 'delta_z']
    data = df[feature_cols].values

    print(f"Original data shape: {data.shape}")

    # Normalize features to [-1, 1] range
    print("Normalizing data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences
    print(f"Creating sequences with length {config.sequence_length}...")
    X, y = [], []
    for i in range(len(data_scaled) - config.sequence_length):
        sequence = data_scaled[i:i + config.sequence_length]
        target = data_scaled[i + config.sequence_length, config.n_features_tau:]
        X.append(sequence)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    print(f"Shape of X (sequences): {X.shape}")
    print(f"Shape of y (targets): {y.shape}")

    # Split data
    print("Splitting data into train, validation, and test sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=VALIDATION_SPLIT_RATIO, random_state=42, shuffle=False
    )

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Maintain order
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train shapes: X={X_train_tensor.shape}, y={y_train_tensor.shape}")
    print(f"Validation shapes: X={X_val_tensor.shape}, y={y_val_tensor.shape}")
    print(f"Test shapes: X={X_test_tensor.shape}, y={y_test_tensor.shape}")

    # --- 3. Build LSTM Model (PyTorch) ---
    print("Building PyTorch LSTM model...")
    model = LSTMModel(config.total_features, config.lstm_hidden_units, config.output_dim, config.lstm_num_layers).to(device)
    print(model)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Learning Rate Scheduler (equivalent to ReduceLROnPlateau)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
        verbose=True
    )

    # --- 4. Train LSTM Model (PyTorch) ---
    print("Training model...")

    train_losses = []
    val_losses = []
    learning_rates = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for i, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * sequences.size(0) # Multiply by batch size

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad(): # Disable gradient calculations
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * sequences.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}, LR: {current_lr:.6f}')

        # Learning Rate Scheduler step
        scheduler.step(epoch_val_loss)

        # Early Stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'Validation loss improved. Saving model state at epoch {epoch+1}.')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve for {epochs_no_improve} epoch(s).')

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            # Restore best model weights
            if best_model_state:
                 model.load_state_dict(best_model_state)
                 print("Restored best model weights.")
            else:
                 print("Warning: Early stopping triggered but no best model state was saved.")
            break

    # --- 5. Evaluate Model (PyTorch) ---
    print("\nEvaluating model on test data...")
    # Ensure the best model is loaded if early stopping occurred
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval() # Set to evaluation mode
    test_loss = 0.0
    all_predictions_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * sequences.size(0)
            all_predictions_scaled.append(outputs.cpu().numpy()) # Move back to CPU for numpy
            all_targets_scaled.append(targets.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")

    y_pred_scaled = np.concatenate(all_predictions_scaled, axis=0)
    y_test_scaled = np.concatenate(all_targets_scaled, axis=0) # This is y_test_tensor as numpy


    # --- 6. Denormalize and Calculate Metrics (same as before) ---
    print("\nDenormalizing predictions and calculating metrics...")
    # Create dummy arrays for scaler inverse transform
    dummy_pred = np.zeros((len(y_pred_scaled), config.total_features))
    dummy_pred[:, config.n_features_tau:] = y_pred_scaled

    dummy_true = np.zeros((len(y_test_scaled), config.total_features))
    dummy_true[:, config.n_features_tau:] = y_test_scaled # Use the scaled test targets

    # Denormalize
    y_pred_denormalized = scaler.inverse_transform(dummy_pred)[:, config.n_features_tau:]
    y_test_denormalized = scaler.inverse_transform(dummy_true)[:, config.n_features_tau:]

    # Calculate MAE
    mae = np.mean(np.abs(y_test_denormalized - y_pred_denormalized), axis=0)
    mean_mae = np.mean(mae)
    print(f"Denormalized Test MAE (x, y, z): {mae}")
    print(f"Mean Denormalized Test MAE: {mean_mae}")

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test_denormalized - y_pred_denormalized)**2, axis=0))
    mean_rmse = np.mean(rmse)
    print(f"Denormalized Test RMSE (x, y, z): {rmse}")
    print(f"Mean Denormalized Test RMSE: {mean_rmse}")

    # --- 7. Plot Training History (adapted for PyTorch lists) ---
    print("\nPlotting training history...")
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, learning_rates, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # --- 8. Save Model and Results ---
    print("\nSaving model and results...")

    # Get the directory from the input file path
    save_dir = os.path.dirname(FILE_PATH)
    model_filename = os.path.join(save_dir, "lstm_model.pth")
    plot_filename = os.path.join(save_dir, "lstm_training_plots.png")
    metrics_filename = os.path.join(save_dir, "lstm_metrics.txt")

    # Save the model with all necessary information
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler,
        'config': {
            'config.sequence_length': config.sequence_length,
            'config.n_features_tau': config.n_features_tau,
            'config.n_features_x': config.n_features_x,
            'config.lstm_hidden_units': config.lstm_hidden_units,
            'config.lstm_num_layers': config.lstm_num_layers
        },
        'performance': {
            'test_loss': avg_test_loss,
            'mae': mae.tolist(),
            'rmse': rmse.tolist(),
            'mean_mae': mean_mae,
            'mean_rmse': mean_rmse
        }
    }, model_filename)
    print(f"Model saved to {model_filename}")

    # Save the training plots
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to {plot_filename}")

    # Save metrics to a text file
    with open(metrics_filename, 'w') as f:
        f.write(f"Test Loss (MSE): {avg_test_loss:.6f}\n")
        f.write(f"Denormalized Test MAE (x, y, z): {mae}\n")
        f.write(f"Mean Denormalized Test MAE: {mean_mae:.6f}\n")
        f.write(f"Denormalized Test RMSE (x, y, z): {rmse}\n")
        f.write(f"Mean Denormalized Test RMSE: {mean_rmse:.6f}\n")
        
        # Add training details
        f.write("\nTraining Details:\n")
        f.write(f"Epochs completed: {len(train_losses)}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Final learning rate: {learning_rates[-1]:.6f}\n")
    print(f"Metrics saved to {metrics_filename}")

if __name__ == "__main__":
    main()

