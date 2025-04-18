import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from src.nn_model import VolumeNet
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test a neural network model")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Run in training or testing mode")
    parser.add_argument("--experiment_folder", type=str, 
                        help="Path to the experiment folder containing output CSV")
    parser.add_argument("--model_path", type=str, default="models/volume_net.pth",
                        help="Path to save or load the model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--early_stopping", type=int, default=50,
                        help="Early stopping patience")
    parser.add_argument("--flip_io", action="store_true",
                        help="Flip inputs and outputs (predict volumes from coordinates)")
    return parser.parse_args()

def load_and_preprocess_data(csv_path, flip_io=False):
    """Load the CSV data and preprocess it for the neural network"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract volumes
    volumes = df[['volume_1', 'volume_2', 'volume_3']].values
    
    # Calculate tip-base difference
    df['delta_x'] = df['tip_x'] - df['base_x']
    df['delta_y'] = df['tip_y'] - df['base_y']
    df['delta_z'] = df['tip_z'] - df['base_z']
    deltas = df[['delta_x', 'delta_y', 'delta_z']].values
    
    # Create scalers for both data types
    scaler_volumes = MinMaxScaler(feature_range=(0, 1))
    volumes_scaled = scaler_volumes.fit_transform(volumes)
    
    scaler_deltas = MinMaxScaler(feature_range=(-1, 1))
    deltas_scaled = scaler_deltas.fit_transform(deltas)
    
    # Flip inputs and outputs if requested
    if flip_io:
        print("Flipping inputs and outputs: Training model to predict volumes from coordinates")
        return deltas_scaled, volumes_scaled, scaler_deltas, scaler_volumes
    else:
        print("Normal mode: Training model to predict coordinates from volumes")
        return volumes_scaled, deltas_scaled, scaler_volumes, scaler_deltas

def create_datasets(volumes, deltas, val_split=0.2):
    """Create PyTorch datasets and dataloaders"""
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(volumes, dtype=torch.float32)
    y_tensor = torch.tensor(deltas, dtype=torch.float32)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset

def train_model(args):
    """Train the neural network model"""
    print("Training neural network...")
    
    # Find CSV file in experiment folder
    if args.experiment_folder:
        csv_files = [f for f in os.listdir(args.experiment_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in {args.experiment_folder}")
            return
        csv_path = os.path.join(args.experiment_folder, csv_files[0])
    else:
        print("Error: Please provide an experiment folder with --experiment_folder")
        return
    
    # Load and preprocess data
    volumes, deltas, scaler_volumes, scaler_deltas = load_and_preprocess_data(
        csv_path, flip_io=args.flip_io)
    
    # Adjust model input/output dimensions if flipped
    input_dim = 3  # Always 3 (either volumes or deltas, both are 3D)
    output_dim = 3  # Always 3 (either deltas or volumes, both are 3D)
    
    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(volumes, deltas, args.val_split)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model, loss, optimizer
    model = VolumeNet(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # For plotting
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= args.early_stopping:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Save model and scalers with appropriate naming
    model_filename = "inverse_volume_net.pth" if args.flip_io else "volume_net.pth"
    model_path = os.path.join(args.experiment_folder, model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save scalers - now saving to experiment folder instead of parent directory
    scalers_filename = "inverse_volume_net_scalers.npz" if args.flip_io else "volume_net_scalers.npz"
    scalers_path = os.path.join(args.experiment_folder, scalers_filename)
    np.savez(
        scalers_path,
        volumes_min=scaler_volumes.min_,
        volumes_scale=scaler_volumes.scale_,
        deltas_min=scaler_deltas.min_,
        deltas_scale=scaler_deltas.scale_
    )
    print(f"Scalers saved to {scalers_path}")
    
    # Save training history - now saving to experiment folder
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    history_path = os.path.join(args.experiment_folder, "volume_net_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    # Plot results and save to experiment folder
    plot_training_results(train_losses, val_losses, args.experiment_folder)
    
    # Evaluate on validation set
    evaluate_model(model, val_loader, scaler_volumes, scaler_deltas, args.experiment_folder)

def evaluate_model(model, dataloader, scaler_volumes, scaler_deltas, save_dir):
    """Evaluate the model and visualize predictions"""
    model.eval()
    
    # Get all validation data
    all_inputs = []
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            preds = model(inputs)
            
            all_inputs.append(inputs.numpy())
            all_targets.append(targets.numpy())
            all_preds.append(preds.numpy())
    
    # Concatenate batches
    all_inputs = np.concatenate(all_inputs)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    
    # Inverse transform to original scale
    inputs_orig = scaler_volumes.inverse_transform(all_inputs)
    targets_orig = scaler_deltas.inverse_transform(all_targets)
    preds_orig = scaler_deltas.inverse_transform(all_preds)
    
    # Calculate errors
    mse = np.mean((preds_orig - targets_orig)**2)
    mae = np.mean(np.abs(preds_orig - targets_orig))
    rmse = np.sqrt(mse)
    
    print("\nValidation Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Plot actual vs predicted for each dimension
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dimensions = ['X', 'Y', 'Z']
    
    for i, ax in enumerate(axes):
        ax.scatter(targets_orig[:, i], preds_orig[:, i], alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(targets_orig[:, i].min(), preds_orig[:, i].min())
        max_val = max(targets_orig[:, i].max(), preds_orig[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel(f'Actual Delta {dimensions[i]}')
        ax.set_ylabel(f'Predicted Delta {dimensions[i]}')
        ax.set_title(f'Delta {dimensions[i]} Prediction')
        ax.grid(True)
    
    # Save plot to experiment folder
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "validation_predictions.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Validation predictions saved to {plot_path}")

def plot_training_results(train_losses, val_losses, save_dir):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot to experiment folder
    plot_path = os.path.join(save_dir, "training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Training plot saved to {plot_path}")

def test_model(args):
    """Test the trained model on new data"""
    print("Testing neural network...")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Look for scalers in the same directory as the model
    model_dir = os.path.dirname(args.model_path)
    scalers_filename = "inverse_volume_net_scalers.npz" if "inverse" in args.model_path else "volume_net_scalers.npz"
    scalers_path = os.path.join(model_dir, scalers_filename)
    
    if not os.path.exists(scalers_path):
        print(f"Error: Scalers not found at {scalers_path}")
        return
    
    scalers = np.load(scalers_path)
    
    # Recreate scalers
    scaler_volumes = MinMaxScaler()
    scaler_volumes.min_ = scalers['volumes_min']
    scaler_volumes.scale_ = scalers['volumes_scale']
    
    scaler_deltas = MinMaxScaler()
    scaler_deltas.min_ = scalers['deltas_min']
    scaler_deltas.scale_ = scalers['deltas_scale']
    
    # Load model
    model = VolumeNet(input_dim=3, output_dim=3)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Find CSV file in experiment folder for testing
    if args.experiment_folder:
        csv_files = [f for f in os.listdir(args.experiment_folder) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in {args.experiment_folder}")
            return
        csv_path = os.path.join(args.experiment_folder, csv_files[0])
    else:
        print("Error: Please provide an experiment folder with --experiment_folder")
        return
    
    # Load test data
    df = pd.read_csv(csv_path)
    volumes = df[['volume_1', 'volume_2', 'volume_3']].values
    
    # Calculate actual tip-base differences
    df['delta_x'] = df['tip_x'] - df['base_x']
    df['delta_y'] = df['tip_y'] - df['base_y']
    df['delta_z'] = df['tip_z'] - df['base_z']
    actual_deltas = df[['delta_x', 'delta_y', 'delta_z']].values
    
    # Scale inputs
    volumes_scaled = scaler_volumes.transform(volumes)
    
    # Convert to tensor
    volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions_scaled = model(volumes_tensor).numpy()
    
    # Inverse transform predictions
    predictions = scaler_deltas.inverse_transform(predictions_scaled)
    
    # Calculate errors
    mse = np.mean((predictions - actual_deltas)**2)
    mae = np.mean(np.abs(predictions - actual_deltas))
    rmse = np.sqrt(mse)
    
    print("\nTest Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # Plot predictions vs actual and save to experiment folder
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dimensions = ['X', 'Y', 'Z']
    
    for i, ax in enumerate(axes):
        ax.plot(actual_deltas[:, i], label='Actual', alpha=0.7)
        ax.plot(predictions[:, i], label='Predicted', alpha=0.7)
        ax.set_title(f'Delta {dimensions[i]} Prediction')
        ax.set_xlabel('Sample')
        ax.set_ylabel(f'Delta {dimensions[i]}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    test_plot_path = os.path.join(model_dir, "test_predictions.png")
    plt.savefig(test_plot_path)
    print(f"Test predictions plot saved to {test_plot_path}")
    plt.show()

    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions, columns=['pred_delta_x', 'pred_delta_y', 'pred_delta_z'])
    predictions_df.to_csv(os.path.join(model_dir, "predictions.csv"), index=False)
    print(f"Predictions saved to {os.path.join(model_dir, 'predictions.csv')}")

def main():
    args = parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)

if __name__ == "__main__":
    main()