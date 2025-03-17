#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.pressure_net import PressureNet
from src import config

def load_data(csv_path):
    """
    Load the CSV file, drop rows with missing values, and compute differences (tip - base)
    Expected CSV columns:
      timestamp, volume_1, volume_2, volume_3, pressure_1, pressure_2, pressure_3,
      img_left, img_right, tip_x, tip_y, tip_z, base_x, base_y, base_z.
    """
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    # Compute differences
    df['dx'] = df['tip_x'] - df['base_x']
    df['dy'] = df['tip_y'] - df['base_y']
    df['dz'] = df['tip_z'] - df['base_z']
    X = df[['dx', 'dy', 'dz']].values
    if config.pressure_only:
        y = df[['pressure_1', 'pressure_2', 'pressure_3']].values
    else:
        y = df[['volume_1', 'volume_2', 'volume_3', 'pressure_1', 'pressure_2', 'pressure_3']].values
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train NN for pressure prediction")
    parser.add_argument("--experiment_folder", type=str, required=True, help="Path to data folder")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    # Find CSV file(s) starting with 'output' in the experiment folder.
    csv_files = glob.glob(os.path.join(args.experiment_folder, "output*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file starting with 'output' found.")
    csv_path = csv_files[0]
    
    # Load and split data.
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Scale inputs and outputs.
    input_scaler = StandardScaler()
    X_train_scaled = input_scaler.fit_transform(X_train)
    X_test_scaled = input_scaler.transform(X_test)
    
    output_scaler = StandardScaler()
    y_train_scaled = output_scaler.fit_transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)
    
    # Convert to torch tensors.
    train_X = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train_scaled, dtype=torch.float32)
    test_X = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_y = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    # Create DataLoader.
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model, loss, optimizer.
    model = PressureNet(input_dim=3, output_dim=config.output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    loss_history = []
    print("Training neural network...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(train_dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{args.epochs}: Loss = {epoch_loss:.4f}")
    
    # Plot training loss.
    plt.figure(figsize=(8,5))
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.savefig(os.path.join(args.experiment_folder, "training_loss_nn.png"))
    plt.close()


    # Evaluate on test set for creating predicted vs actual scatter plot.
    model.eval()
    with torch.no_grad():
        preds = model(test_X)
    preds_np = preds.cpu().numpy()
    # Inverse-transform predictions and true labels to original scale.
    preds_unscaled = output_scaler.inverse_transform(preds_np)
    test_y_unscaled = output_scaler.inverse_transform(test_y.cpu().numpy())

    # Create scatter plots for each output dimension.
    if config.output_dim == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, config.output_dim, figsize=(5 * config.output_dim, 5))
    task_names = ["Pressure 1", "Pressure 2", "Pressure 3"] if config.pressure_only \
                 else ["Volume 1", "Volume 2", "Volume 3", "Pressure 1", "Pressure 2", "Pressure 3"]
    for i in range(config.output_dim):
        axes[i].scatter(test_y_unscaled[:, i], preds_unscaled[:, i], alpha=0.7)
        min_val = min(test_y_unscaled[:, i].min(), preds_unscaled[:, i].min())
        max_val = max(test_y_unscaled[:, i].max(), preds_unscaled[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[i].set_xlim([min_val, max_val])
        axes[i].set_ylim([min_val, max_val])
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(task_names[i])
    plt.suptitle("Predicted vs Actual Values on Test Set")
    scatter_plot_path = os.path.join(args.experiment_folder, "predicted_vs_actual_nn.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Predicted vs Actual plot saved to {scatter_plot_path}")
    
    with torch.no_grad():
        test_preds = model(test_X)
        test_loss = criterion(test_preds, test_y).item()
    test_preds_np = test_preds.cpu().numpy()
    # Inverse transform predictions and ground truth.
    test_preds_unscaled = output_scaler.inverse_transform(test_preds_np)
    test_y_unscaled = output_scaler.inverse_transform(test_y.cpu().numpy())
    mse_unscaled = np.mean((test_preds_unscaled - test_y_unscaled) ** 2)
    print(f"Test MSE (unscaled): {mse_unscaled:.4f}")
    
    # Save model and scalers.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
    }
    save_path = os.path.join(args.experiment_folder, "nn_model.pth")
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
