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
from sklearn.preprocessing._data import StandardScaler
import cv2
import yaml
import time
from zaber_motion import Units
from zaber_motion.ascii import Connection
from src.nn_model import PressureNet
from src import config

# Common functions
def load_model(model_path):
    """Load a trained model and its scalers"""
    # Allow StandardScaler during unpickling
    allowed_globals = {"sklearn.preprocessing._data.StandardScaler": StandardScaler}
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    model = PressureNet(input_dim=3, output_dim=config.output_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_scaler = checkpoint["input_scaler"]
    output_scaler = checkpoint["output_scaler"]
    return model, input_scaler, output_scaler

# Training-specific functions
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

# Testing-specific functions
def move_axis(axis, position):
    """Move a motor axis to a specific position"""
    axis.move_absolute(float(position), Units.LENGTH_MILLIMETRES, False)
    time.sleep(0.1)

def load_projection_matrix(yaml_path):
    """Load camera projection matrix from a YAML file"""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    P = np.array(data["projection_matrix"], dtype=np.float64)
    return P

def detect_base(frame):
    """
    Input: frame - Image from the camera
    Output: x,y - Average coordinates of the base of the robot
    """
    # Transform the image to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Make mask for yellow color
    mask_yellow = cv2.inRange(hsv, config.lower_yellow, config.upper_yellow)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No yellow base detected.")
        return None
    
    # Choose the largest contour.
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    return cx, cy

def detect_target(frame):
    """
    Input: frame - Image from the camera
    Output: x,y - Average coordinates of the green target in the frame
    """
    # Transform the image to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Make mask for green color using config values
    mask_green = cv2.inRange(hsv, config.lower_green, config.upper_green)

    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No green target detected.")
        return None

    # Choose the largest contour.
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy

def triangulate_target(img_left, img_right, P_left_matrix, P_right_matrix, target='base'):
    """
    Input: img_left - Image from left camera
           img_right - Image from right camera
    Output: x,y,z - Coordinates of the target in 3D space
    """
    if target == 'base':
        # Get base points from the images.
        target_left = detect_base(img_left)
        target_right = detect_base(img_right)
    elif target == 'target':
        # Get target points from the images.
        target_left = detect_target(img_left)
        target_right = detect_target(img_right)

    if target_left is None or target_right is None:
        print("Couldn't detect the target in one or both images.")
        return None

    # Convert the points to a 2x1 array required by cv2.triangulatePoints.
    target_left = np.array([[target_left[0]], [target_left[1]]], dtype=np.float32)
    target_right = np.array([[target_right[0]], [target_right[1]]], dtype=np.float32)

    # Triangulate the target point.
    target_4d = cv2.triangulatePoints(P_left_matrix, P_right_matrix, target_left, target_right)

    # Convert from homogeneous coordinates to 3D.
    target_3d = target_4d[:3] / target_4d[3]

    return target_3d

def train_model(args):
    """Train the neural network model"""
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

def test_model(args):
    """Test the neural network model"""
    model, input_scaler, output_scaler = load_model(args.model_path)
    
    # Load the projection matrices
    P_left = load_projection_matrix(config.P_left_yaml)
    P_right = load_projection_matrix(config.P_right_yaml)

    # Open connection on COM3
    connection = Connection.open_serial_port('COM3')
    connection.enable_alerts()

    # connection.enableAlerts()  # (commented out as in MATLAB)
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    print(device_list)

    # Get the axis
    base_rightaxis_1 = device_list[0].get_axis(1)
    base_rightaxis_2 = device_list[1].get_axis(1)
    base_rightaxis_3 = device_list[2].get_axis(1)

    while True:
        input("Move the target and press Enter when ready to predict...")
        try:
            # Take images from both cameras
            cap_left = cv2.VideoCapture(config.cam_left_index, cv2.CAP_DSHOW)
            ret_left, frame_left = cap_left.read()
            cap_left.release()

            cap_right = cv2.VideoCapture(config.cam_right_index, cv2.CAP_DSHOW)
            ret_right, frame_right = cap_right.read()
            cap_right.release()

            if not ret_left or not ret_right:
                print("Error: Could not read frames from cameras.")
                return
            
            # Detect the target in the images
            base_3d = triangulate_target(frame_left, frame_right, P_left, P_right, target='base')
            target_3d = triangulate_target(frame_left, frame_right, P_left, P_right, target='target')
            if base_3d is None or target_3d is None:
                print("Couldn't triangulate the target.")
                return
            
            dx = target_3d[0][0] - base_3d[0][0]
            dy = target_3d[1][0] - base_3d[1][0]
            dz = target_3d[2][0] - base_3d[2][0]
            
        except ValueError:
            print("Invalid input, please enter numeric values.")
            return
        
        input_array = np.array([[dx, dy, dz]])
        input_scaled = input_scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(input_tensor)
        
        pred_np = pred.cpu().numpy()
        # Inverse transform predictions to original scale.
        pred_unscaled = output_scaler.inverse_transform(pred_np)
        pred_unscaled = pred_unscaled.flatten()
        
        if config.pressure_only:
            print("\nPredicted Pressure Values:")
            print(f"Pressure 1: {pred_unscaled[0]:.3f}")
            print(f"Pressure 2: {pred_unscaled[1]:.3f}")
            print(f"Pressure 3: {pred_unscaled[2]:.3f}")
        else:
            print("\nPredicted Values:")
            print(f"Volume 1: {pred_unscaled[0]:.3f}")
            print(f"Volume 2: {pred_unscaled[1]:.3f}")
            print(f"Volume 3: {pred_unscaled[2]:.3f}")
            print(f"Pressure 1: {pred_unscaled[3]:.3f}")
            print(f"Pressure 2: {pred_unscaled[4]:.3f}")
            print(f"Pressure 3: {pred_unscaled[5]:.3f}")

        # Move the axis to the predicted position
        move_axis(base_rightaxis_1, pred_unscaled[0])
        move_axis(base_rightaxis_2, pred_unscaled[1])
        move_axis(base_rightaxis_3, pred_unscaled[2])

def main():
    """Main function to handle command line arguments and execute appropriate mode"""
    parser = argparse.ArgumentParser(description="Train or test neural network for pressure prediction")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    
    # Add arguments for both modes
    parser.add_argument("--experiment_folder", type=str, help="Path to data folder (for train mode)")
    parser.add_argument("--model_path", type=str, help="Path to the saved model checkpoint (required for test mode)")
    
    # Training-specific arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "train":
        if not args.experiment_folder:
            parser.error("--experiment_folder is required for train mode")
        train_model(args)
    else:  # args.mode == "test"
        if not args.model_path:
            parser.error("--model_path is required for test mode")
        test_model(args)

if __name__ == "__main__":
    main()