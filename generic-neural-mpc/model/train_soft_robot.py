import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConfig, NeuralNetConfig, TrainingConfig

def load_real_robot_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='T (ms)')  # Ensure correct order by time
    x_cols = ['tip_x (cm)', 'tip_y (cm)', 'tip_z (cm)', 'tip_velocity_x (cm/s)', 'tip_velocity_y (cm/s)', 'tip_velocity_z (cm/s)']
    u_cols = ['volume_1 (mm)', 'volume_2 (mm)', 'volume_3 (mm)']
    # Also need to drop rows with missing values in x_dot columns
    x_dot_cols = ['tip_velocity_x (cm/s)', 'tip_velocity_y (cm/s)', 'tip_velocity_z (cm/s)',
                 'tip_acceleration_x (cm/ss)', 'tip_acceleration_y (cm/ss)', 'tip_acceleration_z (cm/ss)']
    df = df.dropna(subset=x_cols + u_cols + x_dot_cols)
    x = torch.tensor(df[x_cols].values, dtype=torch.float32)
    u = torch.tensor(df[u_cols].values, dtype=torch.float32)
    x_dot = torch.tensor(df[x_dot_cols].values, dtype=torch.float32)
    xu = torch.cat((x, u), dim=1)
    # Check for NaN/Inf
    print("Any NaN in inputs:", torch.isnan(xu).any().item())
    print("Any NaN in targets:", torch.isnan(x_dot).any().item())
    print("Any Inf in inputs:", torch.isinf(xu).any().item())
    print("Any Inf in targets:", torch.isinf(x_dot).any().item())
    return TensorDataset(xu, x_dot)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(NeuralNetConfig.INPUT_DIM, NeuralNetConfig.HIDDEN_SIZE))
        layers.append(getattr(nn, NeuralNetConfig.ACTIVATION)())
        for _ in range(NeuralNetConfig.HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(NeuralNetConfig.HIDDEN_SIZE, NeuralNetConfig.HIDDEN_SIZE))
            layers.append(getattr(nn, NeuralNetConfig.ACTIVATION)())
        layers.append(nn.Linear(NeuralNetConfig.HIDDEN_SIZE, NeuralNetConfig.OUTPUT_DIM))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train():
    csv_path = TrainingConfig.REAL_DATASET_PATH  # Add this to your config
    print(f"Loading real robot data from {csv_path}...")
    dataset = load_real_robot_data(csv_path)
    # Split into train/val
    num_samples = len(dataset.tensors[0])
    split = int(0.8 * num_samples)
    train_dataset = TensorDataset(dataset.tensors[0][:split], dataset.tensors[1][:split])
    val_dataset = TensorDataset(dataset.tensors[0][split:], dataset.tensors[1][split:])
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE)
    all_train_inputs = train_dataset.tensors[0]
    all_train_outputs = train_dataset.tensors[1]
    input_mean = all_train_inputs.mean(dim=0)
    input_std = all_train_inputs.std(dim=0)
    output_mean = all_train_outputs.mean(dim=0)
    output_std = all_train_outputs.std(dim=0)
    input_std[input_std == 0] = 1.0
    output_std[output_std == 0] = 1.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    criterion = nn.MSELoss()
    print(f"Starting training on {device}...")
    best_val_loss = float('inf')
    for epoch in range(TrainingConfig.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TrainingConfig.NUM_EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
            norm_targets = (targets - output_mean.to(device)) / output_std.to(device)
            optimizer.zero_grad()
            outputs = model(norm_inputs)
            loss = criterion(outputs, norm_targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
                norm_targets = (targets - output_mean.to(device)) / output_std.to(device)
                outputs = model(norm_inputs)
                val_loss = criterion(outputs, norm_targets)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_payload = {
                'state_dict': model.state_dict(),
                'input_mean': input_mean,
                'input_std': input_std,
                'output_mean': output_mean,
                'output_std': output_std,
            }
            torch.save(save_payload, TrainingConfig.MODEL_SAVE_PATH)
            print(f"Model saved to {TrainingConfig.MODEL_SAVE_PATH} with validation loss {avg_val_loss:.6f}")

if __name__ == '__main__':
    os.makedirs(os.path.dirname(TrainingConfig.MODEL_SAVE_PATH), exist_ok=True)
    train()