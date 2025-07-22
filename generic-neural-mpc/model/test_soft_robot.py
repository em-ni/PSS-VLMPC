import os
import sys
import torch
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConfig, TrainingConfig
from model.train_soft_robot import NeuralNetwork

def load_real_robot_data(csv_path):
    df = pd.read_csv(csv_path)
    x_cols = ['tip_x (cm)', 'tip_y (cm)', 'tip_z (cm)', 'tip_velocity_x (cm/s)', 'tip_velocity_y (cm/s)', 'tip_velocity_z (cm/s)']
    u_cols = ['volume_1 (mm)', 'volume_2 (mm)', 'volume_3 (mm)']
    x_dot_cols = ['tip_velocity_x (cm/s)', 'tip_velocity_y (cm/s)', 'tip_velocity_z (cm/s)',
                 'tip_acceleration_x (cm/ss)', 'tip_acceleration_y (cm/ss)', 'tip_acceleration_z (cm/ss)']
    df = df.dropna(subset=x_cols + u_cols + x_dot_cols)
    x = torch.tensor(df[x_cols].values, dtype=torch.float32)
    u = torch.tensor(df[u_cols].values, dtype=torch.float32)
    x_dot = torch.tensor(df[x_dot_cols].values, dtype=torch.float32)
    xu = torch.cat((x, u), dim=1)
    return xu, x_dot

def test_model():
    print(f"Loading model from {TrainingConfig.MODEL_SAVE_PATH}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    payload = torch.load(TrainingConfig.MODEL_SAVE_PATH, map_location=device)
    model = NeuralNetwork().to(device)
    model.load_state_dict(payload['state_dict'])
    model.eval()
    input_mean = payload['input_mean'].to(device)
    input_std = payload['input_std'].to(device)
    output_mean = payload['output_mean'].to(device)
    output_std = payload['output_std'].to(device)
    print("Model loaded successfully.")
    # Load test data
    csv_path = TrainingConfig.REAL_DATASET_PATH  # Add this to your config
    xu, x_dot_true = load_real_robot_data(csv_path)
    xu = xu.to(device)
    x_dot_true = x_dot_true.to(device)
    # Test on a random sample
    idx = np.random.randint(0, xu.shape[0])
    sample_xu = xu[idx:idx+1]
    sample_true_x_dot = x_dot_true[idx:idx+1]
    with torch.no_grad():
        norm_input = (sample_xu - input_mean) / input_std
        norm_pred_x_dot = model(norm_input)
        pred_x_dot = (norm_pred_x_dot * output_std) + output_mean
    print("\n--- Test Sample ---")
    print(f"State (x): {sample_xu[0, :SystemConfig.STATE_DIM].cpu().numpy()}")
    print(f"Input (u): {sample_xu[0, SystemConfig.STATE_DIM:].cpu().numpy()}")
    print(f"True x_dot:      {sample_true_x_dot.cpu().numpy()[0]}")
    print(f"Predicted x_dot: {pred_x_dot.cpu().numpy()[0]}")

if __name__ == '__main__':
    test_model()