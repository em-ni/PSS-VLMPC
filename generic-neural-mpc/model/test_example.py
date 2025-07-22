import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConfig, TrainingConfig
from model.train_example import NeuralNetwork, true_system_dynamics_dt

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

    # --- GENERIC TEST SAMPLE CREATION ---
    # Create a random test sample with the correct dimensions from config
    x_test = torch.rand(1, SystemConfig.STATE_DIM, dtype=torch.float32)
    u_test = torch.rand(1, SystemConfig.CONTROL_DIM, dtype=torch.float32)
    xu_test = torch.cat((x_test, u_test), dim=1).to(device)
    
    # Get true dynamics
    true_x_dot = true_system_dynamics_dt(x_test, u_test)

    # Get model prediction
    with torch.no_grad():
        norm_input = (xu_test - input_mean) / input_std
        norm_pred_x_dot = model(norm_input)
        pred_x_dot = (norm_pred_x_dot * output_std) + output_mean

    print("\n--- Test Sample ---")
    print(f"State (x): {x_test.numpy()[0]}")
    print(f"Control (u): {u_test.numpy()[0]}")
    print(f"True x_dot:      {true_x_dot.numpy()[0]}")
    print(f"Predicted x_dot: {pred_x_dot.cpu().numpy()[0]}")

if __name__ == '__main__':
    test_model()