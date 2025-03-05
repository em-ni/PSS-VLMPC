#!/usr/bin/env python
import argparse
import torch
import gpytorch
import numpy as np
from sklearn.preprocessing._data import StandardScaler
import torch.nn as nn

pressure_only = True
if pressure_only:
    output_dim = 3
else:
    output_dim = 6

# Define the same network architecture as in training.
class PressureNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(PressureNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def load_model(model_path):
    # Allow StandardScaler during unpickling.
    allowed_globals = {"sklearn.preprocessing._data.StandardScaler": StandardScaler}
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    model = PressureNet(input_dim=3, output_dim=output_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_scaler = checkpoint["input_scaler"]
    output_scaler = checkpoint["output_scaler"]
    return model, input_scaler, output_scaler

def main():
    parser = argparse.ArgumentParser(
        description="Test NN for pressure prediction at a single point"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g. nn_model.pth)")
    args = parser.parse_args()
    
    model, input_scaler, output_scaler = load_model(args.model_path)
    
    # Ask user for desired coordinates (difference between tip and base)
    # print("Enter desired coordinates (difference between tip and base):")
    try:
        # dx = float(input("dx: "))
        # dy = float(input("dy: "))
        # dz = float(input("dz: "))
        # For testing purposes, here we use hard-coded values:
        dx = -0.6989994
        dy = 0.028108835
        dz = -1.4100609
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
    
    print("\nPredicted Pressure Values:")
    print(f"Pressure 1: {pred_unscaled[0]:.3f}")
    print(f"Pressure 2: {pred_unscaled[1]:.3f}")
    print(f"Pressure 3: {pred_unscaled[2]:.3f}")

if __name__ == "__main__":
    main()
