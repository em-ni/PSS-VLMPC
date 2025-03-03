#!/usr/bin/env python
import argparse
import torch
import gpytorch
import numpy as np
from sklearn.preprocessing._data import StandardScaler

# Define the same multitask GP model as in training
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def load_model(model_path):
    """
    Loads the saved model, likelihood, input scaler, and (optionally) the output scaler.
    We force a full checkpoint load (weights_only=False) so that the saved scalers are loaded.
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    input_scaler = checkpoint['scaler']
    # Load the output scaler if it was saved; otherwise, outputs will be unscaled.
    output_scaler = checkpoint.get('output_scaler', None)
    num_tasks = 6  # for our problem: 3 volumes and 3 pressures

    # Create dummy training data (values don't matter here)
    dummy_train_x = torch.zeros(1, 3)
    dummy_train_y = torch.zeros(1, num_tasks)
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(dummy_train_x, dummy_train_y, likelihood, num_tasks=num_tasks)
    
    # Load saved state dictionaries
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    
    model.eval()
    likelihood.eval()
    return model, likelihood, input_scaler, output_scaler

def main():
    parser = argparse.ArgumentParser(
        description="Predict pressures and volumes using a trained GPyTorch model"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model file (e.g. gpytorch_model.pth)")
    args = parser.parse_args()
    
    # Load the model, likelihood, and scalers
    model, likelihood, input_scaler, output_scaler = load_model(args.model_path)
    
    # Ask the user for desired coordinates (difference between tip and base)
    print("Enter desired coordinates (difference between tip and base):")
    try:
        dx = float(input("dx: "))
        dy = float(input("dy: "))
        dz = float(input("dz: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    
    # Scale the input using the loaded input scaler
    input_array = np.array([[dx, dy, dz]])
    input_scaled = input_scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Predict using the trained model
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(input_tensor))
        pred_means = prediction.mean  # shape: (1, num_tasks)
    
    pred_means_np = pred_means.cpu().numpy()  # shape: (1,6)
    
    # If an output scaler was saved, inverse transform the predictions
    if output_scaler is not None:
        pred_unscaled = output_scaler.inverse_transform(pred_means_np)
    else:
        pred_unscaled = pred_means_np
    
    pred_unscaled = pred_unscaled.flatten()
    
    # Display the predictions
    print("\nPredicted Values:")
    print(f"Volume 1:   {pred_unscaled[0]:.3f}")
    print(f"Volume 2:   {pred_unscaled[1]:.3f}")
    print(f"Volume 3:   {pred_unscaled[2]:.3f}")
    print(f"Pressure 1: {pred_unscaled[3]:.3f}")
    print(f"Pressure 2: {pred_unscaled[4]:.3f}")
    print(f"Pressure 3: {pred_unscaled[5]:.3f}")

if __name__ == "__main__":
    main()


    """
    tip : 8.227515,4.108507,-9.315143
    base : 8.092597,3.4163816,-7.492409
    difference : 0.134918  ,  0.6921253  ,  -1.8227334

    """