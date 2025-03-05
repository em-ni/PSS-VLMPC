#!/usr/bin/env python
import argparse
import torch
import gpytorch
import numpy as np
from gpytorch.kernels import MaternKernel, LinearKernel, ScaleKernel, MultitaskKernel
from sklearn.preprocessing._data import StandardScaler

pressure_only = True
if pressure_only:
    num_tasks = 3
else:
    num_tasks = 6

# Define the same multitask GP model as in training.
# This version uses a composite kernel: ScaleKernel(MaternKernel(nu=3/2) + LinearKernel())
# and sets the multitask kernel rank to 2.
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # Use a composite base kernel (Matern + Linear) wrapped in a ScaleKernel.
        base_kernel = ScaleKernel(MaternKernel(nu=3/2) + LinearKernel())
        # Use a multitask kernel with rank 2 (matching training)
        self.covar_module = MultitaskKernel(base_kernel, num_tasks=num_tasks, rank=2)
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def load_model(model_path):
    """
    Loads the saved model, likelihood, input scaler, and output scaler.
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    input_scaler = checkpoint['input_scaler']
    output_scaler = checkpoint['output_scaler']
    
    # Create dummy training data (the values here don't matter)
    dummy_train_x = torch.zeros(1, 3)
    dummy_train_y = torch.zeros(1, num_tasks)
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MultitaskGPModel(dummy_train_x, dummy_train_y, likelihood, num_tasks=num_tasks)
    
    # Load saved state dictionaries (this will now match the composite kernel architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    
    model.eval()
    likelihood.eval()
    return model, likelihood, input_scaler, output_scaler

def main():
    parser = argparse.ArgumentParser(
        description="Predict pressures (or volumes+pressures) using a trained GPyTorch model"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model file (e.g. gpytorch_model.pth)")
    args = parser.parse_args()
    
    # Load the model, likelihood, and scalers.
    model, likelihood, input_scaler, output_scaler = load_model(args.model_path)
    
    # Ask the user for desired coordinates (difference between tip and base).
    print("Enter desired coordinates (difference between tip and base):")
    try:
        # For testing, here we use hard-coded values:
        dx = -0.6989994
        dy = 0.028108835
        dz = -1.4100609
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return
    
    # Scale the input using the loaded input scaler.
    input_array = np.array([[dx, dy, dz]])
    input_scaled = input_scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Predict using the trained model.
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(input_tensor))
        pred_means = prediction.mean  # shape: (1, num_tasks)
    
    pred_means_np = pred_means.cpu().numpy()  # shape: (1, num_tasks)
    
    # Inverse transform predictions to the original scale.
    pred_unscaled = output_scaler.inverse_transform(pred_means_np)
    pred_unscaled = pred_unscaled.flatten()
    
    # Display the predictions.
    print("\nPredicted Values:")
    if pressure_only:
        print(f"Pressure 1: {pred_unscaled[0]:.3f}")
        print(f"Pressure 2: {pred_unscaled[1]:.3f}")
        print(f"Pressure 3: {pred_unscaled[2]:.3f}")
    else:
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
    expected output: 114,114,114,  6.078997020770636,7.392328736296217,6.958050216976569,

    tip: 7.3871756, 3.4346023, -8.884726, 
    base: 8.085175, 3.4064934, -7.474665
    difference : -0.6989994  ,  0.028108835  ,  -1.4100609
    expected output: 116.4,115.6,117.73333333333333,  6.147222044953783,8.31467858246453,10.104273447576311

    """