#!/usr/bin/env python
import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import gpytorch
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define a multitask GP model using GPyTorch
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.num_tasks = num_tasks
        # Use a multitask mean module; here we use a constant mean for each task.
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        # Use a multitask kernel. We use an RBF kernel shared across tasks with a rank-1 coregionalization.
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def load_data(csv_path):
    """
    Load the CSV file, drop rows with missing values, and compute the differences between tip and base coordinates.
    
    The CSV is expected to have the following columns:
    timestamp, volume_1, volume_2, volume_3, pressure_1, pressure_2, pressure_3,
    img_left, img_right, tip_x, tip_y, tip_z, base_x, base_y, base_z.
    """
    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    df.dropna(inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing values from the dataset.")
    
    # Compute differences (tip - base)
    df['dx'] = df['tip_x'] - df['base_x']
    df['dy'] = df['tip_y'] - df['base_y']
    df['dz'] = df['tip_z'] - df['base_z']
    
    # Input features: [dx, dy, dz]
    X = df[['dx', 'dy', 'dz']].values
    # Outputs: volumes and pressures (6 outputs)
    y = df[['volume_1', 'volume_2', 'volume_3', 'pressure_1', 'pressure_2', 'pressure_3']].values
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train multi-task GP model using GPyTorch for Soft Continuum Robot data")
    parser.add_argument("--experiment_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to use as test set")
    parser.add_argument("--training_iterations", type=int, default=100, help="Number of training iterations")
    args = parser.parse_args()
    
    # Look for CSV file(s) starting with 'output' in the experiment folder
    csv_files = glob.glob(os.path.join(args.experiment_folder, "output*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file starting with 'output' found in the experiment folder.")
    csv_path = csv_files[0]
    
    # Load data and split into train/test sets
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    
    # Scale inputs using StandardScaler for better numerical stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert data to torch tensors
    train_x = torch.tensor(X_train_scaled, dtype=torch.float32)
    # For multitask GP in GPyTorch, training labels should be of shape (n_samples, num_tasks)
    train_y = torch.tensor(y_train, dtype=torch.float32)
    test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.float32)
    
    num_tasks = train_y.shape[1]  # 6 outputs
    
    # Use a multitask Gaussian likelihood
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    # Initialize our model
    model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=num_tasks)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    loss_history = []
    print("Training the multitask GP model using GPyTorch...")
    for i in tqdm(range(args.training_iterations), desc="Training Iterations"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        loss_history.append(loss.item())
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{args.training_iterations} - Loss: {loss.item():.3f}")
        optimizer.step()
    
    # Plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Negative Marginal Log Likelihood")
    plt.title("Training Loss")
    loss_plot_path = os.path.join(args.experiment_folder, "training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training loss plot saved to {loss_plot_path}")
    
    # Switch to evaluation mode
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(test_x))
        pred_means = predictions.mean  # shape: (n_test, num_tasks)
        mse = torch.mean((pred_means - test_y) ** 2)
        print(f"Test Mean Squared Error: {mse.item():.4f}")
    
    # Convert predictions and test labels to numpy arrays for plotting
    pred_means_np = pred_means.cpu().numpy()
    test_y_np = test_y.cpu().numpy()
    
    # Create scatter plots for predicted vs. actual values for each output
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()
    task_names = ["Volume 1", "Volume 2", "Volume 3", "Pressure 1", "Pressure 2", "Pressure 3"]
    for i in range(num_tasks):
        axs[i].scatter(test_y_np[:, i], pred_means_np[:, i], alpha=0.7)
        # Compute combined min and max for the axis limits
        min_val = min(test_y_np[:, i].min(), pred_means_np[:, i].min())
        max_val = max(test_y_np[:, i].max(), pred_means_np[:, i].max())
        axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        axs[i].set_xlim([min_val, max_val])
        axs[i].set_ylim([min_val, max_val])
        axs[i].set_xlabel("Actual")
        axs[i].set_ylabel("Predicted")
        axs[i].set_title(task_names[i])
    plt.suptitle("Predicted vs Actual Values on Test Set")
    scatter_plot_path = os.path.join(args.experiment_folder, "predicted_vs_actual.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Predicted vs. Actual plot saved to {scatter_plot_path}")
    
    # Save the trained model state and scaler for later use
    model_out = os.path.join(args.experiment_folder, "gpytorch_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "likelihood_state_dict": likelihood.state_dict(),
        "scaler": scaler,
    }, model_out)
    print(f"Trained model and scaler saved to {model_out}")

if __name__ == "__main__":
    main()
