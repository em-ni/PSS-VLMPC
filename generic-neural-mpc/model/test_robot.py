import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from train_robot import StatePredictor, load_and_prepare_data
from config import TrainingConfig

def plot_predictions(y_true, y_pred, save_path):
    """Generates Predicted vs. Actual plots and saves the figure to a file."""
    state_labels = [
        'Tip Position X (cm)', 'Tip Position Y (cm)', 'Tip Position Z (cm)',
        'Tip Velocity X (cm/s)', 'Tip Velocity Y (cm/s)', 'Tip Velocity Z (cm/s)'
    ]
    
    num_states = y_true.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i in range(num_states):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel("Actual Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.set_title(state_labels[i], fontsize=14)
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout(pad=3.0)
    plt.suptitle("Predicted vs. Actual State Values on Test Set", fontsize=20, y=1.02)
    
    # --- MODIFIED LINE ---
    # Instead of showing the plot, save it to a file.
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")
    # plt.show() # We no longer need this line


if __name__ == "__main__":
    # --- 1. Load data and scalers ---
    # ... (rest of the script is identical)
    print("--- Loading data, model, and scalers ---")
    try:
        X, y = load_and_prepare_data(TrainingConfig.REAL_DATASET_PATH)
        input_scaler = joblib.load(TrainingConfig.INPUT_SCALER_PATH)
        output_scaler = joblib.load(TrainingConfig.OUTPUT_SCALER_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file: {e.filename}")
        print("Please run the training script first to generate the model and scaler files.")
        exit()

    # --- 2. Recreate the exact same test split ---
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TrainingConfig.TEST_SIZE, random_state=42
    )

    # --- 3. Load the trained model ---
    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]
    model = StatePredictor(input_dim, output_dim)
    model.load_state_dict(torch.load(TrainingConfig.MODEL_PATH))
    model.eval() 
    print("Model loaded successfully.")

    # --- 4. Prepare test data and make predictions ---
    print("\n--- Making predictions on the test set ---")
    X_test_scaled = input_scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).numpy()

    predictions = output_scaler.inverse_transform(predictions_scaled)

    # --- 5. Evaluate the model ---
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Performance Metrics on Test Set ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("(R² score of 1.0 is perfect, 0.0 means the model is no better than predicting the mean)")

    # --- 6. Plot the results ---
    print("\n--- Generating plots ---")
    plot_predictions(y_test, predictions, TrainingConfig.PLOT_OUTPUT_PATH)