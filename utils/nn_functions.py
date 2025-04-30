

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from src import config
from src.config import STATE_DIM, VOLUME_DIM, initial_pos
from src.nn_model import VolumeNet


def load_model_and_scalers(model_path, scalers_path):
    # print(f"Loading model from: {model_path}")
    # print(f"Loading scalers from: {scalers_path}")
    try:
        scalers_data = np.load(scalers_path)
        scaler_volumes = MinMaxScaler(); scaler_deltas = MinMaxScaler()
        required_keys = ['volumes_min', 'volumes_scale', 'deltas_min', 'deltas_scale']
        for key in required_keys:
            if key not in scalers_data:
                print(f"FATAL ERROR: Required key '{key}' not found in scaler file '{scalers_path}'. Available keys: {list(scalers_data.keys())}"); return None, None, None, None
        scaler_volumes.min_ = scalers_data['volumes_min']; scaler_volumes.scale_ = scalers_data['volumes_scale']
        scaler_deltas.min_ = scalers_data['deltas_min']; scaler_deltas.scale_ = scalers_data['deltas_scale']
        # print("Scalers loaded and initialized.")
        # print(f"  Volume Scaler Min: {scaler_volumes.min_}"); print(f"  Volume Scaler Scale: {scaler_volumes.scale_}")
        # print(f"  Delta Scaler Min: {scaler_deltas.min_}"); print(f"  Delta Scaler Scale: {scaler_deltas.scale_}")
        nn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {nn_device}")
        model = VolumeNet(input_dim=VOLUME_DIM, output_dim=STATE_DIM)
        model.load_state_dict(torch.load(model_path, map_location=nn_device))
        model = model.to(nn_device); model.eval()
        # print("NN model loaded successfully.")
        return model, scaler_volumes, scaler_deltas, nn_device
    except FileNotFoundError: print(f"FATAL ERROR: Model or Scaler file not found.\n  Model: {model_path}\n  Scalers: {scalers_path}"); return None, None, None, None
    except Exception as e: print(f"FATAL ERROR during NN loading: {e}"); import traceback; traceback.print_exc(); return None, None, None, None

def predict_tip(command, scaler_volumes, scaler_deltas, model, device):
    """
    Predict the tip position based on the command using the neural network model.
    """
    volumes = np.zeros(3, dtype=np.float32)
    volumes[0] = command[0]
    volumes[1] = command[1] 
    volumes[2] = command[2]
    
    # Reshape to 2D array (1 sample, 3 features) for scikit-learn
    volumes_2d = volumes.reshape(1, -1)
    
    # Scale inputs
    volumes_scaled = scaler_volumes.transform(volumes_2d)

    # Convert to tensor and move to the appropriate device (GPU if available)
    volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32).to(device)

    # Make predictions
    with torch.no_grad():
        # Run the model on the device (GPU if available)
        predictions_tensor = model(volumes_tensor)
        # Move results back to CPU for numpy conversion
        predictions_scaled = predictions_tensor.cpu().numpy()

    # Inverse transform predictions
    predictions = scaler_deltas.inverse_transform(predictions_scaled)
    return predictions[0]


def get_initial_state():
    """Get initial state by inferring the neural network model with zero input."""
    
    # Load the model and scalers
    model, scaler_volumes, scaler_deltas, device = load_model_and_scalers(config.MODEL_PATH, config.SCALERS_PATH)
    if model is None or scaler_volumes is None or scaler_deltas is None:
        print("Error loading model or scalers. Cannot get initial state.")
        return None
    
    # Create a zero command
    command = np.zeros(VOLUME_DIM, dtype=np.float32)
    # Predict the tip position
    initial_state = predict_tip(command, scaler_volumes, scaler_deltas, model, device)
    return initial_state
