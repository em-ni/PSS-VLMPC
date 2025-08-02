"""
Last dataset 2/8/25:

Processed 750051 time steps
Time range: 0.00 - 15001.00 ms
Complete dataset saved to: results/sim_dataset.csv
Creating neural network training data...
Created 750050 training samples
Input features shape: (750050, 11)
Target outputs shape: (750050, 6)

Dataset Statistics:
==================================================
Total samples: 750050
Input dimensions: 11
Output dimensions: 6
Time step (dt): 0.02

Input ranges:
  rod1_torque_x_k: [-0.137872, 0.145008]
  rod1_torque_y_k: [-0.144178, 0.156151]
  rod2_torque_x_k: [-0.052293, 0.052114]
  rod2_torque_y_k: [-0.051869, 0.057591]
  tip_position_x_k: [-0.703178, 0.704306]
  tip_position_y_k: [-0.705243, 0.706738]
  tip_position_z_k: [-0.800292, 0.496858]
  tip_velocity_x_k: [-0.624930, 0.636474]
  tip_velocity_y_k: [-0.642704, 0.624778]
  tip_velocity_z_k: [-0.789484, 1.034373]
  dt: [0.020000, 0.020000]

Output ranges:
  tip_position_x_k+1: [-0.703178, 0.704306]
  tip_position_y_k+1: [-0.705243, 0.706738]
  tip_position_z_k+1: [-0.800292, 0.496858]
  tip_velocity_x_k+1: [-0.624930, 0.636474]
  tip_velocity_y_k+1: [-0.642704, 0.624778]
  tip_velocity_z_k+1: [-0.789484, 1.034373]
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import glob


def load_simulation_data(data_path="results"):
    """
    Load all .dat files from the results directory.
    
    Parameters:
    -----------
    data_path : str
        Path to the directory containing .dat files
        
    Returns:
    --------
    list
        List of loaded simulation data dictionaries
    """
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data folder '{data_path}' not found.")
    
    all_data_files = glob.glob(os.path.join(data_path, "*.dat"))
    if not all_data_files:
        raise FileNotFoundError(f"No .dat files found in '{data_path}'")
    
    print(f"Found {len(all_data_files)} .dat files in '{data_path}'. Loading...")
    
    all_sim_data = []
    for data_file_path in sorted(all_data_files):  # Sort to ensure consistent rod ordering
        try:
            with open(data_file_path, "rb") as fptr:
                data = pickle.load(fptr)
                all_sim_data.append(data)
                print(f"Loaded: {os.path.basename(data_file_path)}")
        except Exception as err:
            print(f"Error loading {data_file_path}: {err}")
            continue
    
    return all_sim_data


def extract_tip_data(position_collection, velocity_collection):
    # Tip is the last node (-1 index)
    tip_position = position_collection[:, -1]
    tip_velocity = velocity_collection[:, -1]
    
    return tip_position, tip_velocity


def process_simulation_data(all_sim_data):
    """
    Process simulation data and create dataset for neural network training.
    
    Parameters:
    -----------
    all_sim_data : list
        List of simulation data dictionaries for each rod
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns for neural network training
    """
    if len(all_sim_data) < 2:
        raise ValueError("Need at least 2 rods for the specified neural network input format")
    
    # Extract data from both rods
    rod1_data = all_sim_data[0]
    rod2_data = all_sim_data[1]
    
    # Get time arrays
    times_rod1 = np.array(rod1_data["time"])
    times_rod2 = np.array(rod2_data["time"])
    
    # Find common time range (intersection)
    min_time = max(times_rod1.min(), times_rod2.min())
    max_time = min(times_rod1.max(), times_rod2.max())
    
    # Filter data to common time range
    rod1_mask = (times_rod1 >= min_time) & (times_rod1 <= max_time)
    rod2_mask = (times_rod2 >= min_time) & (times_rod2 <= max_time)
    
    times_rod1_filtered = times_rod1[rod1_mask]
    times_rod2_filtered = times_rod2[rod2_mask]
    
    # Check if we have synchronized data
    if not np.allclose(times_rod1_filtered, times_rod2_filtered, rtol=1e-10):
        print("Warning: Time arrays are not synchronized. Using rod1 times as reference.")
        reference_times = times_rod1_filtered
    else:
        reference_times = times_rod1_filtered
    
    n_steps = len(reference_times)
    print(f"Processing {n_steps} time steps from {min_time:.6f}s to {max_time:.6f}s")
    
    # Initialize data arrays
    dataset = []
    
    # Process each time step
    for i in range(n_steps):
        current_time = reference_times[i]
        
        # Find corresponding indices in both datasets
        idx1 = np.argmin(np.abs(times_rod1 - current_time))
        idx2 = np.argmin(np.abs(times_rod2 - current_time))
        
        # Extract torques (assuming they're stored in external_torques)
        if "torques" in rod1_data and len(rod1_data["torques"]) > idx1:
            # Torques are typically stored as (3, n_elements) - sum over elements to get total torque
            rod1_torques = np.array(rod1_data["torques"][idx1])
            if rod1_torques.ndim > 1:
                rod1_torque = np.sum(rod1_torques, axis=1)  # Sum over elements
            else:
                rod1_torque = rod1_torques
        else:
            rod1_torque = np.zeros(3)  # Default if no torque data
            
        if "torques" in rod2_data and len(rod2_data["torques"]) > idx2:
            rod2_torques = np.array(rod2_data["torques"][idx2])
            if rod2_torques.ndim > 1:
                rod2_torque = np.sum(rod2_torques, axis=1)  # Sum over elements
            else:
                rod2_torque = rod2_torques
        else:
            rod2_torque = np.zeros(3)  # Default if no torque data
        
        # Extract positions and velocities
        rod1_positions = np.array(rod1_data["position"][idx1])
        rod1_velocities = np.array(rod1_data["velocity"][idx1])
        
        # Get tip data (assuming tip is the second rod's tip in a double rod system)
        # For a connected double rod, the overall tip is at the end of rod2
        if len(all_sim_data) > 1:
            rod2_positions = np.array(rod2_data["position"][idx2])
            rod2_velocities = np.array(rod2_data["velocity"][idx2])
            tip_position, tip_velocity = extract_tip_data(rod2_positions, rod2_velocities)
        else:
            # If only one rod, use its tip
            tip_position, tip_velocity = extract_tip_data(rod1_positions, rod1_velocities)
        
        # Create data row
        row_data = {
            'T': current_time,
            'rod1_torque_x': rod1_torque[0],
            'rod1_torque_y': rod1_torque[1],
            'rod2_torque_x': rod2_torque[0],
            'rod2_torque_y': rod2_torque[1],
            'tip_position_x': tip_position[0],
            'tip_position_y': tip_position[1],
            'tip_position_z': tip_position[2],
            'tip_velocity_x': tip_velocity[0],
            'tip_velocity_y': tip_velocity[1],
            'tip_velocity_z': tip_velocity[2]
        }
        
        dataset.append(row_data)
    
    return pd.DataFrame(dataset)


def create_neural_network_dataset(df, DT):
    """
    Create input-output pairs for neural network training.
    
    For each time step k, create:
    Input X (11x1): [u(k), x(k), dt] where
        - u(k): [rod1_torque(k), rod2_torque(k)] (4x1)
        - x(k): [tip_position(k), tip_velocity(k)] (6x1)  
        - dt: simulation time step (1x1)
    
    Output Y (6x1): [tip_position(k+1), tip_velocity(k+1)]
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with simulation data
    DT : float
        Simulation time step
        
    Returns:
    --------
    tuple
        (X, Y) where X is input features and Y is target outputs
    """
    n_samples = len(df) - 1  # We lose one sample since we need k+1
    
    # Input features (11 dimensions: 4 torques + 6 state + 1 dt)
    X = np.zeros((n_samples, 11))
    # Output targets (6 dimensions) 
    Y = np.zeros((n_samples, 6))
    
    for i in range(n_samples):
        # Current time step k
        current_row = df.iloc[i]
        # Next time step k+1
        next_row = df.iloc[i + 1]
        
        # Input X: [u(k), x(k), dt]
        X[i, :] = [
            # u(k): torques at time k (4x1)
            current_row['rod1_torque_x'], current_row['rod1_torque_y'],
            current_row['rod2_torque_x'], current_row['rod2_torque_y'],
            # x(k): state at time k (6x1)
            current_row['tip_position_x'], current_row['tip_position_y'], current_row['tip_position_z'],
            current_row['tip_velocity_x'], current_row['tip_velocity_y'], current_row['tip_velocity_z'],
            # dt: time step (1x1)
            DT
        ]
        
        # Output Y: state at time k+1 (6x1)
        Y[i, :] = [
            next_row['tip_position_x'], next_row['tip_position_y'], next_row['tip_position_z'],
            next_row['tip_velocity_x'], next_row['tip_velocity_y'], next_row['tip_velocity_z']
        ]
    
    return X, Y


def main():
    """
    Main function to convert simulation data to CSV format for neural network training.
    """
    # Configuration
    DATA_PATH = "results"
    OUTPUT_CSV = "results/sim_dataset.csv"
    from main import simulation_params
    DT = simulation_params['mpc_dt']
    
    try:
        # Load simulation data
        print("Loading simulation data...")
        all_sim_data = load_simulation_data(DATA_PATH)
        
        if len(all_sim_data) < 2:
            print("Warning: Less than 2 rods found. Neural network expects 2 rods.")
        
        # Process data into DataFrame
        print("Processing simulation data...")
        df = process_simulation_data(all_sim_data)
        
        print(f"Processed {len(df)} time steps")
        print(f"Time range: {df['T'].min():.2f} - {df['T'].max():.2f} ms")
        
        # Save complete dataset
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Complete dataset saved to: {OUTPUT_CSV}")
        
        # Create neural network input-output pairs
        print("Creating neural network training data...")
        X, Y = create_neural_network_dataset(df, DT)
        
        print(f"Created {X.shape[0]} training samples")
        print(f"Input features shape: {X.shape}")
        print(f"Target outputs shape: {Y.shape}")
        
        # Save features and targets separately
        feature_columns = [
            'rod1_torque_x_k', 'rod1_torque_y_k',
            'rod2_torque_x_k', 'rod2_torque_y_k',
            'tip_position_x_k', 'tip_position_y_k', 'tip_position_z_k',
            'tip_velocity_x_k', 'tip_velocity_y_k', 'tip_velocity_z_k',
            'dt'
        ]
        
        target_columns = [
            'tip_position_x_k+1', 'tip_position_y_k+1', 'tip_position_z_k+1',
            'tip_velocity_x_k+1', 'tip_velocity_y_k+1', 'tip_velocity_z_k+1'
        ]
        
        # Print some statistics
        print("\nDataset Statistics:")
        print("=" * 50)
        print(f"Total samples: {len(X)}")
        print(f"Input dimensions: {X.shape[1]}")
        print(f"Output dimensions: {Y.shape[1]}")
        print(f"Time step (dt): {DT}")
        
        print("\nInput ranges:")
        for i, col in enumerate(feature_columns):
            print(f"  {col}: [{X[:, i].min():.6f}, {X[:, i].max():.6f}]")
        
        print("\nOutput ranges:")
        for i, col in enumerate(target_columns):
            print(f"  {col}: [{Y[:, i].min():.6f}, {Y[:, i].max():.6f}]")
        
        print(f"\nDataset generation complete!")
        print(f"Files generated:")
        print(f"  - {OUTPUT_CSV}: Complete time series data")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()