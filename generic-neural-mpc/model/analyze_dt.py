import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Set the path to your dataset file
FILEPATH = 'model/data/output_exp_2025-07-22_12-23-07.csv' 

def analyze_and_plot_dt(filepath):
    """
    Calculates and plots the distribution of time steps (dt) from a 
    trajectory dataset, correctly handling resets between trajectories.
    """
    try:
        print(f"Loading data from '{filepath}'...")
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at '{filepath}'.")
        print("Please update the FILEPATH variable in the script.")
        return

    # Clean up column names to be safe
    df.columns = df.columns.str.strip().str.replace(' \(.*\)', '', regex=True)

    if 'trajectory' not in df.columns or 'T' not in df.columns:
        print("Error: The CSV must contain 'trajectory' and 'T' columns.")
        return

    print("Calculating dt for each trajectory...")
    all_dts = []

    # Group by 'trajectory' to ensure dt is only calculated within a single run
    for traj_id, group in df.groupby('trajectory'):
        # We need at least 2 points in a trajectory to calculate a time difference
        if len(group) < 2:
            continue
        
        # Sort by time just in case the data is not ordered
        group = group.sort_values(by='T')
        
        # .diff() calculates the difference with the previous row.
        # The first dt in each group will be NaN, which is exactly what we want.
        dts_in_milliseconds = group['T'].diff()
        
        # Convert to seconds and drop the NaN values
        dts_in_seconds = (dts_in_milliseconds / 1000.0).dropna()
        
        # Add the valid dt values from this trajectory to our master list
        all_dts.extend(dts_in_seconds.tolist())

    if not all_dts:
        print("Could not calculate any valid dt values. Please check your data.")
        return

    # Convert to a NumPy array for easier statistical calculations
    dts_array = np.array(all_dts)

    # --- Print Statistics ---
    mean_dt = np.mean(dts_array)
    std_dt = np.std(dts_array)
    min_dt = np.min(dts_array)
    max_dt = np.max(dts_array)
    median_dt = np.median(dts_array)
    mean_freq = 1 / mean_dt if mean_dt > 0 else 0

    print("\n--- Time Step (dt) Statistics ---")
    print(f"Total valid time steps calculated: {len(dts_array)}")
    print(f"Mean dt:      {mean_dt:.6f} seconds")
    print(f"Median dt:    {median_dt:.6f} seconds")
    print(f"Std Dev of dt:{std_dt:.6f} seconds")
    print(f"Min dt:       {min_dt:.6f} seconds")
    print(f"Max dt:       {max_dt:.6f} seconds")
    print(f"-----------------------------------")
    print(f"Mean Sampling Frequency: {mean_freq:.2f} Hz")
    print("-----------------------------------\n")

    # --- Plotting ---
    print("Generating plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Use seaborn for a nice histogram with a kernel density estimate
    sns.histplot(dts_array, bins=75, kde=True, color='royalblue', stat='density')
    
    plt.title('Frequency Distribution of Time Steps (dt)', fontsize=16)
    plt.xlabel('Time Step (dt) in Seconds', fontsize=12)
    plt.ylabel('Normalized Frequency (Density)', fontsize=12)
    
    # Add a vertical line for the mean to make it easy to see
    plt.axvline(
        mean_dt, 
        color='crimson', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean = {mean_dt:.4f}s ({mean_freq:.1f} Hz)'
    )
    
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig('model/data/dt_distribution.png', bbox_inches='tight', dpi=300)
    print("Plot saved as 'dt_distribution.png'.")


if __name__ == "__main__":
    analyze_and_plot_dt(FILEPATH)