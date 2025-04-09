# Load the csv file
import pandas as pd
import numpy as np
csv_path = r"data/exp_2025-04-04_19-17-42/output_exp_2025-04-04_19-17-42.csv"
df = pd.read_csv(csv_path)

# Extract volumes
volumes = df[['volume_1', 'volume_2', 'volume_3']].values

# Calculate tip-base difference
df['delta_x'] = df['tip_x'] - df['base_x']
df['delta_y'] = df['tip_y'] - df['base_y']
df['delta_z'] = df['tip_z'] - df['base_z']
deltas = df[['delta_x', 'delta_y', 'delta_z']].values

redundant_indexes = []

# Find a subset of deltas points that are close to each other
for i, point in enumerate(deltas):
    for j, other_point in enumerate(deltas):
        if i != j:  # Make sure we're not comparing the same point
            distance = np.linalg.norm(point - other_point)
            if distance < 0.1:
                # print(f"Close points: {point} and {other_point}")
                # Check if the corresponding volumes are different
                volume1 = volumes[i]
                volume2 = volumes[j]
                volume_diff = np.abs(volume1 - volume2)
                if np.any(volume_diff > 0.5):
                    print(f"Volume difference is big: {volume_diff} for points {point} and {other_point}")
                    redundant_indexes.append(i)
                else:
                    pass
                    # print(f"Volume difference is small: {volume_diff} for points {point} and {other_point}")
                break


# Plot all the redundant points in a color and the rest in another color using pyvista
import pyvista as pv

# Create a plotter
plotter = pv.Plotter()

# Check if we have any redundant points
if redundant_indexes:
    # Create arrays for redundant and non-redundant points
    
    # Identify non-redundant indexes
    non_redundant_indexes = [i for i in range(len(deltas)) if i not in redundant_indexes]
    
    # Create point clouds
    redundant_cloud = pv.PolyData(np.array([deltas[i] for i in redundant_indexes]))
    non_redundant_cloud = pv.PolyData(np.array([deltas[i] for i in non_redundant_indexes]))
    
    # Add points to the plotter with different colors
    plotter.add_mesh(non_redundant_cloud, color='blue', point_size=1, 
                    render_points_as_spheres=True, label='Non-redundant points')
    plotter.add_mesh(redundant_cloud, color='red', point_size=10, 
                    render_points_as_spheres=True, label='Redundant points')
    
    # Add a legend
    plotter.add_legend()
else:
    # If no redundant points, just plot all points in one color
    all_points = pv.PolyData(deltas)
    plotter.add_mesh(all_points, color='blue', point_size=10, render_points_as_spheres=True)

# Show the plot
plotter.show()
