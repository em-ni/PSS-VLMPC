import numpy as np
import trimesh
import pandas as pd
from scipy.spatial import Delaunay

def load_point_cloud(filename):
    """Load a point cloud from a CSV file and compute the difference between tip and base coordinates."""
    # Load CSV
    df = pd.read_csv(filename)
    
    # Compute point cloud as difference between tip and base coordinates
    points = df[['tip_x', 'tip_y', 'tip_z']].values - df[['base_x', 'base_y', 'base_z']].values
    
    return points

def reconstruct_mesh(point_cloud):
    """Reconstruct a watertight mesh from a point cloud using Delaunay triangulation."""
    tri = Delaunay(point_cloud)
    mesh = trimesh.Trimesh(vertices=point_cloud, faces=tri.simplices)
    return mesh

def sample_points_in_mesh(mesh, num_samples=1000):
    """Sample random points inside a watertight mesh using rejection sampling."""
    # Compute the bounding box
    bbox_min, bbox_max = mesh.bounds
    samples = []

    while len(samples) < num_samples:
        # Generate a random point in the bounding box
        random_point = np.random.uniform(bbox_min, bbox_max)
        
        # Check if the point is inside the mesh using ray casting
        if mesh.contains(random_point.reshape(1, -1)):
            samples.append(random_point)

    return np.array(samples)

# Load or generate a point cloud
pcd = load_point_cloud(r"../data/exp_2025-04-02_11-43-59/output_exp_2025-04-02_11-43-59.csv")  # Fixed path format

# Reconstruct mesh from point cloud
mesh = reconstruct_mesh(pcd)

# Sample points inside the reconstructed mesh
sampled_points = sample_points_in_mesh(mesh, num_samples=1000)

# Save the sampled points as a new point cloud
# np.savetxt("sampled_points.csv", sampled_points, delimiter=",", header="x,y,z", comments="")

# Visualization
mesh.show()