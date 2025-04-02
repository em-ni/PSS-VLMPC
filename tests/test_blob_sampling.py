import numpy as np
import alphashape
from scipy.spatial import ConvexHull
import pandas as pd
import pyvista as pv

def generate_surface(points, alpha=1.0):
    """
    Creates an alpha shape (concave hull) of the given point cloud.
    Returns the alpha shape and the convex hull (for faster sampling checks).
    """
    alpha_shape = alphashape.alphashape(points, alpha)
    convex_hull = ConvexHull(points)
    return alpha_shape, convex_hull

def sample_point_in_hull(hull, num_samples=1):
    """
    Generates random points inside the convex hull using rejection sampling.
    """
    min_bounds, max_bounds = np.min(hull.points, axis=0), np.max(hull.points, axis=0)
    sampled_points = []
    
    while len(sampled_points) < num_samples:
        rand_point = np.random.uniform(min_bounds, max_bounds)
        if is_inside_hull(rand_point, hull):
            sampled_points.append(rand_point)
    
    return np.array(sampled_points)

def is_inside_hull(point, hull):
    """
    Checks if a point is inside the convex hull using the half-space representation.
    """
    new_point = np.append(hull.points, [point], axis=0)
    try:
        _ = ConvexHull(new_point)
        return True
    except:
        return False

def load_point_cloud_from_csv(file_path):
    """
    Loads point cloud from CSV file and computes the relative coordinates.
    """
    df = pd.read_csv(file_path)
    tip_coords = df[['tip_x', 'tip_y', 'tip_z']].values
    base_coords = df[['base_x', 'base_y', 'base_z']].values
    relative_coords = tip_coords - base_coords
    return relative_coords

# Example usage
if __name__ == "__main__":
    # Load point cloud from CSV
    file_path = r"data/exp_2025-04-02_11-43-59/output_exp_2025-04-02_11-43-59.csv"  # Change this to your actual CSV file path
    point_cloud = load_point_cloud_from_csv(file_path)
    
    # Create surface
    alpha_shape, convex_hull = generate_surface(point_cloud, alpha=1.0)
    # Visualize the surface with PyVista

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add the original points to the plot
    point_cloud_pv = pv.PolyData(point_cloud)
    plotter.add_mesh(point_cloud_pv, color="red", point_size=5, render_points_as_spheres=True, label="Points")

    # Add convex hull to the plot
    hull_points = convex_hull.points
    hull_faces = np.column_stack((np.ones(len(convex_hull.simplices), dtype=int) * 3, 
                                 convex_hull.simplices)).astype(int)
    hull_mesh = pv.PolyData(hull_points, hull_faces)
    plotter.add_mesh(hull_mesh, color="lightgreen", opacity=0.3, label="Convex Hull")

    # Add the alpha shape
    # Extract coordinates from the shapely geometry
    if hasattr(alpha_shape, '__geo_interface__'):
        geo = alpha_shape.__geo_interface__
        
        # Extract coordinates based on geometry type
        if geo['type'] == 'Polygon':
            coords = np.array(geo['coordinates'][0])
        elif geo['type'] == 'MultiPolygon':
            all_coords = []
            for polygon in geo['coordinates']:
                all_coords.extend(polygon[0])
            coords = np.array(all_coords)
            
        # Create surface from coordinates
        if coords.shape[1] == 3:  # Ensure we have 3D points
            cloud = pv.PolyData(coords)
            surface = cloud.delaunay_3d().extract_surface()
            plotter.add_mesh(surface, color="blue", opacity=0.5, label="Alpha Shape")

    # Add a legend
    plotter.add_legend()

    # Display the plot
    plotter.show()
    
    # Sample points inside the hull
    sampled_points = sample_point_in_hull(convex_hull, num_samples=1000)
    print("Sampled Points:", sampled_points)
