import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
from src.config import STATE_DIM, VOLUME_DIM, initial_pos
from utils.nn_functions import get_initial_state

# --- Utility Functions  ---
def pick_goal(point_cloud):
    random_index = np.random.randint(0, len(point_cloud))
    return point_cloud[random_index]

# --- Trajectory Generation Function  ---
def generate_snapped_trajectory(point_cloud, num_waypoints, T_sim, N_steps, from_initial_pos=False):
    """Generates a trajectory by interpolating waypoints and snapping to the point cloud."""
    if point_cloud is None or len(point_cloud) < num_waypoints:
        print("Error: Point cloud issue. Returning static trajectory.")
        default_point = pick_goal(point_cloud) if point_cloud is not None and len(point_cloud) > 0 else np.zeros(STATE_DIM)
        return np.tile(default_point, (N_steps + 1, 1))

    print(f"Generating trajectory with {num_waypoints} waypoints...")
    kdtree = KDTree(point_cloud)
    
    # Get waypoints - if from_initial_pos is True, start from initial position
    if from_initial_pos:
        initial_state = get_initial_state()
        if initial_state is None:
            print("Failed to get initial state, using random waypoints")
            waypoints = np.array([pick_goal(point_cloud) for _ in range(num_waypoints)])
        else:
            print("Starting trajectory from initial position")
            waypoints = np.zeros((num_waypoints, STATE_DIM))
            waypoints[0] = initial_state
            for i in range(1, num_waypoints):
                waypoints[i] = pick_goal(point_cloud)
    else:
        waypoints = np.array([pick_goal(point_cloud) for _ in range(num_waypoints)])
    
    print("Selected Waypoints:\n", np.round(waypoints, 3))

    waypoint_times = np.linspace(0, T_sim, num_waypoints)
    sim_times = np.linspace(0, T_sim, N_steps + 1)

    try:
        # Use linear interpolation for simplicity and robustness
        interp_func = interp1d(waypoint_times, waypoints, axis=0, kind='linear', bounds_error=False, fill_value=(waypoints[0], waypoints[-1]))
        ideal_trajectory = interp_func(sim_times)
    except ValueError as e:
         print(f"Interpolation failed: {e}. Holding first waypoint.")
         return np.tile(waypoints[0], (N_steps + 1, 1))

    snapped_trajectory = np.zeros_like(ideal_trajectory)
    print("Snapping trajectory to point cloud...")
    for i, point in enumerate(ideal_trajectory):
        # Check for NaN or Inf values
        if np.isnan(point).any() or np.isinf(point).any():
            print(f"Warning: NaN/Inf values detected at index {i}. Using nearest valid point.")
            # Find the nearest valid point in the trajectory (before or after)
            valid_indices = [j for j in range(len(ideal_trajectory)) 
                            if not (np.isnan(ideal_trajectory[j]).any() or np.isinf(ideal_trajectory[j]).any())]
            if not valid_indices:
                # If no valid points, use the first waypoint
                snapped_trajectory[i] = waypoints[0]
                continue
            # Find closest valid index
            closest_idx = valid_indices[np.argmin(np.abs(np.array(valid_indices) - i))]
            point = ideal_trajectory[closest_idx]
        
        # Reshape point to 2D array for KDTree query
        point_2d = point.reshape(1, -1)
        distance, index = kdtree.query(point_2d, k=1)
        snapped_trajectory[i] = point_cloud[index[0]]
    
    # Smooth the trajectory while keeping all points within the point cloud
    print("Smoothing the trajectory...")
    # Use a larger window for more aggressive smoothing
    window_size = min(51, N_steps // 2 * 2 + 1)  # Make window_size much larger for smoother trajectory
    print(f"Using smoothing window size: {window_size}")
    smoothed_trajectory = np.zeros_like(snapped_trajectory)
    
    # Apply smoothing while maintaining endpoints
    smoothed_trajectory[0] = snapped_trajectory[0]  # Keep start point
    smoothed_trajectory[-1] = snapped_trajectory[-1]  # Keep end point
    
    # Apply moving average smoothing for interior points
    for i in range(1, len(snapped_trajectory) - 1):
        # Calculate window boundaries
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(len(snapped_trajectory), i + half_window + 1)
        
        # Get average of points in window
        window_average = np.mean(snapped_trajectory[start_idx:end_idx], axis=0)
        
        # Snap the smoothed point back to the point cloud
        point_2d = window_average.reshape(1, -1)
        distance, index = kdtree.query(point_2d, k=1)
        smoothed_trajectory[i] = point_cloud[index[0]]
    
    print("Trajectory generated, snapped, and smoothed.")
    return smoothed_trajectory