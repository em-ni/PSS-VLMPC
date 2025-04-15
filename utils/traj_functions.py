import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, splprep, splev 
from src.config import STATE_DIM
from utils.nn_functions import get_initial_state
from scipy.signal import savgol_filter

def pick_goal(point_cloud):
    """Selects a random point from the point cloud."""
    if point_cloud is None or len(point_cloud) == 0:
        print("Warning: Point cloud is empty or None in pick_goal.")
        return np.zeros(STATE_DIM) # Return a default point
    random_index = np.random.randint(0, len(point_cloud))
    return point_cloud[random_index]

def generate_denser_point_cloud(original_point_cloud, density_factor=3):
    """
    Create a denser point cloud by interpolating between existing points.
    
    Args:
        original_point_cloud: The original point cloud
        density_factor: How many times denser the result should be
        
    Returns:
        A denser point cloud
    """
    if len(original_point_cloud) < 2:
        return original_point_cloud
        
    # Create a dense point cloud through interpolation
    dense_point_cloud = []
    for i in range(len(original_point_cloud)-1):
        p1 = original_point_cloud[i]
        p2 = original_point_cloud[i+1]
        
        # Add original point
        dense_point_cloud.append(p1)
        
        # Add interpolated points
        for j in range(1, density_factor):
            t = j / density_factor
            interpolated = p1 * (1-t) + p2 * t
            dense_point_cloud.append(interpolated)
            
    # Add final point
    dense_point_cloud.append(original_point_cloud[-1])
    
    return np.array(dense_point_cloud)

# --- Trajectory Generation Function ---
def generate_snapped_trajectory(
        point_cloud, 
        num_waypoints, 
        T_sim, 
        N_steps, 
        from_initial_pos=True,
        num_iterations = 100,          # Number of smoothing/projection iterations
        smoothing_window_factor = 0.1, # Relative window size for SavGol (smaller factor -> less smoothing per step)
        poly_order = 2                 # Polynomial order for SavGol 
    ):
    """
    Generates a trajectory using spline interpolation, snaps it, and then
    iteratively smooths and re-projects it onto the point cloud to reduce
    zig-zagging while staying within the cloud.

    Args:
        point_cloud (np.ndarray): Workspace points.
        num_waypoints (int): Number of initial waypoints.
        T_sim (float): Simulation time.
        N_steps (int): Number of points in the final trajectory.
        from_initial_pos (bool): Start from initial state.
        num_iterations (int): How many times to apply smoothing and projection.
        smoothing_window_factor (float): Window size for SavGol as fraction of N_steps.
        poly_order (int): Polynomial order for SavGol filter.

    Returns:
        np.ndarray: The iteratively smoothed and snapped trajectory.
    """
    # --- Initial checks and waypoint generation ---
    if point_cloud is None or len(point_cloud) < 2:
        print("Error: Point cloud issue.")
        default_point = pick_goal(point_cloud) if point_cloud is not None and len(point_cloud) > 0 else np.zeros(STATE_DIM)
        return np.tile(default_point, (N_steps + 1, 1))

    num_waypoints = max(2, min(num_waypoints, len(point_cloud)))
    kdtree = KDTree(point_cloud)

    # --- Waypoint Selection ---
    waypoints = np.zeros((num_waypoints, STATE_DIM))
    if from_initial_pos:
        initial_state = get_initial_state()
        if initial_state is None:
            print("Warning: No initial state, using random.")
            dist, idx = kdtree.query(pick_goal(point_cloud).reshape(1,-1)) # Ensure first wp is valid
            waypoints[0] = point_cloud[idx[0]]
        else:
            dist, idx = kdtree.query(initial_state.reshape(1, -1))
            waypoints[0] = point_cloud[idx[0]] # Snap initial state
    else:
         dist, idx = kdtree.query(pick_goal(point_cloud).reshape(1,-1)) # Ensure first wp is valid
         waypoints[0] = point_cloud[idx[0]]

    for i in range(1, num_waypoints):
        waypoints[i] = pick_goal(point_cloud) # Could add checks for distance if needed

    # --- Spline interpolation (robust version) ---
    waypoint_times = np.linspace(0, T_sim, num_waypoints)
    sim_times = np.linspace(0, T_sim, N_steps + 1)
    ideal_trajectory = None
    try:
        # (Spline generation logic with unique point handling and fallback)
        unique_indices = [0] + [i for i in range(1, num_waypoints) if not np.allclose(waypoints[i], waypoints[i-1], atol=1e-6) and waypoint_times[i] > waypoint_times[i-1]]
        if len(unique_indices) < 2: raise ValueError("Need >= 2 unique waypoints for spline")
        unique_waypoints = waypoints[unique_indices]; unique_waypoint_times = waypoint_times[unique_indices]
        spline_k = min(3, len(unique_indices) - 1)
        if spline_k >= 1:
            coords = [unique_waypoints[:, d] for d in range(STATE_DIM)]
            tck, u = splprep(coords, u=unique_waypoint_times, s=0, k=spline_k)
            ideal_trajectory_coords = splev(sim_times, tck)
            ideal_trajectory = np.vstack(ideal_trajectory_coords).T
        else: raise ValueError("Not enough points for k=1 spline")
    except Exception as e_spline:
        print(f"Spline failed ({e_spline}), using linear.")
        try:
            interp_func = interp1d(waypoint_times, waypoints, axis=0, kind='linear', bounds_error=False, fill_value=(waypoints[0], waypoints[-1]))
            ideal_trajectory = interp_func(sim_times)
        except ValueError as e_linear:
            print(f"Linear failed ({e_linear}), holding first.")
            # Use the already validated first waypoint
            return np.tile(waypoints[0], (N_steps + 1, 1))
    if ideal_trajectory is None:
        print("Failed ideal trajectory gen. Holding first.")
        return np.tile(waypoints[0], (N_steps + 1, 1)) # Use validated first waypoint

    # --- Step 1: Initial Snap ---
    current_snapped_path = np.zeros_like(ideal_trajectory)
    # print("Performing initial snap...")
    last_valid_snap_point = None
    try: # Handle first point
        dist, idx = kdtree.query(ideal_trajectory[0].reshape(1,-1)); current_snapped_path[0] = point_cloud[idx[0]]
        last_valid_snap_point = current_snapped_path[0]
    except Exception:
        current_snapped_path[0] = waypoints[0] # Fallback to initial waypoint
        last_valid_snap_point = current_snapped_path[0]

    for i in range(1, N_steps + 1):
        point = ideal_trajectory[i]
        if np.isnan(point).any() or np.isinf(point).any():
            current_snapped_path[i] = last_valid_snap_point # Reuse previous
            continue
        try:
            dist, idx = kdtree.query(point.reshape(1, -1), k=1)
            if np.isinf(dist[0]): raise ValueError("Infinite distance during initial snap")
            current_snapped_path[i] = point_cloud[idx[0]]
            last_valid_snap_point = current_snapped_path[i]
        except Exception as e:
             # print(f"Warning: Initial snap failed at step {i}: {e}. Reusing previous.")
             current_snapped_path[i] = last_valid_snap_point # Reuse previous on error

    # --- Step 2: Iterative Smoothing and Projection ---
    # print(f"Starting {num_iterations} iterations of smoothing and projection...")

    # Calculate window size for SavGol based on factor
    window_size = int((N_steps + 1) * smoothing_window_factor)
    # Ensure window size is odd and appropriate for SavGol
    window_size = max(poly_order + 1, window_size) # Must be >= poly_order + 1
    if window_size % 2 == 0: window_size += 1 # Make odd
    # Ensure window size is not larger than the data length
    window_size = min(window_size, N_steps + 1)

    can_smooth = (window_size > poly_order) and ((N_steps + 1) >= window_size)
    if not can_smooth:
        print(f"Warning: Trajectory too short or window too small for SavGol (N={N_steps+1}, win={window_size}, poly={poly_order}). Skipping iterative smoothing.")
        return current_snapped_path # Return the initial snap

    for iteration in range(num_iterations):
        # --- Smooth Step ---
        smoothed_intermediate_path = np.copy(current_snapped_path) # Work on a copy
        try:
            # Apply SavGol filter to each dimension
            for d in range(STATE_DIM):
                 smoothed_intermediate_path[:, d] = savgol_filter(
                     current_snapped_path[:, d],
                     window_length=window_size, # Window size for SavGol
                     polyorder=poly_order,
                     mode='interp' # Interpolate boundaries
                 )
            # Crucially, DO NOT fix endpoints here, let them be smoothed/projected

        except Exception as e:
            print(f"Warning: SavGol filtering failed during iteration {iteration + 1}: {e}. Skipping smoothing for this iteration.")
            # Keep smoothed_intermediate_path as current_snapped_path if smoothing fails
            smoothed_intermediate_path = np.copy(current_snapped_path)


        # --- Project Step ---
        projected_path = np.zeros_like(smoothed_intermediate_path)
        last_valid_proj_point = None
        try: # Handle first point projection
            dist, idx = kdtree.query(smoothed_intermediate_path[0].reshape(1,-1)); projected_path[0] = point_cloud[idx[0]]
            last_valid_proj_point = projected_path[0]
        except Exception:
            projected_path[0] = current_snapped_path[0] # Fallback to previous iteration start
            last_valid_proj_point = projected_path[0]


        for i in range(1, N_steps + 1):
            point = smoothed_intermediate_path[i]
            if np.isnan(point).any() or np.isinf(point).any():
                projected_path[i] = last_valid_proj_point # Reuse previous valid projection
                continue
            try:
                dist, idx = kdtree.query(point.reshape(1, -1), k=1)
                if np.isinf(dist[0]): raise ValueError("Infinite distance during projection")
                projected_path[i] = point_cloud[idx[0]]
                last_valid_proj_point = projected_path[i] # Update last valid point for this iter
            except Exception as e:
                # print(f"Warning: Projection failed at step {i}, iter {iteration + 1}: {e}. Reusing previous projection.")
                projected_path[i] = last_valid_proj_point # Reuse previous on error

        # Update the path for the next iteration
        current_snapped_path = projected_path

    return current_snapped_path

    