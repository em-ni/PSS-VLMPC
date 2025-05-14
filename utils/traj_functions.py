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

def generate_straight_trajectory(distance=1.5, num_points=100, axis='x'):
    """
    Generate a straight line trajectory along the specified axis.

    Args:
        distance (float): The distance to travel along the axis.
        num_points (int): The number of points in the trajectory.
        axis (str or int): The axis of movement. Can be 'x' (or 0), 'y' (or 1), or 'z' (or 2).

    Returns:
        np.ndarray: The generated straight line trajectory.
    """
    initial_state = get_initial_state()
    # print(f"Initial state for straight trajectory: {initial_state}")

    if initial_state is None:
        print("Warning: Initial state is None in generate_straight_trajectory. Returning empty array.")
        # Assuming STATE_DIM is 3 based on current usage, otherwise, this needs to be dynamic
        return np.empty((0, 3))

    # Ensure initial_state has at least 3 dimensions as assumed by indexing
    if len(initial_state) < 3:
        print(f"Warning: Initial state has fewer than 3 dimensions: {initial_state}. Returning empty array.")
        return np.empty((0, 3))

    if num_points <= 0:
        return np.empty((0, 3))
    if num_points == 1:
        return np.array([initial_state[:3]])

    # Determine the axis index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if isinstance(axis, str):
        axis_index = axis_map.get(axis.lower())
        if axis_index is None:
            raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")
    elif isinstance(axis, int):
        if axis not in [0, 1, 2]:
            raise ValueError("Invalid axis index. Must be 0 (x), 1 (y), or 2 (z).")
        axis_index = axis
    else:
        raise TypeError("Axis must be a string ('x', 'y', 'z') or an integer (0, 1, 2).")

    trajectory_points = np.tile(initial_state[:3], (num_points, 1))
    
    start_value = initial_state[axis_index]
    end_value = start_value + distance
    
    axis_values = np.linspace(start_value, end_value, num_points)
    trajectory_points[:, axis_index] = axis_values
    
    return trajectory_points

def generate_circle_trajectory(radius=0.5, num_points=100, plane='xy'):
    """
    Generate a circular trajectory on a specified plane.
    The circle is centered at the initial state's projection onto that plane.

    Args:
        radius (float): The radius of the circle.
        num_points (int): The number of points in the trajectory.
        plane (str): The plane of the circle. Can be 'xy', 'yz', or 'xz'.

    Returns:
        np.ndarray: The generated circular trajectory (num_points, 3).
    """
    initial_state = get_initial_state()

    if initial_state is None:
        print("Warning: Initial state is None in generate_circle_trajectory. Returning empty array.")
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))

    if len(initial_state) < 3:
        print(f"Warning: Initial state has fewer than 3 dimensions: {initial_state}. Returning empty array.")
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))

    if num_points <= 0:
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))
    if num_points == 1: # A single point doesn't make a circle, return initial state
        return np.array([initial_state[:3]])


    trajectory_points = np.zeros((num_points, 3))
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False) # endpoint=False to avoid duplicate for closed circle

    plane_lower = plane.lower()

    if plane_lower == 'xy':
        center_x, center_y, const_z = initial_state[0], initial_state[1], initial_state[2]
        trajectory_points[:, 0] = center_x + radius * np.cos(angles)
        trajectory_points[:, 1] = center_y + radius * np.sin(angles)
        trajectory_points[:, 2] = const_z
    elif plane_lower == 'yz':
        const_x, center_y, center_z = initial_state[0], initial_state[1], initial_state[2]
        trajectory_points[:, 0] = const_x
        trajectory_points[:, 1] = center_y + radius * np.cos(angles)
        trajectory_points[:, 2] = center_z + radius * np.sin(angles)
    elif plane_lower == 'xz':
        center_x, const_y, center_z = initial_state[0], initial_state[1], initial_state[2]
        trajectory_points[:, 0] = center_x + radius * np.cos(angles)
        trajectory_points[:, 1] = const_y
        trajectory_points[:, 2] = center_z + radius * np.sin(angles)
    else:
        raise ValueError("Invalid plane. Must be 'xy', 'yz', or 'xz'.")

    return trajectory_points

def generate_spiral_trajectory(
    radius_start=0.05,
    radius_end=0.2,
    height_change=0.1,
    num_points=100,
    num_revolutions=2,
    plane='xy'
):
    """
    Generate a spiral trajectory starting from the initial state.
    The spiral expands/contracts its radius and changes height over a number of revolutions.

    Args:
        radius_start (float): The starting radius of the spiral.
        radius_end (float): The ending radius of the spiral.
        height_change (float): The total change in height along the axis perpendicular
                               to the spiral plane. Positive values move "up" or "out"
                               along that axis, negative values move "down" or "in".
        num_points (int): The number of points in the trajectory.
        num_revolutions (float): The number of full revolutions the spiral makes.
        plane (str): The plane of the spiral's main expansion.
                     Can be 'xy', 'yz', or 'xz'.

    Returns:
        np.ndarray: The generated spiral trajectory (num_points, 3).
    """
    initial_state = get_initial_state()

    if initial_state is None:
        print("Warning: Initial state is None in generate_spiral_trajectory. Returning empty array.")
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))

    if len(initial_state) < 3:
        print(f"Warning: Initial state has fewer than 3 dimensions: {initial_state}. Returning empty array.")
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))

    if num_points <= 0:
        return np.empty((0, STATE_DIM if 'STATE_DIM' in globals() else 3))
    
    # For a single point, return the initial position (consistent with other trajectory types)
    # The spiral starts *at* initial_state[0:3] and evolves from there.
    # If radius_start is > 0, the first point of the spiral path itself would be offset.
    # However, for num_points = 1, it's conventional to just return the start.
    if num_points == 1:
        return np.array([initial_state[:3]])

    trajectory_points = np.zeros((num_points, 3))
    
    # Angles for the spiral
    # If num_revolutions is 0, this creates a line of points with angle 0.
    # If num_points is 1, linspace(x,y,1) gives [x] - handled by above check.
    angles = np.linspace(0, 2 * np.pi * num_revolutions, num_points)
    
    # Radii for each point in the spiral
    radii = np.linspace(radius_start, radius_end, num_points)
    
    # Height offsets for each point in the spiral (along the axis perpendicular to the plane)
    # The spiral starts at the initial_state's height component and changes by height_change.
    height_offsets = np.linspace(0, height_change, num_points)

    base_point = initial_state[:3]
    plane_lower = plane.lower()

    if plane_lower == 'xy':
        # Spiral in XY plane, height changes along Z
        center_x, center_y, start_z = base_point[0], base_point[1], base_point[2]
        trajectory_points[:, 0] = center_x + radii * np.cos(angles)
        trajectory_points[:, 1] = center_y + radii * np.sin(angles)
        trajectory_points[:, 2] = start_z + height_offsets
    elif plane_lower == 'yz':
        # Spiral in YZ plane, "height" (depth) changes along X
        start_x, center_y, center_z = base_point[0], base_point[1], base_point[2]
        trajectory_points[:, 0] = start_x + height_offsets # X is the perpendicular axis
        trajectory_points[:, 1] = center_y + radii * np.cos(angles)
        trajectory_points[:, 2] = center_z + radii * np.sin(angles)
    elif plane_lower == 'xz':
        # Spiral in XZ plane, "height" (width) changes along Y
        center_x, start_y, center_z = base_point[0], base_point[1], base_point[2]
        trajectory_points[:, 0] = center_x + radii * np.cos(angles)
        trajectory_points[:, 1] = start_y + height_offsets # Y is the perpendicular axis
        trajectory_points[:, 2] = center_z + radii * np.sin(angles)
    else:
        raise ValueError("Invalid plane. Must be 'xy', 'yz', or 'xz'.")

    return trajectory_points






