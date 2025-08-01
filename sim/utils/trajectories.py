import numpy as np


def build_trajectories(n_trajectories, trajectory_time, max_torque, pause_time):
    """
    Build trajectories by sampling constant torques for each rod.
    Rod 1: constant torques between -max_torque and max_torque for x and y components
    Rod 2: constant torques between -max_torque/2 and max_torque/2 for x and y components
    z component is set to 0
    Each trajectory holds the same torque value for the entire duration.
    """
    trajectories = []
    
    print(f"\nBuilding {n_trajectories} trajectories...")
    print(f"Each trajectory holds a constant torque for {trajectory_time}s")
    
    for traj_idx in range(n_trajectories):
        # Sample one constant torque for the entire trajectory duration
        # Rod 1: full torque range for x and y components
        rod1_torque_x = np.random.uniform(-max_torque, max_torque)
        rod1_torque_y = np.random.uniform(-max_torque, max_torque)
        rod1_torque = np.array([rod1_torque_x, rod1_torque_y, 0.0])
        
        # Rod 2: half torque range for x and y components  
        rod2_torque_x = np.random.uniform(-max_torque/2, max_torque/2)
        rod2_torque_y = np.random.uniform(-max_torque/2, max_torque/2)
        rod2_torque = np.array([rod2_torque_x, rod2_torque_y, 0.0])
        
        trajectory = {
            'rod1_torque': rod1_torque,
            'rod2_torque': rod2_torque,
            'start_time': traj_idx * (trajectory_time + pause_time),
            'end_time': (traj_idx + 1) * trajectory_time + traj_idx * pause_time
        }
        
        trajectories.append(trajectory)
    
    return trajectories

def get_current_torques(trajectories, current_time, trajectory_time, pause_time):
    """
    Get the current torques based on time and trajectory execution.
    Returns zero torques during pause periods.
    Returns constant torques during active trajectory periods.
    """
    if not trajectories:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    # Find which trajectory we should be executing
    trajectory_cycle_time = trajectory_time + pause_time
    cycle_time = current_time % trajectory_cycle_time
    trajectory_idx = int(current_time // trajectory_cycle_time)
    
    # Check if we're beyond all trajectories
    if trajectory_idx >= len(trajectories):
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    # Check if we're in a pause period
    if cycle_time >= trajectory_time:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
    
    # We're in an active trajectory period - return the constant torque
    trajectory = trajectories[trajectory_idx]
    return trajectory['rod1_torque'], trajectory['rod2_torque']