import os
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import splprep, splev
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.nn_model import VolumeNet
from tests.test_blob_sampling import generate_surface, sample_point_in_hull, is_inside_hull, load_point_cloud_from_csv

class SimRobotEnv(gym.Env):
    """
    A simulated robot environment using a pre-trained neural network model.
    """
    def __init__(self):
        super(SimRobotEnv, self).__init__()

        # Load the point cloud from a CSV file and get the mesh 
        file_path = r"data/exp_2025-04-04_19-17-42/output_exp_2025-04-04_19-17-42.csv"
        self.point_cloud = load_point_cloud_from_csv(file_path)
        self.alpha_shape, self.convex_hull = generate_surface(self.point_cloud, alpha=1.0)
        print("Loaded point cloud")

        # Continuous action space: 4 continuous values from 0 to max_stroke, 3 bending + 1 elongation
        self.action_space = spaces.Box(
            low=0.0, 
            high=config.max_stroke, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Observation space: tip(3) + goal(3) + distance(1) + last_action(4)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(11,), 
            dtype=np.float32
        )
        
        # Store the last action for observation
        self.last_action = np.zeros(4, dtype=np.float32)
        
        # Initialize neural network model and scalers
        self.init_nn()
        print("Neural network initialized.")

        # Initialize tip position with commands at initial position
        self.tip = self.predict_tip(np.zeros(3, dtype=np.float32))  # Initial tip position
        
        # Curriculum learning properties
        self.curriculum_points = 10  # Start with 10 points
        self.success_counter = 0
        self.success_threshold = 5  # Number of successes before advancing curriculum
        
        if config.pick_random_goal:
            self.goal = self.pick_goal()
        else:
            if config.use_trajectory:
                self.spline_points = None
                self.pick_trajectory()
                self.current_trajectory_index = 0
            elif config.N_points > 0:
                self.current_trajectory_index = 0
                # Pick a fixed number of goals based on curriculum level
                self.goals = self.pick_smart_goals(min(self.curriculum_points, config.N_points))
                # Set the first goal as the initial goal
                self.goal = self.goals[0]
            else:
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")

        # Scale max steps based on the number of goals - minimum 100 steps
        self.max_steps = max(30 * min(self.curriculum_points, config.N_points) if config.N_points > 0 else 100, 100)
        self.current_step = 0
        self.distances = []
        self.episode_rewards = 0  # Track total rewards for curriculum advancement

    def init_nn(self):
        """
        Initialize the neural network model and scalers.
        """
        # Load scalers
        scalers_path = r"data/exp_2025-04-04_19-17-42/volume_net_scalers.npz"
        scalers = np.load(scalers_path)

        # Recreate scalers
        self.scaler_volumes = MinMaxScaler()
        self.scaler_volumes.min_ = scalers['volumes_min']
        self.scaler_volumes.scale_ = scalers['volumes_scale']

        self.scaler_deltas = MinMaxScaler()
        self.scaler_deltas.min_ = scalers['deltas_min']
        self.scaler_deltas.scale_ = scalers['deltas_scale']

        # Check if CUDA is available and set device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load model
        model_path = r"data/exp_2025-04-04_19-17-42/volume_net.pth"
        self.model = VolumeNet(input_dim=3, output_dim=3)

        # Load state dict to the appropriate device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_goal(self):
        return self.goal

    def pick_goal(self):
        """
        Pick a random goal within the workspace.
        """
        # Pick a random index
        random_index = np.random.randint(0, len(self.point_cloud))
                
        # Return the randomly selected point
        return self.point_cloud[random_index]
    
    def pick_goals(self, num_goals):
        """
        Pick a fixed number of random goals within the workspace.
        """
        goals = []
        for _ in range(num_goals):
            goal = self.pick_goal()
            goals.append(goal)
        return np.array(goals)
    
    def pick_trajectory(self):
        # Build a trajectory starting from the tip position to the goal position

        # Pick random points in the convex hull
        num_points = 5
        random_points = []
        for i in range(num_points):
            random_points.append(self.pick_goal())
        # Interpolate between the tip and the goal passing through the random points
        # Take last point of the previous trajectory as the starting point
        if self.spline_points is not None:
            self.trajectory_points = np.vstack([self.spline_points[-1], random_points])
        else:
            self.trajectory_points = np.vstack([self.pick_goal(), random_points])

        # Sort points by distance from the tip position (initial position) for a more natural path
        distances = np.linalg.norm(self.trajectory_points - self.tip, axis=1)
        sorted_indices = np.argsort(distances)
        self.trajectory_points = self.trajectory_points[sorted_indices]

        # Create linear interpolation between control points
        # First, create a parameter vector representing distance along the path
        distances = np.zeros(len(self.trajectory_points))
        for i in range(1, len(self.trajectory_points)):
            distances[i] = distances[i-1] + np.linalg.norm(
            self.trajectory_points[i] - self.trajectory_points[i-1])

        # Normalize the distance parameter to [0, 1]
        distances = distances / distances[-1]

        # Create a finer, evenly spaced set of points for the linear interpolation
        num_points = 100000
        u_fine = np.linspace(0, 1, num_points)

        # Interpolate each coordinate independently
        x = np.interp(u_fine, distances, self.trajectory_points[:, 0])
        y = np.interp(u_fine, distances, self.trajectory_points[:, 1])
        z = np.interp(u_fine, distances, self.trajectory_points[:, 2])

        # Combine the interpolated coordinates
        self.spline_points = np.column_stack((x, y, z))

        # Get all points from the point cloud
        point_cloud = self.convex_hull.points

        # For each point in the trajectory, check if it's too far from any point in the cloud
        # If so, replace it with the closest point in the cloud
        max_distance_threshold = 0.5  
        for i in range(len(self.spline_points)):
            # Calculate distances from current trajectory point to all points in the cloud
            distances_to_cloud = np.linalg.norm(point_cloud - self.spline_points[i], axis=1)
            
            # Find the closest point and its distance
            min_distance = np.min(distances_to_cloud)
            closest_point_idx = np.argmin(distances_to_cloud)
            
            # If the closest point is too far, replace the trajectory point
            if min_distance > max_distance_threshold:
                self.spline_points[i] = point_cloud[closest_point_idx]

        # Initialize trajectory tracking
        self.current_trajectory_index = 0
        self.goal = self.spline_points[self.current_trajectory_index]

    def pick_smart_goals(self, num_goals):
        """Pick a sequence of goals that form a logical path."""
        goals = []
        
        # Start from the current tip position
        last_point = self.tip.copy()
        
        for _ in range(num_goals):
            # Sample several candidate points
            candidates = [self.pick_goal() for _ in range(10)]
            
            # Pick the one that's closest to the last point but still far enough to be interesting
            distances = [np.linalg.norm(c - last_point) for c in candidates]
            best_idx = np.argmin([abs(d - 2.0) for d in distances])  # Try to maintain ~2.0 distance
            
            next_point = candidates[best_idx]
            goals.append(next_point)
            last_point = next_point
        
        return np.array(goals)

    def predict_tip(self, command):
        """
        Predict the tip position based on the command using the neural network model.
        """
        volumes = np.zeros(3, dtype=np.float32)
        volumes[0] = config.initial_pos + command[0]
        volumes[1] = config.initial_pos + command[1] 
        volumes[2] = config.initial_pos + command[2]
        
        # Reshape to 2D array (1 sample, 3 features) for scikit-learn
        volumes_2d = volumes.reshape(1, -1)
        
        # Scale inputs
        volumes_scaled = self.scaler_volumes.transform(volumes_2d)

        # Convert to tensor and move to the appropriate device (GPU if available)
        volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32).to(self.device)

        # Make predictions
        with torch.no_grad():
            # Run the model on the device (GPU if available)
            predictions_tensor = self.model(volumes_tensor)
            # Move results back to CPU for numpy conversion
            predictions_scaled = predictions_tensor.cpu().numpy()

        # Inverse transform predictions
        predictions = self.scaler_deltas.inverse_transform(predictions_scaled)
        return predictions[0]

    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        # Actions [motor1, motor2, motor3, elongation (3 motors combined)]
        elongation = action[3]
        
        # Create a new command vector with the first 3 actions adjusted by elongation
        command = np.zeros(3, dtype=np.float32)
        for i in range(3):
            # Combine each action with elongation, ensuring it stays within bounds
            combined_action = min(action[i] + elongation, config.max_stroke)
            command[i] = max(0, combined_action)  # Ensure non-negative
        
        self.tip = self.predict_tip(command)
        distance = np.linalg.norm(self.tip - self.goal)
        self.distances = np.append(self.distances, distance)
        terminated = False
        self.current_step += 1
        
        # Base reward is negative distance
        reward = -distance
        
        # Truncate episode if we exceed max steps
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {"step": self.current_step, "distance": distance}

        # Handle trajectory-based training
        if config.use_trajectory:
            # Existing trajectory code
            if self.current_trajectory_index < len(self.spline_points) - 1:
                self.goal = self.spline_points[self.current_trajectory_index]
                self.current_trajectory_index += 1
            else:
                terminated = True
                print("Reached the end of the trajectory. Picking new one")
                self.pick_trajectory()
                self.current_trajectory_index = 0
        
        # Handle multi-goal training
        elif config.N_points > 0:
            # Calculate average distance over last 10 steps 
            window_size = min(10, len(self.distances))
            avg_distance = np.mean(self.distances[-window_size:])
            
            # Progress ratio (how far along the sequence we are)
            progress_ratio = self.current_trajectory_index / len(self.goals)
            
            # If we're close enough to the current goal
            if avg_distance < 0.5:
                # Add progress-based bonus reward (higher rewards for later goals)
                progress_bonus = 10.0 * (1.0 + progress_ratio)
                reward += progress_bonus
                
                # Move to the next goal
                print(f"Reached goal {self.current_trajectory_index} with distance {distance}. Moving to next one")
                self.current_trajectory_index += 1
                
                if self.current_trajectory_index < len(self.goals):
                    # We still have more goals in this sequence
                    self.goal = self.goals[self.current_trajectory_index]
                else:
                    # Completed all goals in the sequence
                    self.success_counter += 1
                    reward += 50.0  # Major bonus for completing all goals
                    print(f"\033[32m COMPLETED ALL {len(self.goals)} GOALS! Success counter: {self.success_counter}/{self.success_threshold} \033[0m")
                    
                    # Check if we should advance curriculum
                    if self.success_counter >= self.success_threshold:
                        old_points = self.curriculum_points
                        self.curriculum_points = min(self.curriculum_points + 5, config.N_points)
                        self.success_counter = 0
                        print(f"\033[33m CURRICULUM ADVANCED: {old_points} -> {self.curriculum_points} points \033[0m")
                        
                        # Update max steps for new curriculum level
                        self.max_steps = max(30 * self.curriculum_points, 100)
                    
                    # Reset the trajectory index and pick new goals
                    self.goals += self.pick_smart_goals(min(self.curriculum_points, config.N_points))
                    self.goal = self.goals[self.current_trajectory_index]
        
        # Store the current action for the next observation
        self.last_action = action.copy()
        
        # Add total rewards for curriculum tracking
        self.episode_rewards += reward
        
        # Create the enhanced observation
        observation = np.concatenate([
            self.tip,                # Current tip position (3)
            self.goal,               # Current goal position (3)
            [distance],              # Distance to goal (1)
            self.last_action,        # Last action taken (4)
        ])
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)
        self.current_step = 0
        self.distances = []
        self.episode_rewards = 0
        
        # Reset the last action
        self.last_action = np.zeros(4, dtype=np.float32)
        
        if config.pick_random_goal:
            # Change goal with 40% probability for random goal mode
            if np.random.random() < 0.4:
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")
        
        elif config.N_points > 0:
            # Reset the trajectory index and pick new goals
            self.current_trajectory_index = 0

        
        distance = np.linalg.norm(self.tip - self.goal)

        # Create the observation
        observation = np.concatenate([
            self.tip,                # Current tip position (3)
            self.goal,               # Current goal position (3)
            [distance],              # Distance to goal (1)
            self.last_action,        # Last action taken (4)
        ])
        
        return observation, {}

    def render(self, mode='human'):
        pass

    def save_interactive_3d_visualization(self, points, spline_points):
        """Save an interactive 3D visualization of the trajectory and workspace as HTML."""
        try:
            import plotly.graph_objects as go
            import numpy as np
            import os
            import time
            
            # Create a new figure
            fig = go.Figure()
            
            # Add the convex hull as a mesh
            hull = self.convex_hull
            
            # Extract hull simplices for visualization
            i, j, k = np.array(hull.simplices).T
            
            # Add the hull as a mesh surface
            fig.add_trace(go.Mesh3d(
                x=hull.points[:, 0],
                y=hull.points[:, 1],
                z=hull.points[:, 2],
                i=i, j=j, k=k,
                opacity=0.3,
                color='cyan',
                name='Workspace'
            ))
            
            # Add control points
            fig.add_trace(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Control Points'
            ))
            
            # Add the spline trajectory
            fig.add_trace(go.Scatter3d(
                x=spline_points[:, 0],
                y=spline_points[:, 1],
                z=spline_points[:, 2],
                mode='lines',
                line=dict(color='blue', width=4),
                name='Trajectory'
            ))
            
            # Add start point
            fig.add_trace(go.Scatter3d(
                x=[spline_points[0, 0]],
                y=[spline_points[0, 1]],
                z=[spline_points[0, 2]],
                mode='markers',
                marker=dict(size=8, color='magenta', symbol='circle'),
                name='Start'
            ))
            
            # Add end point
            fig.add_trace(go.Scatter3d(
                x=[spline_points[-1, 0]],
                y=[spline_points[-1, 1]],
                z=[spline_points[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name='End'
            ))
            
            # Add current tip position - FIXED: changed 'star' to 'diamond'
            fig.add_trace(go.Scatter3d(
                x=[self.tip[0]],
                y=[self.tip[1]],
                z=[self.tip[2]],
                mode='markers',
                marker=dict(size=10, color='orange', symbol='diamond'),
                name='Current Tip'
            ))
            
            # Add origin point
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(size=8, color='black', symbol='circle'),
                name='Origin'
            ))
            
            # Update layout for better visualization
            fig.update_layout(
                title='Interactive Robot Trajectory and Workspace',
                scene=dict(
                    xaxis_title='X (mm)',
                    yaxis_title='Y (mm)',
                    zaxis_title='Z (mm)',
                    aspectmode='data'  # Preserve aspect ratio
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(x=0, y=1),
                template='plotly_white'
            )
            
            # Create timestamped filename
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            viz_dir = os.path.join(config.data_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            filename = os.path.join(viz_dir, f"interactive_3d_{timestamp}.html")
            
            # Save as an HTML file that can be opened in a browser
            fig.write_html(
                filename,
                include_plotlyjs='cdn',  # Use CDN for smaller file size
                full_html=True,
                include_mathjax='cdn'
            )
            
            print(f"Interactive 3D visualization saved to {filename}")
            print(f"Open this file in a web browser to view and interact with the 3D model")
            
        except Exception as e:
            print(f"Error creating interactive 3D visualization: {e}")
            import traceback
            traceback.print_exc()