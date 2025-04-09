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
    Observation: 3D coordinates of the simulated tip wrt to the base (shape: 3,)
    Action: A 3D discrete vector representing desired volumes (tau).
    Reward: Negative Euclidean distance from the simulated tip to the goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SimRobotEnv, self).__init__()

        # Load the point cloud from a CSV file and get the mesh 
        print("Loading point cloud...")
        file_path = r"data/exp_2025-04-04_19-17-42/output_exp_2025-04-04_19-17-42.csv"  # Change this to your actual CSV file path
        point_cloud = load_point_cloud_from_csv(file_path)
        self.alpha_shape, self.convex_hull = generate_surface(point_cloud, alpha=1.0)
        
        # Continuous action space: 4 continuous values from 0 to max_stroke
        self.action_space = spaces.Box(
            low=0.0, 
            high=config.max_stroke, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Observation space: 3D coordinates of the tip wrt to the base (shape: 3,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        # Initialize neural network model and scalers
        self.init_nn()
        print("Neural network initialized.")

        # Initialize tip position with commands at initial position
        self.tip = self.predict_tip(np.zeros(3, dtype=np.float32))  # Initial tip position
        if config.pick_random_goal:
            self.goal = self.pick_goal()
        else:

            if config.use_trajectory:
                self.spline_points = None
                self.pick_trajectory()
                self.current_trajectory_index = 0
            else:
                # self.goal = config.rl_goal
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")


        self.max_steps = 100
        self.current_step = 0

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

        # Load model
        model_path = r"data/exp_2025-04-04_19-17-42/volume_net.pth"
        self.model = VolumeNet(input_dim=3, output_dim=3)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()



    def get_goal(self):
        return self.goal

    def pick_goal(self):
        """
        Pick a random goal within the workspace.
        """
        # return sample_point_in_hull(self.convex_hull, num_samples=1)[0]
        # Get the vertices from the convex hull - these are points from the original point cloud
        hull_vertices = self.convex_hull.points
        
        # Pick a random index
        random_index = np.random.randint(0, len(hull_vertices))
        
        # Return the randomly selected point
        return hull_vertices[random_index]
    
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
        
        # Create interactive 3D visualization
        # self.save_interactive_3d_visualization(self.trajectory_points, self.spline_points)


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

        # Convert to tensor
        volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            predictions_scaled = self.model(volumes_tensor).numpy()

        # Inverse transform predictions
        predictions = self.scaler_deltas.inverse_transform(predictions_scaled)
        return predictions[0]


    def set_goal(self, goal):
        self.goal = goal

    def step(self, action):
        # Action is now directly a continuous vector [motor1, motor2, motor3, elongation]
        elongation = action[3]
        
        # Create a new command vector with the first 3 actions adjusted by elongation
        command = np.zeros(3, dtype=np.float32)
        for i in range(3):
            # Combine each action with elongation, ensuring it stays within bounds
            combined_action = min(action[i] + elongation, config.max_stroke)
            command[i] = max(0, combined_action)  # Ensure non-negative
        

        self.tip = self.predict_tip(command)
        distance = np.linalg.norm(self.tip - self.goal)
        terminated = False
        self.current_step += 1
        reward = 0

        # Same as paper
        reward = -distance  

        ## Old way
        # distance_threshold = 1.9
        
        # if distance > distance_threshold:
        #     # terminated = True
        #     # reward = 0
        #     reward -= 1  # Negative reward for being too far from the goal
        # bonus = 0

        # # Check 3d points alignment with the goal
        # # Compute perpendicular distance from goal to the line defined by the origin and the tip.
        # dist_to_line = np.linalg.norm(np.cross(self.tip, self.goal)) / np.linalg.norm(self.tip)

        # # If the goal lies nearly on the line, add a bonus reward.
        # if dist_to_line < 0.7 and np.linalg.norm(self.goal) > np.linalg.norm(self.tip):
        #     bonus += 5
        #     if distance < 0.5:
        #         bonus += 10
        #         if distance < 0.2:
        #             bonus += 20
        #             if distance < 0.1:
        #                 bonus += 50

        # if not terminated:
        #     # reward = 1/(distance*(self.current_step**2)) + bonus
        #     reward = 1/(0.5*distance+0.0001) + bonus

        ## New way (DONT USE)
        # if distance < 0.5:
        #     reward += 1
        #     if distance < 0.5:
        #         reward += 2
        #         if distance < 0.2:
        #             reward += 4
        # if distance < 1.0:
        #     reward += 1 / (distance**2)  # Inverse distance reward
            # self.goal = self.pick_goal()
            # print(f"Setting new goal at {self.goal}")

        print(f"Step {self.current_step}: Distance {distance} ")
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {"step": self.current_step, "distance": distance}

        if config.use_trajectory:
            # Check if we are at the end of the trajectory
            if self.current_trajectory_index < len(self.spline_points) - 1:
                # Move to the next point in the trajectory
                self.goal = self.spline_points[self.current_trajectory_index]
                self.current_trajectory_index += 1
            else:
                # If we reached the end of the trajectory, set terminated to True
                terminated = True
                print("Reached the end of the trajectory. Picking new one")
                self.pick_trajectory()
                self.current_trajectory_index = 0  # Reset trajectory index

        return self.tip, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)
        self.current_step = 0

        if config.pick_random_goal:
            # Change goal with 20% probability.
            if np.random.random() < 0.4:
                self.goal = self.pick_goal()
                print(f"Setting new goal at {self.goal}")
        return self.tip, {}

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