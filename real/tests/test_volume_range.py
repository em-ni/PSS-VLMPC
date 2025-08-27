import os
import argparse
import numpy as np
import torch
torch.set_float32_matmul_precision('medium')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from src.nn_model import VolumeNet

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize neural network predictions for a range of volumes")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--num_points", type=int, default=10,
                        help="Number of points per dimension for visualization")
    parser.add_argument("--base_coords", type=str, default="0,0,0",
                        help="Base coordinates in format 'x,y,z'")
    parser.add_argument("--volume_range", type=str, default="116,118",
                        help="Range of volumes to test in format 'min,max'")
    parser.add_argument("--viz_mode", type=str, default="vary_all",
                        choices=["grid3d", "vary_vol1", "vary_vol2", "vary_vol3", "vary_all"],
                        help="Visualization mode")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the 3D plot")
    return parser.parse_args()

def visualize_predictions(args):
    # Parse base coordinates
    base_x, base_y, base_z = map(float, args.base_coords.split(','))
    base_coords = np.array([base_x, base_y, base_z])
    
    # Parse volume range
    volume_min, volume_max = map(float, args.volume_range.split(','))
    
    # Calculate middle value for fixed dimensions
    mid_volume = (volume_min + volume_max) / 2
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    # Look for scalers in the same directory as the model
    model_dir = os.path.dirname(args.model_path)
    scalers_filename = "inverse_volume_net_scalers.npz" if "inverse" in args.model_path else "volume_net_scalers.npz"
    scalers_path = os.path.join(model_dir, scalers_filename)
    
    if not os.path.exists(scalers_path):
        print(f"Error: Scalers not found at {scalers_path}")
        return
    
    # Load scalers
    scalers = np.load(scalers_path)
    
    # Recreate scalers
    scaler_volumes = MinMaxScaler()
    scaler_volumes.min_ = scalers['volumes_min']
    scaler_volumes.scale_ = scalers['volumes_scale']
    
    scaler_deltas = MinMaxScaler()
    scaler_deltas.min_ = scalers['deltas_min']
    scaler_deltas.scale_ = scalers['deltas_scale']
    
    # Load model
    model = VolumeNet(input_dim=3, output_dim=3)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Create volume inputs based on visualization mode
    num_points = args.num_points
    lin_space = np.linspace(volume_min, volume_max, num_points)
    
    if args.viz_mode == "grid3d":
        # Create a sparse 3D grid to avoid too many points
        grid_points = max(3, int(np.cbrt(num_points)))
        lin_space_sparse = np.linspace(volume_min, volume_max, grid_points)
        
        volumes_grid = []
        for vol1 in lin_space_sparse:
            for vol2 in lin_space_sparse:
                for vol3 in lin_space_sparse:
                    volumes_grid.append([vol1, vol2, vol3])
    
    elif args.viz_mode == "vary_vol1":
        volumes_grid = [[vol, mid_volume, mid_volume] for vol in lin_space]
    
    elif args.viz_mode == "vary_vol2":
        volumes_grid = [[mid_volume, vol, mid_volume] for vol in lin_space]
    
    elif args.viz_mode == "vary_vol3":
        volumes_grid = [[mid_volume, mid_volume, vol] for vol in lin_space]
    
    elif args.viz_mode == "vary_all":
        # Vary all three volumes simultaneously
        volumes_grid = [[vol, vol, vol] for vol in lin_space]
    
    volumes_grid = np.array(volumes_grid)
    print(f"Generated {len(volumes_grid)} test points")
    
    # Scale inputs
    volumes_scaled = scaler_volumes.transform(volumes_grid)
    
    # Convert to tensor
    volumes_tensor = torch.tensor(volumes_scaled, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        predictions_scaled = model(volumes_tensor).numpy()
    
    # Inverse transform predictions
    deltas = scaler_deltas.inverse_transform(predictions_scaled)
    
    # Calculate actual tip positions
    tip_positions = base_coords + deltas
    
    # Create the 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Determine coloring based on visualization mode
    if args.viz_mode == "grid3d":
        # Use the sum of volumes for coloring
        color_values = np.sum(volumes_grid, axis=1)
        scatter = ax.scatter(
            tip_positions[:, 0], 
            tip_positions[:, 1], 
            tip_positions[:, 2], 
            c=color_values, 
            cmap='viridis', 
            alpha=0.6,
            s=50
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sum of Volumes')
    
    elif args.viz_mode in ["vary_vol1", "vary_vol2", "vary_vol3"]:
        vol_idx = {"vary_vol1": 0, "vary_vol2": 1, "vary_vol3": 2}[args.viz_mode]
        color_values = volumes_grid[:, vol_idx]
        scatter = ax.scatter(
            tip_positions[:, 0], 
            tip_positions[:, 1], 
            tip_positions[:, 2], 
            c=color_values, 
            cmap='viridis', 
            alpha=0.8,
            s=80
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label(f'Volume {vol_idx+1}')
        
        # Connect points with a line to show progression
        ax.plot(
            tip_positions[:, 0], 
            tip_positions[:, 1], 
            tip_positions[:, 2], 
            'k--', 
            alpha=0.5
        )
    
    elif args.viz_mode == "vary_all":
        scatter = ax.scatter(
            tip_positions[:, 0], 
            tip_positions[:, 1], 
            tip_positions[:, 2], 
            c=volumes_grid[:, 0], 
            cmap='viridis', 
            alpha=0.8,
            s=80
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('Volume Value (all equal)')
        
        # Connect points with a line
        ax.plot(
            tip_positions[:, 0], 
            tip_positions[:, 1], 
            tip_positions[:, 2], 
            'k--', 
            alpha=0.5
        )
    
    # Plot the base point
    ax.scatter([base_x], [base_y], [base_z], color='red', s=150, marker='*', label='Base')
    
    # Add arrows from base to first and last points
    if args.viz_mode in ["vary_vol1", "vary_vol2", "vary_vol3", "vary_all"]:
        # Arrow to first point (lowest volume)
        first_point = tip_positions[0]
        ax.quiver(
            base_x, base_y, base_z,
            first_point[0]-base_x, first_point[1]-base_y, first_point[2]-base_z,
            color='blue', alpha=0.7, arrow_length_ratio=0.1
        )
        
        # Arrow to last point (highest volume)
        last_point = tip_positions[-1]
        ax.quiver(
            base_x, base_y, base_z,
            last_point[0]-base_x, last_point[1]-base_y, last_point[2]-base_z,
            color='green', alpha=0.7, arrow_length_ratio=0.1
        )
    
    # Add labels and title
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    title = f'Predicted Coordinates for Volumes {volume_min}-{volume_max}'
    ax.set_title(title)
    
    # Add a legend
    ax.legend()
    
    # Set specific axis limits and view angle
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.view_init(elev=-50, azim=-100, roll=-80)
    
    # Add text with volume information
    if args.viz_mode != "grid3d":
        if args.viz_mode == "vary_all":
            info_text = "All volumes vary equally from {:.1f} to {:.1f}".format(volume_min, volume_max)
        elif args.viz_mode == "vary_vol1":
            info_text = "Volume 1 varies {:.1f}-{:.1f}, others = {:.1f}".format(
                volume_min, volume_max, mid_volume)
        elif args.viz_mode == "vary_vol2":
            info_text = "Volume 2 varies {:.1f}-{:.1f}, others = {:.1f}".format(
                volume_min, volume_max, mid_volume)
        elif args.viz_mode == "vary_vol3":
            info_text = "Volume 3 varies {:.1f}-{:.1f}, others = {:.1f}".format(
                volume_min, volume_max, mid_volume)
        
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=11, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Save the plot if requested
    if args.output_path:
        plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {args.output_path}")
    
    plt.tight_layout()
    plt.show()

def main():
    args = parse_args()
    visualize_predictions(args)

if __name__ == "__main__":
    main()