from src.tracker import Tracker
from src.durability import Durability
from src.points_cloud import PointsCloud
import src.config as config

if __name__ == "__main__":

    # Import the configuration
    experiment_name = config.experiment_name
    save_dir = config.save_dir
    csv_path = config.csv_path

    # Initialize the classes
    durability = Durability(save_dir, csv_path)
    tracker = Tracker(experiment_name, save_dir, csv_path)
    points_cloud = PointsCloud(csv_path)

    # Move the robot and save volumes and images
    durability.run()

    # Triangulate the points to get 3d coordinates and plot the points cloud
    tracker.run()
    points_cloud.plot_points()