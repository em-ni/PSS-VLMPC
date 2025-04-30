from src.pressure_loader import PressureLoader
from src.tracker import Tracker
from src.explorer import Explorer
from src.points_cloud import PointsCloud
import src.config as config

if __name__ == "__main__":

    # Import the configuration
    experiment_name = config.experiment_name
    save_dir = config.save_dir
    csv_path = config.csv_path

    # Load pressure 
    offsets = []
    pressure_loader = PressureLoader(save_offsets=True)
    offsets = pressure_loader.load_pressure()

    # Initialize the classes
    explorer = Explorer(save_dir, csv_path, offsets)
    tracker = Tracker(experiment_name, save_dir, csv_path)
    points_cloud = PointsCloud(csv_path)

    try:
        # Move the robot and save volumes, pressures and images
        explorer.run()

        # Triangulate the points to get 3d coordinates and plot the points cloud
        tracker.run()
        points_cloud.get_points_from_csv()
        points_cloud.plot_points()
    except Exception as e:
        print("An error occurred.")
        print(e)
        