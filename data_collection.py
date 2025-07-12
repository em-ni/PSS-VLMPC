from src.pressure_loader import PressureLoader
from src.tracker import Tracker
from src.explorer import Explorer
from src.points_cloud import PointsCloud
import src.config as config
import argparse
import threading


def main(realtime: bool):
    # Import the configuration
    experiment_name = config.experiment_name
    save_dir = config.save_dir
    csv_path = config.csv_path

    # # Load pressure 
    # offsets = []
    # pressure_loader = PressureLoader(save_offsets=True)
    # offsets = pressure_loader.load_pressure()
    offsets = [0.0, 0.0, 0.0]  # Placeholder for offsets, replace with actual values if needed

    # Initialize the classes
    explorer = Explorer(save_dir, csv_path, offsets, realtime)
    tracker = Tracker(experiment_name, save_dir, csv_path, realtime)

    if realtime:
        try:

            # Start explorer in a thread
            explorer_thread = threading.Thread(target=explorer.run_realtime)
            explorer_thread.start()
            
            # Start tracker in a thread
            tracker_thread = threading.Thread(target=tracker.run_realtime_tracking)
            tracker_thread.start()

            # Wait for the threads to finish
            explorer_thread.join()
            tracker_thread.join()

            print("Realtime execution completed successfully.")
        except Exception as e:
            print("An error occurred during realtime execution.")
            print(e)
    else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection script.")
    parser.add_argument('--realtime', '-rt', action='store_true', help='Enable realtime mode')
    args = parser.parse_args()
    main(args.realtime)
