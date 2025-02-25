from src.tracker import Tracker
from src.durability import Durability
from src.points_cloud import PointsCloud

if __name__ == "__main__":
    tracker = Tracker()
    durability = Durability()
    points_cloud = PointsCloud()

    # Move the robot and save volumes and images
    durability.run()

    # Triangulate the points and plot the points cloud
    tracker.run()
    points_cloud.plot_points()