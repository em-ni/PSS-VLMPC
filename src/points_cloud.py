import csv
import os
import pyvista as pv

class PointsCloud:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.base_points = []
        self.tip_points = []
        self.base_avg = []
        self.get_points_from_csv()
        pass

    def get_points_from_csv(self):
        # csv file columns: timestamp - volume_1 - volume_2 - volume_3 - img_1 - img_2 - tip_x - tip_y - tip_z - base_x - base_y - 
        # Base coordinates
        with open(self.csv_path, mode="r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                # Get the base coordinates
                base_x = row[9]
                base_y = row[10]
                base_z = row[11]

                # Get the tip coordinates
                tip_x = row[6]
                tip_y = row[7]
                tip_z = row[8]

                # Append the coordinates to the points
                self.base_points.append([base_x, base_y, base_z])
                self.tip_points.append([tip_x, tip_y, tip_z])

        # Compute the average of the base points
        base_x_avg = sum(float(point[0]) for point in self.base_points) / len(self.base_points)
        base_y_avg = sum(float(point[1]) for point in self.base_points) / len(self.base_points)
        base_z_avg = sum(float(point[2]) for point in self.base_points) / len(self.base_points)
        self.base_avg = [base_x_avg, base_y_avg, base_z_avg]
        
        return
    
    def plot_points(self):
        # Create a pyvista plot
        plotter = pv.Plotter()
        
        # Add the base point to the plot as a sphere
        plotter.add_mesh(pv.Sphere(center=self.base_avg, radius=0.1), color="yellow")

        # Add the tip points to the plot as spheres
        for tip_point in self.tip_points:
            tip_point_float = [float(coord) for coord in tip_point]
            plotter.add_mesh(pv.Sphere(center=tip_point_float, radius=0.01), color="red")
        
        # Show the plot
        plotter.show()


if __name__ == "__main__":
    csv_path = "C:\Users\dogro\Desktop\Emanuele\github\sorolearn\data\exp_2025-02-25_16-13-52\output_exp_2025-02-25_16-13-52.csv"
    points_cloud = PointsCloud(csv_path)
    points_cloud.plot_points()