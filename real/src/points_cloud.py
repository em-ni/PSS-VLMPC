import csv
import pyvista as pv

class PointsCloud:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.base_points = []
        self.tip_points = []
        self.base_avg = []
        self.diff = []
        pass

    def get_points_from_csv(self):
        # csv file columns: timestamp - volume_1 - volume_2 - volume_3 - pressure_1 - pressure_2 - pressure_3 - img_left - img_right - tip_x - tip_y - tip_z - base_x - base_y - 
        # Base coordinates
        with open(self.csv_path, mode="r") as csvfile:
            reader = csv.DictReader(csvfile)  # Use DictReader to read the CSV
            next(reader)  # Skip the header
            for row in reader:
                print("Processing row number: ", reader.line_num, end="\r", flush=True)
                # Get the base coordinates
                base_x = row['base_x']
                base_y = row['base_y']
                base_z = row['base_z']

                # Get the tip coordinates
                tip_x = row['tip_x']
                tip_y = row['tip_y']
                tip_z = row['tip_z']

                # Skip the row if any of the coordinates is missing
                if base_x == "" or base_y == "" or base_z == "" or tip_x == "" or tip_y == "" or tip_z == "" or base_x == "nan" or base_y == "nan" or base_z == "nan" or tip_x == "nan" or tip_y == "nan" or tip_z == "nan":
                    print("Skipping row with missing coordinates.")
                    continue

                # Compute the difference between the tip and base coordinates
                diff_x = float(tip_x) - float(base_x)
                diff_y = float(tip_y) - float(base_y)
                diff_z = float(tip_z) - float(base_z)

                # Append the coordinates to the points
                self.base_points.append([base_x, base_y, base_z])
                self.tip_points.append([tip_x, tip_y, tip_z])
                self.diff.append([diff_x, diff_y, diff_z])

        # Compute the average of the base points
        base_x_avg = sum(float(point[0]) for point in self.base_points) / len(self.base_points)
        base_y_avg = sum(float(point[1]) for point in self.base_points) / len(self.base_points)
        base_z_avg = sum(float(point[2]) for point in self.base_points) / len(self.base_points)
        self.base_avg = [base_x_avg, base_y_avg, base_z_avg]
        print("Done processing points.")
        
        return
    
    def plot_points(self):
        # Create a pyvista plot
        plotter = pv.Plotter()

        # Add the base point to the plot as a sphere
        plotter.add_mesh(pv.Sphere(center=[0, 0, 0], radius=0.01), color="yellow")

        # Add the tip points to the plot as spheres
        total_points = len(self.tip_points)
        for diff_point in self.diff:
            print(f"Adding point number: ", self.diff.index(diff_point), " / {total_points}", end="\r", flush=True)
            diff_point_float = [float(coord) for coord in diff_point]
            plotter.add_mesh(pv.Sphere(center=diff_point_float, radius=0.01), color="red")
        
        # Optional: Add coordinate axes and a bounding grid to provide spatial context.
        plotter.add_axes(line_width=2)
        plotter.show_bounds(grid="back", color="gray")

        # Show the plot
        plotter.show()


if __name__ == "__main__":
    csv_path = r"C:\Users\dogro\Desktop\Emanuele\github\ISS-VLMPC\data\exp_2025-07-01_15-18-16\output_exp_2025-07-01_15-18-16.csv"
    points_cloud = PointsCloud(csv_path)
    points_cloud.get_points_from_csv()
    points_cloud.plot_points()