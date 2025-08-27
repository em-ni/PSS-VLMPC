import cv2
import numpy as np
import yaml
import os
import pyvista as pv

# ---------------------------
# 1. Load the two images
# ---------------------------
img1_path = os.path.join("data", "test", "cam_left.png")
img2_path = os.path.join("data", "test", "cam_right.png")

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    print("Error: Could not load one or both images.")
    exit()

# ---------------------------
# 2. Detect Colors in the Images
# ---------------------------
# Define HSV ranges for red, green, blue, and yellow.
# Note: Red usually spans two ranges in HSV.
color_ranges = {
    "red": [((0, 80, 0), (5, 255, 255)), ((172, 80, 0), (180, 255, 255))],
    "blue": [((100, 150, 0), (140, 255, 255))],
    "yellow": [((23, 88, 0), (36, 254, 255))],
}


def detect_color_center(hsv_img, ranges):
    """
    Given an HSV image and a list of (lower, upper) tuples for a color,
    return the centroid (x, y) of the largest blob found.
    """
    mask = None
    # Combine all ranges (use bitwise OR)
    for lower, upper in ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        current_mask = cv2.inRange(hsv_img, lower_np, upper_np)
        mask = current_mask


    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Choose the largest contour.
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# Convert both images to HSV
hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# For each color, detect the centroid in both images.
points_cam1 = {}
points_cam2 = {}

for color, ranges in color_ranges.items():
    pt1 = detect_color_center(hsv1, ranges)
    pt2 = detect_color_center(hsv2, ranges)
    if pt1 is None or pt2 is None:
        print(f"Warning: Could not detect {color} in one of the images.")
    else:
        points_cam1[color] = pt1
        points_cam2[color] = pt2

# Optionally, draw the detected points on the images.
for color, pt in points_cam1.items():
    cv2.circle(img1, pt, 5, (0, 0, 255), -1)
    cv2.putText(
        img1,
        color,
        (pt[0] + 5, pt[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )
for color, pt in points_cam2.items():
    cv2.circle(img2, pt, 5, (0, 0, 255), -1)
    cv2.putText(
        img2,
        color,
        (pt[0] + 5, pt[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
    )


# ---------------------------
# 3. Load Projection Matrices from YAML Files
# ---------------------------
def load_projection_matrix(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    P = np.array(data["projection_matrix"], dtype=np.float64)
    return P


# File paths for the projection matrices.
P_right_yaml = os.path.join("calibration", "calibration_images_camright_640x480p", "projection_matrix.yaml")
P_left_yaml = os.path.join("calibration", "calibration_images_camleft_640x480p", "projection_matrix.yaml")

P_right_matrix = load_projection_matrix(P_right_yaml)
P_left_matrix = load_projection_matrix(P_left_yaml)

print("Projection Matrix for right camera:\n", P_right_matrix)
print("Projection Matrix for left camera:\n", P_left_matrix)

# ---------------------------
# 4. Triangulate the Detected Points
# ---------------------------
triangulated_points = {}
for color in points_cam1:
    if color in points_cam2:
        pt1 = points_cam1[color]
        pt2 = points_cam2[color]
        # Prepare points as 2x1 arrays (in homogeneous pixel coordinates).
        pts1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
        pts2 = np.array([[pt2[0]], [pt2[1]]], dtype=np.float64)

        # Triangulate the 3D point.
        point_4d = cv2.triangulatePoints(P_left_matrix, P_right_matrix, pts1, pts2)
        # Convert from homogeneous coordinates to 3D.
        point_3d = point_4d / point_4d[3]
        triangulated_points[color] = point_3d[:3].ravel()
        print(f"{color} point 3D coordinate: {triangulated_points[color]}")
    else:
        print(
            f"Warning: {color} was not detected in both images; skipping triangulation."
        )

# ---------------------------
# 5. Display the Results (Optional)
# ---------------------------
# cv2.imshow("Camera 1", img1)
# cv2.imshow("Camera 2", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Create a PyVista plotter instance
plotter = pv.Plotter()

# For each triangulated point, add a small sphere and a label.
for color_name, point in triangulated_points.items():
    # Create a sphere centered at the triangulated point.
    sphere = pv.Sphere(radius=0.01, center=point)
    # Add the sphere to the scene using the color name (e.g., "red", "green", etc.)
    plotter.add_mesh(sphere, color=color_name, specular=0.5)
    # Add a label near the point
    plotter.add_point_labels(
        [point], [color_name], font_size=12, point_color="black", shape_opacity=0.5
    )

# Optional: Add coordinate axes and a bounding grid to provide spatial context.
plotter.add_axes(line_width=2)
plotter.show_bounds(grid="back", color="gray")

# Display the interactive 3D visualization.
plotter.show()
