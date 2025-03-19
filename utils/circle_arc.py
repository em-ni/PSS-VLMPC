import numpy as np


def calculate_circle_through_points(p1, p2, p3, num_points=100):
    # Convert input points to numpy arrays
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    
    # Compute two vectors on the plane and the normal
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Create an orthonormal basis in the plane: u along v1 and v perpendicular to u in the plane
    u = v1 / np.linalg.norm(v1)
    v = np.cross(normal, u)
    
    # Project points onto 2D coordinates in the (u,v) plane. Let p1 be the origin.
    a2d = np.array([0, 0])
    b2d = np.array([np.dot(p2 - p1, u), np.dot(p2 - p1, v)])
    c2d = np.array([np.dot(p3 - p1, u), np.dot(p3 - p1, v)])
    
    # Compute circumcenter in 2D for points a2d, b2d, c2d.
    d = 2 * (b2d[0] * c2d[1] - b2d[1] * c2d[0])
    if abs(d) < 1e-6:
        # Points are collinear, so we cannot define a unique circle.
        return np.empty((0, 3))
    center_x = (c2d[1] * (b2d[0]**2 + b2d[1]**2) - b2d[1] * (c2d[0]**2 + c2d[1]**2)) / d
    center_y = (b2d[0] * (c2d[0]**2 + c2d[1]**2) - c2d[0] * (b2d[0]**2 + b2d[1]**2)) / d
    center_2d = np.array([center_x, center_y])
    
    # Convert the 2D center back to the 3D coordinate system.
    center_3d = p1 + center_x * u + center_y * v
    
    # Compute the radius from the 2D center
    radius = np.linalg.norm(b2d - center_2d)
    
    # Compute the angles for p2 and p3 relative to the center in the 2D plane.
    angle_p2 = np.arctan2(b2d[1] - center_y, b2d[0] - center_x)
    angle_p3 = np.arctan2(c2d[1] - center_y, c2d[0] - center_x)
    
    # Ensure we take the minimal angular difference.
    delta = angle_p3 - angle_p2
    if delta > np.pi:
        delta -= 2 * np.pi
    elif delta < -np.pi:
        delta += 2 * np.pi

    # Create an array of angles spanning from p2 to p3.
    angles = np.linspace(angle_p2, angle_p2 + delta, num_points)
    
    # Calculate the arc points in 3D using the (u, v) basis.
    arc_points = []
    for theta in angles:
        point = center_3d + radius * (np.cos(theta) * u + np.sin(theta) * v)
        arc_points.append(point)
    
    return np.array(arc_points)