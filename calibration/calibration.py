import cv2
import numpy as np
import yaml
import os

# Camera name
# camera_name = 'Misumi_200x200p'
# camera_name = "Misumi_400x380p"
# camera_name = "videoscope_1280x720p"
camera_name = "cam_left_640x480p"
# camera_name = "cam_right_640x480p"

# Define the chessboard size
chessboard_size = (9, 6)
# Take frame size from name
frame_size = (
    int(camera_name.split("_")[1].split("x")[0]),
    int(camera_name.split("_")[1].split("x")[1].split("p")[0]),
)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboard_size[0], 0 : chessboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Folder name calibration_images + camera name
folder_name = "calibration_images_" + camera_name
first = 1
last = 20

# Capture images of the calibration pattern
images = [
    cv2.imread(os.path.join(folder_name, f"image_{i}.jpg"))
    for i in range(first, last + 1)
]  # Assuming 20 images

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow("img", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Perform camera calibration to obtain camera matrix, distortion coefficients, and extrinsic parameters.
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, frame_size, None, None
)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)

# ---- Compute the Projection Matrix for the first calibration view ----
# Select the first set of rotation and translation vectors.
rvec = rvecs[0]
tvec = tvecs[0]

# Convert the rotation vector to a rotation matrix.
R, _ = cv2.Rodrigues(rvec)

# Create the extrinsic matrix by concatenating R and t.
extrinsic_matrix = np.hstack((R, tvec))

# Compute the 3x4 projection matrix: P = K * [R | t]
projection_matrix = camera_matrix @ extrinsic_matrix

print("\nProjection Matrix:")
print(projection_matrix)

# ---- Save the Projection Matrix to a YAML file under the images folder ----

# Ensure the directory exists
os.makedirs(folder_name, exist_ok=True)

data_to_save = {"projection_matrix": projection_matrix.tolist()}
save_path = os.path.join(folder_name, "projection_matrix.yaml")

with open(save_path, "w") as f:
    yaml.dump(data_to_save, f)

print(f"\nProjection matrix saved to '{save_path}'")
