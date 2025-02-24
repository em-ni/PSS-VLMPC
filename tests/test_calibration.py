import cv2
import os

# Print the current working directory
print("Current working directory:", os.getcwd())

# Path to the calibration file (use absolute path)
calibration_file = os.path.abspath('em/calibration.yaml')

# Print the absolute path to ensure it's correct
print("Calibration file path:", calibration_file)

# Load the calibration file
fs = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)

if not fs.isOpened():
    print("Error: Could not open settings file.")
else:
    print("Settings file loaded successfully.")

    fx = fs.getNode("Camera.fx").real()
    fy = fs.getNode("Camera.fy").real()
    cx = fs.getNode("Camera.cx").real()
    cy = fs.getNode("Camera.cy").real()

    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")

    fs.release()
