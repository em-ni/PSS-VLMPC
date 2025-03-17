#!/usr/bin/env python
import argparse
import torch
import numpy as np
from sklearn.preprocessing._data import StandardScaler
import cv2
import torch
import yaml
from src.pressure_net import PressureNet
from src import config
import time
from zaber_motion import Units
from zaber_motion.ascii import Connection

def move_axis(axis, position):
    axis.move_absolute(float(position), Units.LENGTH_MILLIMETRES, False)
    time.sleep(0.1)

def load_model(model_path):
    # Allow StandardScaler during unpickling.
    allowed_globals = {"sklearn.preprocessing._data.StandardScaler": StandardScaler}
    with torch.serialization.safe_globals(allowed_globals):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    model = PressureNet(input_dim=3, output_dim=config.output_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_scaler = checkpoint["input_scaler"]
    output_scaler = checkpoint["output_scaler"]
    return model, input_scaler, output_scaler

def load_projection_matrix(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    P = np.array(data["projection_matrix"], dtype=np.float64)
    return P

def detect_base(frame):
    """
    Input: frame - Image from the camera
    Output: x,y - Average coordinates of the base of the robot
    """
    # Transform the image to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Make mask for yellow color
    mask_yellow = cv2.inRange(hsv, config.lower_yellow, config.upper_yellow)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No yellow base detected.")
        return None
    
    # Choose the largest contour.
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])

    return cx, cy

def detect_target(frame):
    """
    Input: frame - Image from the camera
    Output: x,y - Average coordinates of the green target in the frame
    """
    # Transform the image to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Make mask for green color using config values
    mask_green = cv2.inRange(hsv, config.lower_green, config.upper_green)

    # Find contours in the mask.
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No green target detected.")
        return None

    # Choose the largest contour.
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    return cx, cy

def triangulate_target(img_left, img_right, P_left_matrix, P_right_matrix, target='base'):
    """
    Input: img_left - Image from left camera
           img_right - Image from right camera
    Output: x,y,z - Coordinates of the target in 3D space
    """
    if target == 'base':
        # Get base points from the images.
        target_left = detect_base(img_left)
        target_right = detect_base(img_right)
    elif target == 'target':
        # Get target points from the images.
        target_left = detect_target(img_left)
        target_right = detect_target(img_right)

    if target_left is None or target_right is None:
        print("Couldn't detect the target in one or both images.")
        return None

    # Convert the points to a 2x1 array required by cv2.triangulatePoints.
    target_left = np.array([[target_left[0]], [target_left[1]]], dtype=np.float32)
    target_right = np.array([[target_right[0]], [target_right[1]]], dtype=np.float32)

    # Triangulate the target point.
    target_4d = cv2.triangulatePoints(P_left_matrix, P_right_matrix, target_left, target_right)

    # Convert from homogeneous coordinates to 3D.
    target_3d = target_4d[:3] / target_4d[3]

    return target_3d

def main():
    parser = argparse.ArgumentParser(
        description="Test NN for pressure prediction at a single point"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model checkpoint (e.g. nn_model.pth)")
    args = parser.parse_args()
    
    model, input_scaler, output_scaler = load_model(args.model_path)
    
    # Load the projection matrices
    P_left = load_projection_matrix(config.P_left_yaml)
    P_right = load_projection_matrix(config.P_right_yaml)

    # Open connection on COM3
    connection = Connection.open_serial_port('COM3')
    connection.enable_alerts()

    # connection.enableAlerts()  # (commented out as in MATLAB)
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    print(device_list)

    # Get the axis
    base_rightaxis_1 = device_list[0].get_axis(1)
    base_rightaxis_2 = device_list[1].get_axis(1)
    base_rightaxis_3 = device_list[2].get_axis(1)

    while True:
        input("Move the target and press Enter when ready to predict...")
        try:
            # Take images from both cameras
            cap_left = cv2.VideoCapture(config.cam_left_index, cv2.CAP_DSHOW)
            ret_left, frame_left = cap_left.read()
            cap_left.release()

            cap_right = cv2.VideoCapture(config.cam_right_index, cv2.CAP_DSHOW)
            ret_right, frame_right = cap_right.read()
            cap_right.release()

            if not ret_left or not ret_right:
                print("Error: Could not read frames from cameras.")
                return
            
            # Detect the target in the images
            base_3d = triangulate_target(frame_left, frame_right, P_left, P_right, target='base')
            target_3d = triangulate_target(frame_left, frame_right, P_left, P_right, target='target')
            if base_3d is None or target_3d is None:
                print("Couldn't triangulate the target.")
                return
            
            dx = target_3d[0][0] - base_3d[0][0]
            dy = target_3d[1][0] - base_3d[1][0]
            dz = target_3d[2][0] - base_3d[2][0]

            # # For testing purposes, here we use hard-coded values:
            # # tip : 6.878252,4.057358,-9.675416
            # # base: 4.257098,3.4624796,-9.606976
            # dx = 6.878252 - 4.257098
            # dy = 4.057358 - 3.4624796
            # dz = -9.675416 - -9.606976

            # # Expected output: 118.8,118.0,118.0,6.72568056078031,7.597437599419604,6.049265586710018
            
        except ValueError:
            print("Invalid input, please enter numeric values.")
            return
        
        input_array = np.array([[dx, dy, dz]])
        input_scaled = input_scaler.transform(input_array)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred = model(input_tensor)
        
        pred_np = pred.cpu().numpy()
        # Inverse transform predictions to original scale.
        pred_unscaled = output_scaler.inverse_transform(pred_np)
        pred_unscaled = pred_unscaled.flatten()
        
        if config.pressure_only:
            print("\nPredicted Pressure Values:")
            print(f"Pressure 1: {pred_unscaled[0]:.3f}")
            print(f"Pressure 2: {pred_unscaled[1]:.3f}")
            print(f"Pressure 3: {pred_unscaled[2]:.3f}")
        else:
            print("\nPredicted Values:")
            print(f"Volume 1: {pred_unscaled[0]:.3f}")
            print(f"Volume 2: {pred_unscaled[1]:.3f}")
            print(f"Volume 3: {pred_unscaled[2]:.3f}")
            print(f"Pressure 1: {pred_unscaled[3]:.3f}")
            print(f"Pressure 2: {pred_unscaled[4]:.3f}")
            print(f"Pressure 3: {pred_unscaled[5]:.3f}")

        # Safety check
        

        # Move the axis to the predicted position
        move_axis(base_rightaxis_1, pred_unscaled[0])
        move_axis(base_rightaxis_2, pred_unscaled[1])
        move_axis(base_rightaxis_3, pred_unscaled[2])

if __name__ == "__main__":
    main()
