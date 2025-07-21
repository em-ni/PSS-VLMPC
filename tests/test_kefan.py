import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import threading
import signal
import time
import os
import cv2
from test_mpc import apply_transformation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pressure_loader import PressureLoader
import src.config as config
from src.robot_env import RobotEnv
import joblib
from scipy.spatial import Delaunay, KDTree
from src.tracker import Tracker

class WorkspaceManager:
    """
    Manages the robot's workspace using a point cloud.
    
    This class operates in a NORMALIZED (0-1) coordinate space. It combines a 
    Delaunay triangulation to check if a point is inside the volume, and a 
    KD-Tree to quickly find the closest point on the surface if a target is outside.
    """
    def __init__(self, normalized_point_cloud):
        """
        Initializes the manager by building structures from NORMALIZED data.
        
        Args:
            normalized_point_cloud (np.ndarray): An array of shape (n, 3) of the 
                                                 workspace points, scaled to a 0-1 range.
        """
        print("--- Initializing WorkspaceManager ---")
        if not isinstance(normalized_point_cloud, np.ndarray) or normalized_point_cloud.ndim != 2 or normalized_point_cloud.shape[1] != 3:
            raise ValueError("Input must be a NumPy array of shape (n, 3)")
        
        self.points = normalized_point_cloud
        
        print(f"Building Delaunay triangulation from {len(self.points)} normalized points...")
        self.tessellation = Delaunay(self.points)
        
        print(f"Building KD-Tree from {len(self.points)} normalized points...")
        self.kdtree = KDTree(self.points)
        
        print("WorkspaceManager is ready.")

    def get_valid_point(self, normalized_target_point):
        """
        Ensures the normalized target point is within the workspace.
        
        If the point is inside, it's returned as is.
        If it's outside, the closest point on the workspace surface is returned.
        
        Args:
            normalized_target_point (np.ndarray): A NumPy array of shape (3,) in a 0-1 range.
            
        Returns:
            np.ndarray: A valid normalized point of shape (3,) guaranteed to be in the workspace.
        """
        if self.tessellation.find_simplex(normalized_target_point) >= 0:
            return normalized_target_point
        else:
            _, index = self.kdtree.query(normalized_target_point)
            return self.points[index]


def load_inverse_model():
    """
    Loads the scikit-learn model and scalers saved by the training script.
    """
    model_path = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\inverse_nn_model.joblib"
    scaler_X_path = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\x_scaler.joblib"
    scaler_Y_path = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\v_scaler.joblib"

    print("--- Loading Model and Scalers ---")
    try:
        model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_Y = joblib.load(scaler_Y_path)
        print(f"Successfully loaded model from {model_path}")
        return model, scaler_X, scaler_Y
    except FileNotFoundError as e:
        print(f"FATAL: Could not load model files. Error: {e}")
        sys.exit()

def get_image(cam_index, timestamp):
    """
    Input: cam_index - Index of the camera
    Output: img - Image from the camera
    """
    save_dir = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\images"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # cap = cv2.VideoCapture(cam_index)
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Cannot access the camera")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        cap.release()
        return

    photo_name = f"cam_{cam_index}_{timestamp}.png"
    photo_path = os.path.join(save_dir, photo_name)
    cv2.imwrite(photo_path, frame)
    # print(f"Photo saved at {photo_path}")

    cap.release()
    cv2.destroyAllWindows()

    return frame, photo_path


def main():
    POINT_CLOUD_PATH = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\allData_matrix.csv"
    MODEL, SCALER_X, SCALER_Y = load_inverse_model()
    full_df = pd.read_csv(POINT_CLOUD_PATH)
    tip_position_data = full_df.iloc[:, 3:6].dropna().values
    cam_left_index = 0
    cam_right_index = 3
        
    # Normalize the point cloud data BEFORE creating the manager
    normalized_tip_data = SCALER_X.transform(tip_position_data)
    workspace = WorkspaceManager(normalized_tip_data)

    # Load pressure 
    offsets = []
    pressure_loader = PressureLoader()
    offsets = pressure_loader.load_pressure()
    
    # Create robot environment
    env = RobotEnv()

    # Init tracker
    tracker = Tracker(config.experiment_name, config.save_dir, config.csv_path, realtime=False)

    csv_path = r"C:\Users\dogro\Desktop\Emanuele\github\sorolearn\tests\kefan\results.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("timestamp,raw_tip_x,raw_tip_y,raw_tip_z,"
                    "tip_3d_x,tip_3d_y,tip_3d_z,"
                    "base_3d_x,base_3d_y,base_3d_z,"
                    "pred_vol_x,pred_vol_y,pred_vol_z\n")
    else:
        # Clean the file if it exists
        with open(csv_path, 'w') as f:
            f.write("timestamp,raw_tip_x,raw_tip_y,raw_tip_z,"
                    "tip_3d_x,tip_3d_y,tip_3d_z,"
                    "base_3d_x,base_3d_y,base_3d_z,"
                    "pred_vol_x,pred_vol_y,pred_vol_z\n")
    print(f"CSV file ready at {csv_path}")

    # Initialize the camera
    cap_left = cv2.VideoCapture(cam_left_index, cv2.CAP_DSHOW)
    cap_right = cv2.VideoCapture(cam_right_index, cv2.CAP_DSHOW)

    random_indices = np.random.choice(len(tip_position_data), size=100, replace=False)
    for idx in random_indices:
        raw_tip_pos = tip_position_data[idx]
        # print(f"\nProcessing target position: {np.round(raw_tip_pos, 2)}")

        # Normalize the raw tip position
        raw_tip_pos = np.array(raw_tip_pos).reshape(1, -1)
        tip_input_normalized = SCALER_X.transform(raw_tip_pos)

        # Get a valid point in the workspace
        valid_tip_input_normalized = workspace.get_valid_point(tip_input_normalized[0])
        tip_input = valid_tip_input_normalized.reshape(1, -1)

        # Predict the volume using the model
        pred_vol_scaled = MODEL.predict(tip_input)
        pred_vol = SCALER_Y.inverse_transform(pred_vol_scaled)[0] - [116.0, 115.9, 115.8]  # Adjusting for Zaber min positions
        # print(f"Predicted volumes: {np.round(pred_vol, 2)}")

        # Apply control to the robot
        print(f"Point number {idx}, Applying control input {pred_vol} to the robot")
        env.robot_api.send_command(pred_vol)

        # Read the frames from the cameras
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Error: Couldn't read the frames.")
            break

        # Triangulate the points
        tip_3d, base_3d, _ = tracker.triangulate(frame_left, frame_right)

        # Save results in CSV withy columns: raw_tip_pos, tip_3d, base_3d, pred_vol
        timestamp = time.time()
        # Flatten tip_3d, base_3d, pred_vol if they are arrays/lists
        tip_3d_flat = np.array(tip_3d).flatten()
        base_3d_flat = np.array(base_3d).flatten()
        pred_vol_flat = np.array(pred_vol).flatten()

        with open(csv_path, 'a') as f:
            f.write(
            f"{timestamp},"
            f"{','.join(map(str, raw_tip_pos[0]))},"
            f"{tip_3d_flat[0]},{tip_3d_flat[1]},{tip_3d_flat[2]},"
            f"{base_3d_flat[0]},{base_3d_flat[1]},{base_3d_flat[2]},"
            f"{pred_vol_flat[0]},{pred_vol_flat[1]},{pred_vol_flat[2]}\n"
            )
        # print(f"Data saved at {csv_path}")
        time.sleep(0.5)




if __name__ == "__main__":
    main()