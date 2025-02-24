import os
import socket
import threading
import time
import cv2
import numpy as np
import yaml
import csv

P1_yaml = os.path.join("calibration_images_cam4_640x480p", "projection_matrix.yaml")
P2_yaml = os.path.join("calibration_images_cam2_640x480p", "projection_matrix.yaml")

today = time.strftime("%Y-%m-%d")
time_now = time.strftime("%H-%M-%S")
experiment_name = "exp_" + today + "_" + time_now
save_dir = os.path.join(".", "data", experiment_name)
output_file = os.path.join(save_dir, f"output_{experiment_name}.csv")

# Colors range for detection
# Yellow
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Red
lower_red1 = np.array([0, 140, 90])
upper_red1 = np.array([6, 255, 255])
lower_red2 = np.array([174, 140, 90])
upper_red2 = np.array([179, 255, 255])

# Blue
lower_blue = np.array([100, 100, 50])
upper_blue = np.array([130, 255, 255])


class Tracker:
    def __init__(self):
        self.sim_server_bool = "1"
        self.sim_server_thread = None
        self.P1_matrix = None
        self.P2_matrix = None

        # Load the projection matrices
        self.P1_matrix = self.load_projection_matrix(P1_yaml)
        print("Projection Matrix for Camera 1 (P1):\n", self.P1_matrix)
        self.P2_matrix = self.load_projection_matrix(P2_yaml)
        print("Projection Matrix for Camera 2 (P2):\n", self.P2_matrix)

    def detect_tip(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the red tip of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make masks for red color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Get the red tip points
        red_tip = cv2.bitwise_and(frame, frame, mask=mask_red)
        red_tip_points = np.where(red_tip > 0)
        red_tip_points = np.column_stack((red_tip_points[1], red_tip_points[0]))

        if red_tip_points.size == 0:
            return np.nan, np.nan

        # Get average of coorinates of point in red tip
        red_tip_x = red_tip_points[:, 0]
        red_tip_y = red_tip_points[:, 1]
        x_tip = np.mean(red_tip_x)
        y_tip = np.mean(red_tip_y)

        return x_tip, y_tip

    def detect_base(self, frame):
        """
        Input: frame - Image from the camera
        Output: x,y - Average coordinates of the base of the robot
        """
        # Transform the image to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Make mask for yellow color
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Get the yellow base points
        yellow_base = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        yellow_base_gray = cv2.cvtColor(yellow_base, cv2.COLOR_BGR2GRAY)
        _, yellow_base_thresh = cv2.threshold(
            yellow_base_gray, 127, 255, cv2.THRESH_BINARY
        )
        yellow_base_edges = cv2.Canny(yellow_base_thresh, 50, 150)
        yellow_base_points = np.where((yellow_base_edges > 0))
        yellow_base_points = np.column_stack(
            (yellow_base_points[1], yellow_base_points[0])
        )

        # Get average coordinates of yellow base
        yellow_base_x = yellow_base_points[:, 0]
        yellow_base_y = yellow_base_points[:, 1]
        x_base = np.mean(yellow_base_x)
        y_base = np.mean(yellow_base_y)

        return x_base, y_base

    def get_image(self, cam_index, timestamp):
        """
        Input: cam_index - Index of the camera
        Output: img - Image from the camera
        """

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cap = cv2.VideoCapture(cam_index)
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
        print(f"Photo saved at {photo_path}")

        cap.release()
        cv2.destroyAllWindows()

        return frame

    def load_projection_matrix(self, yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        P = np.array(data["projection_matrix"], dtype=np.float64)
        return P

    def parse_data(self, data):
        """
        Message protocol:
        - volume signal: "V 100 200 300\n"
        - wait signal "W\n"
        - exit signal "E\n"
        """
        # Parse the received data and update the pressure values
        command = ""
        volume_values = None
        msg = data.decode()
        if msg.startswith("E"):
            command = "exit"
        elif msg.startswith("W"):
            command = "wait"
        elif msg.startswith("V"):
            command = "measure"

        # Extract the pressure values
        if command == "measure":
            volume_values = [int(p) for p in msg.strip().split()[1:]]
        return command, volume_values

    def run(self):
        """
        main function
        """
        if self.sim_server_bool == "1":
            # Start the simulation server
            self.sim_server_thread = threading.Thread(
                target=self.sim_server, daemon=True
            )
            self.sim_server_thread.start()

        # Start the server
        self.run_server()

    def run_server(self, host="127.0.0.1", port=12345):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Allow the socket to reuse the address immediately after closing
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            print(f"Server listening on {host}:{port}")
            try:
                while True:
                    conn, addr = s.accept()
                    conn.settimeout(1)  # Set a timeout if needed
                    with conn:
                        with conn.makefile("rb") as f:
                            while True:
                                try:
                                    line = f.readline()
                                    if not line:
                                        break  # Client closed connection
                                    command, volume_values = self.parse_data(line)
                                    if command == "exit":
                                        print("Exiting connection...")
                                        break
                                    elif command == "wait":
                                        print("Waiting for the next command...")
                                        continue
                                    elif command == "measure":
                                        # Get current timestamp
                                        timestamp = time.time()
                                        try:
                                            # Take images from the cameras
                                            img1 = self.get_image(4, timestamp)
                                            img2 = self.get_image(2, timestamp)
                                        except Exception as e:
                                            print(f"Error in get_image: {e}")
                                            continue

                                        try:
                                            # Triangulate the points
                                            tip_3d, base_3d = self.triangulate(
                                                img1, img2
                                            )
                                            print("Tip coordinates:", tip_3d)
                                            print("Base coordinates:", base_3d)

                                            # Save the data
                                            self.save_data(
                                                volume_values,
                                                tip_3d,
                                                base_3d,
                                                timestamp,
                                            )
                                        except Exception as e:
                                            print(f"Error in triangulate: {e}")
                                            continue

                                except socket.timeout:
                                    continue
                        print("Connection closed")
            except KeyboardInterrupt:
                print("Server shutdown requested. Exiting...")

    def save_data(self, volume_values, tip_3d, base_3d, timestamp):
        """
        Save data in a csv with columns:
        timestamp - pressure_1 - pressure_2 - ... - tip_x - tip_y - tip_z - base_x - base_y - base_z
        """
        header = (
            ["timestamp"]
            + [f"pressure_{i+1}" for i in range(len(volume_values))]
            + ["tip_x", "tip_y", "tip_z", "base_x", "base_y", "base_z"]
        )
        file_exists = os.path.isfile(output_file)

        with open(output_file, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(header)

            tip_list = (
                tip_3d.flatten().tolist() if hasattr(tip_3d, "flatten") else tip_3d
            )
            base_list = (
                base_3d.flatten().tolist() if hasattr(base_3d, "flatten") else base_3d
            )

            writer.writerow([timestamp] + volume_values + tip_list + base_list)

    def sim_server(self, host="127.0.0.1", port=12345):
        """
        Simulate a server that sends data to the client
        """
        time.sleep(1)  # Give the server time to start
        print("Starting sim_server...")
        try:
            # Create a socket and connect to the server
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))

            while True:
                # Send a message to the server
                msg = "V 100 200 300\n"  # example pressure values
                # msg = "W\n"  # Wait for the next command
                # msg = "E\n"  # Exit the server
                s.sendall(msg.encode())
                time.sleep(0.1)  # Add small delay between sends

        except ConnectionRefusedError:
            print("Could not connect to server. Is it running?")
        except Exception as e:
            print(f"Error in sim_server: {e}")
        finally:
            s.close()

    def triangulate(self, img1, img2):
        """
        Input: img1 - Image from camera 1
                img2 - Image from camera 2
        Output: x,y,z - Coordinates of the robot tip in 3D space
                x,y,z - Coordinates of the robot base in 3D space
        """

        # Get tip and base points from the images
        tip1 = self.detect_tip(img1)
        base1 = self.detect_base(img1)
        tip2 = self.detect_tip(img2)
        base2 = self.detect_base(img2)

        # Triangulate the points
        tip_4d = cv2.triangulatePoints(self.P1_matrix, self.P2_matrix, tip1, tip2)
        tip_3d = tip_4d[:3] / tip_4d[3]
        base_4d = cv2.triangulatePoints(self.P1_matrix, self.P2_matrix, base1, base2)
        base_3d = base_4d[:3] / base_4d[3]

        return tip_3d, base_3d


if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()
