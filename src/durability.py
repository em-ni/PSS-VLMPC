import socket
import threading
import time
import numpy as np
from zaber_motion import Units
from zaber_motion.ascii import Connection

class Durability:
    def __init__(self):
        self.msg = "W\n"

    
    def get_msg(self):
        return self.msg    

    def move(self):
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()
        
        # demonstration
        offset = 0  # 21 mm in comment, but offset is set to 0
        initial_pos = 110 + offset  # mm

        # PositionSweep Settings
        steps = 5
        stroke = 3  # mm
        stepSize = stroke / steps

        try:
            # connection.enableAlerts()  # (commented out as in MATLAB)
            device_list = connection.detect_devices()
            print("Found {} devices.".format(len(device_list)))
            print(device_list)
            
            # Home each axis
            axis_1 = device_list[0].get_axis(1)
            axis_1.home()
            axis_2 = device_list[1].get_axis(1)
            axis_2.home()
            axis_3 = device_list[2].get_axis(1)
            axis_3.home()

            # Move each axis to the minimum position
            axis_1.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_2.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_3.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            
            userInput = input("Enter 2 to continue:\n")
            
            if userInput == '2':
                position_matrix = np.zeros((3, 216))
                
                i = 1
                j = 0
                k = 0
                i_flipFlag = 1
                j_flipFlag = 1
                k_flipFlag = 1
                stepCounter = 0  # Python index starts at 0
                
                position_i = initial_pos
                position_j = initial_pos
                position_k = initial_pos
                
                while i <= steps + 1:
                    while j <= steps:
                        while k <= steps:
                            if k == 0 and k_flipFlag == -1:
                                k_flipFlag = -k_flipFlag
                                position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                                print(i, j, k)
                                axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                                axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                                axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                                time.sleep(0.2)
                                new_msg = f"V {position_i}, {position_j}, {position_k}\n"
                                self.update_msg(new_msg)
                                position_k = initial_pos + k * stepSize
                                stepCounter += 1
                                break
                            if k == steps and k_flipFlag == 1:
                                k_flipFlag = -k_flipFlag
                                position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                                print(i, j, k)
                                axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                                axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                                axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                                time.sleep(0.2)
                                new_msg = f"V {position_i}, {position_j}, {position_k}\n"
                                self.update_msg(new_msg)
                                position_k = initial_pos + k * stepSize
                                stepCounter += 1
                                break

                            position_matrix[:, stepCounter] = [position_i, position_j, position_k]
                            print(i, j, k)
                            axis_1.move_absolute(position_i, Units.LENGTH_MILLIMETRES, True)
                            axis_2.move_absolute(position_j, Units.LENGTH_MILLIMETRES, True)
                            axis_3.move_absolute(position_k, Units.LENGTH_MILLIMETRES, True)
                            new_msg = f"V {position_i}, {position_j}, {position_k}\n"
                            self.update_msg(new_msg)
                            time.sleep(0.2)
                            position_k = initial_pos + k * stepSize
                            k = k + k_flipFlag
                            stepCounter += 1
                        if j == 0 and j_flipFlag == -1:
                            j_flipFlag = -j_flipFlag
                            position_j = initial_pos + j * stepSize
                            break
                        if j == steps and j_flipFlag == 1:
                            j_flipFlag = -j_flipFlag
                            position_j = initial_pos + j * stepSize
                            break
                        j = j + j_flipFlag
                        position_j = initial_pos + j * stepSize
                    position_i = initial_pos + i * stepSize
                    i = i + 1
            
            axis_1.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_2.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            axis_3.move_absolute(initial_pos, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.2)
            exit_msg = "E\n"
            self.update_msg(exit_msg)
            
        except Exception as exception:
            connection.close()
            raise exception
            
        connection.close()

    def run(self):
        """
        Run the durability test
        """
        # Start the server in a separate thread
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()

        # Cover the whole workspace
        self.move()


    def start_server(self, host="127.0.0.1", port=12345):
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
                if self.get_msg() == "E\n":
                    break

                # Send a message to the server
                msg = self.get_msg()
                s.sendall(msg.encode())
                time.sleep(0.1)  # Add small delay between sends

        except ConnectionRefusedError:
            print("Could not connect to server. Is it running?")
        except Exception as e:
            print(f"Error in sim_server: {e}")
        finally:
            s.close()

    def update_msg(self, msg):
        self.msg = msg

    
