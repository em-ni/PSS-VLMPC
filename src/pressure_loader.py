import os
import socket
import struct
import sys
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from zaber_motion import Units
from zaber_motion.ascii import Connection


class PressureLoader:
    def __init__(self, save_offsets=False):
        self.p_des = config.init_pressure
        self.pressure_values = [0.0, 0.0, 0.0]
        self.increment = 0.1

        self.listen = True
        self.sock = None
        self.connection = None
        self.save_offsets = save_offsets

        print("Initializing motors...")
        # Open connection on COM3
        self.connection = Connection.open_serial_port('COM3')
        self.connection.enable_alerts()

        # connection.enableAlerts()  # (commented out as in MATLAB)
        device_list = self.connection.detect_devices()
        print("Found {} devices.".format(len(device_list)))
        print(device_list)
        
        # Get the axis
        self.axis_1 = device_list[0].get_axis(1)
        self.axis_2 = device_list[1].get_axis(1)
        self.axis_3 = device_list[2].get_axis(1)

        # Home each axis if home_first is True
        if config.home_first: 
            self.axis_1.home()
            self.axis_2.home()
            self.axis_3.home()

        # Move each axis to the initial position
        self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        print("Motors initialized and moved to initial position.")
        time.sleep(1)

    def get_pressure_values(self):
        return self.pressure_values

    def listen_pressure_udp(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((config.UDP_IP, config.UDP_PRESSURE_PORT))

        print(f"Listening on {config.UDP_IP}:{config.UDP_PRESSURE_PORT}")
        try:
            while self.listen:
                data, addr = self.sock.recvfrom(1024)  # Adjust buffer size if necessary
                try:
                    # Attempt to decode as a double precision float 
                    values_double = struct.unpack('ddd', data)
                    # print(f"Received double: {values_double[0]} from {addr}")
                    # print(f"Received double: {values_double[1]} from {addr}")
                    # print(f"Received double: {values_double[2]} from {addr}")
                    cur_pressure_1 = values_double[0]
                    cur_pressure_2 = values_double[1]
                    cur_pressure_3 = values_double[2]
                    cur_pressure_values = [cur_pressure_1, cur_pressure_2, cur_pressure_3]
                    
                    self.set_pressure_values(cur_pressure_values)
                except:
                    print(f"Received unhandled data type: {data} from {addr}")
                    pass

        except KeyboardInterrupt:
            print("Stopping receiver.")
            # self.sock.close()

    def set_pressure_values(self, pressure_values):
        # # Use the mapping to correctly assign pressure values
        # axis_mapping = {
        #     0: 2,  # Index 0 from incoming data goes to axis 2 -> 1
        #     1: 1,  # Index 1 from incoming data goes to axis 1 -> 0
        #     2: 3   # Index 2 from incoming data goes to axis 3 -> 2
        # }

        self.pressure_values[0] = pressure_values[1]
        self.pressure_values[1] = pressure_values[0]
        self.pressure_values[2] = pressure_values[2]

    def load_pressure(self):

        print("IMPORTANT: Be sure MATLAB is running and sending pressure data.\n")
        input("Press Enter to move each axis until the desired pressure is reached...")

        # Start the UDP listener in a separate thread
        udp_thread = threading.Thread(target=self.listen_pressure_udp)
        udp_thread.daemon = True
        udp_thread.start()

        # Bring every motor to the desired pressure
        offset_1 = 0.0
        offset_2 = 0.0
        offset_3 = 0.0
        while self.pressure_values[0] < self.p_des:
            self.axis_1.move_absolute(config.initial_pos + offset_1, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.5)
            offset_1 += self.increment
            print(f"Axis 1 position: {config.initial_pos + offset_1} mm, pressure: {self.pressure_values[0]} bar", end="\r", flush=True)
            if offset_1 > config.max_stroke:
                print("Max stroke reached for axis 1. Stopping.")
                break
        print(f"Axis 1 absolute position: {config.initial_pos + offset_1} mm")
            
        while self.pressure_values[1] < self.p_des:
            self.axis_2.move_absolute(config.initial_pos + offset_2, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.5)
            offset_2 += self.increment
            print(f"Axis 2: {config.initial_pos + offset_2} mm, pressure: {self.pressure_values[1]} bar", end="\r", flush=True)
            if offset_2 > config.max_stroke:
                print("Max stroke reached for axis 2. Stopping.")
                break
        print(f"Axis 2 absolute position: {config.initial_pos + offset_2} mm")
            
        while self.pressure_values[2] < self.p_des:
            self.axis_3.move_absolute(config.initial_pos + offset_3, Units.LENGTH_MILLIMETRES, False)
            time.sleep(0.5)
            offset_3 += self.increment
            print(f"Axis 3: {config.initial_pos + offset_3} mm, pressure: {self.pressure_values[2]} bar", end="\r", flush=True)
            if offset_3 > config.max_stroke:
                print("Max stroke reached for axis 3. Stopping.")
                break
        print(f"Axis 3 absolute position: {config.initial_pos + offset_3} mm")
        
        print(f"Pressure loaded with offsets: {offset_1}, {offset_2}, {offset_3}")

        # Stop listening
        self.listen = False 
        
        # Close the socket to unblock the recvfrom() call
        if self.sock:
            self.sock.close()

        # Stop the UDP listener
        udp_thread.join(timeout=1)  # Wait for the thread to finish (if needed)
        if udp_thread.is_alive():
            print("UDP thread is still running. Stopping it.")
            udp_thread.join()
        print("UDP thread stopped.")

        # Close motors connection
        self.connection.close()
        print("Connection closed.")

        if self.save_offsets:
            # Save offsets to a file
            os.makedirs(config.offsets_path, exist_ok=True)
            with open(os.path.join(config.offsets_path, "offsets.txt"), "w") as f:
                f.write(f"offset_1: {offset_1}\n")
                f.write(f"offset_2: {offset_2}\n")
                f.write(f"offset_3: {offset_3}\n")

        return [offset_1, offset_2, offset_3]

if __name__ == "__main__":
    pressure_loader = PressureLoader()
    offsets = []
    offsets = pressure_loader.load_pressure()