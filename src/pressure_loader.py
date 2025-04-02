import socket
import struct
import threading
import time
import src.config as config
from zaber_motion import Units
from zaber_motion.ascii import Connection


class PressureLoader:
    def __init__(self):
        self.p_des = config.init_pressure
        self.pressure_values = [0.0, 0.0, 0.0]
        self.increment = 0.1

        print("Initializing motors...")
        # Open connection on COM3
        connection = Connection.open_serial_port('COM3')
        connection.enable_alerts()

        # connection.enableAlerts()  # (commented out as in MATLAB)
        device_list = connection.detect_devices()
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

        input("Press Enter to move each axis unitl the desired pressure is reached...")
        self.load_pressure()
        print("Pressure loaded.")


    def get_pressure_values(self):
        return self.pressure_values

    def listen_pressure_udp(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((config.UDP_IP, config.UDP_PORT))

        print(f"Listening on {config.UDP_IP}:{config.UDP_PORT}")
        try:
            while True:
                data, addr = sock.recvfrom(1024)  # Adjust buffer size if necessary
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
            sock.close()

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
            time.sleep(2)
            offset_1 += self.increment
            print(f"Axis 1: {config.initial_pos + offset_1} mm")
            print(f"Axis 1 pressure: {self.pressure_values[0]} bar")
            if offset_1 > config.max_stroke:
                print("Max stroke reached for axis 1. Stopping.")
                break
        print(f"Axis 1 absolute position: {config.initial_pos + offset_1} mm")
            
        while self.pressure_values[1] < self.p_des:
            self.axis_2.move_absolute(config.initial_pos + offset_2, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            offset_2 += self.increment
            print(f"Axis 2: {config.initial_pos + offset_2} mm")
            print(f"Axis 2 pressure: {self.pressure_values[1]} bar")
            if offset_2 > config.max_stroke:
                print("Max stroke reached for axis 2. Stopping.")
                break
            print(f"Axis 2: {config.initial_pos + offset_2} mm")
            
        while self.pressure_values[2] < self.p_des:
            self.axis_3.move_absolute(config.initial_pos + offset_3, Units.LENGTH_MILLIMETRES, False)
            time.sleep(1)
            offset_3 += self.increment
            print(f"Axis 3: {config.initial_pos + offset_3} mm")
            print(f"Axis 3 pressure: {self.pressure_values[2]} bar")
            if offset_3 > config.max_stroke:
                print("Max stroke reached for axis 3. Stopping.")
                break
            print(f"Axis 3: {config.initial_pos + offset_3} mm")
