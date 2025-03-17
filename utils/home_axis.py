
from zaber_motion import Units
from zaber_motion.ascii import Connection

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

# Home each axis to home
base_rightaxis_1.home()
base_rightaxis_2.home()
base_rightaxis_3.home()

