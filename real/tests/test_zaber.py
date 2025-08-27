from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("COM3") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    device = device_list[0]

    axis_1 = device.get_axis(1)
    if not axis_1.is_homed():
      axis_1.home()

    axis_2 = device.get_axis(2)
    if not axis_2.is_homed():
      axis_2.home()

    axis_3 = device.get_axis(3)
    if not axis_3.is_homed():
      axis_3.home()

    # Move by an additional 5mm
    # axis_1.move_absolute(v1_min, Units.LENGTH_MILLIMETRES)
    # axis.move_relative(5, Units.LENGTH_MILLIMETRES)
                
