from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("COM3") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    device = device_list[0]

    axis = device.get_axis(1)
    if not axis.is_homed():
      axis.home()

    # Move to 10mm
    axis.move_absolute(10, Units.LENGTH_MILLIMETRES)

    # Move by an additional 5mm
    axis.move_relative(5, Units.LENGTH_MILLIMETRES)


    x_min, x_max = 1, 11  # x from 1 to 100
    y_min, y_max = 1, 11  # y from 1 to 100
    z_min, z_max = 1, 11  # z from 1 to 100

    for i, x in enumerate(range(x_min, x_max)):
        # For y, alternate order based on the x index
        if i % 2 == 0:
            y_iter = range(y_min, y_max)
        else:
            y_iter = range(y_max - 1, y_min - 1, -1)

        for j, y in enumerate(y_iter):
            # For z, you can alternate based on y (or some other logic)
            if j % 2 == 0:
                z_iter = range(z_min, z_max)
            else:
                z_iter = range(z_max - 1, z_min - 1, -1)

            for z in z_iter:
                # Process the (x, y, z) point
                print(f"Processing point: x={x}, y={y}, z={z}")

