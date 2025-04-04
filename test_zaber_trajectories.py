import numpy as np
import src.config as config
from zaber_motion.ascii import Connection
from zaber_motion import Units
import time

def run():
    NUMBER_OF_TRAJECTORIES = 2
    way_points = []
    # Open connection on COM3
    connection = Connection.open_serial_port('COM3')
    connection.enable_alerts()
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    print(device_list)
    scope_1 = device_list[0].oscilloscope
    print(f'Oscilloscope can store {scope_1.get_max_buffer_size()} samples.')
    scope_1.clear()
    scope_1.add_channel(1, 'pos')
    scope_1.set_timebase(10, Units.TIME_MILLISECONDS)
    scope_1.set_delay(0)
    
    # Get the axis
    axis_1 = device_list[0].get_axis(1)
    axis_2 = device_list[1].get_axis(1)
    axis_3 = device_list[2].get_axis(1)
    print('Starting experiment')
    scope_1.start()
    time.sleep(5)  # Wait for the oscilloscope to start
    for i in range(NUMBER_OF_TRAJECTORIES):
        next_points = np.random.uniform(0, 3*config.max_stroke, size=(3,))
        way_points.append(config.initial_pos + next_points[0])
        print(f"Trajectory {i+1}:")
        print(f"Next points: {next_points}")

        # Move the motors to the next points
        axis_1.move_absolute(config.initial_pos + next_points[0], Units.LENGTH_MILLIMETRES, False)
        axis_2.move_absolute(config.initial_pos + next_points[1], Units.LENGTH_MILLIMETRES, False)
        axis_3.move_absolute(config.initial_pos + next_points[2], Units.LENGTH_MILLIMETRES, True)
        time.sleep(1)  
        print("Motors moved to next points.")

    # Get readings from the oscilloscope
    data = scope_1.read()
    print('Writing results')
    pos = data[0]
    pos_samples = pos.get_data(Units.LENGTH_MILLIMETRES)
    with open('scope.csv', 'wt') as file:
        file.write('Time (ms),Trajectory Position (mm),Measured Position (mm)\n')
        for i in range(len(pos_samples)):
            file.write(f'{pos.get_sample_time(i, Units.TIME_MILLISECONDS)},')
            file.write(f'{pos_samples[i]}\n')

    # Plot the results
    import matplotlib.pyplot as plt
    # Plot the position data
    plt.figure(figsize=(12, 10))
    
    # First subplot: Position vs Time
    plt.subplot(2, 1, 1)
    plt.plot(pos.get_sample_times(Units.TIME_MILLISECONDS), pos_samples, label='Position')
    plt.xlabel('Time (ms)')
    plt.ylabel('Position (mm)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid()
    
    # Second subplot: Waypoints
    plt.subplot(2, 1, 2)
    
    # Plot waypoints with connecting lines
    plt.plot(range(len(way_points)), way_points, 'ro-', label='Way points')

    plt.xlabel('Waypoint Number')
    plt.ylabel('Position (mm)')
    plt.title('Waypoints')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

    # Close the connection
    connection.close()
    print("Connection closed.")

if __name__ == "__main__":
    run()
