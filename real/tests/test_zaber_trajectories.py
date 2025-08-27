import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from zaber_motion.ascii import Connection
from zaber_motion import Units
import time

def run():
    NUMBER_OF_TRAJECTORIES = 5
    way_points = []

    # Open connection on COM3
    connection = Connection.open_serial_port('COM3')
    connection.enable_alerts()
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    print(device_list)
    scope_1 = device_list[0].oscilloscope
    print(f'Oscilloscope 1 can store {scope_1.get_max_buffer_size()} samples.')
    scope_1.clear()
    scope_1.add_channel(1, 'pos')
    scope_1.add_channel(1, 'encoder.pos')
    scope_1.set_timebase(10, Units.TIME_MILLISECONDS)
    scope_1.set_delay(10, Units.TIME_MILLISECONDS)

    scope_2 = device_list[1].oscilloscope
    print(f'Oscilloscope 2 can store {scope_2.get_max_buffer_size()} samples.')
    scope_2.clear()
    scope_2.add_channel(1, 'pos')
    scope_2.add_channel(1, 'encoder.pos')
    scope_2.set_timebase(10, Units.TIME_MILLISECONDS)
    scope_2.set_delay(10, Units.TIME_MILLISECONDS)

    scope_3 = device_list[2].oscilloscope
    print(f'Oscilloscope 3 can store {scope_3.get_max_buffer_size()} samples.')
    scope_3.clear()
    scope_3.add_channel(1, 'pos')
    scope_3.add_channel(1, 'encoder.pos')
    scope_3.set_timebase(10, Units.TIME_MILLISECONDS)
    scope_3.set_delay(10, Units.TIME_MILLISECONDS)
    
    # Get the axis
    axis_1 = device_list[0].get_axis(1)
    axis_2 = device_list[1].get_axis(1)
    axis_3 = device_list[2].get_axis(1)
    print('Starting experiment')
    scope_1.start()
    scope_2.start()
    scope_3.start()
    time.sleep(0.01)  # Wait for the oscilloscopes to start
    for i in range(NUMBER_OF_TRAJECTORIES):
        next_points = np.random.uniform(0, 3*config.max_stroke, size=(3,))
        way_points.append(config.initial_pos + next_points[0])
        print(f"Trajectory {i+1}:")
        print(f"Next points: {next_points}")

        # Move the motors to the next points
        axis_1.move_absolute(config.initial_pos + next_points[0], Units.LENGTH_MILLIMETRES, False)
        axis_2.move_absolute(config.initial_pos + next_points[1], Units.LENGTH_MILLIMETRES, False)
        axis_3.move_absolute(config.initial_pos + next_points[2], Units.LENGTH_MILLIMETRES, True)
        axis_1.wait_until_idle()
        axis_2.wait_until_idle()    
        axis_3.wait_until_idle()
        print("Motors moved to next points.")


    # Get readings from the oscilloscopes
    data1 = scope_1.read()
    data2 = scope_2.read()
    data3 = scope_3.read()
    print('Writing results')

    pos1 = data1[0]
    pos_samples1 = pos1.get_data(Units.LENGTH_MILLIMETRES)
    encoder1 = data1[1]
    encoder_samples1 = encoder1.get_data(Units.LENGTH_MILLIMETRES)

    pos2 = data2[0]
    pos_samples2 = pos2.get_data(Units.LENGTH_MILLIMETRES)
    encoder2 = data2[1]
    encoder_samples2 = encoder2.get_data(Units.LENGTH_MILLIMETRES)

    pos3 = data3[0]
    pos_samples3 = pos3.get_data(Units.LENGTH_MILLIMETRES)
    encoder3 = data3[1]
    encoder_samples3 = encoder3.get_data(Units.LENGTH_MILLIMETRES)

    with open('scope.csv', 'wt') as file:
        file.write('Time (ms),Trajectory Position 1 (mm),Measured Position 1 (mm),Trajectory Position 2 (mm),Measured Position 2 (mm),Trajectory Position 3 (mm),Measured Position 3 (mm)\n')
        for i in range(min(len(pos_samples1), len(pos_samples2), len(pos_samples3))):
            file.write(f'{pos1.get_sample_time(i, Units.TIME_MILLISECONDS)},')
            file.write(f'{pos_samples1[i]},{encoder_samples1[i]},')
            file.write(f'{pos_samples2[i]},{encoder_samples2[i]},')
            file.write(f'{pos_samples3[i]},{encoder_samples3[i]}\n')

    # Plot the results
    import matplotlib.pyplot as plt
    # Plot the position data
    plt.figure(figsize=(12, 12))

    # First subplot: Position vs Time for all scopes
    plt.subplot(2, 1, 1)
    plt.plot(pos1.get_sample_times(Units.TIME_MILLISECONDS), pos_samples1, label='Position 1')
    plt.plot(pos2.get_sample_times(Units.TIME_MILLISECONDS), pos_samples2, label='Position 2')
    plt.plot(pos3.get_sample_times(Units.TIME_MILLISECONDS), pos_samples3, label='Position 3')
    plt.xlabel('Time (ms)')
    plt.ylabel('Position (mm)')
    plt.title('Position vs Time (All Scopes)')
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
