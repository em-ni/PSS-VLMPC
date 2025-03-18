import threading
import time
import signal
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import matplotlib.animation as animation
from src.tracker import Tracker
import src.config as config

# Global stop event to allow threads to exit cleanly
stop_event = threading.Event()

# Signal handler to allow graceful exit on Ctrl+C
def signal_handler(sig, _frame):
    print("Exiting... Signal:", sig)
    stop_event.set()  # Signal threads to stop
    time.sleep(0.5)   # Allow some time for cleanup
    sys.exit(0)

# Function to update tracker data in a separate thread.
def update_tracker(tracker):
    while not stop_event.is_set():
        tracker.real_time_tracking()
        time.sleep(1)

if __name__ == "__main__":
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Instantiate tracker
    tracker = Tracker(config.experiment_name, config.save_dir, config.csv_path)

    # Start tracker update in background thread.
    tracker_thread = threading.Thread(target=update_tracker, args=(tracker,))
    tracker_thread.daemon = True
    tracker_thread.start()

    # Set up the figure and 3D axis.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("3D Real-Time Tracking")

    # Create empty scatter plots for the base (yellow) and tip (red).
    base_scatter = ax.scatter([], [], [], s=10, c="yellow")
    tip_scatter = ax.scatter([], [], [], s=10, c="red")

    # Animation update function.
    def animate(frame):
        base = tracker.get_current_base()
        tip = tracker.get_current_tip()
        
        # Prepare data lists for base and tip.
        base_x, base_y, base_z = [], [], []
        tip_x, tip_y, tip_z = [], [], []
        
        if base is not None:
            base = base.ravel()
            base_x = [base[0]]
            base_y = [base[1]]
            base_z = [base[2]]
        if tip is not None:
            tip = tip.ravel()
            tip_x = [tip[0]]
            tip_y = [tip[1]]
            tip_z = [tip[2]]

        # Compute the difference between tip and base if both exist.
        if base_x and base_y and base_z and tip_x and tip_y and tip_z:
            dif_x = [tip_x[0] - base_x[0]]
            dif_y = [tip_y[0] - base_y[0]]
            dif_z = [tip_z[0] - base_z[0]]
        else:
            dif_x, dif_y, dif_z = [], [], []

        # Plot the base at the origin (yellow) and the tip difference (red).
        base_scatter._offsets3d = ([0], [0], [0])
        tip_scatter._offsets3d = (dif_x, dif_y, dif_z)
        return base_scatter, tip_scatter

    # Create the animation. Adjust the interval as needed.
    anim = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)

    plt.show()
