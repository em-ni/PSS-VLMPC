import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from src.tracker import Tracker
import src.config as config
from zaber_motion import Units
from zaber_motion.ascii import Connection
import threading
import time
import signal
import sys

# Robot API for real-time interaction.
class RealRobotAPI:
    def __init__(self):
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

        input("Press Enter to move each axis to the initial position...")
        self.reset_robot()
        print("Motors initialized.")

        print("Initializing tracker...")
        self.tracker = Tracker(config.experiment_name, config.save_dir, config.csv_path)
        print("Tracker initialized.")

        # Global stop event to allow threads to exit cleanly
        self.stop_event = threading.Event()

        print("Robot API initialized.\n")
    
    def send_command(self, action):
        # Move each axis by the action vector
        self.axis_1.move_absolute(config.initial_pos + float(action[0]), Units.LENGTH_MILLIMETRES, True)
        self.axis_2.move_absolute(config.initial_pos + float(action[1]), Units.LENGTH_MILLIMETRES, True)
        self.axis_3.move_absolute(config.initial_pos + float(action[2]), Units.LENGTH_MILLIMETRES, True)
        time.sleep(0.1)
    
    def get_current_tip(self):
        tip = self.tracker.get_current_tip()
        base = self.tracker.get_current_base()
        if tip is None or base is None:
            print("Tracker data not available, returning default zero vector.")
            return np.zeros(3, dtype=np.float32)
        # Convert to numpy arrays and subtract
        dif = np.array(tip) - np.array(base)
        # Flatten the result to match shape (3,)
        dif = np.squeeze(dif)
        return dif.astype(np.float32)

    def get_signal_handler(self):
        return self.signal_handler
    
    def get_tracker(self):
        return self.tracker

    def reset_robot(self):
        # Move each axis to the initial position
        self.axis_1.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_2.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        self.axis_3.move_absolute(config.initial_pos, Units.LENGTH_MILLIMETRES, False)
        time.sleep(1)

    # Signal handler to allow graceful exit on Ctrl+C
    def signal_handler(self, sig, _frame):
        print("Exiting... Signal:", sig)
        self.stop_event.set()  # Signal threads to stop
        time.sleep(0.5)   # Allow some time for cleanup
        sys.exit(0)

    # Function to update tracker data in a separate thread.
    def update_tracker(self, tracker):
        while not self.stop_event.is_set():
            self.tracker.real_time_tracking()
            time.sleep(1)


class RobotEnv(gym.Env):
    """
    A real robot environment.
    Observation: concatenated 3D coordinates of the tip wrt to the base (shape: 3,)
    Action: a 3D discrete vector (each element is an integer from 0 to config.max_stroke, inclusive)
    Reward: negative Euclidean distance from the tip to the goal.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobotEnv, self).__init__()
        # Change action space to a discrete multi-dimensional space.
        self.action_space = spaces.MultiDiscrete([config.max_stroke + 1] * 3)
        # Observation space remains the same.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        
        self.tip = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.goal = np.array([2.5, -1.1, 0.0], dtype=np.float32)
        self.max_steps = 200
        self.current_step = 0

        self.robot_api = RealRobotAPI()

    def step(self, action):
        self.robot_api.send_command(action)
        self.tip = self.robot_api.get_current_tip()
        distance = np.linalg.norm(self.tip - self.goal)
        reward = -distance
        terminated = False
        self.current_step += 1
        if distance < 1:
            terminated = True
        # print(f"Step {self.current_step}: Terminated {terminated} Distance {distance} ", end="\r", flush=True)
        print(f"Step {self.current_step}: Terminated {terminated} Distance {distance} ")
        truncated = self.current_step >= self.max_steps  # episode timeout
        info = {}
        return self.tip, reward, terminated, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("Resetting environment...")
        super().reset(seed=seed)
        self.robot_api.reset_robot()
        self.tip = self.robot_api.get_current_tip()
        self.current_step = 0
        return self.tip, {}

    def render(self, mode='human'):
        pass


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import matplotlib.animation as animation
import src.config as config


if __name__ == '__main__':
    # Create a real robot environment.
    env = RobotEnv()
    signal_handler = env.robot_api.get_signal_handler()

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Start a background thread to update tracker data for real-time plotting.
    tracker_thread = threading.Thread(target=env.robot_api.update_tracker, args=(env.robot_api.get_tracker(),))
    tracker_thread.daemon = True
    tracker_thread.start()

    # Function to run RL training and evaluation.
    def run_rl(env=env, reward_threshold=-0.5, success_rate=0.9, patience=10):
        # Initialize the model
        model = PPO('MlpPolicy', env, verbose=1)
        
        # Train the model (this handles all the episodes internally)
        print("Starting training...")
        model.learn(total_timesteps=5)  # Adjust timesteps as needed
        print("Training complete")
        
        # Save the trained model
        model.save("ppo_robot_model")
        
        # Evaluate the trained model
        print("\nEvaluating trained policy...")
        recent_rewards = []
        success_count = 0
        eval_episodes = 2
        
        for episode in range(eval_episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Use deterministic=True for evaluation
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
            recent_rewards.append(total_reward)
            if terminated:  # If we reached the goal
                success_count += 1
                
            print(f"Eval episode {episode+1}: Total reward = {total_reward:.2f}, Success = {terminated}")
        
        # Calculate statistics
        avg_reward = np.mean(recent_rewards)
        success_ratio = success_count / eval_episodes
        
        print(f"\nEvaluation results:")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {success_ratio:.2f}")
        
        return model


    # Start the RL training and evaluation in a separate thread.
    rl_thread = threading.Thread(target=run_rl, args=(env,))
    rl_thread.daemon = True
    rl_thread.start()

    # Set up the figure and 3D axis for the real-time plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("3D Real-Time Tracking")
    ax.view_init(elev=-50, azim=-100, roll=-80)

    # Create empty scatter plots for the base (yellow) and tip difference (red).
    base_scatter = ax.scatter([], [], [], s=10, c="yellow")
    tip_scatter = ax.scatter([], [], [], s=10, c="red")
    goal_scatter = ax.scatter([2.5], [-1.1], [0], s=10, c="green")

    # Animation update function.
    def animate(frame, plot_tracker=env.robot_api.get_tracker()):
        base = plot_tracker.get_current_base()
        tip = plot_tracker.get_current_tip()
        
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
    anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, fargs=(env.robot_api.get_tracker(),), interval=50)
    plt.show()
