from gymnasium.utils.env_checker import check_env
from stable_baselines3 import A2C, PPO
import threading
import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import signal
import src.config as config
from src.robot_env import RobotEnv
from src.custom_policy import CustomPolicy
from src.robot_train_monitor import RobotTrainingMonitor
from utils.circle_arc import calculate_circle_through_points
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="RL agent for real soft robot")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, 
                        help="Run in training or testing mode")
    parser.add_argument("--model_path", type=str, 
                        help="Path to the trained policy model (required for test mode)")
    args = parser.parse_args()

    # Validate that model_path is provided when mode is test
    if args.mode == "test" and not args.model_path:
        parser.error("--model_path is required when mode is test")

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
    def train(env=env):
        # Initialize the model
        # model = PPO('MlpPolicy', env, verbose=1)
        model = A2C("MlpPolicy", env, learning_rate=0.0005, ent_coef=0.3, verbose=1)
        # model = A2C(CustomPolicy, env, verbose=1)

        # Create the metrics callback
        metrics_callback = RobotTrainingMonitor()
        
        # Train with callback
        model.learn(total_timesteps=10000, callback=metrics_callback)
        print("\n--- Training complete. Starting evaluation ---")

        # Evaluate after training (with the trained model)
        obs, _ = env.reset()
        total_reward = 0
        eval_episodes = 5
        
        for i in range(eval_episodes):
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                env.render()
            
            obs, _ = env.reset()
            total_reward += episode_reward
            
            print(f"Eval episode {i+1}: Reward = {episode_reward:.4f}")
        
        print(f"\nEvaluation results:")
        print(f"Average reward: {total_reward/eval_episodes:.4f}")

        # Save the trained policy to a file.
        model.save(os.path.join(config.data_dir, "trained_policy.zip"))
        os.kill(os.getpid(), signal.SIGTERM)
        return model
    
    # Function to run RL testing.
    def test(env):
        """Load and test a pre-trained RL policy."""
        # Load the trained policy
        try:
            model = A2C.load(args.model_path)
            print("Loaded A2C policy")
        except Exception as e:
            print(f"Failed to load policy: {e}")
            return
        
        # Run the policy in a continuous loop
        episode_count = 0
        
        while True:
            episode_count += 1
            print(f"Starting episode {episode_count}")
            
            # Handle different reset return types
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]
            else:
                obs = reset_result
            
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Apply action to environment
                step_result = env.step(action)
                
                # Handle different return types
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                
                # Break if episode takes too long
                if step_count > 1000:
                    print("Episode taking too long, resetting")
                    break
            
            print(f"Episode {episode_count} finished with reward {episode_reward} after {step_count} steps")

    if args.mode == "train":
        # Start the RL training and evaluation in a separate thread.
        rl_thread = threading.Thread(target=train, args=(env,))
        rl_thread.daemon = True
        rl_thread.start()
    elif args.mode == "test":
        # Start the RL testing in a separate thread.
        rl_thread = threading.Thread(target=test, args=(env,))
        rl_thread.daemon = True
        rl_thread.start()

    # Set up the figure and 3D axis for the real-time plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_title("3D Real-Time Tracking")
    ax.view_init(elev=-50, azim=-100, roll=-80)

    # Create empty scatter plots for the base (yellow) and tip difference (red).
    base_scatter = ax.scatter([], [], [], s=40, c="yellow")
    tip_scatter = ax.scatter([], [], [], s=60, c="red")
    goal_coordinates = env.get_goal()
    goal_scatter = ax.scatter([goal_coordinates[0]], [goal_coordinates[1]], [goal_coordinates[2]], s=60, c="green")
    body_scatter = ax.scatter([], [], [], s=5, c="blue")

    # Animation update function.
    def animate(frame, plot_tracker=env.robot_api.get_tracker()):
        base = plot_tracker.get_current_base()
        tip = plot_tracker.get_current_tip()
        body = plot_tracker.get_current_body()
        
        base_x, base_y, base_z = [], [], []
        tip_x, tip_y, tip_z = [], [], []
        body_x, body_y, body_z = [], [], []
        
        if base is not None:
            base = base.ravel()
            base_x, base_y, base_z = [base[0]], [base[1]], [base[2]]
        if tip is not None:
            tip = tip.ravel()
            tip_x, tip_y, tip_z = [[x] for x in tip]
        if body is not None:
            body = body.ravel()
            body_x, body_y, body_z = [body[0]], [body[1]], [body[2]]
        
        if base_x and base_y and base_z and tip_x and tip_y and tip_z:
            dif_x, dif_y, dif_z = [tip_x[0] - base_x[0]], [tip_y[0] - base_y[0]], [tip_z[0] - base_z[0]]
        else:
            dif_x, dif_y, dif_z = [], [], []

        if body_x and body_y and body_z:
            body_dif_x, body_dif_y, body_dif_z = [body_x[0]-base_x[0]], [body_y[0]-base_y[0]], [body_z[0]-base_z[0]]
        else:
            body_dif_x, body_dif_y, body_dif_z = [], [], []

        # Plot the base at the origin (yellow) and the tip difference (red).
        base_scatter._offsets3d = ([0], [0], [0])
        tip_scatter._offsets3d = (dif_x, dif_y, dif_z)
        body_scatter._offsets3d = (body_dif_x, body_dif_y, body_dif_z)

        # Clear previous circle line if it exists
        for line in ax.get_lines():
            line.remove()
        
        # Draw circle if we have all three points
        if base is not None and tip is not None and body is not None:
            circle_points = calculate_circle_through_points(body-base, tip-base, [0,0,0])
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
                    label="Circle", color="blue", linewidth=5)
        return base_scatter, tip_scatter, body_scatter

    # Create the animation. Adjust the interval as needed.
    anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, fargs=(env.robot_api.get_tracker(),), interval=50)
    plt.show()
