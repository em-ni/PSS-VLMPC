from stable_baselines3 import A2C, PPO
from sb3_contrib import TRPO
from stable_baselines3.common.callbacks import CheckpointCallback
import threading
import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import src.config as config
from src.robot_env import RobotEnv
from src.rl_train_monitor import RobotTrainingMonitor
from src.sim_robot_env import SimRobotEnv
from src.mpc_sim_env import MpcSimEnv
from utils.circle_arc import calculate_circle_through_points
import argparse
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

class FixedTRPOPolicy(ActorCriticPolicy):
    """Custom policy class that handles device correctly for TRPO"""
    def __init__(self, *args, **kwargs):
        # Remove device from kwargs before passing to parent
        if 'device' in kwargs:
            self._device = kwargs.pop('device')
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self):
        # This is the correct way to initialize the MLP extractor in SB3
        from stable_baselines3.common.torch_layers import MlpExtractor
        
        # Force the device to CPU for initialization
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device='cpu'  # Force CPU initialization
        )
        # Move to the correct device after initialization
        if hasattr(self, '_device') and torch.cuda.is_available():
            self.mlp_extractor = self.mlp_extractor.to(self._device)

# Check if CUDA is available for RL training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"RL training will use device: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RL agent for soft robot (real or simulated)")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, 
                        help="Run in training or testing mode")
    parser.add_argument("--model_path", type=str, 
                        help="Path to the trained policy model (required for test mode, optional for train mode)")
    parser.add_argument("--sim", action="store_true", 
                        help="Use simulated environment instead of real robot")
    # parser.add_argument("--nn_path", type=str,
    #                     help="Path to the neural network model (required when --sim is used)")
    # parser.add_argument("--csv_path", type=str,
    #                     help="Path to the CSV file with collected data")
    args = parser.parse_args()

    # Validate that model_path is provided when mode is test
    if args.mode == "test" and not args.model_path:
        parser.error("--model_path is required when mode is test")
        
    # # Validate that nn_path is provided when sim is used
    # if args.sim and not args.nn_path:
    #     parser.error("--nn_path is required when using simulation (--sim)")

    if args.sim:
        # Create a simulated environment with the neural network model
        print("Using simulated environment")
        # env = SimRobotEnv()
        env = MpcSimEnv()

    else:
        # Create a real robot environment.
        print("Using real robot environment")
        # Check if the environment is valid
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
        # Determine algorithm type
        algorithm = "TRPO"
        # algorithm = "PPO" 

        model_trpo = TRPO(
            policy=FixedTRPOPolicy,  # Keep using our custom policy class
            env=env, 
            policy_kwargs=dict(
                # Much larger network architecture - adjust both pi (policy) and vf (value function) networks
                net_arch=dict(
                    # Policy network - 5 layers with increasing capacity
                    pi=[512, 512, 384, 256, 128],
                    # Value network - 5 layers with increasing capacity
                    vf=[512, 512, 384, 256, 128]
                ),
                activation_fn=torch.nn.ReLU
            ),
            # Keep these parameters the same
            cg_max_steps=20,        
            cg_damping=0.12,        
            target_kl=0.015,        
            verbose=1
        )
        
        # Detect algorithm from model path if provided
        if args.model_path:
            if "trpo" in args.model_path.lower():
                algorithm = "TRPO"
        
        # Initialize the model or load from checkpoint if provided
        if args.model_path and os.path.exists(args.model_path):
            print(f"Loading {algorithm} model from {args.model_path} to continue training...")
            try:
                if algorithm == "TRPO":
                    model = TRPO.load(args.model_path, env=env, device=device)
                else:  # PPO
                    model = PPO.load(args.model_path, env=env, device=device)
                print(f"Successfully loaded {algorithm} model for continued training on {device}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Creating new {algorithm} model instead...")
                if algorithm == "TRPO":
                    # First, create the model WITHOUT specifying device
                    model = model_trpo
                    
                    # Then manually move policy to GPU after initialization
                    if torch.cuda.is_available():
                        # Force load policy to GPU
                        print("Moving TRPO policy to CUDA...")
                        model.policy.to(torch.device("cuda"))
                        model.device = torch.device("cuda")
                        
                        # Verify device placement
                        print(f"Policy device: {next(model.policy.parameters()).device}")
                        
                        # Monitor GPU memory usage
                        t = torch.cuda.get_device_properties(0).total_memory
                        r = torch.cuda.memory_reserved(0)
                        a = torch.cuda.memory_allocated(0)
                        f = r-a  # free inside reserved
                        print(f"GPU Memory: {a/1024**2:.1f}MB allocated, {f/1024**2:.1f}MB free, {r/1024**2:.1f}MB reserved, {t/1024**2:.1f}MB total")
                else:  # PPO
                    model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64, 
                               n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1, device=device)
        else:
            print(f"Starting new training run with {algorithm}...")
            if algorithm == "TRPO":
                # First, create the model WITHOUT specifying device
                model = model_trpo
                
                # Then manually move policy to GPU after initialization
                if torch.cuda.is_available():
                    # Force load policy to GPU
                    print("Moving TRPO policy to CUDA...")
                    model.policy.to(torch.device("cuda"))
                    model.device = torch.device("cuda")
                    
                    # Verify device placement
                    print(f"Policy device: {next(model.policy.parameters()).device}")
                    
                    # Monitor GPU memory usage
                    t = torch.cuda.get_device_properties(0).total_memory
                    r = torch.cuda.memory_reserved(0)
                    a = torch.cuda.memory_allocated(0)
                    f = r-a  # free inside reserved
                    print(f"GPU Memory: {a/1024**2:.1f}MB allocated, {f/1024**2:.1f}MB free, {r/1024**2:.1f}MB reserved, {t/1024**2:.1f}MB total")
            else:  # PPO
                model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=2048, batch_size=64, 
                           n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=1, device=device)
        
        
        # Create directory for saving checkpoints
        checkpoint_dir = os.path.join(config.data_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create the checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=100000, 
            save_path=checkpoint_dir,
            name_prefix="robot_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        
        # Create the metrics callback
        metrics_callback = RobotTrainingMonitor()
        
        # Use both callbacks during training
        model.learn(total_timesteps=10000000, callback=[metrics_callback, checkpoint_callback])
        
        print("\n--- Training complete. Starting evaluation ---")

        # Evaluate after training (with the trained model)
        obs, _ = env.reset()
        total_reward = 0
        eval_episodes = 100
        
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
        model.save(os.path.join(config.data_dir, "policy", "trained_policy.zip"))
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
        if args.sim:
            train(env=env)
        else:
            # Start the RL training and evaluation in a separate thread.
            rl_thread = threading.Thread(target=train, args=(env,))
            rl_thread.daemon = True
            rl_thread.start()
    elif args.mode == "test":
        # Start the RL testing in a separate thread.
        rl_thread = threading.Thread(target=test, args=(env,))
        rl_thread.daemon = True
        rl_thread.start()

    if not args.sim:
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

        # Add this outside the animation function (run once)
        def plot_hull_wireframe(hull, ax, color='lightgreen', alpha=0.1):
            # Plot the hull wireframe
            for simplex in hull.simplices:
                pts = hull.points[simplex]
                # Draw just the edges for better performance
                for i in range(3):
                    xi, yi, zi = pts[i]
                    xj, yj, zj = pts[(i+1)%3]
                    line, = ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, alpha=alpha)
        
        # Animation update function.
        def animate(frame, env=env):
            try:
                base = env.robot_api.get_tracker().get_current_base()
                tip = env.robot_api.get_tracker().get_current_tip()
                body = env.robot_api.get_tracker().get_current_body()
                
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

                # Update the goal coordinates
                goal_coordinates = env.get_goal()

                # Plot the base at the origin (yellow) and the tip difference (red).
                base_scatter._offsets3d = ([0], [0], [0])
                tip_scatter._offsets3d = (dif_x, dif_y, dif_z)
                body_scatter._offsets3d = (body_dif_x, body_dif_y, body_dif_z)
                goal_scatter._offsets3d = ([goal_coordinates[0]], [goal_coordinates[1]], [goal_coordinates[2]])

                # Clear previous circle line if it exists, BUT NOT HULL LINES
                for line in list(ax.get_lines()):
                    if not hasattr(line, 'is_hull_line'):  # Only remove non-hull lines
                        line.remove()
                
                # Draw circle if we have all three points
                if base is not None and tip is not None and body is not None:
                    circle_points = calculate_circle_through_points(body-base, tip-base, [0,0,0])
                    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
                            label="Circle", color="blue", linewidth=5)
                    
                # Periodically perform garbage collection (every 50 frames)
                if frame % 50 == 0:
                    import gc
                    gc.collect()

                # plot_hull_wireframe(env.convex_hull, ax) # too slow

                return base_scatter, tip_scatter, body_scatter, goal_scatter
            
            except Exception as e:
                print(f"Animation error: {e}")
                return base_scatter, tip_scatter, body_scatter, goal_scatter

        # Create the animation. Adjust the interval as needed.
        anim = animation.FuncAnimation(fig, animate, cache_frame_data=False, fargs=(env,), interval=50)
        plt.show()
