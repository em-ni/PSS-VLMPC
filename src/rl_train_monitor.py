from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import src.config as config

class RobotTrainingMonitor(BaseCallback):
    """
    Custom callback to monitor training metrics and generate plots
    """
    def __init__(self, verbose=1, save_freq=100, plot_freq=1000):
        super(RobotTrainingMonitor, self).__init__(verbose)
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.distances_to_goal = []
        self.current_episode_reward = 0
        self.current_length = 0
        self.last_timesteps = 0
        self.timesteps_since_last_record = 0
        
        # Training progress tracking
        self.start_time = None
        self.timestamps = []
        self.timesteps_history = []
        self.rolling_rewards_10 = []
        self.rolling_rewards_50 = []
        self.rolling_rewards_100 = []
        
        # Settings
        self.save_freq = save_freq  # Episodes between saves
        self.plot_freq = plot_freq  # Episodes between plots
        
        # Output directory setup
        self.output_dir = os.path.join(config.data_dir, "training_metrics", 
                                      f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _on_training_start(self):
        self.start_time = time.time()
        print(f"Training metrics will be saved to: {self.output_dir}")
        
    def _calculate_metrics(self):
        # Calculate rolling averages if we have enough data
        if len(self.episode_rewards) >= 10:
            self.rolling_rewards_10.append(np.mean(self.episode_rewards[-10:]))
        else:
            self.rolling_rewards_10.append(np.mean(self.episode_rewards))
            
        if len(self.episode_rewards) >= 50:
            self.rolling_rewards_50.append(np.mean(self.episode_rewards[-50:]))
        else:
            self.rolling_rewards_50.append(np.mean(self.episode_rewards))
            
        if len(self.episode_rewards) >= 100:
            self.rolling_rewards_100.append(np.mean(self.episode_rewards[-100:]))
        else:
            self.rolling_rewards_100.append(np.mean(self.episode_rewards))
            
        self.timestamps.append(time.time() - self.start_time)
        self.timesteps_history.append(self.num_timesteps)
        
    def _save_metrics(self):
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'length': self.episode_lengths,
            'timesteps': self.timesteps_history,
            'time_elapsed': self.timestamps,
            'rolling_reward_10': self.rolling_rewards_10,
            'rolling_reward_50': self.rolling_rewards_50,
            'rolling_reward_100': self.rolling_rewards_100
        })
        
        metrics_df.to_csv(os.path.join(self.output_dir, 'training_metrics.csv'), index=False)
        
    def _generate_plots(self):
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, label='Episode Reward')
        if len(self.rolling_rewards_10) > 0:
            plt.plot(self.rolling_rewards_10, label='10-ep avg', linewidth=2)
        if len(self.rolling_rewards_50) > 0:
            plt.plot(self.rolling_rewards_50, label='50-ep avg', linewidth=2)
        if len(self.rolling_rewards_100) > 0:
            plt.plot(self.rolling_rewards_100, label='100-ep avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Reward vs Timesteps
        plt.subplot(2, 2, 3)
        plt.plot(self.timesteps_history, self.episode_rewards, alpha=0.5)
        if len(self.rolling_rewards_10) > 0:
            plt.plot(self.timesteps_history, self.rolling_rewards_10, linewidth=2)
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.title('Reward vs Timesteps')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Training Speed
        if len(self.timestamps) > 1:
            plt.subplot(2, 2, 4)
            # Calculate timesteps per second
            timesteps = np.array(self.timesteps_history)
            times = np.array(self.timestamps)
            if len(timesteps) > 10:
                # Use a window to smooth the calculation
                window_size = min(10, len(timesteps)-1)
                steps_per_sec = []
                for i in range(window_size, len(timesteps)):
                    steps_delta = timesteps[i] - timesteps[i-window_size]
                    time_delta = times[i] - times[i-window_size]
                    if time_delta > 0:
                        steps_per_sec.append(steps_delta / time_delta)
                plt.plot(range(window_size, len(timesteps)), steps_per_sec)
                plt.xlabel('Episode')
                plt.ylabel('Steps/second')
                plt.title('Training Speed')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_plots.png'), dpi=150)
        plt.close()
        
    def _on_step(self):
        # Update metrics for current episode
        reward = self.locals.get("rewards")[0]
        self.current_episode_reward += reward
        self.current_length += 1
        self.timesteps_since_last_record += 1
        
        # Extract distance to goal if available in info
        info = self.locals.get("infos")[0]
        if info and 'distance_to_goal' in info:
            distance = info['distance_to_goal']
            # Store for later analysis
            if len(self.distances_to_goal) < self.num_timesteps:
                self.distances_to_goal.append(distance)
            else:
                self.distances_to_goal[-1] = distance  # Update latest
        
        # When episode ends
        if self.locals.get("dones")[0]:
            # Record episode stats
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_length)
            
            # Calculate additional metrics
            self._calculate_metrics()
            
            # Print episode summary
            print(f"\n--- Episode {len(self.episode_rewards)} Completed ---")
            print(f"Episode reward: {self.current_episode_reward:.4f}")
            print(f"Episode length: {self.current_length}")
            
            if len(self.rolling_rewards_10) > 0:
                print(f"Avg reward (last 10): {self.rolling_rewards_10[-1]:.4f}")
            if len(self.rolling_rewards_50) > 0:
                print(f"Avg reward (last 50): {self.rolling_rewards_50[-1]:.4f}")
                
            elapsed = time.time() - self.start_time
            steps_per_second = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(f"Total steps: {self.num_timesteps}, Steps/sec: {steps_per_second:.1f}")
            print(f"Time elapsed: {elapsed:.1f} seconds\n")
            
            # Save metrics periodically
            if len(self.episode_rewards) % self.save_freq == 0:
                self._save_metrics()
                print(f"Metrics saved after {len(self.episode_rewards)} episodes")
                
            # Generate plots periodically
            if len(self.episode_rewards) % self.plot_freq == 0:
                self._generate_plots()
                print(f"Training plots updated after {len(self.episode_rewards)} episodes")
            
            # Reset episode counters
            self.current_episode_reward = 0
            self.current_length = 0
            self.timesteps_since_last_record = 0
            
        return True
        
    def _on_training_end(self):
        # Final metrics and plots
        self._save_metrics()
        self._generate_plots()
        
        # Generate a final summary plot
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 1, 1)
        plt.plot(self.episode_rewards, label='Episode Reward', alpha=0.3)
        plt.plot(self.rolling_rewards_100, label='100-ep Moving Avg', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress Summary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_summary.png'), dpi=200)
        
        # Print final statistics
        print("\n===== TRAINING COMPLETE =====")
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Total timesteps: {self.num_timesteps}")
        print(f"Best episode reward: {max(self.episode_rewards):.4f}")
        if len(self.rolling_rewards_100) > 0:
            print(f"Best 100-episode average: {max(self.rolling_rewards_100):.4f}")
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.1f} seconds")
        print(f"Average training speed: {self.num_timesteps/total_time:.1f} steps/second")
        print(f"All metrics and plots saved to: {self.output_dir}")
        print("============================\n")