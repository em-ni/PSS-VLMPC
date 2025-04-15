from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
import src.config as config

class RobotTrainingMonitor(BaseCallback):
    """
    Simplified and robust training monitor that maintains a single plot window
    """
    def __init__(self, verbose=1):
        super(RobotTrainingMonitor, self).__init__(verbose)
        
        # Basic metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.rolling_avg_rewards = []
        
        # Episode tracking
        self.current_reward = 0
        self.current_length = 0
        self.current_distances = []
        
        # Initialize single persistent figure
        self.fig = None
        self.axs = None
        
        # Time tracking
        self.start_time = time.time()
        
        # Output directory
        self.output_dir = os.path.join(config.data_dir, "training_metrics", 
                                      f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training metrics will be saved to: {self.output_dir}")
        
        # Create the persistent figure immediately
        self._setup_figure()
    
    def _setup_figure(self):
        """Create a simple, persistent figure with just two subplots"""
        try:
            # Use TkAgg backend for better interactive performance
            plt.switch_backend('TkAgg')
            
            # Create figure with just 2 subplots - rewards and distances
            self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
            self.fig.canvas.manager.set_window_title('Training Progress')
            
            # Configure plots
            self.axs[0].set_title('Rewards')
            self.axs[0].set_xlabel('Episode')
            self.axs[0].set_ylabel('Reward')
            self.axs[0].grid(True, alpha=0.3)
            
            self.axs[1].set_title('Distance to Goal')
            self.axs[1].set_xlabel('Episode')
            self.axs[1].set_ylabel('Distance')
            self.axs[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.ion()  # Interactive mode
            plt.show(block=False)
            print("Training monitor initialized with live plot")
            
        except Exception as e:
            print(f"WARNING: Could not create training plot: {e}")
            self.fig = None
            self.axs = None
    
    def _update_plot(self):
        """Update the existing plot without creating new figures"""
        if self.fig is None or self.axs is None:
            return
            
        try:
            # Clear previous plots
            self.axs[0].clear()
            self.axs[1].clear()
            
            # Plot rewards
            episodes = range(1, len(self.episode_rewards) + 1)
            self.axs[0].plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Reward')
            
            # Add rolling average if we have enough episodes
            if len(self.rolling_avg_rewards) > 0:
                self.axs[0].plot(episodes[-len(self.rolling_avg_rewards):], 
                                 self.rolling_avg_rewards, 'r-', 
                                 linewidth=2, label='10-ep Avg')
            
            # Plot distances if available
            if self.episode_distances:
                distance_episodes = range(1, len(self.episode_distances) + 1)
                self.axs[1].plot(distance_episodes, self.episode_distances, 'g-', label='Final Distance')
            
            # Re-add labels and grid
            self.axs[0].set_title('Rewards')
            self.axs[0].set_xlabel('Episode')
            self.axs[0].set_ylabel('Reward')
            self.axs[0].grid(True, alpha=0.3)
            self.axs[0].legend(loc='upper left')
            
            self.axs[1].set_title('Distance to Goal')
            self.axs[1].set_xlabel('Episode')
            self.axs[1].set_ylabel('Distance')
            self.axs[1].grid(True, alpha=0.3)
            
            # Add current training speed in the title
            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            self.fig.suptitle(f'Training Progress - {steps_per_sec:.1f} steps/sec')
            
            # Update layout and draw
            plt.tight_layout()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Save the current plot
            plt.savefig(os.path.join(self.output_dir, 'training_progress.png'), dpi=100)
            
        except Exception as e:
            print(f"WARNING: Failed to update plot: {e}")
    
    def _on_step(self):
        """Simple step update"""
        # Get reward and info from environment
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        
        # Safely extract reward
        reward = rewards[0] if rewards is not None and len(rewards) > 0 else 0
        
        # Safely extract info
        info = infos[0] if infos is not None and len(infos) > 0 else {}
        
        # Update episode tracking
        self.current_reward += reward
        self.current_length += 1
        
        # Track distance if available
        if info and 'distance' in info:
            distance = info['distance']
            self.current_distances.append(distance)
        
        # Episode completed
        if dones is not None and len(dones) > 0 and dones[0]:
            # Record episode stats
            self.episode_rewards.append(self.current_reward)
            self.episode_lengths.append(self.current_length)
            
            # Record final distance if available
            if self.current_distances:
                self.episode_distances.append(self.current_distances[-1])
            
            # Calculate rolling average reward
            window = min(10, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-window:])
            self.rolling_avg_rewards.append(avg_reward)
            
            # Print episode summary
            episode_num = len(self.episode_rewards)
            print(f"\nEpisode {episode_num}: reward={self.current_reward:.2f}, length={self.current_length}")
            print(f"10-ep avg reward: {avg_reward:.2f}")
            
            # Update plot every episode (or less frequently for better performance)
            if episode_num % 5 == 0:  # Update every 5 episodes
                self._update_plot()
            
            # Save CSV every 25 episodes
            if episode_num % 25 == 0:
                self._save_csv()
            
            # Reset trackers
            self.current_reward = 0
            self.current_length = 0
            self.current_distances = []
            
        return True
    
    def _save_csv(self):
        """Save simple metrics to CSV"""
        try:
            import pandas as pd
            data = {
                'episode': range(1, len(self.episode_rewards) + 1),
                'reward': self.episode_rewards,
                'length': self.episode_lengths
            }
            
            if self.rolling_avg_rewards:
                # Pad with NaN for earlier episodes
                padding = [np.nan] * (len(self.episode_rewards) - len(self.rolling_avg_rewards))
                data['avg_reward'] = padding + self.rolling_avg_rewards
                
            if self.episode_distances:
                # Only include if we have distances
                data['final_distance'] = self.episode_distances + [np.nan] * (len(self.episode_rewards) - len(self.episode_distances))
                
            pd.DataFrame(data).to_csv(os.path.join(self.output_dir, 'metrics.csv'), index=False)
            
        except Exception as e:
            print(f"Failed to save CSV: {e}")
    
    def _on_training_end(self):
        """Clean up on training end"""
        # Final save
        self._update_plot()
        self._save_csv()
        
        # Save a final high-quality plot
        if self.fig:
            plt.savefig(os.path.join(self.output_dir, 'final_training_progress.png'), dpi=150)
            plt.close(self.fig)
            
        print(f"\nTraining complete! Metrics saved to {self.output_dir}")