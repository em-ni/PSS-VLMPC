from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
import time
from datetime import datetime
import src.config as config

class RobotTrainingMonitor(BaseCallback):
    """
    Robust training monitor with fallback modes and reliable plotting
    """
    def __init__(self, verbose=1, plot_frequency=5):
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
        
        # Plot settings
        self.plot_frequency = plot_frequency
        self.fig = None
        self.axs = None
        self.use_interactive = True
        
        # Time tracking
        self.start_time = time.time()
        
        # Output directory
        self.output_dir = config.METRICS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Training metrics will be saved to: {self.output_dir}")
        
        # Try to set up plotting
        self._setup_figure()
    
    def _setup_figure(self):
        """Create a simple figure with fallback options"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # Try GUI backends first, then fallback to non-interactive
            backends_to_try = ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg', 'WebAgg', 'Agg']
            for backend in backends_to_try:
                try:
                    matplotlib.use(backend, force=True)
                    print(f"Using matplotlib backend: {backend}")
                    break
                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
            
            # Create figure with 2 subplots
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
            
            # Ensure interactive mode is on and show the plot
            try:
                plt.ion()  # Interactive mode
                self.fig.show()  # Force window to appear
                plt.pause(0.5)  # Longer initial pause to ensure window appears
                self.use_interactive = True
                print("Interactive plotting enabled")
            except Exception as e:
                print(f"Interactive mode failed: {e}")
                self.use_interactive = False
            
            print("Training monitor plot initialized")
            
        except Exception as e:
            print(f"WARNING: Could not set up matplotlib plotting: {e}")
            print("Will save plots to files only")
            self.fig = None
            self.axs = None
            self.use_interactive = False
    
    def _update_plot(self):
        """Update the plot, with robust fallbacks"""
        if not self.episode_rewards:
            return  # Nothing to plot yet
            
        try:
            import matplotlib.pyplot as plt
            
            # If figure doesn't exist, create it
            if self.fig is None or self.axs is None:
                self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
            
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
            
            # Update layout
            plt.tight_layout()
            
            # Try to update the interactive plot, if enabled
            if self.use_interactive:
                try:
                    self.fig.canvas.draw()  # More reliable than draw_idle
                    if hasattr(self.fig.canvas, 'flush_events'):
                        self.fig.canvas.flush_events()
                    plt.pause(0.1)  # Longer pause for better visibility
                    
                    # Ensure window is visible (sometimes needed on Windows)
                    if hasattr(self.fig.canvas.manager, 'window'):
                        if hasattr(self.fig.canvas.manager.window, 'lift'):
                            self.fig.canvas.manager.window.lift()  # Tk
                        elif hasattr(self.fig.canvas.manager.window, 'activateWindow'):
                            self.fig.canvas.manager.window.activateWindow()  # Qt
                except Exception as e:
                    print(f"Interactive update failed: {e}")
                    self.use_interactive = False  # Disable for future updates
            
            # Always save to file (works even if interactive fails)
            filepath = os.path.join(self.output_dir, 'training_progress.png')
            plt.savefig(filepath, dpi=100)
            
            # Log update 
            episode_num = len(self.episode_rewards)
            print(f"Updated plot for episode {episode_num} saved to {filepath}")
            
        except Exception as e:
            print(f"WARNING: Failed to update plot: {e}")
    
    def _on_step(self):
        """Process each step update"""
        # Extract info from training state
        rewards = self.locals.get("rewards", [0])
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        
        # Safely extract reward
        reward = rewards[0] if len(rewards) > 0 else 0
        
        # Safely extract info
        info = infos[0] if len(infos) > 0 else {}
        
        # Update episode tracking
        self.current_reward += reward
        self.current_length += 1
        
        # Track distance if available
        if info and 'distance' in info:
            distance = info['distance']
            self.current_distances.append(distance)
        
        # Episode completed
        if len(dones) > 0 and dones[0]:
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
            
            # Update plot periodically
            if episode_num % self.plot_frequency == 0:
                self._update_plot()
            
            # Save CSV more frequently than plots
            if episode_num % 10 == 0:
                self._save_csv()
            
            # Reset trackers for next episode
            self.current_reward = 0
            self.current_length = 0
            self.current_distances = []
            
        return True
    
    def _save_csv(self):
        """Save metrics to CSV file"""
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
            
            filepath = os.path.join(self.output_dir, 'metrics.csv')
            pd.DataFrame(data).to_csv(filepath, index=False)
            print(f"Metrics saved to {filepath}")
            
        except Exception as e:
            print(f"Failed to save CSV: {e}")
    
    def _on_training_end(self):
        """Clean up on training end"""
        # Final saves
        self._update_plot()
        self._save_csv()
        
        try:
            import matplotlib.pyplot as plt
            
            # Save a final high-quality plot
            if self.fig:
                final_path = os.path.join(self.output_dir, 'final_training_progress.png')
                plt.savefig(final_path, dpi=150)
                print(f"Final plot saved to {final_path}")
                plt.close(self.fig)
        except Exception as e:
            print(f"Failed to save final plot: {e}")
            
        print(f"\nTraining complete! All metrics saved to {self.output_dir}")