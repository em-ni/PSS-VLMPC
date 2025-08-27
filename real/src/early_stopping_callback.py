from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from collections import deque, defaultdict

class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback that monitors the distance between tip and goal.
    Training stops when the distances consistently remain below threshold
    for a specified number of consecutive episodes.
    
    :param distance_threshold: Maximum acceptable distance between tip and goal
    :param consecutive_episodes: Number of consecutive successful episodes required
    :param success_percentage: Percentage of steps in episode below threshold to count as success
    :param verbose: Verbosity level
    """
    def __init__(self, distance_threshold=0.5, consecutive_episodes=5, 
                 success_percentage=90, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.distance_threshold = distance_threshold
        self.consecutive_episodes = consecutive_episodes
        self.success_percentage = success_percentage / 100.0
        self.successful_episodes = 0
        self.episode_distances = defaultdict(list)  # Track distances per environment
        
    def _init_callback(self):
        self.episode_distances = defaultdict(list)
        self.successful_episodes = 0
        
    def _on_step(self) -> bool:
        # Get infos and dones from all environments
        infos = self.locals.get('infos')
        dones = self.locals.get('dones')
        
        # Track distances for each environment separately
        for env_idx, (info, done) in enumerate(zip(infos, dones)):
            if 'distance' in info:
                self.episode_distances[env_idx].append(info['distance'])
            
            # If episode ended, evaluate performance for this environment
            if done:
                if len(self.episode_distances[env_idx]) > 0:
                    # Calculate percentage of steps with distance below threshold
                    steps_below_threshold = sum(1 for d in self.episode_distances[env_idx] 
                                               if d < self.distance_threshold)
                    success_rate = steps_below_threshold / len(self.episode_distances[env_idx])
                    avg_distance = np.mean(self.episode_distances[env_idx])
                    
                    if self.verbose > 0:
                        print(f"Env {env_idx} episode completed - Avg distance: {avg_distance:.4f}, "
                              f"Success rate: {success_rate*100:.1f}%")
                    
                    # Check if this episode was successful
                    if success_rate >= self.success_percentage:
                        self.successful_episodes += 1
                        if self.verbose > 0:
                            print(f"Successful episodes: {self.successful_episodes}/{self.consecutive_episodes}")
                    else:
                        # Reset counter if episode wasn't successful
                        self.successful_episodes = 0
                        if self.verbose > 0:
                            print(f"Episode below success threshold. Resetting counter.")
                
                # Clear distances for this environment
                self.episode_distances[env_idx] = []
                
                # Check if we've met the early stopping criteria
                if self.successful_episodes >= self.consecutive_episodes:
                    if self.verbose > 0:
                        print(f"Early stopping: {self.consecutive_episodes} consecutive successful episodes")
                    return False  # Stop training
        
        return True  # Continue training