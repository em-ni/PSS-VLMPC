from stable_baselines3.common.callbacks import BaseCallback


class RobotTrainingMonitor(BaseCallback):
    """
    Custom callback to monitor and log training metrics without requiring a second environment
    """
    def __init__(self, verbose=1):
        super(RobotTrainingMonitor, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_length = 0
        
    def _on_step(self):
        # Update metrics
        reward = self.locals.get("rewards")[0]
        self.current_episode_reward += reward
        self.current_length += 1
        
        # When episode ends
        if self.locals.get("dones")[0]:
            # Log end of episode stats
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_length)
            
            # Print metrics
            print(f"\n--- Episode {len(self.episode_rewards)} Completed ---")
            print(f"Episode reward: {self.current_episode_reward:.4f}")
            print(f"Episode length: {self.current_length}\n\n")
            
            if len(self.episode_rewards) >= 5:
                avg_reward = sum(self.episode_rewards[-5:]) / 5
                print(f"Average reward (last 5): {avg_reward:.4f}\n\n")
                
            # Reset episode counter
            self.current_episode_reward = 0
            self.current_length = 0
            
        return True