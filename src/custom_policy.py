import torch
import time
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy


class CustomPolicy(ActorCriticPolicy):
    """
    Custom policy that loads pre-trained weights for the feature extractor.
    Can be used with PPO or other actor-critic algorithms.
    """
    def __init__(self, 
                 observation_space, 
                 action_space, 
                 lr_schedule, 
                 pretrained_weights_path="data/exp_2025-03-17_15-26-06/nn_model.pth",
                 *args, 
                 **kwargs):
        # Initialize the parent class first
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # Load pre-trained weights
        print(f"Loading pre-trained weights from {pretrained_weights_path}")
        try:
            pretrained_weights = torch.load(pretrained_weights_path)
            
            # Load weights into the mlp_extractor
            # Note: This assumes the architecture is compatible
            # You might need to adapt this depending on your model structure
            self.mlp_extractor.load_state_dict(pretrained_weights)
            
            print("Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"Failed to load pre-trained weights: {e}")
            
    def _build_mlp_extractor(self) -> None:
        """
        Override this method if you need a custom feature extractor
        architecture that's different from the default.
        """
        super()._build_mlp_extractor()
        # You can modify the mlp_extractor here if needed

