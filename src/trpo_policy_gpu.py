
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
import torch

class TRPOPolicyGPU(ActorCriticPolicy):
    """Custom policy class that handles device correctly for TRPO"""
    def __init__(self, *args, **kwargs):
        # Remove device from kwargs before passing to parent
        if 'device' in kwargs:
            self._device = kwargs.pop('device')
        super().__init__(*args, **kwargs)
        
    def _build_mlp_extractor(self):
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


def move_to_gpu(model):
    # Manually move policy to GPU after initialization
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
    else:
        print("CUDA is not available. Keeping TRPO model on CPU.")