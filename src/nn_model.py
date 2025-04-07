import torch.nn as nn

# Define a neural network for pressure prediction.
class VolumeNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        super(VolumeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)