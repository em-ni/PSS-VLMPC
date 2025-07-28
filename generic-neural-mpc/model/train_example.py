import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SystemConfig, NeuralNetConfig, TrainingConfig

def true_system_dynamics_dt(x, u):
    """
    Computes the state derivative x_dot for a generic system of coupled oscillators.
    ASSUMPTION: STATE_DIM is even, representing [pos_1, vel_1, pos_2, vel_2, ...].
    """
    if SystemConfig.STATE_DIM % 2 != 0:
        raise ValueError("STATE_DIM must be an even number for this example dynamic system.")

    x_dot = torch.zeros_like(x)
    num_oscillators = SystemConfig.STATE_DIM // 2

    for i in range(num_oscillators):
        pos_idx = 2 * i
        vel_idx = 2 * i + 1
        
        # Derivative of position is velocity
        x_dot[:, pos_idx] = x[:, vel_idx]

        # Derivative of velocity (acceleration) is a non-linear function
        # It depends on its own position (spring), its own velocity (damping), and a control input.
        # It also has a weak coupling to the previous oscillator.
        
        # Base oscillator dynamics
        accel = -torch.sin(x[:, pos_idx]) - 0.5 * x[:, vel_idx]
        
        # Control input (inputs are cycled through the oscillators)
        control_idx = i % SystemConfig.CONTROL_DIM
        accel += u[:, control_idx]
        
        # Coupling with previous oscillator (if not the first one)
        if i > 0:
            prev_pos_idx = 2 * (i - 1)
            accel -= 0.1 * (x[:, pos_idx] - x[:, prev_pos_idx])
            
        x_dot[:, vel_idx] = accel

    return x_dot

def generate_data(num_samples):
    """Generates data that respects the new generic dimensions."""
    x_data = (torch.rand(num_samples, SystemConfig.STATE_DIM) - 0.5) * 2 * np.pi
    u_data = (torch.rand(num_samples, SystemConfig.CONTROL_DIM) - 0.5) * 2
    
    x_dot_data = true_system_dynamics_dt(x_data, u_data)
    
    xu_data = torch.cat((x_data, u_data), dim=1)
    
    return TensorDataset(xu_data, x_dot_data)

# The NeuralNetwork and train() function are already generic and DO NOT need changes.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(NeuralNetConfig.INPUT_DIM, NeuralNetConfig.HIDDEN_SIZE))
        layers.append(getattr(nn, NeuralNetConfig.ACTIVATION)())
        
        for _ in range(NeuralNetConfig.HIDDEN_LAYERS - 1):
            layers.append(nn.Linear(NeuralNetConfig.HIDDEN_SIZE, NeuralNetConfig.HIDDEN_SIZE))
            layers.append(getattr(nn, NeuralNetConfig.ACTIVATION)())
            
        layers.append(nn.Linear(NeuralNetConfig.HIDDEN_SIZE, NeuralNetConfig.OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train():
    print("Generating training and validation data...")
    train_dataset = generate_data(TrainingConfig.NUM_TRAIN_SAMPLES)
    val_dataset = generate_data(TrainingConfig.NUM_VAL_SAMPLES)
    
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE)
    
    all_train_inputs = train_dataset.tensors[0]
    all_train_outputs = train_dataset.tensors[1]
    input_mean = all_train_inputs.mean(dim=0)
    input_std = all_train_inputs.std(dim=0)
    output_mean = all_train_outputs.mean(dim=0)
    output_std = all_train_outputs.std(dim=0)
    input_std[input_std == 0] = 1.0
    output_std[output_std == 0] = 1.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=TrainingConfig.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print(f"Starting training on {device}...")
    best_val_loss = float('inf')
    
    for epoch in range(TrainingConfig.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TrainingConfig.NUM_EPOCHS}"):
            inputs, targets = inputs.to(device), targets.to(device)
            norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
            norm_targets = (targets - output_mean.to(device)) / output_std.to(device)

            optimizer.zero_grad()
            outputs = model(norm_inputs)
            loss = criterion(outputs, norm_targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                norm_inputs = (inputs - input_mean.to(device)) / input_std.to(device)
                norm_targets = (targets - output_mean.to(device)) / output_std.to(device)
                outputs = model(norm_inputs)
                val_loss = criterion(outputs, norm_targets)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_payload = {
                'state_dict': model.state_dict(),
                'input_mean': input_mean,
                'input_std': input_std,
                'output_mean': output_mean,
                'output_std': output_std,
            }
            torch.save(save_payload, TrainingConfig.EX_MODEL_SAVE_PATH)
            print(f"Model saved to {TrainingConfig.EX_MODEL_SAVE_PATH} with validation loss {avg_val_loss:.6f}")

if __name__ == '__main__':
    os.makedirs(os.path.dirname(TrainingConfig.EX_MODEL_SAVE_PATH), exist_ok=True)
    train()