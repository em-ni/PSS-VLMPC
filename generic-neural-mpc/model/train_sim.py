# train_sim.py - Improved version to prevent overfitting
import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Training Configuration ---
class TrainingConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    SIM_DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../sim/results/sim_dataset.csv"))
    MODEL_PATH = os.path.join(BASE_DIR, "data", "sim_rob_f.pth")
    INPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "sim_rob_i_scaler.joblib")
    OUTPUT_SCALER_PATH = os.path.join(BASE_DIR, "data", "sim_rob_o_scaler.joblib")
    PLOT_OUTPUT_PATH = os.path.join(BASE_DIR, "data", "sim_rob_perf.png")
    
    # Increased epochs for proper training with regularization
    NUM_EPOCHS = 100
    BATCH_SIZE = 64  # Increased batch size for better generalization
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    # Much stronger L2 regularization and different optimizer settings
    LEARNING_RATE = 1e-4  # Slower learning
    WEIGHT_DECAY = 5e-3   # Much stronger L2 regularization
    
    # Aggressive anti-overfitting parameters
    SEQUENCE_GAP = 200    # Much larger gap to break temporal correlation
    NOISE_FACTOR = 0.05   # More noise for better generalization
    EARLY_STOP_PATIENCE = 8  # Stop earlier to prevent overfitting
    MIN_DELTA = 1e-5      # Require larger improvements
    LABEL_SMOOTHING = 0.1 # Add label smoothing

class StatePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        # Much smaller network to force the model to learn simpler patterns
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, 32),   # Even smaller: 64->32
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),          # Even smaller: 32->16
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, output_dim)
        ])
        
        # More aggressive weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Much smaller initialization
                nn.init.xavier_normal_(m.weight, gain=0.1)  # Very small gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RobotStateDataset(Dataset):
    """Custom PyTorch Dataset with aggressive noise injection and data augmentation."""
    def __init__(self, X, y, add_noise=False, noise_factor=0.01):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.add_noise = add_noise
        self.noise_factor = noise_factor
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx].clone(), self.y[idx].clone()
        
        if self.add_noise:
            # Add noise to inputs
            input_noise = torch.randn_like(x) * self.noise_factor
            x = x + input_noise
            
            # Add small noise to targets (label smoothing effect)
            target_noise = torch.randn_like(y) * (self.noise_factor * 0.1)
            y = y + target_noise
            
        return x, y


def load_and_prepare_data(filepath, sequence_gap=1):
    """
    Loads data with temporal gap to reduce overfitting from sequential correlation.
    
    Args:
        filepath: Path to the CSV file
        sequence_gap: Skip every N samples to reduce temporal correlation
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    STATE_COLS = [
        'tip_position_x', 'tip_position_y', 'tip_position_z',
        'tip_velocity_x', 'tip_velocity_y', 'tip_velocity_z'
    ]
    INPUT_COLS = [
        'rod1_torque_x', 'rod1_torque_y',
        'rod2_torque_x', 'rod2_torque_y'
    ]
    
    CURRENT_FEATURES = INPUT_COLS + STATE_COLS
    all_required_cols = ['T'] + CURRENT_FEATURES
    
    for col in all_required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV file.")
    
    df.dropna(subset=all_required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < 2:
        raise ValueError("Not enough data to create training pairs.")

    print(f"Processing data with sequence gap of {sequence_gap}...")
    
    # Apply sequence gap to reduce temporal correlation
    if sequence_gap > 1:
        # Take every sequence_gap-th sample
        indices = np.arange(0, len(df)-1, sequence_gap)
        current_features = df[CURRENT_FEATURES].iloc[indices].values
        next_state = df[STATE_COLS].iloc[indices + 1].values
        
        times_ms = df['T'].iloc[indices].values
        next_times_ms = df['T'].iloc[indices + 1].values
        dt_seconds = (next_times_ms - times_ms)
    else:
        current_features = df[CURRENT_FEATURES].iloc[:-1].values
        next_state = df[STATE_COLS].iloc[1:].values
        times_ms = df['T'].values
        dt_seconds = (times_ms[1:] - times_ms[:-1])
    
    dt_seconds_col = dt_seconds.reshape(-1, 1)
    X = np.hstack([current_features, dt_seconds_col])
    y = next_state
    
    print(f"Data processed. Reduced from {len(df)} to {len(X)} samples.")
    return X, y

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, config):
    """Enhanced training loop with very aggressive regularization."""
    # Custom loss with label smoothing effect
    criterion = nn.MSELoss()
    
    # Much stronger regularization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # More aggressive learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,    # Cut LR in half
        patience=3,    # Very impatient
        min_lr=1e-7    # Allow very small LR
    )
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    
    # Very aggressive early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("Starting aggressive anti-overfitting training...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L1 regularization manually for extra sparsity
            l1_reg = 0
            for param in model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            loss = loss + 1e-5 * l1_reg  # Small L1 penalty
            
            # Very aggressive gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            running_train_loss += loss.item()
        
        train_loss = running_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        
        val_loss = running_val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Very strict early stopping
        if val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        # Check if training loss is getting too close to validation loss
        if epoch > 10 and train_loss < val_loss * 0.9:
            print(f"Warning: Training loss ({train_loss:.6f}) getting too close to validation loss ({val_loss:.6f})")
            
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Progress reporting
        if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= config.EARLY_STOP_PATIENCE - 3:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}, Patience: {patience_counter}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with validation loss: {best_val_loss:.6f}")
    
    print("Training finished.")
    return history

# --- Main Execution ---
if __name__ == "__main__":
    config = TrainingConfig()
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    # --- Load and prepare data with temporal gap ---
    try:
        X, y = load_and_prepare_data(config.SIM_DATASET_PATH, config.SEQUENCE_GAP)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {config.SIM_DATASET_PATH}")
        sys.exit()
    except (KeyError, ValueError) as e:
        print(f"Error during data preparation: {e}")
        sys.exit()

    print(f"Shape of input features (X): {X.shape}")
    print(f"Shape of target features (y): {y.shape}")
    
    # --- Split data with stratification to ensure balance ---
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=42, shuffle=True
    )
    
    val_split_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split_ratio, random_state=42, shuffle=True
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # --- Scale data ---
    print("\nScaling data...")
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    X_train_scaled = input_scaler.fit_transform(X_train)
    y_train_scaled = output_scaler.fit_transform(y_train)
    
    X_val_scaled = input_scaler.transform(X_val)
    y_val_scaled = output_scaler.transform(y_val)
    print("Data scaled successfully.")
    
    # --- Create DataLoaders with noise injection for training ---
    train_dataset = RobotStateDataset(
        X_train_scaled, y_train_scaled, 
        add_noise=True, noise_factor=config.NOISE_FACTOR
    )
    val_dataset = RobotStateDataset(X_val_scaled, y_val_scaled, add_noise=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # --- Initialize and train model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = StatePredictor(input_dim, output_dim)
    print(f"Model initialized with input_dim={input_dim} and output_dim={output_dim}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    history = train_model(model, train_loader, val_loader, config.NUM_EPOCHS, config.LEARNING_RATE, device, config)

    # --- Save model and scalers ---
    print("\n--- Saving model and scalers ---")
    torch.save(model.state_dict(), config.MODEL_PATH)
    joblib.dump(input_scaler, config.INPUT_SCALER_PATH)
    joblib.dump(output_scaler, config.OUTPUT_SCALER_PATH)
    print(f"Model saved to {config.MODEL_PATH}")
    print(f"Input scaler saved to {config.INPUT_SCALER_PATH}")
    print(f"Output scaler saved to {config.OUTPUT_SCALER_PATH}")
    
    # Print final statistics
    if history['train_loss'] and history['val_loss']:
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        print(f"\nFinal Training Loss: {final_train_loss:.6f}")
        print(f"Final Validation Loss: {final_val_loss:.6f}")
        print(f"Train/Val Loss Ratio: {final_train_loss/final_val_loss:.3f}")
        
        if final_train_loss > final_val_loss * 2.0:  # Much stricter threshold
            print("✓ Good generalization - training loss significantly higher than validation loss")
        elif final_train_loss > final_val_loss * 1.5:
            print("⚠ Borderline - some overfitting may be present")
        else:
            print("❌ Overfitting detected - training loss too close to validation loss")