import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM layer
        # batch_first=True makes input/output tensors shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to detach hidden state to prevent backpropagating through the entire history
        # Pass (h0, c0) to LSTM
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # We only need the output of the last time step
        # out shape: (batch_size, seq_len, hidden_dim) -> take last time step
        out = self.fc(out[:, -1, :])
        return out