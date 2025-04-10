import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, dropout):
        super(LSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)  # Predict flow only
    
    def forward(self, x):
        # x: (B, 1, F, T) -> (B, T, F)
        x = x.squeeze(1).permute(0, 2, 1)  # (B, T, F)
        batch_size = x.size(0)
        
        # LSTM
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # (B, T, hidden_size)
        
        # Predict
        out = self.fc(out)  # (B, T, 1)
        out = out[:, -12:, :]  # Last 12 steps: (B, 12, 1)
        return out  # (B, 12, 1)