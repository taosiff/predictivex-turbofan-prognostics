"""
PredictiveX Recurrent Deep Learning Architectures (LSTM & GRU) for RUL Estimation.
Supports multi-layer recurrent encoding, dropout regularization, and dense regression heads.
"""

import torch
import torch.nn as nn


class LSTMPrognosticModel(nn.Module):
    """
    Multi-layer Long Short-Term Memory (LSTM) network for RUL estimation.
    Input shape: (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        
        num_directions = 2 if bidirectional else 1
        fc_in_dim = hidden_dim * num_directions
        
        self.regressor = nn.Sequential(
            nn.Linear(fc_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        lstm_out, _ = self.lstm(x)  # (B, T, H * directions)
        last_step = lstm_out[:, -1, :]  # Take final cycle representation
        rul_pred = self.regressor(last_step)
        return rul_pred.squeeze(-1)


class GRUPrognosticModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) network for RUL estimation.
    Computationally more lightweight with comparable expressiveness to LSTM.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        rul_pred = self.regressor(last_step)
        return rul_pred.squeeze(-1)
