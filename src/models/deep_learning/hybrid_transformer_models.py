"""
PredictiveX Advanced Deep Learning Architectures:
1. CNN-LSTM Hybrid (Spatial-Temporal Feature Extractor)
2. Temporal Transformer (Self-Attention over Wear Sequences)
3. PatchTST (Patch-based Channel-Independent Representation for RUL)
"""

import math
import torch
import torch.nn as nn
from einops import rearrange


class CNNLSTMPrognosticModel(nn.Module):
    """
    CNN-LSTM Hybrid Model:
    1D Convolutions extract instantaneous cross-sensor interactions,
    followed by LSTM modeling cumulative temporal wear accumulation.
    """
    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 64,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> Conv requires (B, C, T)
        x_conv_in = x.transpose(1, 2)
        conv_out = self.conv(x_conv_in)  # (B, conv_channels, T)
        # Transpose back to (B, T, conv_channels) for LSTM
        lstm_in = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        last_step = lstm_out[:, -1, :]
        rul_pred = self.regressor(last_step)
        return rul_pred.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerPrognosticModel(nn.Module):
    """
    Standard Temporal Transformer with Multi-Head Self-Attention for RUL Prognostics.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        enc_out = self.transformer_encoder(h)  # (B, T, d_model)
        
        # Mean pooling across temporal dimension
        pooled = enc_out.mean(dim=1)
        rul_pred = self.regressor(pooled)
        return rul_pred.squeeze(-1)


class PatchTSTPrognosticModel(nn.Module):
    """
    Patch-based Time Series Transformer adapted for Remaining Useful Life Estimation.
    Splits the sensor sequence into local temporal patches to capture high-frequency
    degradation semantics while reducing attention complexity.
    """
    def __init__(
        self,
        input_dim: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        
        # Linear patch projection for each sensor channel
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model * input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # Pad sequence if necessary to allow at least one patch
        if T < self.patch_len:
            pad = self.patch_len - T
            x = torch.nn.functional.pad(x, (0, 0, pad, 0))
            T = self.patch_len
            
        # Unfold into patches: (B, num_patches, C, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, num_patches, C, patch_len)
        num_patches = patches.size(1)
        
        # Flatten patches across channels: (B * C, num_patches, patch_len)
        patches_re = rearrange(patches, 'b p c l -> (b c) p l')
        proj = self.patch_proj(patches_re)  # (B * C, num_patches, d_model)
        proj = self.pos_encoder(proj)
        
        enc = self.encoder(proj)  # (B * C, num_patches, d_model)
        pooled = enc.mean(dim=1)  # (B * C, d_model)
        
        # Reshape back to (B, C * d_model)
        pooled_all_channels = rearrange(pooled, '(b c) d -> b (c d)', b=B, c=C)
        rul_pred = self.head(pooled_all_channels)
        return rul_pred.squeeze(-1)
