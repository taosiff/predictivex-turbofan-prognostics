"""
Temporal Convolutional Network (TCN) for Industrial RUL Estimation.
Implements dilated causal 1D convolutions with residual blocks and exponential receptive fields.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ChausalConv1d(nn.Module):
    """
    1D Causal Convolution ensuring predictions at step t only depend on steps <= t.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Trim off trailing padding to preserve strict causality
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TemporalBlock(nn.Module):
    """
    Residual block with two dilated causal convolutions, weight norm, ReLU, and dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.conv2(out))))
        return self.final_relu(out + res)


class TCNPrognosticModel(nn.Module):
    """
    Complete Temporal Convolutional Network for RUL Estimation.
    """
    def __init__(
        self,
        input_dim: int,
        num_channels: list = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.regressor = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> transpose to (B, C, T) for 1D convolution
        x_trans = x.transpose(1, 2)
        feats = self.network(x_trans)  # (B, C_out, T)
        last_step = feats[:, :, -1]    # (B, C_out)
        rul_pred = self.regressor(last_step)
        return rul_pred.squeeze(-1)
