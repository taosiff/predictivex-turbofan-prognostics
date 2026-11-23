"""
PredictiveX Unified Multi-Task Prognostics Architecture.
Combines:
1. Shared Temporal Representation Backbone (Causal TCN / GRU)
2. RUL Regression Head
3. Horizon Failure-Risk Classification Head (P(Failure within H cycles))
4. Health Indicator / Reconstruction Head
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class MultiTaskPrognosticModel(nn.Module):
    """
    Unified multi-task model sharing a temporal convolutional backbone.
    Outputs:
    - rul: predicted continuous RUL
    - failure_risk: probability of failure within hazard horizon (sigmoid)
    - health_indicator: continuous degradation state [0.0, 1.0]
    """
    def __init__(
        self,
        input_dim: int,
        backbone_channels: list = [64, 64],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        # Shared 1D Temporal Convolutions
        layers = []
        in_ch = input_dim
        for ch in backbone_channels:
            layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size // 1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_ch = ch
            
        self.backbone = nn.Sequential(*layers)
        backbone_dim = backbone_channels[-1]

        # 1. RUL Regression Head
        self.rul_head = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 2. Failure-Risk Classification Head (P(Failure <= 15 cycles))
        self.risk_head = nn.Sequential(
            nn.Linear(backbone_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 3. Health Indicator Head
        self.hi_head = nn.Sequential(
            nn.Linear(backbone_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B, T, C) -> transpose to (B, C, T) for 1D convolutions
        x_in = x.transpose(1, 2)
        feat_map = self.backbone(x_in)  # (B, C_out, T)
        
        # Take the final temporal cycle representation
        latent = feat_map[:, :, -1]  # (B, C_out)
        
        rul = self.rul_head(latent).squeeze(-1)
        risk = self.risk_head(latent).squeeze(-1)
        hi = self.hi_head(latent).squeeze(-1)

        return {
            "rul": rul,
            "failure_risk": risk,
            "health_indicator": hi
        }
