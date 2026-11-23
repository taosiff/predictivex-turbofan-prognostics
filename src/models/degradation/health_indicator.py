"""
PredictiveX Health Indicator (HI) & Degradation Progression Modeling.
Computes continuous health index (1.0 -> 0.0) from multivariate sensor streams,
and evaluates degradation Monotonicity and Trendability.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr


class HealthIndicatorAutoencoder(nn.Module):
    """
    Compresses multivariate sensor sequences into a 1D scalar Health Indicator.
    Trained with autoencoding reconstruction loss and monotonicity regularizer.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Restrict HI to [0.0, 1.0]
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, input_dim)
        hi = self.encoder(x)  # (B, 1)
        reconstruction = self.decoder(hi)  # (B, input_dim)
        return hi.squeeze(-1), reconstruction


def compute_monotonicity(hi_trajectories: List[np.ndarray]) -> float:
    """
    Computes average monotonicity across engine trajectories:
    M = (1/K) * sum( | #(diff > 0) - #(diff < 0) | / (N_k - 1) )
    Perfect monotonic degradation approaches 1.0.
    """
    mon_scores = []
    for traj in hi_trajectories:
        if len(traj) < 2:
            continue
        diffs = np.diff(traj)
        pos = np.sum(diffs > 0)
        neg = np.sum(diffs < 0)
        n = len(diffs)
        mon = abs(pos - neg) / n
        mon_scores.append(mon)
    return float(np.mean(mon_scores)) if mon_scores else 0.0


def compute_trendability(hi_trajectories: List[np.ndarray]) -> float:
    """
    Computes average Spearman rank correlation between HI and operating cycles.
    """
    trend_scores = []
    for traj in hi_trajectories:
        if len(traj) < 3:
            continue
        cycles = np.arange(len(traj))
        rho, _ = spearmanr(cycles, traj)
        if not np.isnan(rho):
            trend_scores.append(abs(rho))
    return float(np.mean(trend_scores)) if trend_scores else 0.0
