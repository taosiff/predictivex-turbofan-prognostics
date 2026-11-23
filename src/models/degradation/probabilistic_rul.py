"""
PredictiveX Probabilistic RUL & Uncertainty Estimation.
Implements:
1. Monte Carlo Dropout for Epistemic Uncertainty & Confidence Intervals (Gal & Ghahramani, 2016)
2. Pinball / Quantile Loss Multi-Head Regression for Aleatoric Uncertainty
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn


class MCDropoutEstimator:
    """
    Computes epistemic model uncertainty by sampling stochastic dropout passes at inference time.
    """
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None, num_samples: int = 50):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_samples = num_samples

    def predict_with_uncertainty(
        self,
        X_seq: np.ndarray,
        rul_cap: float = 125.0
    ) -> Dict[str, np.ndarray]:
        """
        Runs num_samples stochastic forward passes with active dropout.
        """
        self.model.train()  # Keep dropout active!
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                preds = self.model(X_tensor)
                samples.append(preds.cpu().numpy().flatten())
                
        samples = np.clip(np.array(samples), 0.0, rul_cap)  # Shape: (num_samples, N)
        
        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        
        # Normalized confidence metric in [0.0, 1.0]
        # Low std -> high confidence
        confidence = 1.0 / (1.0 + (std_pred / 15.0))
        
        return {
            "mean_rul": mean_pred,
            "std_rul": std_pred,
            "lower_bound_p10": p10,
            "upper_bound_p90": p90,
            "confidence": confidence,
            "num_samples": self.num_samples
        }


class QuantileLoss(nn.Module):
    """
    Pinball Loss for simultaneous multi-quantile estimation.
    """
    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds: (B, num_quantiles), targets: (B)
        targets = targets.unsqueeze(1)
        errors = targets - preds
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            err = errors[:, i]
            loss += torch.max((q - 1) * err, q * err).mean()
        return loss
