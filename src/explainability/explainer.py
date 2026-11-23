"""
PredictiveX Explainable AI (XAI) Suite:
1. SHAP (SHapley Additive exPlanations) for Gradient Boosted Trees
2. Permutation Feature Importance for agnostic verification
3. Temporal Gradient Attribution for Deep Learning sequence models
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import shap
import torch


class TreeExplainerWrapper:
    """
    Computes exact Shapley values for tree-based models (XGBoost, LightGBM, CatBoost).
    """
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: pd.DataFrame, max_samples: int = 100) -> Tuple[np.ndarray, Dict[str, float]]:
        X_sample = X.iloc[:max_samples] if len(X) > max_samples else X
        shap_values = self.explainer.shap_values(X_sample)
        
        # Mean absolute SHAP per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1]
        
        ranking = {
            self.feature_names[i]: float(mean_abs_shap[i])
            for i in top_idx[:15]
        }
        return shap_values, ranking


class TemporalGradientExplainer:
    """
    Computes feature and temporal attributions for PyTorch models using Input x Gradient.
    """
    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute(self, X_seq: np.ndarray, sensor_names: List[str]) -> Dict[str, Any]:
        """
        X_seq: (N, T, C)
        Returns:
            channel_importance: mean importance per sensor channel
            temporal_profile: mean importance across time steps [0 .. T-1]
        """
        X_tensor = torch.tensor(X_seq, dtype=torch.float32, requires_grad=True, device=self.device)
        preds = self.model(X_tensor)
        
        # Backward pass with respect to predictions
        grad_outputs = torch.ones_like(preds)
        preds.backward(gradient=grad_outputs)
        
        # Saliency = |Input * Gradient|
        saliency = (X_tensor * X_tensor.grad).abs().detach().cpu().numpy()  # (N, T, C)
        
        # Channel importance: average over batch and time
        channel_importance_vals = saliency.mean(axis=(0, 1))
        sorted_indices = np.argsort(channel_importance_vals)[::-1]
        channel_importance = {
            sensor_names[idx]: float(channel_importance_vals[idx])
            for idx in sorted_indices
        }
        
        # Temporal profile: average over batch and channels
        temporal_profile = saliency.mean(axis=(0, 2)).tolist()
        
        return {
            "channel_importance": channel_importance,
            "temporal_profile": temporal_profile,
            "raw_saliency": saliency
        }
