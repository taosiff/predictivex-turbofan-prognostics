"""
Prognostic Evaluation Metrics: NASA Scoring Function, RMSE, MAE, R²,
Hazard Classification Metrics, and Uncertainty Calibration Metrics.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    brier_score_loss,
    confusion_matrix
)


def nasa_scoring_function(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA C-MAPSS asymmetric scoring function.
    
    d_i = y_pred_i - y_true_i
    s_i = exp(-d_i / 13) - 1  for d_i < 0 (early / conservative prediction)
    s_i = exp(d_i / 10) - 1   for d_i >= 0 (late / dangerous prediction)
    
    Total Score S = sum(s_i)
    
    Parameters:
        y_true: Ground truth remaining useful life (1D array)
        y_pred: Predicted remaining useful life (1D array)
        
    Returns:
        Total NASA prognostic score (float)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    diff = y_pred - y_true
    
    early_mask = diff < 0
    late_mask = diff >= 0
    
    scores = np.zeros_like(diff, dtype=np.float64)
    scores[early_mask] = np.exp(-diff[early_mask] / 13.0) - 1.0
    scores[late_mask] = np.exp(diff[late_mask] / 10.0) - 1.0
    
    return float(np.sum(scores))


def evaluate_rul_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive RUL regression evaluation.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    score = nasa_scoring_function(y_true, y_pred)
    
    diff = y_pred - y_true
    early_count = int(np.sum(diff < 0))
    late_count = int(np.sum(diff > 0))
    exact_count = int(np.sum(diff == 0))
    mean_bias = float(np.mean(diff))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nasa_score": score,
        "mean_bias": mean_bias,
        "early_predictions": early_count,
        "late_predictions": late_count,
        "exact_predictions": exact_count,
        "total_units": len(y_true)
    }


def evaluate_failure_risk(y_true_binary: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluates failure risk hazard probability predictions.
    """
    y_true = np.asarray(y_true_binary).flatten().astype(int)
    y_prob = np.asarray(y_prob).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    brier = float(brier_score_loss(y_true, y_prob))
    
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "brier_score": brier
    }


def evaluate_uncertainty(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> Dict[str, float]:
    """
    Evaluates prediction interval coverage and width.
    """
    y_true = np.asarray(y_true).flatten()
    lower = np.asarray(lower_bound).flatten()
    upper = np.asarray(upper_bound).flatten()
    
    in_interval = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(in_interval))  # Prediction Interval Coverage Probability
    mpiw = float(np.mean(upper - lower))  # Mean Prediction Interval Width
    
    return {
        "coverage_probability": picp,
        "mean_interval_width": mpiw
    }
