"""
PredictiveX Classical Machine Learning Baseline Suite.
Implements:
1. Linear / Ridge Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. LightGBM Regressor
5. CatBoost Regressor
6. Support Vector Regressor (SVR)
"""

from typing import Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib


def get_classical_models(random_state: int = 42, n_jobs: int = -1) -> Dict[str, Any]:
    """
    Returns an initialized dictionary of competitive baseline models.
    """
    models = {
        "Ridge_Regression": Ridge(alpha=10.0, random_state=random_state),
        "Random_Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            n_jobs=n_jobs,
            random_state=random_state
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=random_state,
            tree_method="hist"
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=250,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            thread_count=16,
            verbose=False
        ),
        "Support_Vector_Regression": SVR(
            kernel="rbf",
            C=10.0,
            epsilon=2.0
        )
    }
    return models


class ClassicalModelTrainer:
    """
    Orchestrates training, evaluation, timing, and artifact persistence for tabular baselines.
    """
    def __init__(self, model_name: str, model_instance: Any):
        self.model_name = model_name
        self.model = model_instance
        self.feature_names = []
        self.train_time = 0.0
        self.inference_time_ms = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        start = time.perf_counter()
        self.model.fit(X, y)
        self.train_time = time.perf_counter() - start
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        start = time.perf_counter()
        preds = self.model.predict(X)
        self.inference_time_ms = (time.perf_counter() - start) * 1000.0 / len(X)
        return np.asarray(preds).flatten()

    def get_feature_importances(self, top_k: int = 15) -> Dict[str, float]:
        """
        Extracts feature importances if supported by the model architecture.
        """
        importances = {}
        if hasattr(self.model, "feature_importances_"):
            vals = self.model.feature_importances_
            sorted_idx = np.argsort(vals)[::-1][:top_k]
            for idx in sorted_idx:
                importances[self.feature_names[idx]] = float(vals[idx])
        elif hasattr(self.model, "coef_"):
            vals = np.abs(self.model.coef_).flatten()
            sorted_idx = np.argsort(vals)[::-1][:top_k]
            for idx in sorted_idx:
                importances[self.feature_names[idx]] = float(vals[idx])
        return importances

    def save(self, filepath: str):
        joblib.dump({
            "model_name": self.model_name,
            "model": self.model,
            "feature_names": self.feature_names,
            "train_time": self.train_time,
            "inference_time_ms": self.inference_time_ms
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClassicalModelTrainer":
        data = joblib.load(filepath)
        trainer = cls(data["model_name"], data["model"])
        trainer.feature_names = data["feature_names"]
        trainer.train_time = data["train_time"]
        trainer.inference_time_ms = data["inference_time_ms"]
        return trainer
