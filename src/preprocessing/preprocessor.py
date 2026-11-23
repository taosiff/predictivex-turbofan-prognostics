"""
PredictiveX Scaler & Sensor Preprocessing Pipeline.
Guarantees zero-leakage transformation fitted exclusively on training trajectories.
Supports operating-condition normalization and sensor variance thresholding.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import joblib

from src.datasets.cmapss_loader import SENSOR_COLUMNS, SETTING_COLUMNS, get_constant_sensors


class CMAPSSPreprocessor:
    """
    Stateful preprocessor for C-MAPSS time-series.
    """
    def __init__(
        self,
        scaler_type: str = "minmax",
        remove_constant_sensors: bool = True,
        normalize_operating_conditions: bool = False,
        feature_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.scaler_type = scaler_type.lower()
        self.remove_constant_sensors = remove_constant_sensors
        self.normalize_operating_conditions = normalize_operating_conditions
        self.feature_range = feature_range
        
        self.active_sensors: List[str] = []
        self.removed_sensors: List[str] = []
        self.feature_columns: List[str] = []
        
        self.global_scaler = None
        self.condition_clusterer: Optional[KMeans] = None
        self.regime_scalers: Dict[int, object] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """
        Fits scalers strictly on training engine data.
        """
        # 1. Determine sensor feature list
        if self.remove_constant_sensors:
            self.removed_sensors = get_constant_sensors(train_df)
            self.active_sensors = [s for s in SENSOR_COLUMNS if s not in self.removed_sensors]
        else:
            self.removed_sensors = []
            self.active_sensors = list(SENSOR_COLUMNS)
            
        self.feature_columns = list(self.active_sensors)
        
        # 2. Operating condition normalization (useful for FD002 and FD004)
        if self.normalize_operating_conditions:
            # Cluster the 3 operational settings into regimes
            self.condition_clusterer = KMeans(n_clusters=6, random_state=42, n_init=10)
            regimes = self.condition_clusterer.fit_predict(train_df[SETTING_COLUMNS])
            
            # Fit separate scaler per regime
            for r in range(6):
                mask = (regimes == r)
                if mask.sum() > 0:
                    scaler = self._create_scaler()
                    scaler.fit(train_df.loc[mask, self.active_sensors])
                    self.regime_scalers[r] = scaler
        else:
            self.global_scaler = self._create_scaler()
            self.global_scaler.fit(train_df[self.active_sensors])
            
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fitted scaling to train, validation, or test trajectories.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")
            
        out = df.copy()
        
        if self.normalize_operating_conditions and self.condition_clusterer is not None:
            regimes = self.condition_clusterer.predict(out[SETTING_COLUMNS])
            out["regime"] = regimes
            for r, scaler in self.regime_scalers.items():
                mask = (regimes == r)
                if mask.sum() > 0:
                    out.loc[mask, self.active_sensors] = scaler.transform(out.loc[mask, self.active_sensors])
        else:
            out[self.active_sensors] = self.global_scaler.transform(out[self.active_sensors])
            
        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    def _create_scaler(self):
        if self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.scaler_type == "standard":
            return StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type {self.scaler_type}")

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "CMAPSSPreprocessor":
        return joblib.load(filepath)
