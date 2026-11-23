"""
PredictiveX Sensor Anomaly Detection Suite:
1. LSTM Autoencoder (Reconstruction error across multivariate sensor channels)
2. Isolation Forest (Tabular baseline for multi-sensor outlier scoring)
3. Sensor Channel Attribution (Diagnosing which physical sensor triggers the anomaly)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import joblib


class LSTMAutoencoder(nn.Module):
    """
    Recurrent sequence autoencoder trained on healthy/nominal engine cycles.
    Encoder maps sequence (B, T, C) -> latent vector (B, hidden_dim).
    Decoder reconstructs sequence (B, T, C).
    Unusual sensor drift results in elevated Mean Squared Reconstruction Error.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, _ = x.shape
        _, (h_n, _) = self.encoder(x)  # h_n: (num_layers, B, hidden_dim)
        latent = h_n[-1]  # (B, hidden_dim)
        
        # Repeat latent across T steps for decoder input
        dec_in = latent.unsqueeze(1).repeat(1, T, 1)  # (B, T, hidden_dim)
        dec_out, _ = self.decoder(dec_in)  # (B, T, hidden_dim)
        reconstruction = self.output_layer(dec_out)  # (B, T, C)
        return reconstruction


class SensorAnomalyDetector:
    """
    High-level manager for training and applying anomaly detection on sensor streams.
    """
    def __init__(
        self,
        input_dim: int,
        sensor_names: List[str],
        method: str = "lstm_ae",
        device: Optional[torch.device] = None
    ):
        self.input_dim = input_dim
        self.sensor_names = sensor_names
        self.method = method.lower()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.method == "lstm_ae":
            self.model = LSTMAutoencoder(input_dim=input_dim).to(self.device)
            self.threshold = 0.05
        else:
            self.model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
            self.threshold = 0.0

    def fit_healthy_sequences(
        self,
        healthy_sequences: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 128
    ):
        """
        Trains the autoencoder strictly on nominal/early engine sequences.
        """
        if self.method != "lstm_ae":
            # For Isolation Forest, average over time dimension
            X_tab = healthy_sequences.mean(axis=1)
            self.model.fit(X_tab)
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(healthy_sequences, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                optimizer.zero_grad()
                recon = self.model(X_b)
                loss = criterion(recon, X_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        # Calibrate anomaly threshold as 99th percentile of healthy reconstruction error
        self.model.eval()
        with torch.no_grad():
            X_all = torch.tensor(healthy_sequences, dtype=torch.float32).to(self.device)
            recon_all = self.model(X_all)
            errors = ((recon_all - X_all) ** 2).mean(dim=[1, 2]).cpu().numpy()
            self.threshold = float(np.percentile(errors, 99))

    def detect_anomalies(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        Returns:
            anomaly_scores: 1D array of anomaly scores
            is_anomaly: boolean mask (True if score > threshold)
            channel_attributions: per-sample dict of channel-level residuals
        """
        if self.method != "lstm_ae":
            X_tab = sequences.mean(axis=1)
            raw_scores = -self.model.score_samples(X_tab)
            preds = (self.model.predict(X_tab) == -1)
            # Dummy attribution for Isolation Forest
            attributions = [{s: 0.0 for s in self.sensor_names} for _ in range(len(sequences))]
            return raw_scores, preds, attributions

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            recon = self.model(X_tensor)
            
            # Channel-level squared residuals: (B, C)
            channel_residuals = ((recon - X_tensor) ** 2).mean(dim=1).cpu().numpy()
            # Overall score: mean across all channels
            scores = channel_residuals.mean(axis=1)
            is_anom = scores > self.threshold
            
            attributions = []
            for i in range(len(sequences)):
                attr_dict = {
                    self.sensor_names[c]: float(channel_residuals[i, c])
                    for c in range(self.input_dim)
                }
                attributions.append(attr_dict)

        return scores, is_anom, attributions
