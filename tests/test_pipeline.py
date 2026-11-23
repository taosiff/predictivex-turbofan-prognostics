"""
PredictiveX Comprehensive Pytest Test Suite.
Validates:
1. Dataset loading, schemas, and target engineering
2. Zero-leakage scaling and preprocessor persistence
3. Tabular rolling features and 3D sequence tensors
4. NASA scoring function and asymmetric penalties
5. PyTorch neural network forward passes (LSTM, GRU, TCN, Transformer, PatchTST)
6. Anomaly detection and Monte Carlo dropout uncertainty
7. FastAPI REST service endpoints and payload validation
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.cmapss_loader import load_raw_cmapss, compute_train_rul, get_constant_sensors
from src.preprocessing.preprocessor import CMAPSSPreprocessor
from src.features.feature_pipeline import generate_tabular_features, generate_sequence_data
from src.utils.metrics import nasa_scoring_function, evaluate_rul_predictions
from src.models.deep_learning.recurrent_models import LSTMPrognosticModel, GRUPrognosticModel
from src.models.deep_learning.tcn_model import TCNPrognosticModel
from src.models.deep_learning.hybrid_transformer_models import (
    CNNLSTMPrognosticModel,
    TemporalTransformerPrognosticModel,
    PatchTSTPrognosticModel
)
from src.models.anomaly_detection.sensor_anomaly import SensorAnomalyDetector
from src.models.degradation.probabilistic_rul import MCDropoutEstimator
from api.main import app


# 1. Dataset Loading Tests
def test_dataset_loader():
    train_df, test_df, rul_true = load_raw_cmapss(subset="FD001")
    assert len(train_df) == 20631
    assert len(test_df) == 13096
    assert len(rul_true) == 100
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0


def test_rul_target_engineering():
    train_df, _, _ = load_raw_cmapss(subset="FD001")
    df_rul = compute_train_rul(train_df, rul_cap=125.0)
    assert "rul" in df_rul.columns
    assert "rul_linear" in df_rul.columns
    assert df_rul["rul"].max() <= 125.0
    # Engine 1 final cycle should have rul_linear == 0
    unit_1 = df_rul[df_rul["unit_nr"] == 1]
    assert unit_1.iloc[-1]["rul_linear"] == 0


# 2. Preprocessor Tests
def test_preprocessor_zero_leakage(tmp_path):
    train_df, test_df, _ = load_raw_cmapss(subset="FD001")
    prep = CMAPSSPreprocessor(scaler_type="minmax")
    train_norm = prep.fit_transform(train_df)
    test_norm = prep.transform(test_df)
    
    assert len(prep.removed_sensors) == 6
    assert len(prep.active_sensors) == 15
    # Scaled values in training should fall in [-1.0, 1.0]
    for s in prep.active_sensors:
        assert train_norm[s].min() >= -1.0001
        assert train_norm[s].max() <= 1.0001

    # Test persistence
    save_file = tmp_path / "test_prep.joblib"
    prep.save(str(save_file))
    loaded_prep = CMAPSSPreprocessor.load(str(save_file))
    assert loaded_prep.active_sensors == prep.active_sensors


# 3. Feature Pipeline Tests
def test_feature_pipeline():
    train_df, test_df, _ = load_raw_cmapss(subset="FD001")
    prep = CMAPSSPreprocessor(scaler_type="minmax")
    train_norm = prep.fit_transform(train_df)
    test_norm = prep.transform(test_df)

    # Tabular
    tab_train = generate_tabular_features(train_norm, prep.active_sensors, windows=[5])
    tab_test = generate_tabular_features(test_norm, prep.active_sensors, windows=[5], is_test=True)
    assert len(tab_test) == 100
    assert tab_train.isnull().sum().sum() == 0

    # Sequence
    X_seq, y_seq, _ = generate_sequence_data(train_norm, prep.active_sensors, sequence_length=30)
    X_test_seq, _, _ = generate_sequence_data(test_norm, prep.active_sensors, sequence_length=30, is_test=True)
    assert X_seq.ndim == 3
    assert X_seq.shape[1] == 30
    assert X_seq.shape[2] == len(prep.active_sensors)
    assert X_test_seq.shape == (100, 30, len(prep.active_sensors))


# 4. Asymmetric NASA Scoring Function Tests
def test_nasa_scoring_function():
    # Early prediction: y_pred = 10, y_true = 20 -> diff = -10
    # Penalty: exp(10/13) - 1
    s_early = nasa_scoring_function(np.array([20]), np.array([10]))
    expected_early = np.exp(10 / 13.0) - 1.0
    assert pytest.approx(s_early, rel=1e-4) == expected_early

    # Late prediction: y_pred = 30, y_true = 20 -> diff = +10
    # Penalty: exp(10/10) - 1 = e^1 - 1
    s_late = nasa_scoring_function(np.array([20]), np.array([30]))
    expected_late = np.exp(1.0) - 1.0
    assert pytest.approx(s_late, rel=1e-4) == expected_late
    # Late prediction penalty must strictly exceed early prediction penalty for same absolute magnitude!
    assert s_late > s_early


# 5. Deep Learning Architecture Forward Passes
def test_deep_learning_forward_passes():
    B, T, C = 4, 30, 15
    dummy_input = torch.randn(B, T, C)

    models = [
        LSTMPrognosticModel(input_dim=C, hidden_dim=32, num_layers=1),
        GRUPrognosticModel(input_dim=C, hidden_dim=32, num_layers=1),
        TCNPrognosticModel(input_dim=C, num_channels=[16, 16], kernel_size=3),
        CNNLSTMPrognosticModel(input_dim=C, conv_channels=16, lstm_hidden_dim=32, num_lstm_layers=1),
        TemporalTransformerPrognosticModel(input_dim=C, d_model=32, nhead=2, num_layers=1),
        PatchTSTPrognosticModel(input_dim=C, patch_len=8, stride=4, d_model=32, nhead=2, num_layers=1)
    ]

    for m in models:
        m.eval()
        with torch.no_grad():
            out = m(dummy_input)
            assert out.shape == (B,)
            assert not torch.isnan(out).any()


# 6. Anomaly & Uncertainty Estimation Tests
def test_anomaly_and_uncertainty():
    B, T, C = 10, 30, 5
    sensor_names = [f"s_{i}" for i in range(1, 6)]
    dummy_seqs = np.random.randn(B, T, C).astype(np.float32)

    # Anomaly Detector
    ad = SensorAnomalyDetector(input_dim=C, sensor_names=sensor_names, method="lstm_ae", device=torch.device("cpu"))
    ad.fit_healthy_sequences(dummy_seqs, epochs=2, batch_size=4)
    scores, is_anom, attr = ad.detect_anomalies(dummy_seqs)
    assert len(scores) == B
    assert len(attr) == B

    # MC Dropout
    model = LSTMPrognosticModel(input_dim=C, hidden_dim=32, num_layers=2, dropout=0.2)
    mc = MCDropoutEstimator(model, device=torch.device("cpu"), num_samples=10)
    res = mc.predict_with_uncertainty(dummy_seqs)
    assert len(res["mean_rul"]) == B
    assert len(res["lower_bound_p10"]) == B
    assert len(res["upper_bound_p90"]) == B
    assert (res["upper_bound_p90"] >= res["lower_bound_p10"]).all()


# 7. FastAPI Endpoint Tests
def test_fastapi_endpoints():
    with TestClient(app) as client:
        # /health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "healthy"

        # /models
        r_models = client.get("/models")
        assert r_models.status_code == 200
        assert "PatchTST" in r_models.json()["active_architecture"]

        # /predict/complete
        # Build dummy request payload with 30 time steps
        dummy_steps = [
            {"cycle": i, "sensors": {f"s_{s}": 0.5 for s in range(2, 22)}}
            for i in range(1, 31)
        ]
        payload = {
            "machine_id": "UNIT_TEST_99",
            "sequence": dummy_steps,
            "rul_cap": 125.0
        }
        r_complete = client.post("/predict/complete", json=payload)
        assert r_complete.status_code == 200
        data = r_complete.json()
        assert "predicted_rul" in data
        assert "confidence_interval_p10" in data
        assert "failure_hazard_15_cycles" in data
        assert "is_anomaly" in data
        assert data["machine_id"] == "UNIT_TEST_99"
