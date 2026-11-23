"""
PredictiveX Production FastAPI REST Service.
Provides endpoints for:
- System health and loaded model registry
- Point and uncertainty-bounded RUL prediction
- Horizon failure-risk hazard classification
- Multivariate sensor anomaly detection
- Unified complete prognostic diagnostics
- CSV batch file inference
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.datasets.cmapss_loader import SENSOR_COLUMNS, SENSOR_DESCRIPTIONS
from src.preprocessing.preprocessor import CMAPSSPreprocessor
from src.models.deep_learning.hybrid_transformer_models import PatchTSTPrognosticModel
from src.models.degradation.probabilistic_rul import MCDropoutEstimator
from src.models.anomaly_detection.sensor_anomaly import SensorAnomalyDetector

app = FastAPI(
    title="PredictiveX - Industrial Predictive Maintenance API",
    description="Production Condition-Based Maintenance & RUL Prognostics Platform",
    version="1.0.0"
)

# Global model state cache
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, Any] = {}
PREPROCESSOR: Optional[CMAPSSPreprocessor] = None


def load_artifacts():
    global MODELS, PREPROCESSOR
    checkpoints_dir = Path("checkpoints")
    
    # 1. Load Preprocessor
    prep_path = checkpoints_dir / "classical" / "preprocessor_FD001.joblib"
    if prep_path.exists():
        PREPROCESSOR = CMAPSSPreprocessor.load(str(prep_path))
    else:
        PREPROCESSOR = CMAPSSPreprocessor(scaler_type="minmax")
        # Ensure default active sensors if unpickled
        PREPROCESSOR.active_sensors = [f"s_{i}" for i in [2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
        
    sensor_names = PREPROCESSOR.active_sensors or [f"s_{i}" for i in range(1, 22)]
    input_dim = len(sensor_names)

    # 2. Load PatchTST Model
    patch_path = checkpoints_dir / "deep_learning" / "PatchTST_FD001_best.pt"
    patch_model = PatchTSTPrognosticModel(input_dim=input_dim, patch_len=8, stride=4, d_model=64, nhead=4, num_layers=2).to(device)
    if patch_path.exists():
        patch_model.load_state_dict(torch.load(patch_path, map_location=device)["model_state_dict"])
    patch_model.eval()
    
    MODELS["rul_model"] = patch_model
    MODELS["mc_estimator"] = MCDropoutEstimator(patch_model, device=device, num_samples=30)

    # 3. Load Anomaly Detector
    anomaly_detector = SensorAnomalyDetector(input_dim=input_dim, sensor_names=sensor_names, method="lstm_ae", device=device)
    MODELS["anomaly_detector"] = anomaly_detector

    # 4. Load Hazard Classifier
    hazard_path = checkpoints_dir / "advanced" / "hazard_classifier_FD001.pt"
    hazard_clf = torch.nn.Sequential(
        torch.nn.Linear(input_dim * 30, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    ).to(device)
    if hazard_path.exists():
        hazard_clf.load_state_dict(torch.load(hazard_path, map_location=device))
    hazard_clf.eval()
    MODELS["hazard_classifier"] = hazard_clf


# Load immediately on import
load_artifacts()


@app.on_event("startup")
def startup_event():
    if PREPROCESSOR is None or not MODELS:
        load_artifacts()


# Pydantic Schemas
class SensorTimeStep(BaseModel):
    cycle: int = Field(..., description="Operational cycle index")
    sensors: Dict[str, float] = Field(..., description="Map of active sensor channel values")


class SequencePredictionRequest(BaseModel):
    machine_id: str = Field(default="UNIT_001", description="Equipment Identifier")
    sequence: List[SensorTimeStep] = Field(..., min_length=1, description="List of time steps (up to 30)")
    rul_cap: float = Field(default=125.0, description="RUL piecewise cap")


class CompletePrognosticResponse(BaseModel):
    machine_id: str
    predicted_rul: float
    confidence_interval_p10: float
    confidence_interval_p90: float
    epistemic_std: float
    confidence_score: float
    failure_hazard_15_cycles: float
    is_anomaly: bool
    anomaly_score: float
    top_sensor_attributions: Dict[str, float]
    health_status: str
    inference_duration_ms: float
    model_version: str = "PredictiveX-v1.0"
    disclaimer: str = "Predictions are statistical decision-support estimates, not guaranteed failure dates."


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "loaded_models": list(MODELS.keys())
    }


@app.get("/models")
def list_models():
    return {
        "active_architecture": "PatchTST (Patch-based Temporal Attention)",
        "framework": "PyTorch 2.11.0+cu128",
        "acceleration": "CUDA 12.8 / RTX 4080 SUPER",
        "supported_subsets": ["FD001", "FD002", "FD003", "FD004"],
        "active_sensors": PREPROCESSOR.active_sensors if PREPROCESSOR else []
    }


def _prepare_sequence_tensor(request: SequencePredictionRequest, seq_len: int = 30) -> np.ndarray:
    active_sensors = PREPROCESSOR.active_sensors
    raw_rows = []
    for step in request.sequence:
        row = [step.sensors.get(s, 0.0) for s in active_sensors]
        raw_rows.append(row)
        
    arr = np.array(raw_rows, dtype=np.float32)
    # Apply preprocessor scaling
    if PREPROCESSOR and PREPROCESSOR.is_fitted:
        df_temp = pd.DataFrame(arr, columns=active_sensors)
        arr = PREPROCESSOR.transform(df_temp)[active_sensors].values
        
    cur_len = len(arr)
    if cur_len >= seq_len:
        seq = arr[-seq_len:]
    else:
        pad_width = seq_len - cur_len
        seq = np.pad(arr, ((pad_width, 0), (0, 0)), mode="edge")
        
    return np.expand_dims(seq, axis=0)  # Shape: (1, seq_len, input_dim)


@app.post("/predict/rul")
def predict_rul(request: SequencePredictionRequest):
    seq = _prepare_sequence_tensor(request)
    patch_model = MODELS["rul_model"]
    start = time.perf_counter()
    with torch.no_grad():
        t = torch.tensor(seq, dtype=torch.float32).to(device)
        rul_raw = patch_model(t).item()
        rul_pred = float(np.clip(rul_raw, 0.0, request.rul_cap))
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "machine_id": request.machine_id,
        "predicted_rul": round(rul_pred, 2),
        "latency_ms": round(latency_ms, 3)
    }


@app.post("/predict/failure-risk")
def predict_failure_risk(request: SequencePredictionRequest):
    seq = _prepare_sequence_tensor(request)
    hazard_clf = MODELS["hazard_classifier"]
    start = time.perf_counter()
    with torch.no_grad():
        flat = torch.tensor(seq.reshape(1, -1), dtype=torch.float32).to(device)
        risk = float(hazard_clf(flat).item())
    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "machine_id": request.machine_id,
        "failure_hazard_15_cycles": round(risk, 4),
        "critical_alert": risk >= 0.70,
        "latency_ms": round(latency_ms, 3)
    }


@app.post("/detect/anomalies")
def detect_anomalies(request: SequencePredictionRequest):
    seq = _prepare_sequence_tensor(request)
    detector: SensorAnomalyDetector = MODELS["anomaly_detector"]
    scores, is_anom, attributions = detector.detect_anomalies(seq)
    return {
        "machine_id": request.machine_id,
        "anomaly_score": round(float(scores[0]), 5),
        "is_anomaly": bool(is_anom[0]),
        "channel_residuals": attributions[0]
    }


@app.post("/predict/complete", response_model=CompletePrognosticResponse)
def predict_complete(request: SequencePredictionRequest):
    start = time.perf_counter()
    seq = _prepare_sequence_tensor(request)
    
    # 1. Uncertainty RUL
    mc_estimator: MCDropoutEstimator = MODELS["mc_estimator"]
    uncert = mc_estimator.predict_with_uncertainty(seq, rul_cap=request.rul_cap)
    
    # 2. Hazard Risk
    hazard_clf = MODELS["hazard_classifier"]
    with torch.no_grad():
        flat = torch.tensor(seq.reshape(1, -1), dtype=torch.float32).to(device)
        risk = float(hazard_clf(flat).item())
        
    # 3. Anomaly Detection
    detector: SensorAnomalyDetector = MODELS["anomaly_detector"]
    scores, is_anom, attributions = detector.detect_anomalies(seq)
    
    rul_val = float(uncert["mean_rul"][0])
    if rul_val < 20:
        health_status = "CRITICAL: Urgent Maintenance Required"
    elif rul_val < 50:
        health_status = "WARNING: Incipient Degradation Detected"
    else:
        health_status = "NOMINAL: Operating in Normal Bounds"

    latency_ms = (time.perf_counter() - start) * 1000.0
    
    # Top 5 contributing channels
    top_attrs = dict(sorted(attributions[0].items(), key=lambda x: x[1], reverse=True)[:5])

    return CompletePrognosticResponse(
        machine_id=request.machine_id,
        predicted_rul=round(rul_val, 2),
        confidence_interval_p10=round(float(uncert["lower_bound_p10"][0]), 2),
        confidence_interval_p90=round(float(uncert["upper_bound_p90"][0]), 2),
        epistemic_std=round(float(uncert["std_rul"][0]), 2),
        confidence_score=round(float(uncert["confidence"][0]), 4),
        failure_hazard_15_cycles=round(risk, 4),
        is_anomaly=bool(is_anom[0]),
        anomaly_score=round(float(scores[0]), 5),
        top_sensor_attributions=top_attrs,
        health_status=health_status,
        inference_duration_ms=round(latency_ms, 2)
    )
