"""
Live Host System & Checkpoint Verification Script.
Loads models directly from disk, verifies GPU VRAM allocation,
and evaluates test engines in real time.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.datasets.cmapss_loader import load_raw_cmapss
from src.preprocessing.preprocessor import CMAPSSPreprocessor
from src.features.feature_pipeline import generate_sequence_data
from src.models.deep_learning.hybrid_transformer_models import PatchTSTPrognosticModel
from src.utils.metrics import evaluate_rul_predictions

print("=" * 65)
print("PREDICTIVEX: LIVE HARDWARE & CHECKPOINT AUDIT")
print("=" * 65)
print(f"PyTorch Version:  {torch.__version__}")
print(f"CUDA Available:   {torch.cuda.is_available()}")
print(f"Target GPU:       {torch.cuda.get_device_name(0)}")
print(f"Initial GPU VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")

# Checkpoints existence
checkpoints = list(Path("checkpoints").rglob("*.*"))
print(f"\nTotal Model Checkpoints Found on Disk: {len(checkpoints)}")
for c in checkpoints:
    print(f"  - {c.relative_to(Path('.'))} ({c.stat().st_size:,} bytes)")

# Load raw test data and ground truth
print("\nLoading test dataset...")
_, test_raw, rul_true = load_raw_cmapss(subset="FD001")
prep = CMAPSSPreprocessor.load("checkpoints/classical/preprocessor_FD001.joblib")
test_norm = prep.transform(test_raw)

# Generate sequences and load PatchTST best weights
X_test_seq, _, _ = generate_sequence_data(test_norm, prep.active_sensors, sequence_length=30, is_test=True)
device = torch.device("cuda")

print("Loading PatchTST trained weights onto RTX 4080 SUPER...")
model = PatchTSTPrognosticModel(input_dim=len(prep.active_sensors), patch_len=8, stride=4, d_model=64, nhead=4, num_layers=2).to(device)
ckpt = torch.load("checkpoints/deep_learning/PatchTST_FD001_best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print(f"Active GPU VRAM with Model: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")

with torch.no_grad():
    t_in = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    preds = model(t_in).cpu().numpy().clip(0, 125)

metrics = evaluate_rul_predictions(rul_true, preds)
print("\n" + "=" * 65)
print("LIVE EVALUATION RESULTS ACROSS ALL 100 TEST ENGINES:")
print("=" * 65)
print(f"  Test RMSE:         {metrics['rmse']:.2f} cycles")
print(f"  Test MAE:          {metrics['mae']:.2f} cycles")
print(f"  Test NASA Score:   {metrics['nasa_score']:.1f}")
print(f"  Test R^2 Score:    {metrics['r2']:.4f}")
print(f"  Mean Bias:         {metrics['mean_bias']:.2f} cycles")
print(f"  Early Predictions: {metrics['early_predictions']} / 100")
print(f"  Late Predictions:  {metrics['late_predictions']} / 100")

print("\nSample Test Engine Predictions:")
for i in [0, 9, 24, 49, 99]:
    diff = preds[i] - rul_true[i]
    tag = "EARLY (Safe)" if diff < 0 else "LATE"
    print(f"  Engine #{i+1:3d} -> Predicted: {preds[i]:5.1f} | True RUL: {rul_true[i]:3.0f} | Error: {diff:+5.1f} cycles ({tag})")
print("=" * 65)
