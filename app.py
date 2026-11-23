"""
PredictiveX: Hugging Face Spaces App.
Theme: Midnight Black, Pure White, and Electric Lime Green.
High-performance Gradio interface for turbofan degradation, RUL estimation,
uncertainty quantification, and sensor anomaly attribution.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gradio as gr
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from src.datasets.cmapss_loader import load_raw_cmapss, SENSOR_DESCRIPTIONS
from src.preprocessing.preprocessor import CMAPSSPreprocessor
from src.features.feature_pipeline import generate_sequence_data
from src.models.deep_learning.hybrid_transformer_models import PatchTSTPrognosticModel
from src.models.degradation.probabilistic_rul import MCDropoutEstimator
from src.models.anomaly_detection.sensor_anomaly import SensorAnomalyDetector

# 1. Load System Artifacts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoints_dir = Path("checkpoints")

prep = CMAPSSPreprocessor.load(str(checkpoints_dir / "classical" / "preprocessor_FD001.joblib"))
sensor_names = prep.active_sensors
input_dim = len(sensor_names)

patch_model = PatchTSTPrognosticModel(input_dim=input_dim, patch_len=8, stride=4, d_model=64, nhead=4, num_layers=2).to(device)
patch_weights = checkpoints_dir / "deep_learning" / "PatchTST_FD001_best.pt"
if patch_weights.exists():
    patch_model.load_state_dict(torch.load(patch_weights, map_location=device)["model_state_dict"])
patch_model.eval()

mc_estimator = MCDropoutEstimator(patch_model, device=device, num_samples=30)
anomaly_detector = SensorAnomalyDetector(input_dim=input_dim, sensor_names=sensor_names, method="lstm_ae", device=device)

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

# Load test engines
_, test_raw, rul_true = load_raw_cmapss(subset="FD001")
test_norm = prep.transform(test_raw)


def run_diagnostics(engine_id: int):
    # Filter engine
    unit_df = test_raw[test_raw["unit_nr"] == engine_id].sort_values("time_cycles").copy()
    unit_norm = test_norm[test_norm["unit_nr"] == engine_id].sort_values("time_cycles").copy()
    total_observed_cycles = int(unit_df["time_cycles"].max())
    true_rul_val = float(rul_true[engine_id - 1])

    # Generate sequence & inference
    X_seq_unit, _, _ = generate_sequence_data(unit_norm, sensor_names, sequence_length=30, is_test=True)
    uncert = mc_estimator.predict_with_uncertainty(X_seq_unit, rul_cap=125.0)
    pred_rul = float(uncert["mean_rul"][0])
    p10 = float(uncert["lower_bound_p10"][0])
    p90 = float(uncert["upper_bound_p90"][0])
    conf_score = float(uncert["confidence"][0])

    with torch.no_grad():
        flat = torch.tensor(X_seq_unit.reshape(1, -1), dtype=torch.float32).to(device)
        hazard_prob = float(hazard_clf(flat).item())

    scores, is_anom, attributions = anomaly_detector.detect_anomalies(X_seq_unit)
    anom_score = float(scores[0])
    is_anom_bool = bool(is_anom[0])

    # Status badge
    if hazard_prob > 0.70 or pred_rul < 20:
        status_text = "CRITICAL HAZARD"
        status_color = "#EF4444"
    elif hazard_prob > 0.35 or pred_rul < 50:
        status_text = "INCIPIENT DEGRADATION"
        status_color = "#F59E0B"
    else:
        status_text = "NOMINAL OPERATING"
        status_color = "#00FF66"

    # Degradation plot
    past_cycles = np.arange(1, total_observed_cycles + 1)
    future_cycles = np.arange(total_observed_cycles, int(total_observed_cycles + pred_rul) + 1)
    
    fig_deg = go.Figure()
    fig_deg.add_trace(go.Scatter(
        x=past_cycles,
        y=[125.0 - (c * (125.0 - pred_rul) / total_observed_cycles) for c in past_cycles],
        mode="lines",
        name="Estimated Wear History",
        line=dict(color="#3B82F6", width=2.5)
    ))
    fig_deg.add_trace(go.Scatter(
        x=future_cycles,
        y=np.linspace(pred_rul, 0, len(future_cycles)),
        mode="lines+markers",
        name="PatchTST RUL Projection",
        line=dict(color="#00FF66", width=3),
        marker=dict(size=5, color="#00FF66")
    ))
    upper_future = np.linspace(p90, 0, len(future_cycles))
    lower_future = np.linspace(p10, 0, len(future_cycles))
    fig_deg.add_trace(go.Scatter(
        x=list(future_cycles) + list(future_cycles[::-1]),
        y=list(upper_future) + list(lower_future[::-1]),
        fill="toself",
        fillcolor="rgba(0, 255, 102, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Uncertainty Band"
    ))
    fig_deg.update_layout(
        paper_bgcolor="#080A0F",
        plot_bgcolor="#0D111A",
        font=dict(color="#FFFFFF"),
        xaxis=dict(gridcolor="#1E2433", title="Operational Cycles"),
        yaxis=dict(gridcolor="#1E2433", title="Remaining Useful Life (Cycles)"),
        height=380,
        margin=dict(l=30, r=30, t=30, b=30)
    )

    # Anomaly bar plot
    attr_dict = attributions[0]
    top_items = sorted(attr_dict.items(), key=lambda x: x[1])[-8:]
    sensors = [k for k, _ in top_items]
    resids = [v for _, v in top_items]

    fig_bar = go.Figure(go.Bar(
        x=resids,
        y=sensors,
        orientation="h",
        marker=dict(color="#00FF66", line=dict(color="#FFFFFF", width=0.5))
    ))
    fig_bar.update_layout(
        paper_bgcolor="#080A0F",
        plot_bgcolor="#0D111A",
        font=dict(color="#FFFFFF"),
        xaxis=dict(gridcolor="#1E2433", title="Reconstruction Residual Error"),
        yaxis=dict(gridcolor="#1E2433", title="Sensor"),
        height=380,
        margin=dict(l=30, r=30, t=30, b=30)
    )

    status_html = f"<div style='background-color:#11141D; padding:15px; border-radius:8px; border-left:5px solid {status_color};'><h3 style='margin:0; color:{status_color};'>{status_text}</h3><p style='margin:4px 0; color:#FFFFFF;'>Engine #{engine_id} | Observed Cycles: {total_observed_cycles}</p></div>"

    kpi_metrics = f"""
    - **Predicted RUL**: **{pred_rul:.1f} cycles** (80% Bounds: [{p10:.1f} - {p90:.1f}] cycles)
    - **True Ground Truth RUL**: **{true_rul_val:.0f} cycles** (Absolute Error: {abs(pred_rul - true_rul_val):.1f} cycles)
    - **Hazard Probability (<= 15 cycles)**: **{hazard_prob*100:.1f}%**
    - **Sensor Anomaly Alert**: **{'ANOMALOUS' if is_anom_bool else 'NOMINAL'}** (Residual: {anom_score:.4f})
    - **Epistemic Model Confidence**: **{conf_score*100:.1f}%**
    """

    return status_html, kpi_metrics, fig_deg, fig_bar


# Custom CSS Theme
custom_css = """
body, .gradio-container {
    background-color: #080A0F !important;
    color: #FFFFFF !important;
}
.gr-button-primary {
    background: #00FF66 !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border: none !important;
}
"""

with gr.Blocks(title="PredictiveX - Industrial Turbofan Prognostics", css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    # ⚡ PredictiveX: Industrial Turbofan Prognostics Platform
    ### Multivariate Sensor Degradation Modeling, Uncertainty Bounds & Hazard Risk Estimation
    """)

    with gr.Row():
        with gr.Column(scale=1):
            engine_dropdown = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Select Test Engine Asset (1-100)")
            run_btn = gr.Button("⚡ Run Prognostic Diagnostics", variant="primary")
            status_box = gr.HTML()
            metrics_box = gr.Markdown()

        with gr.Column(scale=2):
            deg_plot = gr.Plot(label="RUL Degradation Forecast & Uncertainty Cone")
            anom_plot = gr.Plot(label="Top Sensor Anomaly Residuals")

    run_btn.click(
        fn=run_diagnostics,
        inputs=[engine_dropdown],
        outputs=[status_box, metrics_box, deg_plot, anom_plot]
    )
    
    # Run once at launch
    demo.load(
        fn=run_diagnostics,
        inputs=[engine_dropdown],
        outputs=[status_box, metrics_box, deg_plot, anom_plot]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
