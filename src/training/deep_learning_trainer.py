"""
PredictiveX Deep Learning Training Engine.
Fully optimized for NVIDIA GeForce RTX 4080 SUPER with:
- CUDA 12.8 acceleration
- Automatic Mixed Precision (AMP FP16/BF16)
- Gradient clipping & AdamW optimizer
- CosineAnnealing / ReduceLROnPlateau scheduling
- Early stopping & best model checkpointing
- Evaluation tracking with official NASA Scoring Function
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.metrics import evaluate_rul_predictions


class DeepLearningTrainer:
    """
    High-performance PyTorch trainer for prognostic time-series models.
    """
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        clip_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints/deep_learning"
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.clip_grad_norm = clip_grad_norm
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
        )
        self.criterion = nn.SmoothL1Loss()
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        
        self.train_history = []
        self.best_val_rmse = float("inf")
        self.best_val_score = float("inf")

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 40,
        patience: int = 10,
        rul_cap: float = 125.0
    ) -> Dict[str, Any]:
        """
        Executes end-to-end training loop with validation early stopping.
        """
        patience_counter = 0
        start_time = time.perf_counter()
        
        print(f"Starting training: {self.model_name} on {self.device} (AMP: {self.use_amp})...")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.perf_counter()
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0
            
            for X_b, y_b in train_loader:
                X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        preds = self.model(X_b)
                        loss = self.criterion(preds, y_b)
                    self.scaler.scale(loss).backward()
                    if self.clip_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    preds = self.model(X_b)
                    loss = self.criterion(preds, y_b)
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    
                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / max(1, num_batches)
            
            # Validation Step
            val_metrics, val_loss = self.evaluate(val_loader, rul_cap=rul_cap)
            self.scheduler.step(val_loss)
            
            epoch_time = time.perf_counter() - epoch_start
            
            self.train_history.append({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_nasa_score": val_metrics["nasa_score"],
                "epoch_time_s": epoch_time,
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            # Check for improvement
            if val_metrics["rmse"] < self.best_val_rmse:
                self.best_val_rmse = val_metrics["rmse"]
                self.best_val_score = val_metrics["nasa_score"]
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_rmse": self.best_val_rmse,
                    "val_nasa_score": self.best_val_score,
                    "model_name": self.model_name
                }, self.best_checkpoint_path)
                improved_tag = " [BEST SAVED]"
            else:
                patience_counter += 1
                improved_tag = ""
                
            if epoch % 5 == 0 or epoch == 1 or improved_tag:
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.2f} | "
                    f"NASA Score: {val_metrics['nasa_score']:.1f} | "
                    f"Time: {epoch_time:.2f}s{improved_tag}"
                )
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
                break

        total_train_time = time.perf_counter() - start_time
        # Load best weights
        self.load_best()
        print(f"Training completed in {total_train_time:.2f}s. Best Val RMSE: {self.best_val_rmse:.2f}")
        return {
            "total_train_time": total_train_time,
            "best_val_rmse": self.best_val_rmse,
            "best_val_score": self.best_val_score,
            "history": self.train_history
        }

    def evaluate(self, data_loader: DataLoader, rul_cap: float = 125.0) -> Tuple[Dict[str, float], float]:
        """
        Runs inference and computes prognostic metrics.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X_b, y_b in data_loader:
                X_b = X_b.to(self.device, non_blocking=True)
                y_b = y_b.to(self.device, non_blocking=True)
                
                preds = self.model(X_b)
                loss = self.criterion(preds, y_b)
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(y_b.cpu().numpy().flatten())

        avg_loss = total_loss / max(1, num_batches)
        preds_arr = np.clip(np.array(all_preds), 0.0, rul_cap)
        targets_arr = np.array(all_targets)
        metrics = evaluate_rul_predictions(targets_arr, preds_arr)
        return metrics, avg_loss

    def predict_sequences(self, X_seq: np.ndarray, batch_size: int = 128, rul_cap: float = 125.0) -> np.ndarray:
        """
        Inference on raw numpy array of sequences.
        """
        self.model.eval()
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for (X_b,) in loader:
                X_b = X_b.to(self.device)
                out = self.model(X_b)
                preds.extend(out.cpu().numpy().flatten())
                
        return np.clip(np.array(preds), 0.0, rul_cap)

    def load_best(self):
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
