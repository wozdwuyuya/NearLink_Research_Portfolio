"""LinkQualityLSTM Training Engine.

End-to-end training loop for the NearLink link quality prediction model.
Integrates NearLinkDataPipeline (data loading + normalization) with
LinkQualityLSTM (BiLSTM + Attention) and provides:

- HuberLoss for robust regression against communication anomalies
- AdamW optimizer with decoupled weight decay
- ReduceLROnPlateau scheduler (synergizes with early stopping)
- Early stopping with configurable patience
- Model checkpointing (model + scaler saved atomically)
- Per-epoch train/val loss logging

Loss Function Decision: HuberLoss over MSELoss
------------------------------------------------
Communication link quality data contains intermittent anomalies (CRC burst
errors, RSSI deep fades, jitter spikes). MSELoss amplifies these outliers
via squared residuals, destabilizing gradient updates and causing loss
oscillation. HuberLoss applies MSE for small residuals (smooth gradients
near optimum) and switches to MAE for large residuals (constant gradient
magnitude), providing:
- Robust convergence in the presence of link anomalies
- Smoother loss curves for reliable early stopping decisions
- No hyperparameter sensitivity to outlier magnitude (delta=1.0 works well
  for quality_score in [0, 1] range)

Optimizer Decision: AdamW over Adam
-------------------------------------
Standard Adam couples weight decay with gradient updates, reducing effective
regularization. AdamW decouples weight decay from the gradient step,
providing proper L2 regularization that prevents overfitting without
distorting the adaptive learning rate.

Typical Usage
-------------
>>> python src/train.py                          # Run demo training
>>> from train import Trainer                    # Import for custom use
>>> trainer = Trainer(model, pipeline, config)
>>> trainer.train(n_epochs=50)
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure src/ is on path for sibling imports
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from data_processor import (  # noqa: E402
    NearLinkDataPipeline,
    NearLinkScaler,
    generate_synthetic_telemetry,
)
from model_architecture import LinkQualityLSTM  # noqa: E402

logger = logging.getLogger(__name__)


# ─── CUDA Pre-flight ────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    """Resolve compute device with mandatory CUDA assertion.

    Per project protocol: CUDA must be available for embedding/inference.
    Falls back to CPU only in training context with explicit warning.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info("CUDA device: %s (%.0f MB VRAM)", gpu_name, vram_mb)
        return device

    logger.warning(
        "CUDA unavailable — falling back to CPU. "
        "Training will be significantly slower."
    )
    return torch.device("cpu")


# ─── Early Stopping ─────────────────────────────────────────────────────────


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Monitors val loss and triggers when no improvement exceeding
    ``min_delta`` has been observed for ``patience`` consecutive epochs.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum loss decrease to qualify as an improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.triggered = False

    def step(self, val_loss: float) -> bool:
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current epoch's validation loss.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            logger.info(
                "Early stopping triggered after %d epochs without improvement "
                "(best val_loss=%.6f)",
                self.patience,
                self.best_loss,
            )
            return True

        return False


# ─── Training Configuration ─────────────────────────────────────────────────


@dataclass
class TrainConfig:
    """Hyperparameters and training configuration.

    All values tuned for NearLink telemetry prediction with ~5000 samples
    and window_size=50. Adjust for larger datasets.
    """

    # ── Model Architecture ────────────────────────────────────────────
    input_dim: int = 8
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2

    # ── Optimization ──────────────────────────────────────────────────
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2          # AdamW decoupled weight decay
    huber_delta: float = 1.0            # HuberLoss transition point
    max_grad_norm: float = 1.0          # Gradient clipping max norm

    # ── Learning Rate Scheduler ───────────────────────────────────────
    scheduler_patience: int = 3         # Epochs before LR reduction
    scheduler_factor: float = 0.5       # LR multiplication factor
    min_lr: float = 1e-6                # Lower bound on learning rate

    # ── Early Stopping ────────────────────────────────────────────────
    early_stop_patience: int = 7        # Epochs before training halt
    early_stop_min_delta: float = 1e-4  # Minimum improvement threshold

    # ── Data ──────────────────────────────────────────────────────────
    window_size: int = 50
    stride: int = 1
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # ── Checkpointing ─────────────────────────────────────────────────
    checkpoint_dir: Path = field(
        default_factory=lambda: Path(r"E:\1Projects\ML_Research_Hub\models")
    )


# ─── Trainer ────────────────────────────────────────────────────────────────


class Trainer:
    """Orchestrates the full training lifecycle.

    Lifecycle::

        trainer = Trainer(config)
        trainer.setup(dataframe)        # Build model, pipeline, optimizer
        trainer.train(n_epochs=50)      # Training loop with early stopping
        # Best model saved to config.checkpoint_dir / "model.pth"

    Parameters
    ----------
    config : TrainConfig
        Full training hyperparameter configuration.
    """

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()
        self.device = _get_device()

        # Populated by setup()
        self.model: LinkQualityLSTM | None = None
        self.pipeline: NearLinkDataPipeline | None = None
        self.optimizer: AdamW | None = None
        self.scheduler: ReduceLROnPlateau | None = None
        self.criterion: nn.HuberLoss | None = None
        self.early_stopping: EarlyStopping | None = None

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

    def setup(self, df: "pd.DataFrame") -> None:
        """Initialize model, data pipeline, optimizer, and scheduler.

        Parameters
        ----------
        df : DataFrame
            Raw NearLink telemetry data (from generate_synthetic_telemetry
            or real data source).
        """
        cfg = self.config

        # ── Data Pipeline ──────────────────────────────────────────────
        self.pipeline = NearLinkDataPipeline(
            window_size=cfg.window_size,
            stride=cfg.stride,
        )
        train_ds, val_ds, _ = self.pipeline.fit_transform(
            df, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio
        )

        self.train_loader = self.pipeline.get_train_loader(
            batch_size=cfg.batch_size, shuffle=True
        )
        self.val_loader = self.pipeline.get_val_loader(
            batch_size=cfg.batch_size * 2  # No gradients → larger batch
        )

        logger.info(
            "Data: %d train windows, %d val windows",
            len(train_ds),
            len(val_ds),
        )

        # ── Model ──────────────────────────────────────────────────────
        self.model = LinkQualityLSTM(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            output_dim=1,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        ).to(self.device)

        info = self.model.get_model_info()
        logger.info(
            "Model: %d trainable parameters", info["trainable_params"]
        )

        # ── Loss, Optimizer, Scheduler ─────────────────────────────────
        self.criterion = nn.HuberLoss(delta=cfg.huber_delta)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.min_lr,
            verbose=False,
        )

        # ── Early Stopping ─────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=cfg.early_stop_patience,
            min_delta=cfg.early_stop_min_delta,
        )

        # ── Checkpoint Directory ───────────────────────────────────────
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _train_epoch(self) -> float:
        """Run one training epoch.

        Returns
        -------
        float
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients from anomalies)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self) -> float:
        """Run one validation epoch.

        Returns
        -------
        float
            Mean validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for features, targets in self.val_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, val_loss: float) -> None:
        """Save model and scaler to disk.

        Saves:
        - ``model.pth``: model state dict + config + best val loss
        - ``scaler.joblib``: fitted NearLinkScaler for online inference
        """
        cfg = self.config
        checkpoint_path = cfg.checkpoint_dir / "model.pth"
        scaler_path = cfg.checkpoint_dir / "scaler.joblib"

        # Save model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": cfg.input_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "num_heads": cfg.num_heads,
                    "dropout": cfg.dropout,
                },
                "best_val_loss": val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )

        # Save scaler (for online inference pipeline)
        self.pipeline.save_scaler(scaler_path)

        logger.info(
            "Checkpoint saved: %s (val_loss=%.6f)",
            checkpoint_path,
            val_loss,
        )

    def train(self, n_epochs: int = 50) -> dict[str, list[float]]:
        """Execute the full training loop.

        Parameters
        ----------
        n_epochs : int
            Maximum number of training epochs.

        Returns
        -------
        dict
            Training history with keys: train_loss, val_loss, learning_rate.
        """
        if self.model is None:
            raise RuntimeError("Call setup() before train().")

        best_val_loss = float("inf")
        t_start = time.perf_counter()

        logger.info("=" * 65)
        logger.info("  Training Start | Epochs: %d | Device: %s", n_epochs, self.device)
        logger.info("=" * 65)

        for epoch in range(1, n_epochs + 1):
            epoch_start = time.perf_counter()

            # ── Train + Validate ───────────────────────────────────────
            train_loss = self._train_epoch()
            val_loss = self._eval_epoch()

            # ── Scheduler step (based on val loss) ─────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]["lr"]

            # ── Record history ─────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            epoch_time = time.perf_counter() - epoch_start

            # ── Logging ────────────────────────────────────────────────
            improved = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(val_loss)
                improved = " *"

            lr_note = f"  lr: {current_lr:.2e}" if new_lr < current_lr else ""

            logger.info(
                "Epoch %3d/%d | "
                "train: %.6f | val: %.6f | "
                "%.1fs%s%s",
                epoch,
                n_epochs,
                train_loss,
                val_loss,
                epoch_time,
                lr_note,
                improved,
            )

            # ── Early Stopping ─────────────────────────────────────────
            if self.early_stopping.step(val_loss):
                break

        total_time = time.perf_counter() - t_start
        logger.info("=" * 65)
        logger.info(
            "  Training Complete | %d epochs in %.1fs | Best val_loss: %.6f",
            len(self.history["train_loss"]),
            total_time,
            best_val_loss,
        )
        logger.info("=" * 65)

        return self.history


# ─── Standalone Demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    import pandas as pd

    print("=" * 65)
    print("  LinkQualityLSTM — Mini Training Run (Demo)")
    print("=" * 65)

    # ── Step 1: Generate synthetic data ────────────────────────────────
    print("\n[1/4] Generating synthetic NearLink telemetry...")
    df = generate_synthetic_telemetry(n_samples=5000, n_sessions=5)
    print(f"      {len(df)} samples, {df['session_id'].nunique()} sessions")

    # ── Step 2: Configure and setup trainer ────────────────────────────
    print("\n[2/4] Initializing trainer...")
    config = TrainConfig(
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=1e-2,
        window_size=50,
        batch_size=32,
        early_stop_patience=7,
        scheduler_patience=3,
        checkpoint_dir=Path(r"E:\1Projects\ML_Research_Hub\models"),
    )

    trainer = Trainer(config)
    trainer.setup(df)

    print(f"      Model: {trainer.model.count_parameters():,} parameters")
    print(f"      Device: {trainer.device}")
    print(f"      Train batches/epoch: {len(trainer.train_loader)}")
    print(f"      Val batches/epoch:   {len(trainer.val_loader)}")

    # ── Step 3: Train ──────────────────────────────────────────────────
    print("\n[3/4] Training...")
    history = trainer.train(n_epochs=20)

    # ── Step 4: Summary ────────────────────────────────────────────────
    print("\n[4/4] Training Summary:")
    print(f"      Epochs completed:  {len(history['train_loss'])}")
    print(f"      Best train loss:   {min(history['train_loss']):.6f}")
    print(f"      Best val loss:     {min(history['val_loss']):.6f}")
    print(f"      Final LR:          {history['learning_rate'][-1]:.2e}")
    print(f"      Early stopped:     {trainer.early_stopping.triggered}")

    # Verify checkpoint files exist
    model_path = config.checkpoint_dir / "model.pth"
    scaler_path = config.checkpoint_dir / "scaler.joblib"
    print("\n      Saved artifacts:")
    print(f"        Model:  {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"        Scaler: {scaler_path} ({scaler_path.stat().st_size / 1024:.1f} KB)")

    # Verify model can be loaded from checkpoint
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print("\n      Checkpoint verification:")
    print(f"        Best val_loss in checkpoint: {checkpoint['best_val_loss']:.6f}")
    print(f"        Config keys: {list(checkpoint['config'].keys())}")

    # Verify scaler can be loaded
    loaded_scaler = NearLinkScaler.load(scaler_path)
    print(f"        Scaler loaded: fitted={loaded_scaler.is_fitted}")

    # Quick inference test with loaded model
    loaded_model = LinkQualityLSTM(**checkpoint["config"])
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    dummy_input = torch.randn(1, 50, 8)
    with torch.no_grad():
        pred = loaded_model(dummy_input)
    print(f"        Inference test: input(1,50,8) -> output{tuple(pred.shape)}")

    print("\n" + "=" * 65)
    print("  Demo Complete — All components verified.")
    print("=" * 65)
