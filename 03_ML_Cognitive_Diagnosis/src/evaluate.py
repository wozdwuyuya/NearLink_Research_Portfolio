"""LinkQualityLSTM Evaluation & Visualization Engine.

Loads trained model checkpoint and scaler, runs inference on the held-out
test set, computes regression metrics (MSE, RMSE, MAE), and generates
publication-quality figures for academic papers.

Generated Figures
-----------------
Figure_1_Prediction_Curve.png
    Overlay of ground-truth vs. predicted link quality for 200 consecutive
    test timesteps, demonstrating temporal tracking fidelity.

Figure_2_Error_Distribution.png
    Histogram + KDE of prediction errors (pred - actual) with fitted normal
    curve, demonstrating that errors are zero-centered and normally distributed.

Metrics
-------
- MSE  (Mean Squared Error): penalizes large deviations
- RMSE (Root MSE): interpretable in original quality_score units
- MAE  (Mean Absolute Error): robust summary of typical error magnitude

Typical Usage
-------------
>>> python src/evaluate.py                  # Full evaluation + figure generation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

# Non-interactive backend for headless figure saving
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

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

# ─── Publication Style Configuration ────────────────────────────────────────

# Academic paper figure defaults: serif fonts, clean spines, high DPI
PUBLICATION_RC: dict = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "lines.antialiased": True,
}

# Color palette (colorblind-safe)
COLOR_TRUTH: str = "#2166AC"     # Blue — ground truth
COLOR_PRED: str = "#D6604D"      # Red-orange — prediction
COLOR_ERROR: str = "#4DAF4A"     # Green — error histogram
COLOR_KDE: str = "#E41A1C"       # Red — KDE curve
COLOR_NORMAL: str = "#377EB8"    # Blue — fitted normal


# ─── Checkpoint Loader ──────────────────────────────────────────────────────


def load_checkpoint(
    model_path: Path,
    scaler_path: Path,
    device: torch.device,
) -> tuple[LinkQualityLSTM, NearLinkScaler, dict]:
    """Load trained model, scaler, and checkpoint metadata.

    Parameters
    ----------
    model_path : Path
        Path to model.pth checkpoint file.
    scaler_path : Path
        Path to scaler.joblib file.
    device : torch.device
        Target device for model placement.

    Returns
    -------
    model : LinkQualityLSTM
        Model loaded with trained weights, in eval mode.
    scaler : NearLinkScaler
        Fitted scaler for data normalization.
    metadata : dict
        Checkpoint metadata (config, best_val_loss, history).
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Reconstruct and load model
    model = LinkQualityLSTM(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load fitted scaler
    scaler = NearLinkScaler.load(scaler_path)

    logger.info(
        "Checkpoint loaded: %d params, best_val_loss=%.6f",
        model.count_parameters(),
        checkpoint["best_val_loss"],
    )

    return model, scaler, checkpoint


# ─── Inference Engine ───────────────────────────────────────────────────────


@torch.no_grad()
def collect_predictions(
    model: LinkQualityLSTM,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and collect all predictions + targets.

    Parameters
    ----------
    model : LinkQualityLSTM
        Model in eval mode.
    dataloader : DataLoader
        Test set DataLoader yielding (features, targets) batches.
    device : torch.device
        Compute device.

    Returns
    -------
    predictions : ndarray (N,)
        Flattened array of model predictions.
    ground_truth : ndarray (N,)
        Flattened array of ground-truth quality scores.
    """
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for features, targets in dataloader:
        features = features.to(device)
        preds = model(features)

        all_preds.append(preds.cpu())
        all_targets.append(targets)

    predictions = torch.cat(all_preds, dim=0).squeeze(-1).numpy()
    ground_truth = torch.cat(all_targets, dim=0).squeeze(-1).numpy()

    logger.info(
        "Inference complete: %d test samples collected", len(predictions)
    )

    return predictions, ground_truth


# ─── Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Parameters
    ----------
    predictions : ndarray (N,)
        Model predictions.
    ground_truth : ndarray (N,)
        Ground-truth values.

    Returns
    -------
    dict
        Keys: mse, rmse, mae, mape, r_squared, max_error.
    """
    errors = predictions - ground_truth

    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    # MAPE (avoid division by zero)
    nonzero_mask = ground_truth != 0
    if nonzero_mask.any():
        mape = float(
            np.mean(np.abs(errors[nonzero_mask] / ground_truth[nonzero_mask])) * 100
        )
    else:
        mape = float("nan")

    # R-squared
    ss_res = np.sum(errors**2)
    ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Max absolute error
    max_error = float(np.max(np.abs(errors)))

    # Shapiro-Wilk normality test on errors (H0: errors are normal)
    # Use a subsample if N > 5000 (Shapiro-Wilk limitation)
    sample = errors[:5000] if len(errors) > 5000 else errors
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r_squared": r_squared,
        "max_error": max_error,
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "n_samples": len(predictions),
    }


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print evaluation metrics to terminal."""
    print("\n" + "=" * 55)
    print("  Test Set Evaluation Metrics")
    print("=" * 55)
    print(f"  Samples:        {metrics['n_samples']}")
    print(f"  MSE:            {metrics['mse']:.6f}")
    print(f"  RMSE:           {metrics['rmse']:.6f}")
    print(f"  MAE:            {metrics['mae']:.6f}")
    print(f"  MAPE:           {metrics['mape']:.2f}%")
    print(f"  R-squared:      {metrics['r_squared']:.6f}")
    print(f"  Max Error:      {metrics['max_error']:.6f}")
    print("-" * 55)
    print(f"  Shapiro-Wilk:   W={metrics['shapiro_stat']:.4f}, "
          f"p={metrics['shapiro_p']:.4e}")
    normality = "NORMAL" if metrics["shapiro_p"] > 0.05 else "NON-NORMAL"
    print(f"  Error Dist:     {normality} (alpha=0.05)")
    print("=" * 55)


# ─── Figure 1: Prediction Curve ────────────────────────────────────────────


def plot_prediction_curve(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Path,
    n_steps: int = 200,
    training_history: dict | None = None,
) -> None:
    """Plot ground-truth vs. predicted link quality for N consecutive steps.

    Creates a publication-quality line chart showing how well the model
    tracks actual link quality dynamics over time.

    Parameters
    ----------
    predictions : ndarray (N,)
        Model predictions.
    ground_truth : ndarray (N,)
        Ground-truth quality scores.
    save_path : Path
        Output file path.
    n_steps : int
        Number of consecutive timesteps to display.
    training_history : dict, optional
        If provided, includes a small inset showing training convergence.
    """
    with plt.rc_context(PUBLICATION_RC):
        # Truncate to n_steps (centered in the test set)
        n = min(n_steps, len(predictions))
        mid = len(predictions) // 2
        start = mid - n // 2
        end = start + n

        t = np.arange(n)
        gt = ground_truth[start:end]
        pred = predictions[start:end]

        # ── Main plot ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[2.5, 1])

        ax = axes[0]
        ax.plot(t, gt, color=COLOR_TRUTH, label="Ground Truth", alpha=0.9)
        ax.plot(
            t, pred, color=COLOR_PRED, label="LSTM Prediction",
            alpha=0.85, linestyle="--", linewidth=1.0,
        )
        ax.fill_between(
            t, gt, pred, alpha=0.12, color=COLOR_PRED, label="Residual",
        )

        ax.set_xlabel("Test Timestep")
        ax.set_ylabel("Link Quality Score")
        ax.set_title("(a) Temporal Prediction vs. Ground Truth")
        ax.legend(loc="upper right", edgecolor="0.8")
        ax.set_xlim(0, n - 1)
        ax.set_ylim(
            min(gt.min(), pred.min()) - 0.02,
            max(gt.max(), pred.max()) + 0.02,
        )

        # ── Inset: training convergence (if history provided) ──────────
        if training_history and len(training_history.get("train_loss", [])) > 1:
            ax_inset = axes[1]
            epochs = range(1, len(training_history["train_loss"]) + 1)
            ax_inset.plot(
                epochs, training_history["train_loss"],
                color="#636363", label="Train Loss", marker="o",
                markersize=3, linewidth=1.0,
            )
            ax_inset.plot(
                epochs, training_history["val_loss"],
                color=COLOR_PRED, label="Val Loss", marker="s",
                markersize=3, linewidth=1.0,
            )
            ax_inset.set_xlabel("Epoch")
            ax_inset.set_ylabel("Huber Loss")
            ax_inset.set_title("(b) Training Convergence")
            ax_inset.legend(loc="upper right", edgecolor="0.8")
            ax_inset.set_yscale("log")
        else:
            axes[1].set_visible(False)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Figure 1 saved: %s", save_path)


# ─── Figure 2: Error Distribution ──────────────────────────────────────────


def plot_error_distribution(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Path,
    metrics: dict[str, float] | None = None,
) -> None:
    """Plot histogram + KDE of prediction errors with normal fit.

    Demonstrates that prediction errors are approximately normally
    distributed and concentrated near zero — a key assumption for
    evaluating model reliability.

    Parameters
    ----------
    predictions : ndarray (N,)
        Model predictions.
    ground_truth : ndarray (N,)
        Ground-truth quality scores.
    save_path : Path
        Output file path.
    metrics : dict, optional
        If provided, annotates the plot with MSE/RMSE/MAE values.
    """
    with plt.rc_context(PUBLICATION_RC):
        errors = predictions - ground_truth

        fig, ax = plt.subplots(figsize=(6.5, 4.5))

        # Histogram (normalized to density)
        n_bins = min(60, max(20, int(np.sqrt(len(errors)))))
        ax.hist(
            errors, bins=n_bins, density=True,
            color=COLOR_ERROR, alpha=0.55, edgecolor="white",
            linewidth=0.4, label="Error Histogram",
        )

        # KDE overlay
        sns.kdeplot(
            x=errors, ax=ax, color=COLOR_KDE, linewidth=1.5,
            label="KDE Estimate", fill=False,
        )

        # Fitted normal distribution
        mu, sigma = float(np.mean(errors)), float(np.std(errors))
        x_range = np.linspace(errors.min(), errors.max(), 300)
        normal_pdf = stats.norm.pdf(x_range, mu, sigma)
        ax.plot(
            x_range, normal_pdf, color=COLOR_NORMAL, linewidth=1.5,
            linestyle="--", label=f"Normal Fit ($\\mu$={mu:.4f}, $\\sigma$={sigma:.4f})",
        )

        # Vertical reference line at zero
        ax.axvline(x=0, color="0.3", linewidth=0.6, linestyle=":", alpha=0.7)

        # Annotate metrics
        if metrics:
            textstr = (
                f"MSE = {metrics['mse']:.5f}\n"
                f"RMSE = {metrics['rmse']:.5f}\n"
                f"MAE = {metrics['mae']:.5f}\n"
                f"$R^2$ = {metrics['r_squared']:.4f}"
            )
            props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8)
            ax.text(
                0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=props, family="monospace",
            )

        ax.set_xlabel("Prediction Error (Predicted $-$ Actual)")
        ax.set_ylabel("Probability Density")
        ax.set_title("Prediction Error Distribution")
        ax.legend(loc="upper left", edgecolor="0.8")

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        logger.info("Figure 2 saved: %s", save_path)


# ─── End-to-End Evaluation Pipeline ────────────────────────────────────────


def run_evaluation(
    model_dir: Path = Path(r"E:\1Projects\ML_Research_Hub\models"),
    figure_dir: Path = Path(r"E:\1Projects\ML_Research_Hub\figures"),
) -> dict[str, float]:
    """Execute full evaluation pipeline: load → infer → metrics → figures.

    Parameters
    ----------
    model_dir : Path
        Directory containing model.pth and scaler.joblib.
    figure_dir : Path
        Directory to save generated figures.

    Returns
    -------
    dict
        Computed evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Step 1: Load checkpoint ────────────────────────────────────────
    print("\n[1/5] Loading checkpoint...")
    model, scaler, checkpoint = load_checkpoint(
        model_dir / "model.pth", model_dir / "scaler.joblib", device,
    )

    # ── Step 2: Reconstruct test set (same pipeline as training) ───────
    print("[2/5] Reconstructing test dataset...")
    df = generate_synthetic_telemetry(n_samples=5000, n_sessions=5, seed=42)
    pipeline = NearLinkDataPipeline(window_size=50, stride=1)
    pipeline.scaler = scaler  # Use the loaded (fitted) scaler directly

    # Re-split and normalize using the loaded scaler
    _, _, test_ds = pipeline.fit_transform(df, train_ratio=0.7, val_ratio=0.15)
    test_loader = pipeline.get_test_loader(batch_size=64)
    print(f"      Test windows: {len(test_ds)}")

    # ── Step 3: Inference ──────────────────────────────────────────────
    print("[3/5] Running test set inference...")
    predictions, ground_truth = collect_predictions(model, test_loader, device)

    # ── Step 4: Metrics ────────────────────────────────────────────────
    print("[4/5] Computing evaluation metrics...")
    metrics = compute_metrics(predictions, ground_truth)
    print_metrics(metrics)

    # ── Step 5: Generate figures ───────────────────────────────────────
    print("[5/5] Generating publication figures...")
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_prediction_curve(
        predictions, ground_truth,
        save_path=figure_dir / "Figure_1_Prediction_Curve.png",
        n_steps=200,
        training_history=checkpoint.get("history"),
    )

    plot_error_distribution(
        predictions, ground_truth,
        save_path=figure_dir / "Figure_2_Error_Distribution.png",
        metrics=metrics,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n  Figures saved to: {figure_dir}/")
    for f in sorted(figure_dir.glob("*.png")):
        print(f"    {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    return metrics


# ─── Standalone Entry Point ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 55)
    print("  LinkQualityLSTM — Evaluation & Visualization")
    print("=" * 55)

    metrics = run_evaluation()

    print("\n  Evaluation pipeline complete.")
