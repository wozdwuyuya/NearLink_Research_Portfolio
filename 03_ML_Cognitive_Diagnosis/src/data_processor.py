"""NearLink Telemetry Data Processing Pipeline.

Converts raw NearLink PHY/MAC layer telemetry logs into normalized,
windowed PyTorch Datasets ready for LinkQualityLSTM training and inference.

Pipeline Overview
-----------------
    raw DataFrame (N, 8+1)
        -> session-aware splitting (no cross-session leakage)
        -> Z-score normalization (fit on train only)
        -> sliding window slicing -> (num_windows, T, 8) features + (num_windows,) targets
        -> NearLinkTelemetryDataset (torch Dataset)
        -> DataLoader -> LinkQualityLSTM

Normalization Decision: Z-Score over MinMax
-------------------------------------------
- RSSI and SNR are approximately Gaussian -> Z-score preserves distribution shape.
- MinMax is brittle against outlier-induced range compression; a single anomalous
  reading shrinks the entire normal range into a narrow band.
- Zero-centered inputs produce healthier gradients in LSTM gate activations
  (forget gate bias init = 1.0 is calibrated for zero-mean inputs).

Data Leakage Prevention
-----------------------
- Scaler is fit ONLY on the training split; val/test use transform() only.
- Sliding windows do not cross session boundaries (each link session is independent).
- Target value (quality_score at t+1) is excluded from the feature window.
- Scaler is persisted with fit statistics (mean, std) for post-hoc auditing.

Typical Usage
-------------
>>> from data_processor import NearLinkDataPipeline
>>> pipeline = NearLinkDataPipeline(window_size=50, target_col="quality_score")
>>> train_ds, val_ds, test_ds = pipeline.fit_transform(
...     df, train_ratio=0.7, val_ratio=0.15, session_col="session_id"
... )
>>> pipeline.save_scaler("checkpoints/scaler.joblib")
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

# Bypass local sklearn.py shadow: temporarily remove src/ from sys.path
_src_dir = str(Path(__file__).resolve().parent)
_saved_paths = sys.path[:]
sys.path = [p for p in sys.path if Path(p).resolve() != Path(_src_dir)]
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path = _saved_paths

# Ensure src/ is on sys.path for cross-module imports (model_architecture, etc.)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from model_architecture import DEFAULT_INPUT_DIM, FEATURE_NAMES  # noqa: E402

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

#: Minimum number of valid timesteps required to produce one sliding window.
MIN_WINDOW_SAMPLES: int = 2

#: Default random seed for reproducible splits.
DEFAULT_SEED: int = 42


# ─── Scaler Wrapper ─────────────────────────────────────────────────────────


@dataclass
class NearLinkScaler:
    """Z-score scaler with persistence, audit logging, and leakage protection.

    Wraps sklearn StandardScaler with:
    - Explicit fit/transform separation (fit_transform only on train).
    - Persistence via joblib (includes fit statistics for auditing).
    - Runtime guard: transform() raises if called before fit().

    Attributes
    ----------
    feature_names : list[str]
        Names of the 8 telemetry features (from model_architecture.FEATURE_NAMES).
    scaler : StandardScaler
        Underlying sklearn scaler (None until fit() is called).
    is_fitted : bool
        Whether fit() has been called.
    fit_stats : dict or None
        Per-feature mean and std after fitting, for audit logging.
    """

    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    scaler: StandardScaler = field(default_factory=StandardScaler, repr=False)
    is_fitted: bool = field(default=False, repr=False)
    fit_stats: dict[str, dict[str, float]] | None = field(default=None, repr=False)

    def fit(self, X: NDArray[np.floating]) -> NearLinkScaler:
        """Fit scaler on training data ONLY.

        Parameters
        ----------
        X : ndarray (N, 8)
            Training feature matrix. N samples, 8 features.

        Returns
        -------
        self
            Fitted scaler instance (for method chaining).

        Raises
        ------
        ValueError
            If X has wrong number of features.
        """
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            )

        self.scaler.fit(X)
        self.is_fitted = True

        # Record fit statistics for audit trail
        self.fit_stats = {}
        for i, name in enumerate(self.feature_names):
            self.fit_stats[name] = {
                "mean": float(self.scaler.mean_[i]),
                "std": float(self.scaler.scale_[i]),
            }

        logger.info(
            "Scaler fitted on %d samples. Mean range: [%.4f, %.4f]",
            X.shape[0],
            self.scaler.mean_.min(),
            self.scaler.mean_.max(),
        )
        return self

    def transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Transform features using fitted scaler.

        Parameters
        ----------
        X : ndarray (N, 8)
            Feature matrix to normalize.

        Returns
        -------
        ndarray (N, 8)
            Z-score normalized features.

        Raises
        ------
        RuntimeError
            If scaler has not been fit yet (data leakage guard).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler.transform() called before fit() — "
                "this indicates data leakage. Fit on train split only."
            )
        return self.scaler.transform(X)

    def fit_transform(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit on X and return transformed X. Shorthand for fit().transform().

        Parameters
        ----------
        X : ndarray (N, 8)
            Training feature matrix.

        Returns
        -------
        ndarray (N, 8)
            Z-score normalized training features.
        """
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        """Persist fitted scaler and metadata to disk.

        Saves two files:
        - ``{path}``: joblib-serialized StandardScaler + metadata dict.

        Parameters
        ----------
        path : str or Path
            Output file path (recommended: ``checkpoints/scaler.joblib``).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted scaler.")

        payload = {
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "fit_stats": self.fit_stats,
            "is_fitted": self.is_fitted,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(payload, path)
        logger.info("Scaler saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> NearLinkScaler:
        """Load a previously saved scaler from disk.

        Parameters
        ----------
        path : str or Path
            Path to the joblib file saved by ``save()``.

        Returns
        -------
        NearLinkScaler
            Restored scaler instance ready for transform().
        """
        payload = load(path)
        instance = cls(
            feature_names=payload["feature_names"],
            scaler=payload["scaler"],
            is_fitted=payload["is_fitted"],
            fit_stats=payload["fit_stats"],
        )
        logger.info("Scaler loaded from %s", path)
        return instance

    def get_audit_report(self) -> str:
        """Return a human-readable summary of fit statistics."""
        if not self.is_fitted:
            return "Scaler not fitted."
        lines = ["NearLink Scaler Audit Report", "=" * 40]
        for name in self.feature_names:
            stats = self.fit_stats[name]
            lines.append(
                f"  {name:25s}  mean={stats['mean']:10.4f}  std={stats['std']:10.4f}"
            )
        return "\n".join(lines)


# ─── Sliding Window Dataset ─────────────────────────────────────────────────


class NearLinkTelemetryDataset(Dataset):
    """PyTorch Dataset for windowed NearLink telemetry sequences.

    Converts a normalized time-series DataFrame into overlapping sliding
    windows suitable for LinkQualityLSTM input.

    Each sample is a tuple ``(features, target)`` where:
    - features: ``Tensor(T, F)`` — a window of T timesteps with F features
    - target: ``Tensor(1,)`` — the quality score at timestep T+1

    Sliding window illustration (T=5, stride=1)::

        time:   0   1   2   3   4   5   6   7   8   ...
                |---|---|---|---|---|
                [window 0: t=0..4] -> target = quality[5]
                    |---|---|---|---|---|
                    [window 1: t=1..5] -> target = quality[6]
                        |---|---|---|---|---|
                        [window 2: t=2..6] -> target = quality[7]

    Session Boundary Enforcement
    ----------------------------
    When ``session_ids`` is provided, windows are NOT allowed to span across
    different session IDs. This prevents the model from seeing discontinuous
    telemetry as a continuous signal (e.g., after link handover or device
    reboot).

    Parameters
    ----------
    features : NDArray[np.floating]  (N, F)
        Normalized feature matrix (output of NearLinkScaler.transform).
    targets : NDArray[np.floating]  (N,)
        Quality score per timestep.
    window_size : int
        Number of timesteps per sliding window (= model's seq_len).
    stride : int
        Step size between consecutive windows. Default 1 (maximum overlap).
    session_ids : NDArray or None  (N,)
        Integer session identifiers. Windows crossing session boundaries
        are excluded. None disables session-aware filtering.
    """

    def __init__(
        self,
        features: NDArray[np.floating],
        targets: NDArray[np.floating],
        window_size: int,
        stride: int = 1,
        session_ids: NDArray[np.integer] | None = None,
    ) -> None:
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Feature/target length mismatch: {features.shape[0]} vs {targets.shape[0]}"
            )
        if window_size < MIN_WINDOW_SAMPLES:
            raise ValueError(
                f"window_size must be >= {MIN_WINDOW_SAMPLES}, got {window_size}"
            )
        if features.shape[0] < window_size + 1:
            raise ValueError(
                f"Need at least {window_size + 1} samples to produce one window, "
                f"got {features.shape[0]}"
            )

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.window_size = window_size
        self.stride = stride

        # Build valid window start indices
        self._indices = self._build_indices(session_ids)

        if len(self._indices) == 0:
            raise ValueError(
                "No valid windows could be constructed. Check window_size, "
                "stride, and session boundaries."
            )

        logger.info(
            "NearLinkTelemetryDataset: %d windows (size=%d, stride=%d) "
            "from %d timesteps",
            len(self._indices),
            window_size,
            stride,
            features.shape[0],
        )

    def _build_indices(
        self, session_ids: NDArray[np.integer] | None
    ) -> list[int]:
        """Compute valid window start positions.

        A window starting at index i is valid if:
        1. i + window_size < N (enough room for features + target)
        2. All timesteps i..i+window_size belong to the same session
           (when session_ids is provided)

        Parameters
        ----------
        session_ids : ndarray (N,) or None

        Returns
        -------
        list[int]
            Valid start indices for sliding windows.
        """
        n = self.features.shape[0]
        indices: list[int] = []

        for i in range(0, n - self.window_size, self.stride):
            # Target index is one step after the window
            target_idx = i + self.window_size
            if target_idx >= n:
                break

            # Session boundary check: all timesteps in window + target
            # must belong to the same session
            if session_ids is not None:
                window_sessions = session_ids[i : target_idx + 1]
                if np.unique(window_sessions).size > 1:
                    continue

            indices.append(i)

        return indices

    def __len__(self) -> int:
        """Number of sliding windows in the dataset."""
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one sliding window and its target.

        Parameters
        ----------
        idx : int
            Window index (0-based).

        Returns
        -------
        features : Tensor  (window_size, num_features)
            Normalized telemetry features for T timesteps.
        target : Tensor  (1,)
            Quality score at timestep T+1.
        """
        start = self._indices[idx]
        end = start + self.window_size

        x = self.features[start:end]        # (T, F)
        y = self.targets[end]               # scalar

        return x, y.unsqueeze(0)            # (T, F), (1,)

    def get_window_count(self) -> int:
        """Alias for len(self) for clarity in pipeline logging."""
        return len(self)


# ─── Pipeline Orchestrator ──────────────────────────────────────────────────


class NearLinkDataPipeline:
    """End-to-end data pipeline: raw DataFrame -> train/val/test DataLoaders.

    Handles session-aware splitting, Z-score normalization (fit on train only),
    sliding window construction, and DataLoader creation.

    Expected Input DataFrame Schema
    --------------------------------
    Must contain these columns (matching FEATURE_NAMES + target):
        - crc_error_rate, delay_jitter_ms, rssi_dbm, snr_db,
          throughput_mbps, packet_loss_rate, retransmit_count, signal_var_db2
        - quality_score (target column, configurable via target_col)
        - session_id (optional, for session-aware splitting)

    Example
    -------
    >>> pipeline = NearLinkDataPipeline(window_size=50)
    >>> train_ds, val_ds, test_ds = pipeline.fit_transform(df)
    >>> train_loader = pipeline.get_train_loader(batch_size=32)
    """

    def __init__(
        self,
        window_size: int = 50,
        stride: int = 1,
        target_col: str = "quality_score",
        session_col: str | None = "session_id",
        seed: int = DEFAULT_SEED,
    ) -> None:
        """
        Parameters
        ----------
        window_size : int
            Sliding window length (model's seq_len).
        stride : int
            Step between consecutive windows.
        target_col : str
            Name of the target column in the input DataFrame.
        session_col : str or None
            Name of the session ID column. None disables session awareness.
        seed : int
            Random seed for reproducible train/val/test splits.
        """
        self.window_size = window_size
        self.stride = stride
        self.target_col = target_col
        self.session_col = session_col
        self.seed = seed

        self.scaler = NearLinkScaler()
        self.train_dataset: NearLinkTelemetryDataset | None = None
        self.val_dataset: NearLinkTelemetryDataset | None = None
        self.test_dataset: NearLinkTelemetryDataset | None = None

    def fit_transform(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> tuple[
        NearLinkTelemetryDataset,
        NearLinkTelemetryDataset,
        NearLinkTelemetryDataset,
    ]:
        """Split, normalize, and window the data.

        Splitting strategy:
        - If session_col exists: split at session granularity (all timesteps
          from one session go to the same split) to prevent temporal leakage.
        - Otherwise: simple sequential split (first 70% train, next 15% val,
          last 15% test).

        Parameters
        ----------
        df : DataFrame
            Raw telemetry data with features + target + optional session_id.
        train_ratio : float
            Fraction of sessions/data for training.
        val_ratio : float
            Fraction for validation. Test = 1 - train - val.

        Returns
        -------
        train_ds, val_ds, test_ds : NearLinkTelemetryDataset
            Windowed datasets for each split.

        Raises
        ------
        ValueError
            If required columns are missing or data is too short.
        """
        # ── Validate columns ─────────────────────────────────────────────
        missing = set(FEATURE_NAMES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found")

        session_ids = None
        if self.session_col and self.session_col in df.columns:
            session_ids = df[self.session_col].values

        features_raw = df[FEATURE_NAMES].values.astype(np.float64)
        targets_raw = df[self.target_col].values.astype(np.float64)

        # ── Split ────────────────────────────────────────────────────────
        if session_ids is not None:
            train_f, train_t, train_s, val_f, val_t, val_s, test_f, test_t, test_s = (
                self._split_by_session(
                    features_raw, targets_raw, session_ids,
                    train_ratio, val_ratio,
                )
            )
        else:
            train_f, train_t, train_s, val_f, val_t, val_s, test_f, test_t, test_s = (
                self._split_sequential(
                    features_raw, targets_raw, session_ids,
                    train_ratio, val_ratio,
                )
            )

        # ── Normalize (fit on train ONLY) ────────────────────────────────
        train_f = self.scaler.fit_transform(train_f)
        val_f = self.scaler.transform(val_f)
        test_f = self.scaler.transform(test_f)

        logger.info("Scaler audit:\n%s", self.scaler.get_audit_report())

        # ── Build Datasets ───────────────────────────────────────────────
        self.train_dataset = NearLinkTelemetryDataset(
            train_f, train_t, self.window_size, self.stride, train_s
        )
        self.val_dataset = NearLinkTelemetryDataset(
            val_f, val_t, self.window_size, self.stride, val_s
        )
        self.test_dataset = NearLinkTelemetryDataset(
            test_f, test_t, self.window_size, self.stride, test_s
        )

        return self.train_dataset, self.val_dataset, self.test_dataset

    def _split_by_session(
        self,
        features: NDArray,
        targets: NDArray,
        session_ids: NDArray,
        train_ratio: float,
        val_ratio: float,
    ) -> tuple[NDArray, ...]:
        """Split data by session IDs (no cross-session leakage).

        Shuffles sessions, then assigns whole sessions to train/val/test.
        """
        rng = np.random.RandomState(self.seed)
        unique_sessions = np.unique(session_ids)
        rng.shuffle(unique_sessions)

        n_total = len(unique_sessions)
        n_train = round(n_total * train_ratio)
        n_val = round(n_total * val_ratio)
        # Guarantee: train >= 1, val >= 1 (if >= 3 sessions), test gets remainder
        n_train = max(1, min(n_train, n_total - 2)) if n_total >= 3 else max(1, n_total - 1)
        n_val = max(1, min(n_val, n_total - n_train - 1)) if n_total - n_train >= 2 else 0

        train_sessions = set(unique_sessions[:n_train])
        val_sessions = set(unique_sessions[n_train : n_train + n_val])
        test_sessions = set(unique_sessions[n_train + n_val :])

        def mask(sessions_set: set) -> NDArray[np.bool_]:
            return np.array([s in sessions_set for s in session_ids])

        train_mask = mask(train_sessions)
        val_mask = mask(val_sessions)
        test_mask = mask(test_sessions)

        logger.info(
            "Session split: train=%d sessions (%d samples), "
            "val=%d sessions (%d samples), test=%d sessions (%d samples)",
            len(train_sessions), train_mask.sum(),
            len(val_sessions), val_mask.sum(),
            len(test_sessions), test_mask.sum(),
        )

        return (
            features[train_mask], targets[train_mask], session_ids[train_mask],
            features[val_mask], targets[val_mask], session_ids[val_mask],
            features[test_mask], targets[test_mask], session_ids[test_mask],
        )

    def _split_sequential(
        self,
        features: NDArray,
        targets: NDArray,
        session_ids: NDArray | None,
        train_ratio: float,
        val_ratio: float,
    ) -> tuple[NDArray, ...]:
        """Sequential time-ordered split (preserves temporal order)."""
        n = features.shape[0]
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, n_train + n_val)
        test_idx = slice(n_train + n_val, n)

        s_train = session_ids[train_idx] if session_ids is not None else None
        s_val = session_ids[val_idx] if session_ids is not None else None
        s_test = session_ids[test_idx] if session_ids is not None else None

        return (
            features[train_idx], targets[train_idx], s_train,
            features[val_idx], targets[val_idx], s_val,
            features[test_idx], targets[test_idx], s_test,
        )

    def get_train_loader(
        self, batch_size: int = 32, num_workers: int = 0, shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader for training set."""
        if self.train_dataset is None:
            raise RuntimeError("Call fit_transform() before get_train_loader().")
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def get_val_loader(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> DataLoader:
        """Create DataLoader for validation set."""
        if self.val_dataset is None:
            raise RuntimeError("Call fit_transform() before get_val_loader().")
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_test_loader(
        self, batch_size: int = 64, num_workers: int = 0
    ) -> DataLoader:
        """Create DataLoader for test set."""
        if self.test_dataset is None:
            raise RuntimeError("Call fit_transform() before get_test_loader().")
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def save_scaler(self, path: str | Path) -> None:
        """Persist fitted scaler to disk for online inference."""
        self.scaler.save(path)

    def load_scaler(self, path: str | Path) -> None:
        """Load a previously saved scaler."""
        self.scaler = NearLinkScaler.load(path)

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Return a summary dict of pipeline state."""
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "scaler_fitted": self.scaler.is_fitted,
            "train_windows": len(self.train_dataset) if self.train_dataset else 0,
            "val_windows": len(self.val_dataset) if self.val_dataset else 0,
            "test_windows": len(self.test_dataset) if self.test_dataset else 0,
            "scaler_fit_stats": self.scaler.fit_stats,
        }


# ─── Synthetic Data Generator ───────────────────────────────────────────────


def generate_synthetic_telemetry(
    n_samples: int = 5000,
    n_sessions: int = 5,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Generate realistic synthetic NearLink telemetry data.

    Simulates PHY/MAC layer metrics with:
    - Per-session base parameters (varying signal conditions)
    - Temporal autocorrelation (link quality degrades/recovers gradually)
    - Correlated features (RSSI and SNR positively correlated, etc.)
    - Realistic ranges matching NearLink specifications

    Parameters
    ----------
    n_samples : int
        Total number of timesteps to generate.
    n_sessions : int
        Number of independent link sessions.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Columns: FEATURE_NAMES + ['quality_score', 'session_id', 'timestamp']
    """
    rng = np.random.RandomState(seed)

    # Per-session base parameters
    session_params = {
        "rssi_base": rng.uniform(-80, -40, n_sessions),
        "snr_base": rng.uniform(5, 30, n_sessions),
        "jitter_base": rng.uniform(1, 15, n_sessions),
    }

    samples_per_session = n_samples // n_sessions
    records: list[dict[str, Any]] = []

    for sid in range(n_sessions):
        n = samples_per_session if sid < n_sessions - 1 else n_samples - len(records)
        rssi_base = session_params["rssi_base"][sid]
        snr_base = session_params["snr_base"][sid]
        jitter_base = session_params["jitter_base"][sid]

        # Temporal autocorrelation via cumulative random walk (clamped)
        rssi_walk = np.cumsum(rng.normal(0, 0.5, n))
        rssi = np.clip(rssi_base + rssi_walk, -100, -20)

        snr_walk = np.cumsum(rng.normal(0, 0.3, n))
        snr = np.clip(snr_base + snr_walk, 0, 40)

        # Correlated features
        jitter = np.clip(jitter_base + rng.exponential(2, n) + np.abs(snr_walk), 0, 50)
        crc_error = np.clip(0.01 + 0.002 * jitter + rng.exponential(0.005, n), 0, 0.1)
        throughput = np.clip(10 - 0.1 * jitter - 5 * crc_error + rng.normal(0, 0.5, n), 0, 12)
        pkt_loss = np.clip(crc_error * 2 + rng.exponential(0.01, n), 0, 0.3)
        retransmit = np.clip(rng.poisson(2 + 5 * crc_error, n), 0, 10).astype(float)
        sig_var = np.clip(5 + rng.exponential(3, n) + 0.5 * np.abs(snr_walk), 0, 25)

        # Quality score: composite metric (higher is better)
        quality = np.clip(
            0.3 * ((rssi + 100) / 80)           # normalized RSSI contribution
            + 0.25 * (snr / 40)                   # SNR contribution
            + 0.2 * (1 - crc_error / 0.1)         # low CRC is good
            + 0.15 * (throughput / 12)             # high throughput is good
            + 0.1 * (1 - pkt_loss / 0.3),          # low packet loss is good
            0.0,
            1.0,
        )

        for t in range(n):
            records.append({
                "crc_error_rate": crc_error[t],
                "delay_jitter_ms": jitter[t],
                "rssi_dbm": rssi[t],
                "snr_db": snr[t],
                "throughput_mbps": throughput[t],
                "packet_loss_rate": pkt_loss[t],
                "retransmit_count": retransmit[t],
                "signal_var_db2": sig_var[t],
                "quality_score": quality[t],
                "session_id": sid,
                "timestamp": len(records),
            })

    return pd.DataFrame(records)


# ─── Standalone Verification ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 70)
    print("  NearLink Data Pipeline — End-to-End Verification")
    print("=" * 70)

    # ── Step 1: Generate synthetic data ──────────────────────────────────
    df = generate_synthetic_telemetry(n_samples=5000, n_sessions=5)
    print(f"\n[1] Synthetic data: {len(df)} samples, {df['session_id'].nunique()} sessions")
    print(f"    Columns: {list(df.columns)}")
    print("    Feature ranges:")
    for feat in FEATURE_NAMES:
        print(f"      {feat:25s}  [{df[feat].min():8.3f}, {df[feat].max():8.3f}]")

    # ── Step 2: Run pipeline ─────────────────────────────────────────────
    pipeline = NearLinkDataPipeline(window_size=50, stride=1)
    train_ds, val_ds, test_ds = pipeline.fit_transform(
        df, train_ratio=0.7, val_ratio=0.15
    )

    print("\n[2] Pipeline results:")
    print(f"    Train windows: {len(train_ds)}")
    print(f"    Val windows:   {len(val_ds)}")
    print(f"    Test windows:  {len(test_ds)}")

    # ── Step 3: Scaler audit ─────────────────────────────────────────────
    print("\n[3] Scaler audit:")
    print(pipeline.scaler.get_audit_report())

    # ── Step 4: DataLoader batch verification ────────────────────────────
    train_loader = pipeline.get_train_loader(batch_size=32, shuffle=False)
    x_batch, y_batch = next(iter(train_loader))
    print("\n[4] DataLoader batch verification:")
    print(f"    Features batch shape: {tuple(x_batch.shape)}  "
          f"(expected: (32, 50, 8))")
    print(f"    Targets batch shape:  {tuple(y_batch.shape)}   "
          f"(expected: (32, 1))")
    print(f"    Features dtype: {x_batch.dtype}")
    print(f"    Targets dtype:  {y_batch.dtype}")

    # ── Step 5: Shape alignment with LinkQualityLSTM ─────────────────────
    from model_architecture import LinkQualityLSTM

    model = LinkQualityLSTM(input_dim=DEFAULT_INPUT_DIM, hidden_dim=64)
    model.eval()
    with torch.no_grad():
        pred = model(x_batch)
    print("\n[5] Model alignment check:")
    print(f"    Input to model:  {tuple(x_batch.shape)}")
    print(f"    Model output:    {tuple(pred.shape)}")
    assert pred.shape == y_batch.shape, (
        f"Shape mismatch! Model output {pred.shape} != targets {y_batch.shape}"
    )
    print("    Shapes MATCH — pipeline is fully aligned with model.")

    # ── Step 6: Save/load scaler roundtrip ───────────────────────────────
    scaler_path = Path("checkpoints/test_scaler.joblib")
    pipeline.save_scaler(scaler_path)
    loaded_pipeline = NearLinkDataPipeline(window_size=50)
    loaded_pipeline.load_scaler(scaler_path)

    # Verify roundtrip: transform with loaded scaler should match original
    test_input = df[FEATURE_NAMES].iloc[:5].values.astype(np.float64)
    original_out = pipeline.scaler.transform(test_input)
    loaded_out = loaded_pipeline.scaler.transform(test_input)
    assert np.allclose(original_out, loaded_out), "Scaler roundtrip FAILED"
    print("\n[6] Scaler save/load roundtrip: PASSED")
    print(f"    Saved to: {scaler_path}")

    print("\n" + "=" * 70)
    print("  All checks PASSED — pipeline ready for training.")
    print("=" * 70)
