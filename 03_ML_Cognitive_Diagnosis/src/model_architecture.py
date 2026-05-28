"""LinkQualityLSTM — NearLink Communication Link Quality Online Prediction.

A hybrid LSTM + Multi-Head Attention architecture designed for real-time
prediction of wireless link quality in NearLink (星闪) communication systems.

Architecture Overview
---------------------
    Input Projection (F -> d_model)
        -> Bidirectional LSTM (d_model -> 2*hidden)
        -> Multi-Head Self-Attention (residual)
        -> Layer Normalization
        -> Prediction Head (2*hidden -> output_dim)

Design Rationale
----------------
- **LSTM over Transformer**: Online streaming inference requires O(1) per-step
  complexity via hidden state recurrence. Transformer self-attention is O(T^2)
  and recomputes over the full window, making it unsuitable for latency-critical
  link adaptation. LSTM hidden states can be carried across steps without
  re-encoding history.
- **Bidirectional**: During offline batch evaluation (e.g., post-hoc analysis),
  bidirectional encoding captures both past degradation patterns and future
  recovery trends. For online inference, only the forward direction is active.
- **Self-Attention**: Lightweight attention over LSTM outputs selectively
  amplifies critical timesteps (e.g., sudden CRC spikes) without sacrificing
  the recurrent backbone's streaming capability.

Typical Usage
-------------
>>> import torch
>>> model = LinkQualityLSTM(input_dim=8, hidden_dim=64)
>>> x = torch.randn(32, 50, 8)  # (batch, seq_len, features)
>>> pred = model(x)              # (32, 1) — predicted link quality score
>>> loss = torch.nn.MSELoss()(pred, targets)
>>> loss.backward()

References
----------
.. [1] Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Comp. 1997.
.. [2] Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
.. [3] NearLink (SparkLink) Alliance, "NearLink Technology Standard v1.0".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ─── Input Feature Specification ────────────────────────────────────────────
# The 8-dimensional input feature vector per timestep represents one sampling
# instant of the NearLink physical/MAC layer telemetry:
#
#   Index   Feature                   Unit        Typical Range
#   -----   -------                   ----        -------------
#   [0]     CRC error rate            ratio       [0, 0.1]
#   [1]     Delay jitter              ms          [0, 50]
#   [2]     RSSI (recv signal str.)   dBm         [-100, -20]
#   [3]     SNR (signal-to-noise)     dB          [0, 40]
#   [4]     Throughput                Mbps        [0, 12]
#   [5]     Packet loss rate          ratio       [0, 0.3]
#   [6]     Retransmission count      count/step  [0, 10]
#   [7]     Signal strength variance  dB^2        [0, 25]
#
# All features should be z-score normalized per-link before feeding into
# the model (see LinkDataset in training pipeline).
# ─────────────────────────────────────────────────────────────────────────────

#: Recommended input feature dimension for NearLink telemetry.
DEFAULT_INPUT_DIM: int = 8

#: Feature names corresponding to each input dimension.
FEATURE_NAMES: list[str] = [
    "crc_error_rate",
    "delay_jitter_ms",
    "rssi_dbm",
    "snr_db",
    "throughput_mbps",
    "packet_loss_rate",
    "retransmit_count",
    "signal_var_db2",
]


class _ScaledDotProductAttention(nn.Module):
    """Multi-head compatible scaled dot-product attention.

    Computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

    Parameters
    ----------
    dropout : float
        Attention weight dropout probability.
    """

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : Tensor  (B, H, T, d_k)
        key   : Tensor  (B, H, T, d_k)
        value : Tensor  (B, H, T, d_k)

        Returns
        -------
        output          : Tensor  (B, H, T, d_k)
        attn_weights    : Tensor  (B, H, T, T)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    """Lightweight multi-head self-attention for LSTM output sequences.

    Operates on the encoder output of the LSTM to selectively attend to
    critical timesteps (e.g., sudden CRC spikes, jitter transients).

    Parameters
    ----------
    d_model : int
        Input feature dimension (= 2 * hidden_dim for bidirectional LSTM).
    num_heads : int
        Number of parallel attention heads.
    dropout : float
        Dropout probability for attention weights.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attention = _ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  (B, T, d_model)
            Sequence of LSTM hidden states.

        Returns
        -------
        output       : Tensor  (B, T, d_model)
            Attention-refined sequence representation.
        attn_weights : Tensor  (B, num_heads, T, T)
            Per-head attention weight matrices.
        """
        B, T, _ = x.shape

        # Linear projections: (B, T, d_model) -> (B, H, T, d_k)
        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        attn_out, attn_weights = self.attention(Q, K, V)

        # Concatenate heads: (B, H, T, d_k) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.W_o(attn_out)

        return self.dropout(output), attn_weights


class LinkQualityLSTM(nn.Module):
    """Hybrid BiLSTM + Self-Attention model for NearLink link quality prediction.

    This model ingests a sliding window of physical/MAC layer telemetry and
    predicts the next-step link quality score, which can drive adaptive
    modulation, power control, or handover decisions.

    Architecture::

        x (B, T, F)
            │
            ▼
        Input Projection ─── Linear(F, d_model) + ReLU + Dropout
            │
            ▼
        Bidirectional LSTM ── num_layers, hidden_dim per direction
            │
            ├── residual ──────────────────────────┐
            ▼                                       │
        Multi-Head Self-Attention ── num_heads      │
            │                                       │
            ▼                                       │
        (+) ◄─────── residual connection ◄──────────┘
            │
            ▼
        Layer Normalization
            │
            ▼
        [-1] timestep selection (last valid step)
            │
            ▼
        Prediction Head ── Linear + ReLU + Dropout + Linear -> output_dim

    Parameters
    ----------
    input_dim : int
        Number of input features per timestep (default: 8 for NearLink).
    hidden_dim : int
        LSTM hidden state dimension per direction.
        Effective LSTM output dim = 2 * hidden_dim (bidirectional).
    num_layers : int
        Number of stacked LSTM layers.
    output_dim : int
        Prediction output dimension (1 for scalar quality score).
    num_heads : int
        Number of attention heads in self-attention module.
    dropout : float
        Global dropout probability applied to projection and prediction head.
    bidirectional : bool
        Whether to use bidirectional LSTM. Set False for pure online inference
        where future timesteps are unavailable.

    Input Shape
    -----------
    (batch_size, seq_len, input_dim) — e.g., (32, 50, 8)
        - batch_size: number of independent link observation windows
        - seq_len: sliding window length (number of timesteps)
        - input_dim: feature count per timestep (see FEATURE_NAMES)

    Output Shape
    ------------
    (batch_size, output_dim) — e.g., (32, 1)
        - Predicted link quality score, range depends on training target
          normalization (typically sigmoid output mapped to [0, 1]).
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        num_heads: int = 4,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        self.lstm_output_dim = hidden_dim * self.directions

        # ── Input Projection ────────────────────────────────────────────
        # Maps raw telemetry features into a higher-dimensional latent space
        # that the LSTM can more effectively model.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.lstm_output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        # ── Bidirectional LSTM Encoder ──────────────────────────────────
        # Captures temporal dependencies in link quality dynamics.
        # BatchFirst=True: input/output shape (B, T, feature).
        # dropout between layers only (no dropout on last layer output).
        layer_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.lstm_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout,
            bidirectional=bidirectional,
        )

        # ── Multi-Head Self-Attention ───────────────────────────────────
        # Selectively amplifies critical timesteps (CRC spikes, jitter bursts)
        # from the LSTM output sequence. With residual connection to preserve
        # the LSTM's temporal ordering signal.
        self.attention = MultiHeadSelfAttention(
            d_model=self.lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_layer_norm = nn.LayerNorm(self.lstm_output_dim)

        # ── Prediction Head ─────────────────────────────────────────────
        # Maps the final refined hidden state to the output prediction.
        # Two-layer MLP with ReLU activation for non-linear mapping capacity.
        self.prediction_head = nn.Sequential(
            nn.Linear(self.lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers; orthogonal for LSTM.

        Follows best practices from:
        - Glorot & Bengio (2010) for feed-forward weights
        - Sutskever et al. (2014) for recurrent weights
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Orthogonal init for LSTM recurrent weights (hidden-to-hidden)
        for name, param in self.lstm.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 (Jozefowicz et al., 2015)
                # Improves long-term dependency learning
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  (B, T, input_dim)
            Input feature tensor where:
              - B = batch size (number of independent link windows)
              - T = sequence length (sliding window timesteps)
              - input_dim = 8 (CRC err, jitter, RSSI, SNR, throughput,
                pkt loss, retransmit, signal var)
        lengths : Tensor (B,), optional
            Actual sequence lengths for packed input (for variable-length
            sequences). If None, all timesteps are assumed valid.

        Returns
        -------
        prediction : Tensor  (B, output_dim)
            Predicted link quality score per sample.
        """
        # (B, T, F) -> (B, T, d_model)
        projected = self.input_proj(x)

        # Pack variable-length sequences if lengths provided (training)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                projected, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
        else:
            # (B, T, d_model) -> (B, T, 2*hidden_dim)
            lstm_out, _ = self.lstm(projected)

        # Self-attention with residual connection
        # attn_out: (B, T, 2*hidden_dim), attn_weights: (B, H, T, T)
        attn_out, _ = self.attention(lstm_out)
        refined = self.attn_layer_norm(lstm_out + attn_out)

        # Select last valid timestep: (B, 2*hidden_dim)
        if lengths is not None:
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, refined.size(2))
            last_hidden = refined.gather(1, idx).squeeze(1)
        else:
            last_hidden = refined[:, -1, :]

        # (B, 2*hidden_dim) -> (B, output_dim)
        prediction = self.prediction_head(last_hidden)

        return prediction

    def predict_online(
        self,
        step_input: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Single-step online inference with explicit hidden state management.

        Designed for real-time deployment where each new telemetry sample
        arrives one at a time and the model must update incrementally.

        Parameters
        ----------
        step_input : Tensor  (1, 1, input_dim) or (1, input_dim)
            Single timestep feature vector from NearLink PHY/MAC layer.
        hidden : tuple (h_n, c_n) or None
            Previous LSTM hidden state. None on first call (auto-initialized).

        Returns
        -------
        prediction : Tensor  (1, output_dim)
            Link quality prediction for current step.
        new_hidden : tuple (h_n, c_n)
            Updated LSTM hidden state, to be passed to next call.

        Example
        -------
        >>> model = LinkQualityLSTM()
        >>> model.eval()
        >>> hidden = None
        >>> for sample in telemetry_stream:  # sample shape: (1, 8)
        ...     pred, hidden = model.predict_online(sample, hidden)
        ...     # pred is the quality score for this timestep
        """
        if step_input.dim() == 2:
            step_input = step_input.unsqueeze(1)  # (1, F) -> (1, 1, F)

        # Project input: (1, 1, F) -> (1, 1, d_model)
        projected = self.input_proj(step_input)

        # Single-step LSTM (no attention for latency-critical path)
        lstm_out, new_hidden = self.lstm(projected, hidden)

        # Prediction head: (1, 2*hidden_dim) -> (1, output_dim)
        last_hidden = lstm_out[:, -1, :]
        prediction = self.prediction_head(last_hidden)

        return prediction, new_hidden

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of model parameters.

        Parameters
        ----------
        trainable_only : bool
            If True, count only parameters with requires_grad=True.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_model_info(self) -> dict:
        """Return a summary dict of model configuration and parameter counts."""
        return {
            "architecture": "LinkQualityLSTM",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "lstm_output_dim": self.lstm_output_dim,
            "output_dim": self.prediction_head[-1].out_features,
            "total_params": self.count_parameters(trainable_only=False),
            "trainable_params": self.count_parameters(trainable_only=True),
        }


# ─── Standalone Verification ────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity check: construct model and run a forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = LinkQualityLSTM(
        input_dim=8,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        num_heads=4,
        dropout=0.2,
        bidirectional=True,
    ).to(device)

    info = model.get_model_info()
    print("\nModel Configuration:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Batch inference test
    batch_size, seq_len = 32, 50
    x = torch.randn(batch_size, seq_len, 8, device=device)
    pred = model(x)
    print(f"\nBatch forward:  input {tuple(x.shape)} -> output {tuple(pred.shape)}")

    # Online inference test
    model.eval()
    with torch.no_grad():
        hidden = None
        for t in range(seq_len):
            step = x[0:1, t : t + 1, :]  # (1, 1, 8)
            out, hidden = model.predict_online(step, hidden)
        print(f"Online forward: last step output = {out.item():.4f}")
