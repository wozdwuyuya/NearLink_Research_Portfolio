"""LLM-Augmented Root Cause Analysis Engine — MC Dropout + RAG Fusion.

Implements a confidence-driven System 1 + System 2 diagnostic architecture:

  System 1 (fast, deterministic path):
    LinkQualityLSTM.predict_online() — single-step inference, < 1ms

  System 1.5 (uncertainty quantification):
    MC Dropout — 10 stochastic forward passes on the sliding window
    Dynamic Baseline — EMA(w=50) tracking, triggers on distributional shift
    Micro-trajectory — captures t-2, t-1, t feature evolution

  System 2 (slow, cognitive path):
    RAG Retrieval — FAISS top-2 knowledge chunks from Cyber_Cortex_RAG
    LLM Diagnosis — XiaomiMimoClient generates RCA report

Trigger conditions (ANY of):
  1. mean_pred < EMA_mean - 3 * EMA_std  (quality degradation vs baseline)
  2. pred_variance > VARIANCE_THRESHOLD   (model uncertainty / domain shift)

Architecture::

    Telemetry Stream (8 features per step)
        |
        v
    ┌─────────────────────────────────────────────┐
    │  System 1: Sliding Window + MC Dropout      │
    │  model.train() + 10x forward() → mean, var  │
    └──────┬──────────────────────┬───────────────┘
           │                      │
    mean ≥ baseline         mean < baseline - 3σ
    var ≤ threshold         OR var > threshold
           │                      │
           v                      v
    [NORMAL]                [ANOMALY TRIGGERED]
                                    │
                            ┌───────┴───────┐
                            │ Micro-traject  │ (t-2, t-1, t features)
                            │ RAG Retrieval  │ (FAISS top-2 knowledge)
                            │ LLM Diagnosis  │ (XiaomiMimoClient)
                            └───────┬───────┘
                                    v
                            Structured RCA Report

Typical Usage
-------------
>>> python src/llm_diagnostic.py --mock             # Mock LLM, no API key
>>> python src/llm_diagnostic.py --mock --steps 50  # Short simulation
>>> python src/llm_diagnostic.py                    # Live LLM + RAG
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ── Path Registration ──────────────────────────────────────────────────────
_ml_src = str(Path(__file__).resolve().parent)
if _ml_src not in sys.path:
    sys.path.insert(0, _ml_src)

_cortex_root = str(Path(r"E:\1Projects\Cyber_Cortex_RAG"))
if _cortex_root not in sys.path:
    sys.path.insert(0, _cortex_root)

from data_processor import NearLinkScaler, generate_synthetic_telemetry  # noqa: E402
from model_architecture import FEATURE_NAMES, LinkQualityLSTM  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

WINDOW_SIZE: int = 50           # Must match model's training window
MC_N_FORWARD: int = 10          # Number of MC Dropout forward passes
EMA_WINDOW: int = 50            # Dynamic baseline EMA window
BASELINE_SIGMA: float = 3.0     # Trigger when mean < EMA - sigma * std
VARIANCE_THRESHOLD: float = 5e-3  # MC variance threshold (empirical)
TRAJECTORY_LEN: int = 3         # Micro-trajectory depth (t-2, t-1, t)
RAG_TOP_K: int = 2              # Number of RAG knowledge chunks to retrieve
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


# ── Data Contracts ─────────────────────────────────────────────────────────


@dataclass
class DiagnosticEvent:
    """A single diagnostic event with full context."""

    timestep: int
    mc_mean: float
    mc_variance: float
    ema_baseline: float
    trigger_reason: str
    trajectory: list[dict[str, float]]
    rag_context: list[str]
    root_cause: str
    mitigation: str


# ── MC Dropout Predictor ───────────────────────────────────────────────────


class MCDropoutPredictor:
    """Monte Carlo Dropout uncertainty estimator for LinkQualityLSTM.

    Keeps the model in train() mode to activate Dropout layers during
    inference. Runs N stochastic forward passes on the same input window
    to estimate predictive mean and variance.

    Parameters
    ----------
    model : LinkQualityLSTM
        Loaded model (will be set to train() mode).
    n_forward : int
        Number of MC forward passes per prediction.
    """

    def __init__(self, model: LinkQualityLSTM, n_forward: int = MC_N_FORWARD) -> None:
        self._model = model
        self._n_forward = n_forward
        # CRITICAL: keep Dropout active for MC sampling
        self._model.train()

    @torch.no_grad()
    def predict(
        self, window: torch.Tensor
    ) -> tuple[float, float, list[float]]:
        """Run MC Dropout inference on a sliding window.

        Parameters
        ----------
        window : Tensor (1, T, F)
            Sliding window of normalized telemetry features.

        Returns
        -------
        mean : float
            Mean prediction across N forward passes.
        variance : float
            Variance of predictions (uncertainty estimate).
        samples : list[float]
            Individual prediction samples.
        """
        samples: list[float] = []
        for _ in range(self._n_forward):
            pred = self._model(window)
            samples.append(pred.item())

        arr = np.array(samples)
        return float(arr.mean()), float(arr.var()), samples


# ── Dynamic Baseline ───────────────────────────────────────────────────────


class DynamicBaseline:
    """EMA-based dynamic baseline for anomaly detection.

    Maintains a sliding window of recent predictions and computes
    running mean + standard deviation. Replaces hardcoded thresholds
    with adaptive, distribution-aware detection.

    Parameters
    ----------
    window_size : int
        Number of recent predictions to track.
    """

    def __init__(self, window_size: int = EMA_WINDOW) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._ready = False

    def update(self, value: float) -> None:
        """Add a new prediction to the baseline window."""
        self._window.append(value)
        if len(self._window) >= 10:
            self._ready = True

    @property
    def mean(self) -> float:
        return float(np.mean(self._window)) if self._window else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self._window)) if self._window else 1.0

    @property
    def is_ready(self) -> bool:
        return self._ready

    def is_anomaly(self, mc_mean: float, mc_var: float) -> tuple[bool, str]:
        """Check if current prediction is anomalous.

        Returns
        -------
        is_anomaly : bool
        reason : str
            Description of which condition triggered.
        """
        if not self._ready:
            return False, ""

        reasons: list[str] = []

        # Condition 1: quality degradation vs baseline
        threshold = self.mean - BASELINE_SIGMA * self.std
        if mc_mean < threshold:
            reasons.append(
                f"mean={mc_mean:.4f} < baseline-{BASELINE_SIGMA}σ={threshold:.4f}"
            )

        # Condition 2: high model uncertainty (domain shift)
        if mc_var > VARIANCE_THRESHOLD:
            reasons.append(
                f"variance={mc_var:.6f} > threshold={VARIANCE_THRESHOLD}"
            )

        if reasons:
            return True, " | ".join(reasons)
        return False, ""


# ── RAG Retriever ──────────────────────────────────────────────────────────


class RAGRetriever:
    """Lightweight FAISS retriever for diagnostic context enrichment.

    Loads the Cyber_Cortex_RAG vector index and performs semantic search
    to retrieve relevant knowledge chunks for the diagnostic prompt.

    Parameters
    ----------
    faiss_path : str
        Base path to FAISS index (without .index/.meta extension).
    """

    def __init__(
        self,
        faiss_path: str = r"E:\1Projects\Cyber_Cortex_RAG\data\db\vectors.faiss",
    ) -> None:
        self._faiss_path = faiss_path
        self._vector_store = None
        self._embedding_client = None

    def initialize(self) -> bool:
        """Load FAISS index and embedding model.

        Returns True if initialization succeeded, False otherwise.
        """
        try:
            import faiss  # noqa: F401
            from src.vector_store import (  # type: ignore[import-untyped]
                FAISSVectorStore,
                SentenceTransformerClient,
            )

            self._embedding_client = SentenceTransformerClient(
                model_name=EMBEDDING_MODEL, device="cpu"
            )
            self._vector_store = FAISSVectorStore(dim=384)
            self._vector_store.load(self._faiss_path)

            logger.info(
                "RAGRetriever: loaded %d vectors", self._vector_store.size()
            )
            return True
        except Exception as exc:
            logger.warning("RAGRetriever unavailable: %s", exc)
            return False

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[str]:
        """Retrieve top-k relevant knowledge chunks.

        Parameters
        ----------
        query : str
            Natural language query (anomaly description).
        top_k : int
            Number of results to return.

        Returns
        -------
        list[str]
            Content of retrieved knowledge chunks.
        """
        if self._vector_store is None or self._embedding_client is None:
            return []

        try:
            embedding = self._embedding_client.embed_query(query)
            records = self._vector_store.search(embedding, top_k=top_k)
            return [r.content for r in records if r.content.strip()]
        except Exception as exc:
            logger.warning("RAG retrieval failed: %s", exc)
            return []


# ── LLM Client Abstraction ─────────────────────────────────────────────────


def _try_import_xiaomi_client():
    """Attempt to import XiaomiMimoClient from Cyber_Cortex_RAG."""
    try:
        from src.llm_client import XiaomiMimoClient  # type: ignore[import-untyped]

        return XiaomiMimoClient
    except Exception as exc:
        logger.warning("XiaomiMimoClient unavailable: %s", exc)
        return None


class MockDiagnosticClient:
    """Fallback LLM client with deterministic responses."""

    async def generate(self, prompt: str, **kwargs) -> str:  # noqa: ARG002
        return (
            "【故障根因猜测】\n"
            "MC Dropout 不确定性分析显示模型预测方差显著升高，结合微轨迹中"
            "CRC误码率持续上升和RSSI信号强度阶梯式下降的趋势，推测存在渐进式"
            "多径衰落或同频干扰导致的信道质量恶化。动态基线偏移表明这不是瞬时"
            "异常而是系统性链路退化。\n\n"
            "【缓解建议】\n"
            "1. 短期：启动自适应功率控制，提升发射功率 3-6dB 补偿路径损耗\n"
            "2. 短期：降低调制阶数（64QAM → 16QAM），牺牲吞吐量换取鲁棒性\n"
            "3. 中期：切换至备用信道或触发频率跳变（FHSS）规避干扰频段\n"
            "4. 长期：评估基站切换（Handover）以接入更优链路"
        )

    async def close(self) -> None:
        pass


# ── Diagnostic Prompt Builder ──────────────────────────────────────────────


def build_diagnostic_prompt(
    timestep: int,
    mc_mean: float,
    mc_variance: float,
    ema_baseline: float,
    trigger_reason: str,
    trajectory: list[dict[str, float]],
    rag_context: list[str],
) -> str:
    """Build a rich diagnostic prompt with trajectory, uncertainty, and RAG context.

    Parameters
    ----------
    timestep : int
        Current inference step.
    mc_mean : float
        Mean of MC Dropout predictions.
    mc_variance : float
        Variance of MC Dropout predictions.
    ema_baseline : float
        Current EMA baseline value.
    trigger_reason : str
        Human-readable trigger condition.
    trajectory : list[dict]
        Feature dicts for t-2, t-1, t (raw, un-normalized).
    rag_context : list[str]
        Retrieved knowledge chunks from FAISS.
    """
    # ── Micro-trajectory table ────────────────────────────────────────────
    if trajectory and len(trajectory) >= 2:
        names = list(trajectory[0].keys())
        header = "| 特征 | " + " | ".join(
            f"t-{len(trajectory) - 1 - i}" for i in range(len(trajectory))
        ) + " | 变化趋势 |"
        separator = "|------|" + "|".join(["------"] * len(trajectory)) + "|------|"

        rows = []
        for name in names:
            vals = [step[name] for step in trajectory]
            cells = " | ".join(f"{v:.4f}" for v in vals)
            delta = vals[-1] - vals[0]
            trend = "↑ 上升" if delta > 0.01 else ("↓ 下降" if delta < -0.01 else "→ 持平")
            rows.append(f"| {name} | {cells} | {trend} |")

        trajectory_table = f"{header}\n{separator}\n" + "\n".join(rows)
    else:
        trajectory_table = "(微轨迹数据不足)"

    # ── RAG context ───────────────────────────────────────────────────────
    if rag_context:
        rag_block = "\n\n".join(
            f"[知识片段 {i + 1}]\n{chunk[:500]}" for i, chunk in enumerate(rag_context)
        )
    else:
        rag_block = "(未检索到相关知识)"

    return f"""你是一位资深的星闪 (NearLink) 协议通信专家，精通物理层和MAC层的链路质量分析。
你的任务是基于 MC Dropout 不确定性量化结果、动态基线偏移分析、时序微轨迹趋势、
以及 RAG 检索到的领域知识，输出一份严谨的故障根因分析报告。

## 诊断触发信息

- **时间步**: {timestep}
- **MC Dropout 预测均值**: {mc_mean:.6f} (10次前向传播)
- **MC Dropout 预测方差**: {mc_variance:.6f} (不确定性度量)
- **动态基线 (EMA)**: {ema_baseline:.6f}
- **触发原因**: {trigger_reason}

## 时序微轨迹 (最近 {len(trajectory)} 步)

{trajectory_table}

## RAG 检索上下文 (Cyber_Cortex_RAG 知识库)

{rag_block}

## 特征说明

| 特征 | 正常范围 | 异常判定标准 |
|------|---------|-------------|
| crc_error_rate | [0, 0.1] | > 0.05 偏高 |
| delay_jitter_ms | [0, 50] | > 30ms 异常 |
| rssi_dbm | [-100, -20] | < -70dBm 弱信号 |
| snr_db | [0, 40] | < 10dB 信噪比不足 |
| throughput_mbps | [0, 12] | < 5Mbps 吞吐下降 |
| packet_loss_rate | [0, 0.3] | > 0.15 高丢包 |
| retransmit_count | [0, 10] | > 5 频繁重传 |
| signal_var_db2 | [0, 25] | > 15 信号不稳定 |

## 输出要求

请严格按照以下格式输出（不要输出其他内容）：

【故障根因猜测】
（基于 MC Dropout 不确定性、微轨迹趋势和 RAG 知识，分析最可能的故障原因。
  重点关注：方差升高是否表明域偏移/未知干扰？轨迹趋势是否显示系统性退化？2-3句话）

【缓解建议】
（给出短期和中期的缓解措施，编号列表，3-4条。建议应基于 RAG 检索到的领域知识。）"""


# ── Diagnostic Engine ───────────────────────────────────────────────────────


class DiagnosticEngine:
    """System 2: LLM + RAG root cause analysis engine.

    Orchestrates MC Dropout uncertainty, dynamic baseline, micro-trajectory,
    RAG retrieval, and LLM diagnosis into a unified diagnostic pipeline.

    Parameters
    ----------
    use_mock : bool
        If True, use MockDiagnosticClient (no API key required).
    """

    def __init__(self, use_mock: bool = False) -> None:
        self._use_mock = use_mock
        self._llm_client = None
        self._rag = RAGRetriever()
        self._events: list[DiagnosticEvent] = []

    async def initialize(self) -> None:
        """Initialize LLM client and RAG retriever."""
        # LLM
        if self._use_mock:
            self._llm_client = MockDiagnosticClient()
            logger.info("DiagnosticEngine: MockDiagnosticClient")
        else:
            XiaomiMimoClient = _try_import_xiaomi_client()
            if XiaomiMimoClient is None:
                logger.warning("Falling back to MockDiagnosticClient")
                self._llm_client = MockDiagnosticClient()
            else:
                self._llm_client = XiaomiMimoClient()
                logger.info("DiagnosticEngine: XiaomiMimoClient")

        # RAG
        rag_ok = self._rag.initialize()
        if rag_ok:
            logger.info("DiagnosticEngine: RAG retriever ready")
        else:
            logger.info("DiagnosticEngine: RAG retriever unavailable, proceeding without")

    async def shutdown(self) -> None:
        if self._llm_client:
            await self._llm_client.close()

    async def diagnose(
        self,
        timestep: int,
        mc_mean: float,
        mc_variance: float,
        ema_baseline: float,
        trigger_reason: str,
        trajectory: list[dict[str, float]],
    ) -> DiagnosticEvent:
        """Run full diagnostic pipeline: RAG → Prompt → LLM → Parse.

        Parameters
        ----------
        timestep : int
            Current inference step.
        mc_mean : float
            MC Dropout mean prediction.
        mc_variance : float
            MC Dropout variance.
        ema_baseline : float
            Current EMA baseline.
        trigger_reason : str
            Human-readable trigger description.
        trajectory : list[dict]
            Raw features for t-2, t-1, t.

        Returns
        -------
        DiagnosticEvent
        """
        # ── RAG retrieval ─────────────────────────────────────────────────
        # Build anomaly query from the two features with largest change
        anomaly_query = self._build_anomaly_query(trajectory)
        rag_chunks = self._rag.retrieve(anomaly_query)

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = build_diagnostic_prompt(
            timestep=timestep,
            mc_mean=mc_mean,
            mc_variance=mc_variance,
            ema_baseline=ema_baseline,
            trigger_reason=trigger_reason,
            trajectory=trajectory,
            rag_context=rag_chunks,
        )

        # ── LLM diagnosis ─────────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(
                self._llm_client.generate(prompt), timeout=45.0
            )
        except asyncio.TimeoutError:
            raw = (
                "【故障根因猜测】\nLLM 诊断超时 (45s)。\n\n"
                "【缓解建议】\n1. 检查网络和 API 服务状态"
            )
        except Exception as exc:
            raw = f"【故障根因猜测】\nLLM 异常: {exc}\n\n【缓解建议】\n1. 检查 API 配置"

        root_cause, mitigation = self._parse_response(raw)

        event = DiagnosticEvent(
            timestep=timestep,
            mc_mean=mc_mean,
            mc_variance=mc_variance,
            ema_baseline=ema_baseline,
            trigger_reason=trigger_reason,
            trajectory=trajectory,
            rag_context=rag_chunks,
            root_cause=root_cause,
            mitigation=mitigation,
        )
        self._events.append(event)
        return event

    @staticmethod
    def _build_anomaly_query(trajectory: list[dict[str, float]]) -> str:
        """Extract top-2 features with largest absolute change for RAG query."""
        if len(trajectory) < 2:
            return "星闪通信链路质量异常"

        first, last = trajectory[0], trajectory[-1]
        deltas = {}
        for name in first:
            delta = abs(last[name] - first[name])
            deltas[name] = delta

        # Sort by absolute change, take top 2
        top2 = sorted(deltas.items(), key=lambda x: x[1], reverse=True)[:2]
        keywords = " ".join(name for name, _ in top2)
        return f"星闪 NearLink 通信 {keywords} 链路质量异常 根因分析"

    @staticmethod
    def _parse_response(raw: str) -> tuple[str, str]:
        """Extract root cause and mitigation from LLM response."""
        root_cause, mitigation = "", ""
        section = None

        for line in raw.strip().split("\n"):
            s = line.strip()
            if "故障根因" in s or "Root Cause" in s:
                section = "rc"
                after = s.split("】", 1)
                if len(after) > 1 and after[1].strip():
                    root_cause = after[1].strip()
                continue
            if "缓解建议" in s or "Mitigation" in s:
                section = "mit"
                after = s.split("】", 1)
                if len(after) > 1 and after[1].strip():
                    mitigation = after[1].strip()
                continue
            if section == "rc" and s:
                root_cause += " " + s if root_cause else s
            elif section == "mit" and s:
                mitigation += "\n" + s if mitigation else s

        return root_cause.strip(), mitigation.strip()


# ── Model Loader ───────────────────────────────────────────────────────────


def load_model(
    model_dir: Path,
    device: torch.device,
) -> tuple[LinkQualityLSTM, NearLinkScaler]:
    """Load trained LSTM model and fitted scaler."""
    ckpt = torch.load(
        model_dir / "model.pth", map_location=device, weights_only=False
    )
    model = LinkQualityLSTM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    scaler = NearLinkScaler.load(model_dir / "scaler.joblib")
    logger.info("Loaded model: %d params, best_val_loss=%.6f",
                model.count_parameters(), ckpt["best_val_loss"])
    return model, scaler


# ── Feature Formatting ─────────────────────────────────────────────────────


def features_to_dict(raw: np.ndarray, names: list[str]) -> dict[str, float]:
    """Convert a raw feature vector to a named dictionary."""
    return {n: float(v) for n, v in zip(names, raw)}


# ── Online Simulation ──────────────────────────────────────────────────────


async def run_simulation(
    model_dir: Path = Path(r"E:\1Projects\ML_Research_Hub\models"),
    n_steps: int = 200,
    use_mock: bool = False,
    n_forward: int = MC_N_FORWARD,
) -> list[DiagnosticEvent]:
    """Run the full MC Dropout + Dynamic Baseline + RAG diagnostic simulation.

    Parameters
    ----------
    model_dir : Path
        Directory containing model.pth and scaler.joblib.
    n_steps : int
        Number of timesteps to simulate.
    use_mock : bool
        Use mock LLM client (no API key required).
    n_forward : int
        Number of MC Dropout forward passes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MC Dropout + Dynamic Baseline + RAG Diagnostic Engine")
    print("=" * 65)

    # ── Load System 1: LSTM ───────────────────────────────────────────────
    print("\n[1/5] Loading System 1 (LinkQualityLSTM)...")
    model, scaler = load_model(model_dir, device)
    mc_predictor = MCDropoutPredictor(model, n_forward=n_forward)
    print(f"      MC Dropout: {n_forward} forward passes per step")

    # ── Load Dynamic Baseline ─────────────────────────────────────────────
    print(f"[2/5] Initializing Dynamic Baseline (EMA window={EMA_WINDOW})...")
    baseline = DynamicBaseline(window_size=EMA_WINDOW)

    # ── Initialize System 2: Diagnostic Engine ────────────────────────────
    print("[3/5] Initializing System 2 (LLM + RAG Diagnostic Engine)...")
    engine = DiagnosticEngine(use_mock=use_mock)
    await engine.initialize()

    # ── Generate test telemetry ────────────────────────────────────────────
    print(f"[4/5] Generating test telemetry ({n_steps} steps)...")
    df = generate_synthetic_telemetry(
        n_samples=n_steps + WINDOW_SIZE + 10, n_sessions=1, seed=99
    )
    raw_data = df[FEATURE_NAMES].values
    normalized = scaler.transform(raw_data)

    # ── Online inference loop ──────────────────────────────────────────────
    print("[5/5] Running MC Dropout online inference...")
    print("-" * 65)

    # Micro-trajectory buffer (raw features, for prompt formatting)
    trajectory_buf: deque[dict[str, float]] = deque(maxlen=TRAJECTORY_LEN)

    # Sliding window for MC Dropout (normalized features)
    window_buf: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)

    anomaly_count = 0
    diagnosis_count = 0

    for t in range(n_steps):
        # Append to sliding window
        window_buf.append(normalized[t])
        trajectory_buf.append(features_to_dict(raw_data[t], FEATURE_NAMES))

        # Phase 1: Cold start — fill window, use single-pass predict_online
        if len(window_buf) < WINDOW_SIZE:
            continue

        # Phase 2: MC Dropout inference on full window
        window_tensor = torch.tensor(
            np.stack(list(window_buf)), dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1, 50, 8)

        mc_mean, mc_var, samples = mc_predictor.predict(window_tensor)

        # Update dynamic baseline
        baseline.update(mc_mean)

        # Check anomaly conditions
        is_anomaly, reason = baseline.is_anomaly(mc_mean, mc_var)

        if is_anomaly:
            anomaly_count += 1
            trajectory_list = list(trajectory_buf)

            print(
                f"\n  [t={t:3d}] ANOMALY | "
                f"MC mean={mc_mean:.4f} var={mc_var:.6f} | "
                f"EMA={baseline.mean:.4f}±{baseline.std:.4f}"
            )
            print(f"          Trigger: {reason}")

            # System 2: RAG + LLM diagnosis
            event = await engine.diagnose(
                timestep=t,
                mc_mean=mc_mean,
                mc_variance=mc_var,
                ema_baseline=baseline.mean,
                trigger_reason=reason,
                trajectory=trajectory_list,
            )
            diagnosis_count += 1

            print(f"  RCA:    {event.root_cause[:120]}...")
            if event.rag_context:
                print(f"  RAG:    {len(event.rag_context)} knowledge chunks retrieved")

    # ── Summary ───────────────────────────────────────────────────────────
    total_inferred = n_steps - WINDOW_SIZE + 1
    print("\n" + "=" * 65)
    print("  Simulation Summary")
    print("=" * 65)
    print(f"  Timesteps simulated:    {n_steps}")
    print(f"  MC Dropout inferences:  {total_inferred}")
    print(f"  Anomalies detected:     {anomaly_count}")
    print(f"  LLM diagnoses:          {diagnosis_count}")
    print(f"  Anomaly rate:           {anomaly_count / max(total_inferred, 1) * 100:.1f}%")
    print(f"  MC forward passes:      {n_forward}/step")
    print(f"  Variance threshold:     {VARIANCE_THRESHOLD}")
    print(f"  Baseline sigma:         {BASELINE_SIGMA}")
    print("=" * 65)

    await engine.shutdown()
    return engine._events


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MC Dropout + RAG-LLM Diagnostic Engine for NearLink"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use mock LLM client (no API key required)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Number of timesteps to simulate (default: 200)")
    parser.add_argument("--n-forward", type=int, default=MC_N_FORWARD,
                        help="MC Dropout forward passes (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    events = asyncio.run(
        run_simulation(
            n_steps=args.steps,
            use_mock=args.mock,
            n_forward=args.n_forward,
        )
    )

    # Print full report for first event
    if events:
        e = events[0]
        print("\n" + "=" * 65)
        print("  Full Diagnostic Report (First Event)")
        print("=" * 65)
        print(f"  Timestep:       {e.timestep}")
        print(f"  MC Mean:        {e.mc_mean:.6f}")
        print(f"  MC Variance:    {e.mc_variance:.6f}")
        print(f"  EMA Baseline:   {e.ema_baseline:.6f}")
        print(f"  Trigger:        {e.trigger_reason}")
        print(f"  RAG Chunks:     {len(e.rag_context)}")
        print(f"\n  Root Cause:\n    {e.root_cause}")
        print(f"\n  Mitigation:\n{e.mitigation}")
        print("=" * 65)
