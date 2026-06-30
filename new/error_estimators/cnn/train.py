"""
CNN supervised error estimator training (Phase 1–2).

Sections:
  1. sys.path → repo new/
  2. Config dataclass
  3. build_features (6ch z-score)
  4. block_error_labels
  5. CNNErrorEstimator
  6. masked_huber_loss
  7. estimate_sigma (estimate_empirical_sigma_tdl_a)
  8. collect_snapshots_one_channel
  9. build_dataset_cache
  10. train_loop
  11. eval_phase1b
  12. build_finetune_cache
  13. finetune_loop
  14. argparse / main
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. sys.path → repo new/
# ---------------------------------------------------------------------------
_CNN_DIR = Path(__file__).resolve().parent
_NEW_ROOT = _CNN_DIR.parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.lmmse import EstimatorState, lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import (
    SionnaOFDMGrid,
    estimate_empirical_sigma_tdl_a,
    sample_tdl_a_channel,
    vec_from_h,
)
from config import ExpRunConfig
from experiments.common import full_subcarrier_batch_true_mse, make_policy
from pilot_policy.base import PilotPolicy, initial_subcarriers_uniform
from simulation import run_until_threshold
from utils import measure_subcarrier, resolve_device

LABEL_EPS = 1e-8
NUM_FEATURE_CHANNELS = 6
POLICY_NAMES = ("random", "fixed", "active")
NOISE_OFFSET = {"random": 10_000, "fixed": 0, "active": 20_000}
TDL_CHANNEL_OFFSET = 100_000
VAL_CHANNEL_OFFSET = 500_000
PHASE2_TRAIN_CHANNELS = 5_000
PHASE2_VAL_CHANNELS = 1_000
PHASE2_NOISE_OFFSET = 30_000
EVAL_CHANNEL_OFFSET = 2_000_000
EVAL_CNN_NOISE_OFFSET = 40_000
EVAL_FULL_LMMSE_NOISE_OFFSET = 99_999
GEN_LOG_INTERVAL = 500

DATA_DIR = _CNN_DIR / "data"
CHECKPOINT_DIR = _CNN_DIR / "checkpoints"
FIGURES_DIR = _CNN_DIR / "figures"


# ---------------------------------------------------------------------------
# 2. Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class CnnTrainConfig:
    n_antennas: int = 16
    n_subcarriers: int = 32
    rho_space: float = 0.8
    sigma2: float = 1e-2
    reg_empirical: float = 1e-3
    k0: int = 1
    max_pilots: int = 32
    n_cov_mc: int = 512
    seed: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    huber_delta: float = 1.0
    label_eps: float = LABEL_EPS
    train_channels: int = 12_000
    val_channels: int = 2_000
    batch_size: int = 128
    max_epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_n_mc: int = 100

    @property
    def n_total(self) -> int:
        return self.n_antennas * self.n_subcarriers

    def to_run_config(self) -> ExpRunConfig:
        return ExpRunConfig(
            n_antennas=self.n_antennas,
            n_subcarriers=self.n_subcarriers,
            rho_space=self.rho_space,
            sigma2=self.sigma2,
            reg_empirical=self.reg_empirical,
            initial_pilot_subcarriers=self.k0,
            max_pilots=self.max_pilots,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )

    def to_eval_run_config(self) -> ExpRunConfig:
        """Phase 1b: run to max_pilots with no threshold stop."""
        return replace(
            self.to_run_config(),
            target_pilots=self.max_pilots,
            initial_pilot_subcarriers=self.k0,
        )


# ---------------------------------------------------------------------------
# 3. build_features
# ---------------------------------------------------------------------------
def h_hat_to_H(h_hat: torch.Tensor, na: int, nc: int) -> torch.Tensor:
    """Input: h_hat (N,1). Output: H (Na,Nc) column-stacked by subcarrier."""
    h = h_hat.squeeze(-1)
    return h.view(nc, na).T.contiguous()


def zscore_across_nc(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Input: (Na,Nc). Output: z-scored along subcarrier dim per antenna row."""
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
    return (x - mu) / std


def build_features(
    h_hat: torch.Tensor,
    mask: torch.Tensor,
    n_pilots: int,
    sigma2: float,
    *,
    na: int,
    nc: int,
) -> torch.Tensor:
    """Input: h_hat (N,1), mask (Nc,), n_pilots, sigma2. Output: (6, Na, Nc) float32."""
    h_mat = h_hat_to_H(h_hat, na, nc)
    re = zscore_across_nc(h_mat.real)
    im = zscore_across_nc(h_mat.imag)
    mag = zscore_across_nc(h_mat.abs())

    mask_2d = mask.view(1, nc).expand(na, nc)
    snr = math.log10(1.0 / sigma2)
    pilot_frac = float(n_pilots) / float(nc)
    snr_t = torch.full((na, nc), snr, dtype=torch.float32, device=mag.device)
    pf_t = torch.full((na, nc), pilot_frac, dtype=torch.float32, device=mag.device)

    return torch.stack(
        [re.float(), im.float(), mag.float(), mask_2d.float(), snr_t, pf_t],
        dim=0,
    )


# ---------------------------------------------------------------------------
# 4. block_error_labels
# ---------------------------------------------------------------------------
def block_error_labels(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
    na: int,
    nc: int,
    *,
    label_eps: float = LABEL_EPS,
) -> torch.Tensor:
    """Input: h_hat, h_true (N,1). Output: (Nc,) log block error."""
    y = torch.zeros(nc, device=h_hat.device, dtype=torch.float32)
    for k in range(nc):
        s = k * na
        diff = h_hat[s : s + na] - h_true[s : s + na]
        e_k = diff.abs().pow(2).mean().real
        y[k] = torch.log(e_k + label_eps)
    return y


# ---------------------------------------------------------------------------
# 5. CNNErrorEstimator
# ---------------------------------------------------------------------------
class CNNErrorEstimator(nn.Module):
    """2D CNN: (B, 6, Na, Nc) → (B, Nc) log block-error predictions."""

    def __init__(self, n_subcarriers: int = 32) -> None:
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.conv2d = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, 6, Na, Nc). Output: (B, Nc)."""
        h = self.conv2d(x)
        h = h.mean(dim=2)
        h = self.head(h)
        return h.squeeze(1)


# ---------------------------------------------------------------------------
# 6. masked_huber_loss
# ---------------------------------------------------------------------------
def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Input: pred, target, mask (B,Nc) 1 on SCs in loss. Output: scalar batch loss.

    Mean Huber per sample over masked SCs, then mean over batch.
    """
    residual = pred - target
    abs_r = residual.abs()
    quadratic = 0.5 * residual.pow(2)
    linear = delta * (abs_r - 0.5 * delta)
    huber = torch.where(abs_r <= delta, quadratic, linear)
    denom = mask.sum(dim=1).clamp(min=1.0)
    per_sample = (huber * mask).sum(dim=1) / denom
    return per_sample.mean()


def unused_mask_from_features(x: torch.Tensor) -> torch.Tensor:
    """Input: (B, 6, Na, Nc). Output: (B, Nc) float, 1 on unused (unpiloted) subcarriers."""
    return (x[:, 3, 0, :] < 0.5).to(dtype=torch.float32)


def val_top1_unused(
    pred: torch.Tensor,
    target: torch.Tensor,
    unused_mask: torch.Tensor,
) -> float:
    """Fraction of samples where argmax pred on unused SCs matches argmax target."""
    hits = 0
    total = 0
    for b in range(pred.shape[0]):
        m = unused_mask[b] > 0.5
        if m.sum() < 1:
            continue
        p = pred[b].clone()
        t = target[b].clone()
        p[~m] = float("-inf")
        t[~m] = float("-inf")
        if p.argmax().item() == t.argmax().item():
            hits += 1
        total += 1
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# 7. estimate_sigma
# ---------------------------------------------------------------------------
def load_empirical_sigma(cfg: CnnTrainConfig, device: torch.device, *, n_cov_mc: int | None = None) -> torch.Tensor:
    """Input: config, device. Output: (N,N) empirical Sigma from TDL-A."""
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    n_mc = n_cov_mc if n_cov_mc is not None else cfg.n_cov_mc
    t0 = time.perf_counter()
    sigma = estimate_empirical_sigma_tdl_a(
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        rho_space=cfg.rho_space,
        n_cov_mc=n_mc,
        seed=cfg.seed,
        seed_offset=1_000_000,
        reg_empirical=cfg.reg_empirical,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
    )
    elapsed = time.perf_counter() - t0
    print(f"Empirical Sigma: n_cov_mc={n_mc}, seed_offset=1_000_000, elapsed={elapsed:.1f}s")
    return sigma


# ---------------------------------------------------------------------------
# 8. collect_snapshots_one_channel
# ---------------------------------------------------------------------------
def rollout_policy_name(channel_idx: int) -> str:
    return POLICY_NAMES[channel_idx % 3]


def tdl_channel_seed(cfg: CnnTrainConfig, channel_mc: int) -> int:
    """Input: channel_mc index/offset. Output: seed for TDL draw and first-pilot randomization."""
    return cfg.seed + TDL_CHANNEL_OFFSET + channel_mc


def random_first_subcarrier(channel_seed: int, nc: int) -> int:
    rng = random.Random(channel_seed + 777_777)
    return rng.randrange(nc)


def sample_tdl_channel(
    cfg: CnnTrainConfig,
    *,
    channel_mc: int,
    device: torch.device,
    grid: SionnaOFDMGrid,
) -> torch.Tensor:
    """Input: channel index mc. Output: h_true (N,1)."""
    h_mat = sample_tdl_a_channel(
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        rho_space=cfg.rho_space,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
        seed=cfg.seed + TDL_CHANNEL_OFFSET + channel_mc,
    )
    return vec_from_h(h_mat)


def collect_snapshots_one_channel(
    cfg: CnnTrainConfig,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    channel_mc: int,
    channel_idx: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Incremental rollout; returns list of (X, y_label, loss_mask) per decision step."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = cfg.n_total
    policy_name = rollout_policy_name(channel_idx)
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    noise_seed = cfg.seed + NOISE_OFFSET[policy_name] + channel_mc

    run_cfg = cfg.to_run_config()
    policy = make_policy(policy_name, run_cfg, device, seed=noise_seed)
    policy.reset()

    k_init = random_first_subcarrier(channel_seed, nc)
    gen = torch.Generator(device=device).manual_seed(noise_seed)

    x0, y0 = measure_subcarrier(
        k_init, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
    )
    state = lmmse_initial_update(sigma, x0, y0, cfg.sigma2)
    used: List[int] = [k_init]
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    mask[k_init] = 1.0
    n_pilots = 1

    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    while n_pilots < cfg.max_pilots:
        x_feat = build_features(state.h_hat, mask, n_pilots, cfg.sigma2, na=na, nc=nc)
        y_lab = block_error_labels(state.h_hat, h_true, na, nc, label_eps=cfg.label_eps)
        loss_mask = torch.ones(nc, device=device, dtype=torch.float32)
        rows.append((x_feat, y_lab, loss_mask))

        k = policy.next_subcarrier(state, used)
        x, y = measure_subcarrier(
            k, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
        )
        state = lmmse_incremental_update(state, x, y, cfg.sigma2)
        used.append(k)
        mask[k] = 1.0
        n_pilots += 1

    return rows


def collect_snapshots_bootstrap_mixed_k0(
    cfg: CnnTrainConfig,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    channel_mc: int,
    channel_idx: int,
    device: torch.device,
    cnn_model: Optional[CNNErrorEstimator],
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Phase 2 on-policy rollouts with 75% k0=1 / 25% random k0 in 1..8."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = cfg.n_total
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    noise_seed = cfg.seed + PHASE2_NOISE_OFFSET + channel_mc
    gen = torch.Generator(device=device).manual_seed(noise_seed)

    if channel_idx % 4 == 0:
        k0 = random.Random(channel_seed + 99).randint(1, 8)
        init_scs = initial_subcarriers_uniform(k0, nc)
    else:
        k0 = 1
        init_scs = [random_first_subcarrier(channel_seed, nc)]

    k0_init = init_scs[0]
    x0, y0 = measure_subcarrier(
        k0_init, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
    )
    state = lmmse_initial_update(sigma, x0, y0, cfg.sigma2)
    used: List[int] = [k0_init]
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    mask[k0_init] = 1.0
    n_pilots = 1

    for k in init_scs[1:]:
        x, y = measure_subcarrier(
            k, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
        )
        state = lmmse_incremental_update(state, x, y, cfg.sigma2)
        used.append(k)
        mask[k] = 1.0
        n_pilots += 1

    if cnn_model is None:
        raise ValueError("Phase 2 rollouts require a loaded CNNErrorEstimator.")

    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    cnn_model.eval()
    with torch.no_grad():
        while n_pilots < cfg.max_pilots:
            x_feat = build_features(state.h_hat, mask, n_pilots, cfg.sigma2, na=na, nc=nc)
            y_lab = block_error_labels(state.h_hat, h_true, na, nc, label_eps=cfg.label_eps)
            loss_mask = torch.ones(nc, device=device, dtype=torch.float32)
            rows.append((x_feat, y_lab, loss_mask))

            x_batch = x_feat.unsqueeze(0)
            pred = cnn_model(x_batch).squeeze(0)
            unused = [k for k in range(nc) if k not in set(used)]
            scores = pred.clone()
            for k in used:
                scores[k] = float("-inf")
            k = int(scores.argmax().item())

            x, y = measure_subcarrier(
                k, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
            )
            state = lmmse_incremental_update(state, x, y, cfg.sigma2)
            used.append(k)
            mask[k] = 1.0
            n_pilots += 1

    return rows


def stack_snapshots(
    rows: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("No snapshots to stack.")
    x = torch.stack([r[0] for r in rows], dim=0)
    y = torch.stack([r[1] for r in rows], dim=0)
    m = torch.stack([r[2] for r in rows], dim=0)
    return x, y, m


# ---------------------------------------------------------------------------
# 9. build_dataset_cache
# ---------------------------------------------------------------------------
def _channel_mc_for_split(split: Literal["train", "val"], channel_idx: int, cfg: CnnTrainConfig) -> int:
    if split == "train":
        return channel_idx
    return VAL_CHANNEL_OFFSET + channel_idx


def _n_channels_for_split(split: Literal["train", "val"], cfg: CnnTrainConfig) -> int:
    return cfg.train_channels if split == "train" else cfg.val_channels


def build_dataset_cache(
    cfg: CnnTrainConfig,
    *,
    split: Literal["train", "val"],
    device: torch.device,
    sigma: torch.Tensor,
    n_channels: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate and save data/{split}.pt."""
    n_ch = n_channels if n_channels is not None else _n_channels_for_split(split, cfg)
    out = out_path or (DATA_DIR / f"{split}.pt")
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)

    print(f"build {split}: {n_ch} channels, device={device}")
    all_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    t0 = time.perf_counter()

    for ch in range(n_ch):
        mc = _channel_mc_for_split(split, ch, cfg)
        h_true = sample_tdl_channel(cfg, channel_mc=mc, device=device, grid=grid)
        rows = collect_snapshots_one_channel(
            cfg, h_true=h_true, sigma=sigma, channel_mc=mc, channel_idx=ch, device=device
        )
        all_rows.extend(rows)
        if (ch + 1) % GEN_LOG_INTERVAL == 0 or ch + 1 == n_ch:
            elapsed = time.perf_counter() - t0
            print(f"  {split} channel {ch + 1}/{n_ch}, {len(all_rows)} snapshots, {elapsed:.1f}s")

    x, y, m = stack_snapshots(all_rows)
    meta: Dict[str, Any] = {
        **{k: (v if not isinstance(v, torch.dtype) else str(v)) for k, v in asdict(cfg).items()},
        "split": split,
        "n_channels": n_ch,
        "n_snapshots": int(x.shape[0]),
        "feature_channels": NUM_FEATURE_CHANNELS,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"X": x.cpu(), "y_label": y.cpu(), "loss_mask": m.cpu(), "meta": meta}, out)
    print(f"Saved {out}: X={tuple(x.shape)}, y_label={tuple(y.shape)}, loss_mask={tuple(m.shape)}")
    return out


# ---------------------------------------------------------------------------
# 10. train_loop
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["X"], blob["y_label"], blob["loss_mask"], blob["meta"]


def run_epoch(
    model: CNNErrorEstimator,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    huber_delta: float,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    n_batches = 0

    for x_b, y_b, _m_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        unused_mask = unused_mask_from_features(x_b)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x_b)
        loss = masked_huber_loss(pred, y_b, unused_mask, huber_delta)

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if not train:
            total_top1 += val_top1_unused(pred.detach(), y_b, unused_mask)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_top1 / max(n_batches, 1)


def train_loop(
    cfg: CnnTrainConfig,
    *,
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
) -> None:
    device = resolve_device(cfg.device)
    x_tr, y_tr, m_tr, _ = load_dataset(train_path)
    x_va, y_va, m_va, _ = load_dataset(val_path)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr, m_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_va, y_va, m_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_epoch = -1
    best_top1 = 0.0
    metrics: List[Dict[str, Any]] = []

    print(
        f"train: device={device}, train={len(x_tr)} val={len(x_va)} "
        f"batch={cfg.batch_size}, max_\epochs={cfg.max_epochs}, loss=unused-SC Huber (per-sample mean)"
    )

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_huber, _ = run_epoch(model, train_loader, device=device, optimizer=opt, huber_delta=cfg.huber_delta, train=True)
        val_huber, val_top1 = run_epoch(
            model, val_loader, device=device, optimizer=None, huber_delta=cfg.huber_delta, train=False
        )
        elapsed = time.perf_counter() - t0
        star = ""
        if val_huber < best_val:
            best_val = val_huber
            best_epoch = epoch
            best_top1 = val_top1
            star = " *"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_val_huber": best_val,
                    "best_val_top1": best_top1,
                    "arch": {"class": "CNNErrorEstimator", "n_subcarriers": cfg.n_subcarriers},
                },
                checkpoint_path,
            )

        metrics.append(
            {"epoch": epoch, "train_huber": train_huber, "val_huber": val_huber, "val_top1": val_top1}
        )
        print(
            f"epoch {epoch:03d}/{cfg.max_epochs:03d}  train_huber={train_huber:.4f}  "
            f"val_huber={val_huber:.4f}  val_top1={val_top1:.3f}  {elapsed:.1f}s{star}"
        )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"train done: best_val_huber={best_val:.4f} @ epoch {best_epoch}  "
        f"best_val_top1={best_top1:.3f}  checkpoint={checkpoint_path}"
    )


# ---------------------------------------------------------------------------
# 11. eval_phase1b
# ---------------------------------------------------------------------------
def load_cnn_checkpoint(path: Path, device: torch.device) -> CNNErrorEstimator:
    blob = torch.load(path, map_location=device, weights_only=False)
    nc = int(blob.get("arch", {}).get("n_subcarriers", 32))
    model = CNNErrorEstimator(n_subcarriers=nc).to(device)
    model.load_state_dict(blob["model_state_dict"])
    return model


class _FixedInitPilotPolicy:
    """Wrap a policy so bootstrap uses a fixed first subcarrier (eval fairness)."""

    def __init__(self, k_init: int, inner: PilotPolicy) -> None:
        self._k_init = k_init
        self._inner = inner

    def reset(self) -> None:
        self._inner.reset()

    def initial_subcarriers(self) -> List[int]:
        return [self._k_init]

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        return self._inner.next_subcarrier(state, used)


class CnnPilotPolicy:
    """Greedy argmax predicted block error on unused subcarriers."""

    def __init__(self, model: CNNErrorEstimator, *, na: int, nc: int, sigma2: float, device: torch.device) -> None:
        self._model = model
        self._na = na
        self._nc = nc
        self._sigma2 = sigma2
        self._device = device

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return []

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        mask = torch.zeros(self._nc, device=self._device, dtype=torch.float32)
        for k in used:
            mask[k] = 1.0
        return self.pick_next(state, used, mask, len(used))

    def pick_next(
        self,
        state: EstimatorState,
        used: List[int],
        mask: torch.Tensor,
        n_pilots: int,
    ) -> int:
        self._model.eval()
        x = build_features(state.h_hat, mask, n_pilots, self._sigma2, na=self._na, nc=self._nc)
        with torch.no_grad():
            pred = self._model(x.unsqueeze(0)).squeeze(0)
        scores = pred.clone()
        for k in used:
            scores[k] = float("-inf")
        return int(scores.argmax().item())


def run_closed_loop_mse_curve(
    cfg: CnnTrainConfig,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    device: torch.device,
    channel_mc: int,
    policy_mode: Literal["cnn", "fixed", "active"],
    cnn_model: Optional[CNNErrorEstimator] = None,
) -> List[float]:
    """Return average MSE after each pilot count (length max_pilots) via run_until_threshold."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    run_cfg = cfg.to_eval_run_config()
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    k_init = random_first_subcarrier(channel_seed, nc)

    if policy_mode == "cnn":
        if cnn_model is None:
            raise ValueError("cnn policy requires model")
        noise_seed = cfg.seed + EVAL_CNN_NOISE_OFFSET + channel_mc
        inner: PilotPolicy = CnnPilotPolicy(cnn_model, na=na, nc=nc, sigma2=cfg.sigma2, device=device)
    elif policy_mode == "fixed":
        noise_seed = cfg.seed + NOISE_OFFSET["fixed"] + channel_mc
        inner = make_policy("fixed", run_cfg, device, seed=noise_seed)
    else:
        noise_seed = cfg.seed + NOISE_OFFSET["active"] + channel_mc
        inner = make_policy("active", run_cfg, device, seed=noise_seed)

    policy = _FixedInitPilotPolicy(k_init, inner)
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    trace = run_until_threshold(h_true, sigma, run_cfg, policy, device=device, generator=gen)
    return trace.nmse_true


def eval_phase1b(cfg: CnnTrainConfig, *, checkpoint_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    device = resolve_device(cfg.device)
    model = load_cnn_checkpoint(checkpoint_path, device)
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    sigma = load_empirical_sigma(cfg, device)

    curves: Dict[str, List[List[float]]] = {"cnn": [], "fixed": [], "active": []}
    h_list: List[torch.Tensor] = []

    print(f"eval phase1b: n_mc={cfg.eval_n_mc}, checkpoint={checkpoint_path}")
    for mc in range(cfg.eval_n_mc):
        channel_mc = EVAL_CHANNEL_OFFSET + mc
        h_true = sample_tdl_channel(cfg, channel_mc=channel_mc, device=device, grid=grid)
        h_list.append(h_true)
        for policy in ("cnn", "fixed", "active"):
            curve = run_closed_loop_mse_curve(
                cfg,
                h_true=h_true,
                sigma=sigma,
                device=device,
                channel_mc=channel_mc,
                policy_mode=policy,
                cnn_model=model if policy == "cnn" else None,
            )
            curves[policy].append(curve)

    mean_curves = {name: np.mean(np.asarray(rows, dtype=np.float64), axis=0) for name, rows in curves.items()}
    pilots = np.arange(1, cfg.max_pilots + 1)

    run_cfg = cfg.to_run_config()
    full_mse_values: List[float] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(
            cfg.seed + EVAL_FULL_LMMSE_NOISE_OFFSET + mc
        )
        full_mse_values.append(
            full_subcarrier_batch_true_mse(h_true, sigma, run_cfg, device=device, generator=gen_noise)
        )
    full_mse = float(np.mean(full_mse_values))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "phase1b_average_mse_vs_pilots.png"
    json_path = FIGURES_DIR / "phase1b_average_mse_vs_pilots.json"

    plt.figure(figsize=(8.4, 4.8))
    for name, curve in mean_curves.items():
        plt.semilogy(pilots, curve, linewidth=1.6, label=name)
    plt.axhline(full_mse, color="red", linestyle=":", linewidth=1.2, label="full LMMSE")
    plt.xlabel("n_pilots")
    plt.ylabel("Mean average MSE")
    plt.title("Phase 1b: average MSE vs pilots")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    payload = {
        "n_mc": cfg.eval_n_mc,
        "checkpoint": str(checkpoint_path),
        "pilots": pilots.tolist(),
        "curves": {k: v.tolist() for k, v in mean_curves.items()},
        "full_lmmse_mse": full_mse,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {fig_path}")
    for name, curve in mean_curves.items():
        print(f"  {name} @ n_pilots=32: {curve[-1]:.6e}")
    print(f"  full LMMSE: {full_mse:.6e}")


# ---------------------------------------------------------------------------
# 12. build_finetune_cache
# ---------------------------------------------------------------------------
def build_finetune_cache(
    cfg: CnnTrainConfig,
    *,
    device: torch.device,
    sigma: torch.Tensor,
    checkpoint_path: Path,
) -> None:
    model = load_cnn_checkpoint(checkpoint_path, device)
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    print(f"build-finetune: CNN policy from {checkpoint_path}")

    for split, n_ch, offset in (
        ("phase2_train", PHASE2_TRAIN_CHANNELS, 3_000_000),
        ("phase2_val", PHASE2_VAL_CHANNELS, 3_500_000),
    ):
        out = DATA_DIR / f"{split}.pt"
        all_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        t0 = time.perf_counter()
        n_mixed_k0 = sum(1 for ch in range(n_ch) if ch % 4 == 0)
        n_k0_eq1 = n_ch - n_mixed_k0
        print(
            f"build-finetune {split}: {n_ch} channels (on-policy CNN), "
            f"k0 mix: {n_k0_eq1} at k0=1 ({100.0 * n_k0_eq1 / n_ch:.1f}%), "
            f"{n_mixed_k0} at k0~Uniform[1,8] ({100.0 * n_mixed_k0 / n_ch:.1f}%)"
        )
        for ch in range(n_ch):
            mc = offset + ch
            h_true = sample_tdl_channel(cfg, channel_mc=mc, device=device, grid=grid)
            rows = collect_snapshots_bootstrap_mixed_k0(
                cfg,
                h_true=h_true,
                sigma=sigma,
                channel_mc=mc,
                channel_idx=ch,
                device=device,
                cnn_model=model,
            )
            all_rows.extend(rows)
            if (ch + 1) % GEN_LOG_INTERVAL == 0 or ch + 1 == n_ch:
                print(f"  {split} channel {ch + 1}/{n_ch}, {len(all_rows)} snapshots, {time.perf_counter() - t0:.1f}s")

        x, y, m = stack_snapshots(all_rows)
        meta = {
            **{k: (v if not isinstance(v, torch.dtype) else str(v)) for k, v in asdict(cfg).items()},
            "split": split,
            "n_channels": n_ch,
            "n_snapshots": int(x.shape[0]),
            "on_policy": True,
            "k0_mix": {"k0_eq1": n_k0_eq1, "mixed_k0_uniform_1_8": n_mixed_k0},
        }
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"X": x.cpu(), "y_label": y.cpu(), "loss_mask": m.cpu(), "meta": meta}, out)
        print(f"Saved {out}: {tuple(x.shape)}")


# ---------------------------------------------------------------------------
# 13. finetune_loop
# ---------------------------------------------------------------------------
def finetune_loop(cfg: CnnTrainConfig, *, init_checkpoint: Path) -> None:
    device = resolve_device(cfg.device)
    blob = torch.load(init_checkpoint, map_location=device, weights_only=False)
    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers).to(device)
    model.load_state_dict(blob["model_state_dict"])

    x_tr, y_tr, m_tr, _ = load_dataset(DATA_DIR / "phase2_train.pt")
    x_va, y_va, m_va, _ = load_dataset(DATA_DIR / "phase2_val.pt")

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr, m_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_va, y_va, m_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ckpt = CHECKPOINT_DIR / "phase2_best.pt"
    metrics_path = CHECKPOINT_DIR / "phase2_metrics.json"
    best_val = float("inf")
    best_epoch = -1
    best_top1 = 0.0
    metrics: List[Dict[str, Any]] = []

    print(f"finetune: init={init_checkpoint}, max_epochs={cfg.max_epochs}")
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_huber, _ = run_epoch(model, train_loader, device=device, optimizer=opt, huber_delta=cfg.huber_delta, train=True)
        val_huber, val_top1 = run_epoch(
            model, val_loader, device=device, optimizer=None, huber_delta=cfg.huber_delta, train=False
        )
        elapsed = time.perf_counter() - t0
        star = ""
        if val_huber < best_val:
            best_val = val_huber
            best_epoch = epoch
            best_top1 = val_top1
            star = " *"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_val_huber": best_val,
                    "best_val_top1": best_top1,
                    "arch": {"class": "CNNErrorEstimator", "n_subcarriers": cfg.n_subcarriers},
                },
                ckpt,
            )
        metrics.append({"epoch": epoch, "train_huber": train_huber, "val_huber": val_huber, "val_top1": val_top1})
        print(
            f"finetune epoch {epoch:03d}/{cfg.max_epochs:03d}  train_huber={train_huber:.4f}  "
            f"val_huber={val_huber:.4f}  val_top1={val_top1:.3f}  {elapsed:.1f}s{star}"
        )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"finetune done: best_val_huber={best_val:.4f} @ epoch {best_epoch}  "
        f"best_val_top1={best_top1:.3f}  checkpoint={ckpt}"
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def build_sanity_minibatch(
    cfg: CnnTrainConfig,
    *,
    batch_size: int,
    sigma: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for ch in range(batch_size):
        h_true = sample_tdl_channel(cfg, channel_mc=ch, device=device, grid=grid)
        snap = collect_snapshots_one_channel(
            cfg, h_true=h_true, sigma=sigma, channel_mc=ch, channel_idx=ch, device=device
        )
        rows.append(snap[0])
    return stack_snapshots(rows)


def sanity_check(
    cfg: Optional[CnnTrainConfig] = None,
    *,
    batch_size: int = 12,
    n_epochs: int = 10,
    lr: float = 1e-3,
) -> None:
    """Overfit one minibatch; loss must decrease with finite gradients."""
    cfg = cfg or CnnTrainConfig()
    device = resolve_device(cfg.device)
    sanity_cfg = replace(cfg, n_cov_mc=min(cfg.n_cov_mc, 64))

    print(f"sanity: device={device}, batch_size={batch_size}, epochs={n_epochs}")
    sigma = load_empirical_sigma(sanity_cfg, device, n_cov_mc=sanity_cfg.n_cov_mc)

    print("Building one minibatch from TDL-A rollouts...")
    x, y, _loss_mask = build_sanity_minibatch(cfg, batch_size=batch_size, sigma=sigma, device=device)
    x = x.to(device)
    y = y.to(device)
    unused_mask = unused_mask_from_features(x)

    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on one minibatch ({NUM_FEATURE_CHANNELS} feature channels, unused-SC loss):")
    losses: List[float] = []
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = masked_huber_loss(pred, y, unused_mask, cfg.huber_delta)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}: {loss.item()}")
        loss.backward()

        grad_norm = 0.0
        n_params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                if g.abs().sum() > 0:
                    n_params_with_grad += 1
                grad_norm += g.norm(2).item() ** 2
        grad_norm = grad_norm**0.5

        if epoch == 0 and n_params_with_grad == 0:
            raise RuntimeError("No parameters received non-zero gradients on epoch 0.")

        opt.step()
        lv = loss.item()
        losses.append(lv)
        print(f"epoch {epoch + 1}/{n_epochs}  loss={lv:.6f}  grad_norm={grad_norm:.4e}")

    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}")
    if len(losses) >= 2 and abs(losses[0] - losses[1]) < 1e-12:
        raise RuntimeError("Loss barely changed by epoch 1; check graph connectivity.")

    reduction = 100.0 * (1.0 - losses[-1] / losses[0])
    print(f"sanity PASSED: loss {losses[0]:.6f} -> {losses[-1]:.6f} ({reduction:.1f}% reduction)")


# ---------------------------------------------------------------------------
# 14. argparse / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN supervised error estimator training.")
    parser.add_argument(
        "--phase",
        choices=("sanity", "build", "train", "eval", "build-finetune", "finetune"),
        default="sanity",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--n-channels", type=int, default=None, help="Override channel count for build smoke tests.")
    parser.add_argument("--force-regen", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CnnTrainConfig()
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs

    if args.phase == "sanity":
        sanity_check(cfg)
        return

    device = resolve_device(cfg.device)

    if args.phase == "build":
        train_pt = DATA_DIR / "train.pt"
        val_pt = DATA_DIR / "val.pt"
        if not args.force_regen and train_pt.exists() and val_pt.exists():
            print(f"cache hit: {train_pt}, {val_pt} (use --force-regen to rebuild)")
            return
        sigma = load_empirical_sigma(cfg, device)
        n_override = args.n_channels
        build_dataset_cache(cfg, split="train", device=device, sigma=sigma, n_channels=n_override)
        build_dataset_cache(cfg, split="val", device=device, sigma=sigma, n_channels=n_override)
        return

    if args.phase == "train":
        train_loop(
            cfg,
            train_path=DATA_DIR / "train.pt",
            val_path=DATA_DIR / "val.pt",
            checkpoint_path=CHECKPOINT_DIR / "phase1_best.pt",
            metrics_path=CHECKPOINT_DIR / "phase1_metrics.json",
        )
        return

    if args.phase == "eval":
        eval_phase1b(cfg, checkpoint_path=CHECKPOINT_DIR / "phase1_best.pt")
        return

    if args.phase == "build-finetune":
        phase2_train_pt = DATA_DIR / "phase2_train.pt"
        phase2_val_pt = DATA_DIR / "phase2_val.pt"
        if not args.force_regen and phase2_train_pt.exists() and phase2_val_pt.exists():
            print(f"cache hit: {phase2_train_pt}, {phase2_val_pt} (use --force-regen to rebuild)")
            return
        sigma = load_empirical_sigma(cfg, device)
        build_finetune_cache(cfg, device=device, sigma=sigma, checkpoint_path=CHECKPOINT_DIR / "phase1_best.pt")
        return

    if args.phase == "finetune":
        finetune_loop(cfg, init_checkpoint=CHECKPOINT_DIR / "phase1_best.pt")
        return

    raise ValueError(f"Unknown phase: {args.phase}")


if __name__ == "__main__":
    main()
