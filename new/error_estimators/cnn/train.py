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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. sys.path → repo new/
# ---------------------------------------------------------------------------
_CNN_DIR = Path(__file__).resolve().parent
_NEW_ROOT = _CNN_DIR.parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.cov_update import ledoit_wolf_shrinkage
from channel_estimators.lmmse import EstimatorState, lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import (
    SionnaOFDMGrid,
    estimate_empirical_sigma_tdl_a,
    sample_tdl_a_channel,
    sample_tdl_ofdm_channel,
    vec_from_h,
)
from config import ExpRunConfig
from experiments.common import full_subcarrier_batch_true_mse, make_policy
from pilot_policy.base import PilotPolicy, initial_subcarriers_uniform
from pilot_policy.dmrs_comb import DmrsCombPolicy
from simulation import run_until_threshold
from utils import complex_standard_normal, measure_subcarrier, resolve_device

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
SIGMA_BUCKET_OFFSET = 1_000_000  # time_freq: seed + SIGMA_BUCKET_OFFSET + bucket_stride*idx + k
SIGMA_BUCKET_STRIDE = 100_000  # per-bucket seed stride, must exceed n_cov_mc_per_bucket

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
    label_mode: Literal["current", "counterfactual"] = "current"
    early_stop_patience: Optional[int] = None
    # domain="time_freq" only (no-ops under the default "antenna_snapshot"): single-antenna,
    # joint time-freq grid (n_symbols OFDM symbols x n_freq_subcarriers), swept over
    # tdl_models x speeds buckets, with the DMRS-comb "fixed" baseline instead of the generic
    # 1D bisection (see pilot_policy.dmrs_comb). domain="time_freq" forces n_antennas=1 and
    # n_subcarriers=n_symbols*n_freq_subcarriers (see _check_domain_config).
    domain: Literal["antenna_snapshot", "time_freq"] = "antenna_snapshot"
    n_freq_subcarriers: int = 32
    n_symbols: int = 16
    tdl_models: Tuple[str, ...] = ("A", "B", "C")
    speeds: Tuple[float, ...] = (3.0, 30.0, 120.0)
    n_cov_mc_per_bucket: int = 256
    dmrs_comb_spacing: int = 2
    dmrs_pilot_symbols: Tuple[int, ...] = (2, 6, 10, 14)
    # time_freq only (no-op under "antenna_snapshot", where every sample already shares the one
    # implicit bucket): average the masked Huber loss per (model,speed) bucket first, then mean
    # those 9 bucket-means, instead of one flat per-sample mean across the whole batch. A flat
    # mean lets whichever bucket has the largest native error scale (highest speed) dominate the
    # gradient; bucket-weighting gives every bucket equal say regardless of its loss magnitude.
    bucket_weighted_loss: bool = False
    # time_freq only (no-op under "antenna_snapshot" by default): add a squeeze-excite/FiLM
    # global-context branch to CNNErrorEstimator (see its docstring) so per-cell predictions can
    # be conditioned on the sample's overall regime, which a ~7x7 local receptive field can't see.
    use_global_context: bool = False
    # Checkpoint-selection criterion. "val_huber" (default, preserves prior behavior exactly):
    # select/early-stop on the masked-Huber regression loss over the validation set. "real_eval":
    # select on an actual cnn-policy closed-loop MSE proxy (real_eval_cnn_score) instead -- a
    # diagnostic across one 15-epoch time_freq run found pooled val Huber only weakly correlated
    # with real deployment MSE (Pearson ~0.36-0.41 across epochs; epochs with near-identical val
    # Huber differed by up to 4x in real MSE), so val Huber alone is not a reliable proxy for
    # picking the best checkpoint in this multi-bucket setting.
    select_checkpoint_by: Literal["val_huber", "real_eval"] = "val_huber"
    real_eval_n_mc: int = 30  # held-out channels per bucket for the real_eval proxy (cheaper than eval_n_mc)
    real_eval_criterion: Literal["mean", "worst_bucket"] = "mean"
    # Override CNNErrorEstimator's per-domain default kernel_hw (None = use model_arch_kwargs's
    # domain default, (3,3) for time_freq / (3,5) for antenna_snapshot). time_freq motivation:
    # Doppler-driven decorrelation lives on the symbol (H) axis, so widening only H (e.g. (7,3))
    # grows the time-axis receptive field (7->15 symbols on the 16-symbol grid, over 2 layers)
    # without touching the frequency-axis kernel.
    kernel_hw: Optional[Tuple[int, int]] = None
    # "huber" (default, preserves prior behavior exactly): masked Huber regression on log(e+eps)
    # per candidate. "ranking_ce": cross-entropy over unused candidates against the single true
    # best one (CnnPilotPolicy's actual argmin/argmax deployment decision) instead of regressing
    # the absolute label value -- see masked_ranking_ce_loss docstring for the motivation (label
    # scale varies ~15x across time_freq's 9 buckets, which the deployment decision never needs
    # to resolve, only the within-sample ranking).
    loss_type: Literal["huber", "ranking_ce"] = "huber"

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


def _check_domain_config(cfg: "CnnTrainConfig") -> None:
    """domain='time_freq' requires n_antennas=1 (pilot = one (k,t) RE) and n_subcarriers to be
    the flat n_symbols*n_freq_subcarriers index space (mirrors exp12/13's _check_config)."""
    if cfg.domain != "time_freq":
        return
    if cfg.n_antennas != 1:
        raise ValueError("domain='time_freq' requires n_antennas=1.")
    expected_n = cfg.n_symbols * cfg.n_freq_subcarriers
    if cfg.n_subcarriers != expected_n:
        raise ValueError(
            f"domain='time_freq': n_subcarriers ({cfg.n_subcarriers}) must equal "
            f"n_symbols*n_freq_subcarriers ({cfg.n_symbols}*{cfg.n_freq_subcarriers}={expected_n})."
        )


def feature_grid_shape(cfg: "CnnTrainConfig") -> Tuple[int, int]:
    """Output: (na,nc) to pass to h_hat_to_H/build_features/CNNErrorEstimator -- the h_hat
    block layout (vec_from_h convention) to reshape into a 2D grid for the CNN. antenna_snapshot:
    the grid IS (n_antennas,n_subcarriers). time_freq: the grid is (n_symbols,n_freq_subcarriers)
    -- distinct from the flat (n_antennas=1, n_subcarriers=n_symbols*n_freq_subcarriers) candidate
    indexing used by labels/LMMSE/policies, since vec_from_h's actual block size for a time-freq
    channel is n_symbols (one subcarrier's time samples), not 1."""
    if cfg.domain == "time_freq":
        return cfg.n_symbols, cfg.n_freq_subcarriers
    return cfg.n_antennas, cfg.n_subcarriers


def zscore_mode_for(cfg: "CnnTrainConfig") -> Literal["per_row", "global"]:
    return "global" if cfg.domain == "time_freq" else "per_row"


def mask_to_grid(mask_flat: torch.Tensor, cfg: "CnnTrainConfig") -> torch.Tensor:
    """Input: flat (nc,) pilot-used mask, nc=cfg.n_subcarriers. Output: 2D mask matching
    feature_grid_shape(cfg)'s grid convention. antenna_snapshot: one subcarrier pilot reveals
    all n_antennas at once, so the flat per-subcarrier mask is broadcast across antenna rows
    (unchanged from before). time_freq: one pilot reveals exactly one (symbol,subcarrier) cell,
    so the mask is reshaped (not broadcast) into the genuine 2D grid via the same vec_from_h
    block convention as h_hat_to_H -- strictly more informative than a broadcast."""
    if cfg.domain == "time_freq":
        nt, nc_freq = cfg.n_symbols, cfg.n_freq_subcarriers
        return mask_flat.view(nc_freq, nt).T.contiguous()
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    return mask_flat.view(1, nc).expand(na, nc)


# ---------------------------------------------------------------------------
# 3. build_features
# ---------------------------------------------------------------------------
def h_hat_to_H(h_hat: torch.Tensor, na: int, nc: int) -> torch.Tensor:
    """Input: h_hat (N,1). Output: H (Na,Nc) column-stacked by subcarrier."""
    h = h_hat.squeeze(-1)
    return h.view(nc, na).T.contiguous()


def zscore_across_nc(
    x: torch.Tensor, eps: float = 1e-8, *, mode: Literal["per_row", "global"] = "per_row"
) -> torch.Tensor:
    """Input: (Na,Nc). Output: z-scored, either along the subcarrier dim per row ("per_row",
    antenna-domain default: antennas are exchangeable, no cross-row structure to preserve) or
    jointly over both axes ("global", time-freq domain: both axes carry sample-varying
    structure -- frequency-selectivity differs by TDL model, time-selectivity by speed -- and
    normalizing along either axis alone would flatten out exactly that structure)."""
    if mode == "global":
        mu = x.mean()
        std = x.std(unbiased=False).clamp(min=eps)
        return (x - mu) / std
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)
    return (x - mu) / std


def build_features(
    h_hat: torch.Tensor,
    mask_2d: torch.Tensor,
    n_pilots: int,
    sigma2: float,
    *,
    na: int,
    nc: int,
    n_candidates: Optional[int] = None,
    zscore_mode: Literal["per_row", "global"] = "per_row",
) -> torch.Tensor:
    """Input: h_hat (N,1), mask_2d (Na,Nc) (already broadcast/reshaped by the caller via
    mask_to_grid), n_pilots, sigma2. na/nc: h_hat's reshape grid (see feature_grid_shape).
    n_candidates: total flat candidate count for the pilot-fraction feature's denominator;
    defaults to nc, correct only when the grid IS the flat candidate space (antenna_snapshot,
    where grid_nc==cfg.n_subcarriers) -- must be passed explicitly otherwise (time_freq, where
    nc=n_freq_subcarriers but there are n_symbols*n_freq_subcarriers total flat candidates).
    Output: (6, Na, Nc) float32."""
    h_mat = h_hat_to_H(h_hat, na, nc)
    re = zscore_across_nc(h_mat.real, mode=zscore_mode)
    im = zscore_across_nc(h_mat.imag, mode=zscore_mode)
    mag = zscore_across_nc(h_mat.abs(), mode=zscore_mode)

    n_cand = n_candidates if n_candidates is not None else nc
    snr = math.log10(1.0 / sigma2)
    pilot_frac = float(n_pilots) / float(n_cand)
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


def label_suffix(cfg: "CnnTrainConfig") -> str:
    """Input: config. Output: '_tf' for domain='time_freq', then '_cf' for counterfactual
    label mode (either, both, or neither)."""
    suffix = "_tf" if cfg.domain == "time_freq" else ""
    if cfg.label_mode == "counterfactual":
        suffix += "_cf"
    return suffix


def counterfactual_labels(
    state: EstimatorState,
    h_true: torch.Tensor,
    used: Sequence[int],
    na: int,
    nc: int,
    sigma2: float,
    generator: torch.Generator,
    *,
    label_eps: float = LABEL_EPS,
) -> torch.Tensor:
    """Input: state (h_hat,P), h_true (N,1), used SCs, Na, Nc, sigma2, noise generator.
    Output: (Nc,) log counterfactual NMSE labels.

    For each unused candidate k, simulate hypothetically placing one pilot there (fresh noise,
    one incremental LMMSE update) and label with the resulting full-vector true NMSE against
    h_true. Exploits that a single-subcarrier pilot is a pure block-selector, so the Kalman
    gain/update can be computed directly from P's block columns (O(N*Na^2) per candidate)
    instead of the generic dense (N,N) update, and is vectorized across all unused candidates
    at once via batched (Na,Na) solves. Used (already-piloted) SCs get a label_eps floor;
    they are excluded from the training loss via the unused-SC mask anyway.
    """
    device = state.h_hat.device
    dtype = state.h_hat.dtype
    used_set = set(int(k) for k in used)
    unused = [k for k in range(nc) if k not in used_set]
    y_lab = torch.full((nc,), math.log(label_eps), device=device, dtype=torch.float32)
    if not unused:
        return y_lab

    h_hat = state.h_hat  # (N,1)
    p = state.P  # (N,N)
    k_cand = len(unused)
    k_idx = torch.tensor(unused, device=device, dtype=torch.long)
    starts = k_idx * na
    offsets = torch.arange(na, device=device)
    block_idx = starts[:, None] + offsets[None, :]  # (K,Na)

    p_cols = p[:, block_idx.reshape(-1)].reshape(p.shape[0], k_cand, na).permute(1, 0, 2)  # (K,N,Na)
    p_blocks = p[block_idx[:, :, None], block_idx[:, None, :]]  # (K,Na,Na)
    eye_na = torch.eye(na, device=device, dtype=dtype)
    s_mats = sigma2 * eye_na.unsqueeze(0) + p_blocks  # (K,Na,Na)
    inv_s = torch.linalg.solve(s_mats, eye_na.unsqueeze(0).expand(k_cand, na, na))  # (K,Na,Na)
    gain = p_cols @ inv_s  # (K,N,Na)

    noise = (sigma2**0.5) * complex_standard_normal(
        k_cand, na, 1, device=device, dtype=dtype, generator=generator
    )
    h_true_blocks = h_true[block_idx]  # (K,Na,1)
    h_hat_blocks = h_hat[block_idx]  # (K,Na,1)
    innovation = (h_true_blocks + noise) - h_hat_blocks  # (K,Na,1)

    delta_h = gain @ innovation  # (K,N,1)
    h_new = h_hat.unsqueeze(0) + delta_h  # (K,N,1)
    diff = h_new - h_true.unsqueeze(0)  # (K,N,1)
    e_k = diff.abs().pow(2).mean(dim=(1, 2))  # (K,)
    y_lab[k_idx] = torch.log(e_k + label_eps).float()
    return y_lab


# ---------------------------------------------------------------------------
# 5. CNNErrorEstimator
# ---------------------------------------------------------------------------
class CNNErrorEstimator(nn.Module):
    """2D CNN: (B, 6, H, W) → (B, Nc) log error predictions.

    pool_mode="mean" (antenna_snapshot domain): H is a nuisance axis (e.g. antennas) averaged
    away, one score per W=Nc candidate. pool_mode="per_cell" (time_freq domain): every (H,W)
    cell is its own independent pilot candidate, so no axis is pooled; the (H,W) score grid is
    instead flattened subcarrier-major (W major, H minor: s=w*H+h) to match vec_from_h's block
    convention -- requires H=n_symbols, W=n_freq_subcarriers at the input (see feature_grid_shape).
    """

    def __init__(
        self,
        n_subcarriers: int = 32,
        pool_mode: Literal["mean", "per_cell"] = "mean",
        kernel_hw: Tuple[int, int] = (3, 5),
        use_global_context: bool = False,
    ) -> None:
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.pool_mode = pool_mode
        self.use_global_context = use_global_context
        kh, kw = kernel_hw
        ph, pw = kh // 2, kw // 2
        self.conv2d = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(kh, kw), padding=(ph, pw), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(kh, kw), padding=(ph, pw), bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        if use_global_context:
            # Squeeze-excite / FiLM: global-average-pool the whole (H,W) grid into a per-sample
            # context vector and use it to scale+shift the per-cell features before the head.
            # Lets a cell's prediction be conditioned on the sample's overall regime (e.g. TDL
            # model/speed) inferred from the full grid -- something a 7x7-ish local receptive
            # field cannot see on its own. Last layer zero-init so training starts as identity
            # (gamma=0 -> scale=1, beta=0) and only learns modulation if it helps.
            self.context_mlp = nn.Sequential(
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 64),
            )
            nn.init.zeros_(self.context_mlp[-1].weight)
            nn.init.zeros_(self.context_mlp[-1].bias)
        if pool_mode == "mean":
            self.head = nn.Sequential(
                nn.Conv1d(32, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(16, 1, kernel_size=1),
            )
        elif pool_mode == "per_cell":
            self.head = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(16, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, 6, H, W). Output: (B, Nc)."""
        h = self.conv2d(x)
        if self.use_global_context:
            ctx = h.mean(dim=(2, 3))  # (B,32) global average pool over the whole grid
            gamma, beta = self.context_mlp(ctx).chunk(2, dim=1)  # (B,32) each
            h = h * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        if self.pool_mode == "mean":
            h = h.mean(dim=2)
            h = self.head(h)
            return h.squeeze(1)
        h = self.head(h).squeeze(1)  # (B,H,W)
        return h.permute(0, 2, 1).reshape(h.shape[0], -1)  # (B,W,H) -> flat w*H+h


def model_arch_kwargs(cfg: "CnnTrainConfig") -> Dict[str, Any]:
    """Output: CNNErrorEstimator kwargs appropriate for cfg.domain (see class docstring)."""
    if cfg.domain == "time_freq":
        return {
            "pool_mode": "per_cell",
            "kernel_hw": cfg.kernel_hw if cfg.kernel_hw is not None else (3, 3),
            "use_global_context": cfg.use_global_context,
        }
    return {
        "pool_mode": "mean",
        "kernel_hw": cfg.kernel_hw if cfg.kernel_hw is not None else (3, 5),
        "use_global_context": cfg.use_global_context,
    }


# ---------------------------------------------------------------------------
# 6. masked_huber_loss
# ---------------------------------------------------------------------------
def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
    bucket_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Input: pred, target, mask (B,Nc) 1 on SCs in loss. Output: scalar batch loss.

    Mean Huber per sample over masked SCs, then mean over batch -- unless `bucket_idx` (B,) is
    given and spans more than one value, in which case the per-sample losses are first averaged
    within each bucket present in the batch, then those bucket-means are averaged uniformly. This
    keeps a high-error-scale bucket (e.g. high speed) from dominating the gradient just because
    its per-sample Huber values are larger than a low-error-scale bucket's.
    """
    residual = pred - target
    abs_r = residual.abs()
    quadratic = 0.5 * residual.pow(2)
    linear = delta * (abs_r - 0.5 * delta)
    huber = torch.where(abs_r <= delta, quadratic, linear)
    denom = mask.sum(dim=1).clamp(min=1.0)
    per_sample = (huber * mask).sum(dim=1) / denom

    if bucket_idx is None:
        return per_sample.mean()
    buckets = torch.unique(bucket_idx)
    if buckets.numel() <= 1:
        return per_sample.mean()
    bucket_means = torch.stack([per_sample[bucket_idx == b].mean() for b in buckets])
    return bucket_means.mean()


def masked_ranking_ce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    select_mode: Literal["argmax", "argmin"] = "argmin",
) -> torch.Tensor:
    """Input: pred (B,Nc) raw scores, target (B,Nc) true label, mask (B,Nc) 1 on unused SCs.
    Output: scalar batch loss. Cross-entropy over unused candidates against the single best one
    (argmin target for select_mode="argmin"/counterfactual, argmax for "argmax"/current) --
    directly optimizes the deployment decision (CnnPilotPolicy's argmin/argmax pick) instead of
    matching absolute label magnitudes, which vary ~15x in scale across time_freq's 9 buckets and
    needn't be matched for the ranking to be correct. Discards the graded "how much worse" signal
    Huber keeps; see CnnTrainConfig.loss_type docstring."""
    unused = mask >= 0.5
    masked_pred = pred.masked_fill(~unused, float("-inf"))
    fill = float("inf") if select_mode == "argmin" else float("-inf")
    masked_target = target.masked_fill(~unused, fill)
    best_idx = masked_target.argmin(dim=1) if select_mode == "argmin" else masked_target.argmax(dim=1)
    return F.cross_entropy(masked_pred, best_idx)


def unused_mask_from_features(
    x: torch.Tensor, *, domain: Literal["antenna_snapshot", "time_freq"] = "antenna_snapshot"
) -> torch.Tensor:
    """Input: (B, 6, H, W). Output: (B, Nc) float, 1 on unused (unpiloted) candidates.
    antenna_snapshot: mask channel is constant across H (broadcast), any row works, flat Nc=W.
    time_freq: mask channel is genuinely 2D (no broadcast), so the full plane must be flattened
    the same way the per_cell head flattens its scores (W major, H minor: s=w*H+h), else the
    mask wouldn't align with model predictions / y_label's flat ordering."""
    if domain == "time_freq":
        m = (x[:, 3, :, :] < 0.5).to(dtype=torch.float32)  # (B,H,W)
        return m.permute(0, 2, 1).reshape(m.shape[0], -1)  # (B,W,H) -> flat w*H+h
    return (x[:, 3, 0, :] < 0.5).to(dtype=torch.float32)


def val_top1_unused(
    pred: torch.Tensor,
    target: torch.Tensor,
    unused_mask: torch.Tensor,
    select_mode: Literal["argmax", "argmin"] = "argmax",
) -> float:
    """Fraction of samples where pred's deployment pick on unused SCs (argmax for select_mode=
    "argmax", i.e. label_mode="current"; argmin for "argmin", i.e. label_mode="counterfactual")
    matches target's. Must match CnnPilotPolicy's select_mode for this number to mean anything --
    checking argmax agreement for a counterfactual-mode (argmin-deployed) checkpoint silently
    measures the wrong direction."""
    hits = 0
    total = 0
    fill = float("-inf") if select_mode == "argmax" else float("inf")
    for b in range(pred.shape[0]):
        m = unused_mask[b] > 0.5
        if m.sum() < 1:
            continue
        p = pred[b].clone()
        t = target[b].clone()
        p[~m] = fill
        t[~m] = fill
        p_best = p.argmax().item() if select_mode == "argmax" else p.argmin().item()
        t_best = t.argmax().item() if select_mode == "argmax" else t.argmin().item()
        if p_best == t_best:
            hits += 1
        total += 1
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# 7. estimate_sigma
# ---------------------------------------------------------------------------
def make_grid(cfg: CnnTrainConfig) -> SionnaOFDMGrid:
    """Output: the OFDM grid for this domain -- fft_size is the actual number of frequency
    subcarriers modeled (n_freq_subcarriers in time_freq mode, not the flat n_subcarriers)."""
    if cfg.domain == "time_freq":
        return SionnaOFDMGrid(fft_size=cfg.n_freq_subcarriers)
    return SionnaOFDMGrid(fft_size=cfg.n_subcarriers)


def load_empirical_sigma(cfg: CnnTrainConfig, device: torch.device, *, n_cov_mc: int | None = None) -> torch.Tensor:
    """Input: config, device. Output: (N,N) empirical Sigma from TDL-A."""
    grid = make_grid(cfg)
    n_mc = n_cov_mc if n_cov_mc is not None else cfg.n_cov_mc
    t0 = time.perf_counter()
    sigma = estimate_empirical_sigma_tdl_a(
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        rho_space=cfg.rho_space,
        n_cov_mc=n_mc,
        seed=cfg.seed,
        seed_offset=SIGMA_BUCKET_OFFSET,
        reg_empirical=cfg.reg_empirical,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
    )
    elapsed = time.perf_counter() - t0
    print(f"Empirical Sigma: n_cov_mc={n_mc}, seed_offset={SIGMA_BUCKET_OFFSET}, elapsed={elapsed:.1f}s")
    return sigma


def tdl_buckets(cfg: CnnTrainConfig) -> List[Tuple[str, float]]:
    """time_freq only: the (model,speed) sweep, e.g. 3 models x 3 speeds = 9 buckets."""
    return [(m, s) for m in cfg.tdl_models for s in cfg.speeds]


def bucket_for_channel(cfg: CnnTrainConfig, channel_mc: int) -> Tuple[str, float]:
    """time_freq only: round-robins (model,speed) buckets the same way POLICY_NAMES cycles by
    channel_idx % 3 for the policy mix (not required to be phase-aligned with it)."""
    buckets = tdl_buckets(cfg)
    return buckets[channel_mc % len(buckets)]


def load_sigma_buckets(cfg: CnnTrainConfig, device: torch.device) -> Dict[Tuple[str, float], torch.Tensor]:
    """Input: config, device. Output: {(model,speed): Sigma}. antenna_snapshot: single-entry
    dict via today's load_empirical_sigma (unchanged numerics). time_freq: one Ledoit-Wolf-shrunk
    Sigma per (model,speed) bucket (reg via cfg.reg_empirical), each from cfg.n_cov_mc_per_bucket
    draws of that bucket's own TDL model/speed -- needed because a single shared Sigma (today's
    scheme) would be statistically mismatched for every channel whose true (model,speed)
    second-order statistics differ from whatever single setting it was estimated under."""
    if cfg.domain != "time_freq":
        return {("A", 0.0): load_empirical_sigma(cfg, device)}

    grid = make_grid(cfg)
    buckets = tdl_buckets(cfg)
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor] = {}
    t0 = time.perf_counter()
    for bucket_idx, (model, speed) in enumerate(buckets):
        samples = torch.zeros((cfg.n_cov_mc_per_bucket, cfg.n_subcarriers), device=device, dtype=cfg.dtype)
        for k in range(cfg.n_cov_mc_per_bucket):
            h_tf = sample_tdl_ofdm_channel(
                model=model,
                n_ofdm_symbols=cfg.n_symbols,
                n_subcarriers=cfg.n_freq_subcarriers,
                grid=grid,
                device=device,
                n_antennas=1,
                speed=speed,
                symbol_sampling_frequency=grid.subcarrier_spacing,
                dtype=cfg.dtype,
                seed=cfg.seed + SIGMA_BUCKET_OFFSET + bucket_idx * SIGMA_BUCKET_STRIDE + k,
            )
            samples[k] = vec_from_h(h_tf).squeeze(-1)
        sigma, delta = ledoit_wolf_shrinkage(samples, reg=cfg.reg_empirical, device=device, dtype=cfg.dtype)
        sigma_by_bucket[(model, speed)] = sigma
        print(f"  Sigma bucket model={model} speed={speed:g}: n_cov_mc={cfg.n_cov_mc_per_bucket}, delta={delta:.3f}")
    elapsed = time.perf_counter() - t0
    print(f"Empirical Sigma buckets: {len(buckets)} buckets, elapsed={elapsed:.1f}s")
    return sigma_by_bucket


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
    if cfg.domain == "time_freq":
        model, speed = bucket_for_channel(cfg, channel_mc)
        h_tf = sample_tdl_ofdm_channel(
            model=model,
            n_ofdm_symbols=cfg.n_symbols,
            n_subcarriers=cfg.n_freq_subcarriers,
            grid=grid,
            device=device,
            n_antennas=1,
            speed=speed,
            symbol_sampling_frequency=grid.subcarrier_spacing,
            dtype=cfg.dtype,
            seed=cfg.seed + TDL_CHANNEL_OFFSET + channel_mc,
        )
        return vec_from_h(h_tf)
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


def _make_policy(
    name: str, cfg: CnnTrainConfig, device: torch.device, *, seed: int, channel_mc: int
) -> PilotPolicy:
    """'fixed' -> DmrsCombPolicy in time_freq domain (the generic 1D largest-gap bisection
    doesn't respect the 2D (k,t) grid, see pilot_policy.dmrs_comb); comb_offset varies per
    channel so 'fixed' rollouts aren't bit-identical across channels (DmrsCombPolicy has no
    seed of its own). Otherwise falls through to experiments.common.make_policy unchanged in
    both domains (random/active already proven domain-agnostic by exp12)."""
    if name == "fixed" and cfg.domain == "time_freq":
        comb_offset = channel_mc % cfg.dmrs_comb_spacing
        return DmrsCombPolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.k0,
            max_pilots=cfg.max_pilots,
            device=device,
            n_freq_subcarriers=cfg.n_freq_subcarriers,
            n_symbols=cfg.n_symbols,
            comb_spacing=cfg.dmrs_comb_spacing,
            pilot_symbols=cfg.dmrs_pilot_symbols,
            comb_offset=comb_offset,
        )
    run_cfg = cfg.to_run_config()
    return make_policy(name, run_cfg, device, seed=seed)


def collect_snapshots_one_channel(
    cfg: CnnTrainConfig,
    *,
    h_true: torch.Tensor,
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor],
    channel_mc: int,
    channel_idx: int,
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Incremental rollout; returns list of (X, y_label, loss_mask) per decision step."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = cfg.n_total
    grid_na, grid_nc = feature_grid_shape(cfg)
    zscore_mode = zscore_mode_for(cfg)
    bucket = bucket_for_channel(cfg, channel_mc) if cfg.domain == "time_freq" else ("A", 0.0)
    sigma = sigma_by_bucket[bucket]
    policy_name = rollout_policy_name(channel_idx)
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    noise_seed = cfg.seed + NOISE_OFFSET[policy_name] + channel_mc

    policy = _make_policy(policy_name, cfg, device, seed=noise_seed, channel_mc=channel_mc)
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
        mask_2d = mask_to_grid(mask, cfg)
        x_feat = build_features(
            state.h_hat, mask_2d, n_pilots, cfg.sigma2,
            na=grid_na, nc=grid_nc, n_candidates=nc, zscore_mode=zscore_mode,
        )
        if cfg.label_mode == "counterfactual":
            y_lab = counterfactual_labels(
                state, h_true, used, na, nc, cfg.sigma2, gen, label_eps=cfg.label_eps
            )
        else:
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
            mask_2d = mask.view(1, nc).expand(na, nc)
            x_feat = build_features(state.h_hat, mask_2d, n_pilots, cfg.sigma2, na=na, nc=nc)
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
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor],
    n_channels: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> Path:
    """Generate and save data/{split}.pt."""
    n_ch = n_channels if n_channels is not None else _n_channels_for_split(split, cfg)
    out = out_path or (DATA_DIR / f"{split}{label_suffix(cfg)}.pt")
    grid = make_grid(cfg)

    print(f"build {split}: {n_ch} channels, device={device}")
    all_rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    bucket_idx_rows: List[int] = []
    n_buckets = len(tdl_buckets(cfg)) if cfg.domain == "time_freq" else 1
    t0 = time.perf_counter()

    for ch in range(n_ch):
        mc = _channel_mc_for_split(split, ch, cfg)
        h_true = sample_tdl_channel(cfg, channel_mc=mc, device=device, grid=grid)
        rows = collect_snapshots_one_channel(
            cfg, h_true=h_true, sigma_by_bucket=sigma_by_bucket, channel_mc=mc, channel_idx=ch, device=device
        )
        all_rows.extend(rows)
        ch_bucket_idx = (mc % n_buckets) if cfg.domain == "time_freq" else 0
        bucket_idx_rows.extend([ch_bucket_idx] * len(rows))
        if (ch + 1) % GEN_LOG_INTERVAL == 0 or ch + 1 == n_ch:
            elapsed = time.perf_counter() - t0
            print(f"  {split} channel {ch + 1}/{n_ch}, {len(all_rows)} snapshots, {elapsed:.1f}s")

    x, y, m = stack_snapshots(all_rows)
    bucket_idx = torch.tensor(bucket_idx_rows, dtype=torch.int64)
    meta: Dict[str, Any] = {
        **{k: (v if not isinstance(v, torch.dtype) else str(v)) for k, v in asdict(cfg).items()},
        "split": split,
        "n_channels": n_ch,
        "n_snapshots": int(x.shape[0]),
        "feature_channels": NUM_FEATURE_CHANNELS,
    }
    if cfg.domain == "time_freq":
        meta["bucket_assignment"] = [
            list(bucket_for_channel(cfg, _channel_mc_for_split(split, ch, cfg))) for ch in range(n_ch)
        ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"X": x.cpu(), "y_label": y.cpu(), "loss_mask": m.cpu(), "bucket_idx": bucket_idx, "meta": meta}, out
    )
    print(f"Saved {out}: X={tuple(x.shape)}, y_label={tuple(y.shape)}, loss_mask={tuple(m.shape)}")
    return out


# ---------------------------------------------------------------------------
# 10. train_loop
# ---------------------------------------------------------------------------
def load_dataset(
    path: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    bucket_idx = blob.get("bucket_idx")
    if bucket_idx is None:
        # pre-bucket-weighted-loss cache (or antenna_snapshot, which has only one implicit
        # bucket): every sample gets bucket 0, so bucket-weighted loss reduces to the flat mean.
        bucket_idx = torch.zeros(blob["X"].shape[0], dtype=torch.int64)
    return blob["X"], blob["y_label"], blob["loss_mask"], bucket_idx, blob["meta"]


def run_epoch(
    model: CNNErrorEstimator,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    huber_delta: float,
    train: bool,
    domain: Literal["antenna_snapshot", "time_freq"] = "antenna_snapshot",
    bucket_weighted: bool = False,
    label_mode: Literal["current", "counterfactual"] = "current",
    loss_type: Literal["huber", "ranking_ce"] = "huber",
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    select_mode: Literal["argmax", "argmin"] = "argmin" if label_mode == "counterfactual" else "argmax"
    total_loss = 0.0
    total_top1 = 0.0
    n_batches = 0

    for x_b, y_b, _m_b, b_b in loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        unused_mask = unused_mask_from_features(x_b, domain=domain)
        bucket_idx = b_b.to(device) if bucket_weighted else None

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x_b)
        if loss_type == "ranking_ce":
            loss = masked_ranking_ce_loss(pred, y_b, unused_mask, select_mode=select_mode)
        else:
            loss = masked_huber_loss(pred, y_b, unused_mask, huber_delta, bucket_idx=bucket_idx)

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if not train:
            total_top1 += val_top1_unused(pred.detach(), y_b, unused_mask, select_mode=select_mode)
        n_batches += 1

    return total_loss / max(n_batches, 1), total_top1 / max(n_batches, 1)


def real_eval_cnn_score(
    cfg: CnnTrainConfig,
    model: "CNNErrorEstimator",
    *,
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor],
    grid: "SionnaOFDMGrid",
    device: torch.device,
    n_mc: int,
) -> Dict[str, Any]:
    """Cheap real-deployment proxy for checkpoint selection (see CnnTrainConfig.select_checkpoint_by):
    cnn-policy closed-loop average MSE at max_pilots on n_mc held-out channels per bucket --
    antenna_snapshot has one implicit bucket (sigma_by_bucket's single entry), time_freq has 9.
    Uses the same held-out channel-id range as eval_phase1b(_time_freq) (EVAL_CHANNEL_OFFSET),
    so these channels never overlap train/val."""
    buckets = list(sigma_by_bucket.keys())
    per_bucket_mse: Dict[str, float] = {}
    for bi, (model_name, speed) in enumerate(buckets):
        sigma = sigma_by_bucket[(model_name, speed)]
        mses: List[float] = []
        for mc in range(n_mc):
            channel_mc = EVAL_CHANNEL_OFFSET + bi * 1_000_000 + mc
            if cfg.domain == "time_freq":
                # sample_tdl_channel's time_freq branch would pick its OWN bucket from
                # bucket_for_channel(channel_mc % 9), ignoring model_name/speed here -- sample
                # this bucket's channel directly (mirrors eval_phase1b_time_freq) so the channel
                # actually matches the Sigma used above.
                h_tf = sample_tdl_ofdm_channel(
                    model=model_name, n_ofdm_symbols=cfg.n_symbols, n_subcarriers=cfg.n_freq_subcarriers,
                    grid=grid, device=device, n_antennas=1, speed=speed,
                    symbol_sampling_frequency=grid.subcarrier_spacing, dtype=cfg.dtype,
                    seed=cfg.seed + TDL_CHANNEL_OFFSET + channel_mc,
                )
                h_true = vec_from_h(h_tf)
            else:
                h_true = sample_tdl_channel(cfg, channel_mc=channel_mc, device=device, grid=grid)
            curve = run_closed_loop_mse_curve(
                cfg, h_true=h_true, sigma=sigma, device=device, channel_mc=channel_mc,
                policy_mode="cnn", cnn_model=model,
            )
            mses.append(curve[-1])
        key = f"{model_name}_{speed:g}" if cfg.domain == "time_freq" else "antenna_snapshot"
        per_bucket_mse[key] = float(sum(mses) / len(mses))
    mean_mse = float(sum(per_bucket_mse.values()) / len(per_bucket_mse))
    worst_bucket = max(per_bucket_mse, key=per_bucket_mse.get)
    return {
        "mean_mse": mean_mse,
        "worst_bucket": worst_bucket,
        "worst_mse": per_bucket_mse[worst_bucket],
        "per_bucket_mse": per_bucket_mse,
    }


def train_loop(
    cfg: CnnTrainConfig,
    *,
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
    metrics_path: Path,
) -> None:
    device = resolve_device(cfg.device)
    x_tr, y_tr, m_tr, b_tr, _ = load_dataset(train_path)
    x_va, y_va, m_va, b_va, _ = load_dataset(val_path)

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr, m_tr, b_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_va, y_va, m_va, b_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    arch_kwargs = model_arch_kwargs(cfg)
    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers, **arch_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_real_eval = cfg.select_checkpoint_by == "real_eval"
    sigma_by_bucket: Optional[Dict[Tuple[str, float], torch.Tensor]] = None
    grid = None
    if use_real_eval:
        grid = make_grid(cfg)
        sigma_by_bucket = load_sigma_buckets(cfg, device)

    best_select_metric = float("inf")
    best_epoch = -1
    best_top1 = 0.0
    epochs_no_improve = 0
    metrics: List[Dict[str, Any]] = []

    if cfg.loss_type == "ranking_ce":
        loss_desc = "ranking cross-entropy (argmin/argmax over unused SCs)"
    elif cfg.bucket_weighted_loss:
        loss_desc = "bucket-weighted Huber (per-bucket mean, then mean over buckets)"
    else:
        loss_desc = "unused-SC Huber (per-sample mean)"
    select_desc = (
        f"real_eval({cfg.real_eval_criterion}, n_mc={cfg.real_eval_n_mc})" if use_real_eval else "val_huber"
    )
    print(
        f"train: device={device}, train={len(x_tr)} val={len(x_va)} "
        f"batch={cfg.batch_size}, max_epochs={cfg.max_epochs}, label_mode={cfg.label_mode}, "
        f"domain={cfg.domain}, loss={loss_desc}, select_checkpoint_by={select_desc}"
    )

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.perf_counter()
        train_huber, _ = run_epoch(
            model, train_loader, device=device, optimizer=opt, huber_delta=cfg.huber_delta,
            train=True, domain=cfg.domain, bucket_weighted=cfg.bucket_weighted_loss,
            label_mode=cfg.label_mode, loss_type=cfg.loss_type,
        )
        val_huber, val_top1 = run_epoch(
            model, val_loader, device=device, optimizer=None, huber_delta=cfg.huber_delta,
            train=False, domain=cfg.domain, bucket_weighted=cfg.bucket_weighted_loss,
            label_mode=cfg.label_mode, loss_type=cfg.loss_type,
        )

        real_eval_result: Optional[Dict[str, Any]] = None
        if use_real_eval:
            real_eval_result = real_eval_cnn_score(
                cfg, model, sigma_by_bucket=sigma_by_bucket, grid=grid, device=device, n_mc=cfg.real_eval_n_mc,
            )
            select_metric = (
                real_eval_result["worst_mse"] if cfg.real_eval_criterion == "worst_bucket"
                else real_eval_result["mean_mse"]
            )
        else:
            select_metric = val_huber

        elapsed = time.perf_counter() - t0
        star = ""
        if select_metric < best_select_metric:
            best_select_metric = select_metric
            best_epoch = epoch
            best_top1 = val_top1
            epochs_no_improve = 0
            star = " *"
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_epoch": best_epoch,
                    "best_select_metric": best_select_metric,
                    "best_val_huber": val_huber,
                    "best_val_top1": best_top1,
                    "arch": {
                        "class": "CNNErrorEstimator",
                        "n_subcarriers": cfg.n_subcarriers,
                        "pool_mode": arch_kwargs["pool_mode"],
                        "kernel_hw": list(arch_kwargs["kernel_hw"]),
                        "use_global_context": arch_kwargs["use_global_context"],
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        record = {"epoch": epoch, "train_huber": train_huber, "val_huber": val_huber, "val_top1": val_top1}
        log_line = (
            f"epoch {epoch:03d}/{cfg.max_epochs:03d}  train_huber={train_huber:.4f}  "
            f"val_huber={val_huber:.4f}  val_top1={val_top1:.3f}"
        )
        if real_eval_result is not None:
            record["real_eval_mean_mse"] = real_eval_result["mean_mse"]
            record["real_eval_worst_bucket"] = real_eval_result["worst_bucket"]
            record["real_eval_worst_mse"] = real_eval_result["worst_mse"]
            log_line += (
                f"  real_eval_mean={real_eval_result['mean_mse']:.4e}  "
                f"real_eval_worst={real_eval_result['worst_bucket']}({real_eval_result['worst_mse']:.4e})"
            )
        metrics.append(record)
        print(f"{log_line}  {elapsed:.1f}s{star}")

        if cfg.early_stop_patience is not None and epochs_no_improve >= cfg.early_stop_patience:
            print(f"early stopping at epoch {epoch} (no val improvement for {cfg.early_stop_patience} epochs)")
            break

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metric_str = f"{best_select_metric:.4e}" if use_real_eval else f"{best_select_metric:.4f}"
    print(
        f"train done: best_{select_desc}={metric_str} @ epoch {best_epoch}  "
        f"best_val_top1={best_top1:.3f}  checkpoint={checkpoint_path}"
    )


# ---------------------------------------------------------------------------
# 11. eval_phase1b
# ---------------------------------------------------------------------------
def load_cnn_checkpoint(path: Path, device: torch.device) -> CNNErrorEstimator:
    blob = torch.load(path, map_location=device, weights_only=False)
    arch = blob.get("arch", {})
    nc = int(arch.get("n_subcarriers", 32))
    pool_mode = arch.get("pool_mode", "mean")
    kernel_hw = tuple(arch.get("kernel_hw", (3, 5)))
    use_global_context = arch.get("use_global_context", False)
    model = CNNErrorEstimator(
        n_subcarriers=nc, pool_mode=pool_mode, kernel_hw=kernel_hw, use_global_context=use_global_context
    ).to(device)
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
    """Greedy pilot selection from predicted per-subcarrier scores on unused subcarriers.

    select_mode="argmax": pilot the subcarrier with the highest predicted score (current-error
    label paradigm: pilot the worst-currently-estimated SC).
    select_mode="argmin": pilot the subcarrier with the lowest predicted score (counterfactual
    label paradigm: pilot wherever placing a pilot is predicted to leave the lowest resulting NMSE).
    """

    def __init__(
        self,
        model: CNNErrorEstimator,
        *,
        cfg: CnnTrainConfig,
        device: torch.device,
        select_mode: Literal["argmax", "argmin"] = "argmax",
    ) -> None:
        self._model = model
        self._cfg = cfg
        self._na = cfg.n_antennas
        self._nc = cfg.n_subcarriers
        self._sigma2 = cfg.sigma2
        self._device = device
        self._select_mode = select_mode
        self._grid_na, self._grid_nc = feature_grid_shape(cfg)
        self._zscore_mode = zscore_mode_for(cfg)

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
        mask_2d = mask_to_grid(mask, self._cfg)
        x = build_features(
            state.h_hat, mask_2d, n_pilots, self._sigma2,
            na=self._grid_na, nc=self._grid_nc, n_candidates=self._nc, zscore_mode=self._zscore_mode,
        )
        with torch.no_grad():
            pred = self._model(x.unsqueeze(0)).squeeze(0)
        scores = pred.clone()
        if self._select_mode == "argmin":
            for k in used:
                scores[k] = float("inf")
            return int(scores.argmin().item())
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
    """Return average MSE after each pilot count (length max_pilots) via run_until_threshold.
    sigma is the single Sigma to use for this channel -- the caller resolves which bucket's
    Sigma that is (eval_phase1b: the one antenna-domain Sigma; eval_phase1b_time_freq: that
    channel's own (model,speed) bucket)."""
    nc = cfg.n_subcarriers
    run_cfg = cfg.to_eval_run_config()
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    k_init = random_first_subcarrier(channel_seed, nc)

    if policy_mode == "cnn":
        if cnn_model is None:
            raise ValueError("cnn policy requires model")
        noise_seed = cfg.seed + EVAL_CNN_NOISE_OFFSET + channel_mc
        select_mode = "argmin" if cfg.label_mode == "counterfactual" else "argmax"
        inner: PilotPolicy = CnnPilotPolicy(cnn_model, cfg=cfg, device=device, select_mode=select_mode)
    elif policy_mode == "fixed":
        noise_seed = cfg.seed + NOISE_OFFSET["fixed"] + channel_mc
        inner = _make_policy("fixed", cfg, device, seed=noise_seed, channel_mc=channel_mc)
    else:
        noise_seed = cfg.seed + NOISE_OFFSET["active"] + channel_mc
        inner = _make_policy("active", cfg, device, seed=noise_seed, channel_mc=channel_mc)

    policy = _FixedInitPilotPolicy(k_init, inner)
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    trace = run_until_threshold(h_true, sigma, run_cfg, policy, device=device, generator=gen)
    return trace.nmse_true


def eval_phase1b(cfg: CnnTrainConfig, *, checkpoint_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    device = resolve_device(cfg.device)
    model = load_cnn_checkpoint(checkpoint_path, device)
    grid = make_grid(cfg)
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
    suffix = label_suffix(cfg)
    fig_path = FIGURES_DIR / f"phase1b_average_mse_vs_pilots{suffix}.png"
    json_path = FIGURES_DIR / f"phase1b_average_mse_vs_pilots{suffix}.json"

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


def eval_phase1b_time_freq(cfg: CnnTrainConfig, *, checkpoint_path: Path) -> None:
    """Phase 1b eval for domain='time_freq': a 3x3 (tdl_models x speeds) grid of panels, each
    with its own cnn/fixed/active/full-LMMSE curves -- a single pooled curve (as in
    eval_phase1b) would hide whether the CNN generalizes evenly across buckets or just wins on
    whichever it saw most."""
    import matplotlib.pyplot as plt
    import numpy as np

    device = resolve_device(cfg.device)
    model = load_cnn_checkpoint(checkpoint_path, device)
    grid = make_grid(cfg)
    sigma_by_bucket = load_sigma_buckets(cfg, device)

    suffix = label_suffix(cfg)
    pilots = np.arange(1, cfg.max_pilots + 1)
    n_models, n_speeds = len(cfg.tdl_models), len(cfg.speeds)
    fig, axes = plt.subplots(
        n_models, n_speeds, figsize=(5.0 * n_speeds, 4.0 * n_models), squeeze=False
    )
    payload: Dict[str, Any] = {"n_mc": cfg.eval_n_mc, "checkpoint": str(checkpoint_path), "pilots": pilots.tolist(), "buckets": {}}

    print(f"eval phase1b (time_freq): n_mc={cfg.eval_n_mc}/bucket, {n_models}x{n_speeds} buckets, checkpoint={checkpoint_path}")
    for bi, model_name in enumerate(cfg.tdl_models):
        for si, speed in enumerate(cfg.speeds):
            sigma = sigma_by_bucket[(model_name, speed)]
            curves: Dict[str, List[List[float]]] = {"cnn": [], "fixed": [], "active": []}
            h_list: List[torch.Tensor] = []

            for mc in range(cfg.eval_n_mc):
                channel_mc = EVAL_CHANNEL_OFFSET + (bi * n_speeds + si) * 1_000_000 + mc
                h_tf = sample_tdl_ofdm_channel(
                    model=model_name,
                    n_ofdm_symbols=cfg.n_symbols,
                    n_subcarriers=cfg.n_freq_subcarriers,
                    grid=grid,
                    device=device,
                    n_antennas=1,
                    speed=speed,
                    symbol_sampling_frequency=grid.subcarrier_spacing,
                    dtype=cfg.dtype,
                    seed=cfg.seed + TDL_CHANNEL_OFFSET + channel_mc,
                )
                h_true = vec_from_h(h_tf)
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

            ax = axes[bi][si]
            for name, curve in mean_curves.items():
                ax.semilogy(pilots, curve, linewidth=1.4, label=name)
            ax.axhline(full_mse, color="red", linestyle=":", linewidth=1.0, label="full LMMSE")
            ax.set_title(f"TDL-{model_name}, speed={speed:g} m/s")
            ax.set_xlabel("n_pilots")
            ax.set_ylabel("Mean average MSE")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            if bi == 0 and si == 0:
                ax.legend(fontsize=8)

            payload["buckets"][f"{model_name}_{speed:g}"] = {
                "curves": {k: v.tolist() for k, v in mean_curves.items()},
                "full_lmmse_mse": full_mse,
            }
            print(
                f"  TDL-{model_name} speed={speed:g}: "
                + ", ".join(f"{k}={v[-1]:.4e}" for k, v in mean_curves.items())
                + f", full_lmmse={full_mse:.4e}"
            )

    fig.suptitle(f"Phase 1b (time-freq): average MSE vs pilots, n_mc={cfg.eval_n_mc}/bucket")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"phase1b_average_mse_vs_pilots{suffix}.png"
    json_path = FIGURES_DIR / f"phase1b_average_mse_vs_pilots{suffix}.json"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {fig_path}")


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

    x_tr, y_tr, m_tr, b_tr, _ = load_dataset(DATA_DIR / "phase2_train.pt")
    x_va, y_va, m_va, b_va, _ = load_dataset(DATA_DIR / "phase2_val.pt")

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr, m_tr, b_tr), batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(x_va, y_va, m_va, b_va), batch_size=cfg.batch_size, shuffle=False, num_workers=0
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
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = make_grid(cfg)
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for ch in range(batch_size):
        h_true = sample_tdl_channel(cfg, channel_mc=ch, device=device, grid=grid)
        snap = collect_snapshots_one_channel(
            cfg, h_true=h_true, sigma_by_bucket=sigma_by_bucket, channel_mc=ch, channel_idx=ch, device=device
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
    _check_domain_config(cfg)
    device = resolve_device(cfg.device)
    sanity_cfg = replace(
        cfg, n_cov_mc=min(cfg.n_cov_mc, 64), n_cov_mc_per_bucket=min(cfg.n_cov_mc_per_bucket, 64)
    )

    print(f"sanity: device={device}, domain={cfg.domain}, batch_size={batch_size}, epochs={n_epochs}")
    sigma_by_bucket = load_sigma_buckets(sanity_cfg, device)

    print(f"Building one minibatch from {cfg.domain} rollouts...")
    x, y, _loss_mask = build_sanity_minibatch(cfg, batch_size=batch_size, sigma_by_bucket=sigma_by_bucket, device=device)
    x = x.to(device)
    y = y.to(device)
    unused_mask = unused_mask_from_features(x, domain=cfg.domain)

    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers, **model_arch_kwargs(cfg)).to(device)
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
    parser.add_argument("--n-cov-mc", type=int, default=None, help="Override empirical Sigma MC sample count (default 512).")
    parser.add_argument("--force-regen", action="store_true")
    parser.add_argument(
        "--label-mode",
        choices=("current", "counterfactual"),
        default=None,
        help="current = per-SC current-error labels (existing baseline); "
        "counterfactual = look-ahead label from simulating a pilot at each unused SC.",
    )
    parser.add_argument("--train-channels", type=int, default=None)
    parser.add_argument("--val-channels", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--huber-delta", type=float, default=None)
    parser.add_argument("--eval-n-mc", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument(
        "--domain",
        choices=("antenna_snapshot", "time_freq"),
        default=None,
        help="antenna_snapshot (default) = today's static Na x Nc TDL-A snapshot; "
        "time_freq = single-antenna joint (OFDM symbol x subcarrier) grid swept over "
        "--tdl-models x --speeds buckets.",
    )
    parser.add_argument("--tdl-models", nargs="+", default=None, help="time_freq only, e.g. --tdl-models A B C")
    parser.add_argument("--speeds", nargs="+", type=float, default=None, help="time_freq only, e.g. --speeds 3 30 120")
    parser.add_argument("--n-cov-mc-per-bucket", type=int, default=None, help="time_freq only (default 256).")
    parser.add_argument("--n-freq-subcarriers", type=int, default=None, help="time_freq only (default 32).")
    parser.add_argument("--n-symbols", type=int, default=None, help="time_freq only (default 16).")
    parser.add_argument(
        "--bucket-weighted-loss",
        action="store_true",
        default=False,
        help="time_freq only (no-op under antenna_snapshot): average masked Huber per "
        "(model,speed) bucket, then mean those bucket-means, instead of one flat per-sample "
        "mean -- keeps the highest-speed bucket's larger loss scale from dominating the gradient.",
    )
    parser.add_argument(
        "--use-global-context",
        action="store_true",
        default=False,
        help="Add a squeeze-excite/FiLM global-context branch to CNNErrorEstimator so per-cell "
        "predictions can be conditioned on the sample's overall regime (see class docstring).",
    )
    parser.add_argument(
        "--select-checkpoint-by",
        choices=("val_huber", "real_eval"),
        default=None,
        help="Checkpoint-selection criterion (default val_huber). real_eval selects on an actual "
        "cnn-policy closed-loop MSE proxy instead -- see CnnTrainConfig.select_checkpoint_by.",
    )
    parser.add_argument("--real-eval-n-mc", type=int, default=None, help="Held-out channels per bucket for --select-checkpoint-by=real_eval (default 30).")
    parser.add_argument(
        "--kernel-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Override CNNErrorEstimator's per-domain default conv kernel (H,W), e.g. --kernel-hw 7 3 "
        "to widen the time_freq symbol-axis receptive field.",
    )
    parser.add_argument(
        "--loss-type",
        choices=("huber", "ranking_ce"),
        default=None,
        help="Training loss (default huber). ranking_ce: cross-entropy over unused candidates "
        "against the single true best one, directly optimizing the argmin/argmax deployment "
        "decision instead of regressing the absolute label value -- see masked_ranking_ce_loss.",
    )
    parser.add_argument(
        "--real-eval-criterion",
        choices=("mean", "worst_bucket"),
        default=None,
        help="real_eval reduction across buckets: mean (default) or worst_bucket.",
    )
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
    if args.n_cov_mc is not None:
        cfg.n_cov_mc = args.n_cov_mc
    if args.label_mode is not None:
        cfg.label_mode = args.label_mode
    if args.train_channels is not None:
        cfg.train_channels = args.train_channels
    if args.val_channels is not None:
        cfg.val_channels = args.val_channels
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.huber_delta is not None:
        cfg.huber_delta = args.huber_delta
    if args.eval_n_mc is not None:
        cfg.eval_n_mc = args.eval_n_mc
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.bucket_weighted_loss:
        cfg.bucket_weighted_loss = True
    if args.use_global_context:
        cfg.use_global_context = True
    if args.select_checkpoint_by is not None:
        cfg.select_checkpoint_by = args.select_checkpoint_by
    if args.real_eval_n_mc is not None:
        cfg.real_eval_n_mc = args.real_eval_n_mc
    if args.real_eval_criterion is not None:
        cfg.real_eval_criterion = args.real_eval_criterion
    if args.kernel_hw is not None:
        cfg.kernel_hw = tuple(args.kernel_hw)
    if args.loss_type is not None:
        cfg.loss_type = args.loss_type
    if args.tdl_models is not None:
        cfg.tdl_models = tuple(args.tdl_models)
    if args.speeds is not None:
        cfg.speeds = tuple(args.speeds)
    if args.n_cov_mc_per_bucket is not None:
        cfg.n_cov_mc_per_bucket = args.n_cov_mc_per_bucket
    if args.n_freq_subcarriers is not None:
        cfg.n_freq_subcarriers = args.n_freq_subcarriers
    if args.n_symbols is not None:
        cfg.n_symbols = args.n_symbols
    if args.domain is not None:
        cfg.domain = args.domain
    if cfg.domain == "time_freq":
        cfg.n_antennas = 1
        cfg.n_subcarriers = cfg.n_symbols * cfg.n_freq_subcarriers
        if args.label_mode is None:
            # current mode didn't clearly beat fixed/active in the antenna domain; no reason
            # to carry that weaker default into a new, pricier run unless explicitly requested.
            cfg.label_mode = "counterfactual"
    _check_domain_config(cfg)

    if args.phase == "sanity":
        sanity_check(cfg)
        return

    device = resolve_device(cfg.device)
    suffix = label_suffix(cfg)

    if args.phase == "build":
        train_pt = DATA_DIR / f"train{suffix}.pt"
        val_pt = DATA_DIR / f"val{suffix}.pt"
        if not args.force_regen and train_pt.exists() and val_pt.exists():
            print(f"cache hit: {train_pt}, {val_pt} (use --force-regen to rebuild)")
            return
        sigma_by_bucket = load_sigma_buckets(cfg, device)
        n_override = args.n_channels
        build_dataset_cache(cfg, split="train", device=device, sigma_by_bucket=sigma_by_bucket, n_channels=n_override)
        build_dataset_cache(cfg, split="val", device=device, sigma_by_bucket=sigma_by_bucket, n_channels=n_override)
        return

    if args.phase == "train":
        train_loop(
            cfg,
            train_path=DATA_DIR / f"train{suffix}.pt",
            val_path=DATA_DIR / f"val{suffix}.pt",
            checkpoint_path=CHECKPOINT_DIR / f"phase1{suffix}_best.pt",
            metrics_path=CHECKPOINT_DIR / f"phase1{suffix}_metrics.json",
        )
        return

    if args.phase == "eval":
        if cfg.domain == "time_freq":
            eval_phase1b_time_freq(cfg, checkpoint_path=CHECKPOINT_DIR / f"phase1{suffix}_best.pt")
        else:
            eval_phase1b(cfg, checkpoint_path=CHECKPOINT_DIR / f"phase1{suffix}_best.pt")
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
