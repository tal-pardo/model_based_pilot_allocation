"""
Model A: CNN pilot scorer training (TDL-A channels).

Phase 0: sanity_check() — overfit one minibatch, verify loss decreases and gradients flow.
Phase 1: full dataset cache + train() on CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data_generator import complex_standard_normal, empirical_covariance, pilot_matrix_from_indices
from estimators import RecursiveLMMSEState, recursive_lmmse_init, recursive_lmmse_step
from pilots import (
    FixedPilotSampler,
    PilotScheduleConfig,
    active_subcarrier_score_J,
    num_timesteps_from_pilot_growth,
    ordered_subcarriers_from_vec,
    vec_indices_for_subcarriers,
)
from sionna_channels import SionnaOFDMGrid, sample_tdl_ofdm_channel, vec_from_H

RolloutPolicy = Literal["random", "active"]
SplitName = Literal["train", "val"]
LABEL_EPS = 1e-8
NUM_FEATURE_CHANNELS = 7  # |H_hat|, Re, Im, mask, SNR, t/T, pilot_frac (no innovation)

TRAIN_CHANNELS = 12_000
VAL_CHANNELS = 2_000
VAL_SEED_OFFSET = 500_000
DATA_DIR = Path("data/cnn_pilot_scorer")
CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "model_a_phase1_best.pt"
METRICS_PATH = CHECKPOINT_DIR / "model_a_phase1_metrics.json"
SWEEP_DIR = CHECKPOINT_DIR / "sweep"
SWEEP_RESULTS_CSV = SWEEP_DIR / "results.csv"
SWEEP_WINNERS_YAML = SWEEP_DIR / "winners.yaml"
DEFAULT_MODEL_WIDTH = 64
DEFAULT_MODEL_DEPTH = 3
GEN_LOG_INTERVAL = 500


@dataclass
class TrainConfig:
    n_antennas: int = 16
    n_subcarriers: int = 32
    rho_space: float = 0.7
    sigma2: float = 1e-2
    initial_pilot_subcarriers: int = 2
    final_pilot_subcarriers: int = 8
    pilots_added_per_step: int = 1
    n_cov_mc: int = 300
    seed: int = 0
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    huber_delta: float = 1.0
    label_eps: float = LABEL_EPS


@dataclass
class Phase1TrainConfig:
    train_channels: int = TRAIN_CHANNELS
    val_channels: int = VAL_CHANNELS
    batch_size: int = 128
    max_epochs: int = 40
    min_epochs: int = 10
    early_stop_patience: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    width: int = DEFAULT_MODEL_WIDTH
    depth: int = DEFAULT_MODEL_DEPTH
    max_train_label_sc: Optional[int] = None


@dataclass
class SweepTrainConfig(Phase1TrainConfig):
    """Faster early-stop defaults for hyperparameter sweep runs."""

    max_epochs: int = 35
    min_epochs: int = 5
    early_stop_patience: int = 5


@dataclass
class ModelArchConfig:
    width: int = DEFAULT_MODEL_WIDTH
    depth: int = DEFAULT_MODEL_DEPTH
    n_feature_channels: int = NUM_FEATURE_CHANNELS

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelArchConfig":
        return cls(
            width=int(d.get("width", DEFAULT_MODEL_WIDTH)),
            depth=int(d.get("depth", DEFAULT_MODEL_DEPTH)),
            n_feature_channels=int(d.get("n_feature_channels", NUM_FEATURE_CHANNELS)),
        )


@dataclass
class SweepRunSpec:
    run_id: str
    stage: str
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    width: int = DEFAULT_MODEL_WIDTH
    depth: int = DEFAULT_MODEL_DEPTH
    huber_delta: float = 1.0
    max_train_label_sc: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SweepRunSpec":
        max_sc = d.get("max_train_label_sc")
        return cls(
            run_id=str(d["run_id"]),
            stage=str(d.get("stage", "?")),
            lr=float(d.get("lr", 1e-3)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            batch_size=int(d.get("batch_size", 128)),
            width=int(d.get("width", DEFAULT_MODEL_WIDTH)),
            depth=int(d.get("depth", DEFAULT_MODEL_DEPTH)),
            huber_delta=float(d.get("huber_delta", 1.0)),
            max_train_label_sc=None if max_sc is None else int(max_sc),
        )

    def to_phase1(self) -> SweepTrainConfig:
        return SweepTrainConfig(
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            width=self.width,
            depth=self.depth,
            max_train_label_sc=self.max_train_label_sc,
        )

    def to_train_cfg(self, device: str) -> TrainConfig:
        return TrainConfig(device=device, huber_delta=self.huber_delta)


def resolve_device(device_str: str) -> torch.device:
    raw = device_str.lower().strip()
    if raw == "gpu":
        raw = "cuda"
    if raw == "cuda" or raw.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(raw if raw.startswith("cuda:") else "cuda:0")
    if raw != "cpu":
        raise ValueError(f"device must be 'cpu' or 'cuda'; got {device_str!r}.")
    return torch.device("cpu")


def h_to_H(h: torch.Tensor, n_antennas: int, n_subcarriers: int) -> torch.Tensor:
    n = n_antennas * n_subcarriers
    if h.shape != (n, 1):
        raise ValueError(f"Expected h shape ({n}, 1), got {tuple(h.shape)}.")
    return h.view(n_subcarriers, n_antennas).T.contiguous()


def sample_tdl_a_channel(
    cfg: TrainConfig,
    *,
    grid: SionnaOFDMGrid,
    device: torch.device,
    seed: Optional[int],
) -> torch.Tensor:
    h_mat = sample_tdl_ofdm_channel(
        model="A",
        n_ofdm_symbols=1,
        n_subcarriers=cfg.n_subcarriers,
        grid=grid,
        device=device,
        n_antennas=cfg.n_antennas,
        rho_space=cfg.rho_space,
        dtype=cfg.dtype,
        seed=seed,
    )
    return vec_from_H(h_mat)


def estimate_sigma_hat_tdl_a(cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    n = cfg.n_antennas * cfg.n_subcarriers
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    samples = torch.zeros((cfg.n_cov_mc, n), device=device, dtype=cfg.dtype)
    for k in range(cfg.n_cov_mc):
        samples[k] = sample_tdl_a_channel(
            cfg, grid=grid, device=device, seed=cfg.seed + 1_000_000 + k
        ).squeeze(-1)
    return empirical_covariance(samples, device=device, dtype=cfg.dtype)


def empirical_mse(h_hat: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    err = h_hat - h_true
    return (err.abs().pow(2).mean()).real


def zscore_across_subcarriers(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = x.mean()
    std = x.std(unbiased=False)
    return (x - mu) / (std + eps)


def build_features(
    cfg: TrainConfig,
    *,
    h_hat: torch.Tensor,
    mask: torch.Tensor,
    decision_step: int,
    n_decision_steps: int,
) -> torch.Tensor:
    """Return X of shape (7, Nc) float32 — no innovation/residual channel."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    h_mat = h_to_H(h_hat, na, nc)
    mag = h_mat.abs().mean(dim=0)
    re = h_mat.real.mean(dim=0)
    im = h_mat.imag.mean(dim=0)
    mag = zscore_across_subcarriers(mag)
    re = zscore_across_subcarriers(re)
    im = zscore_across_subcarriers(im)

    snr = math.log10(1.0 / cfg.sigma2)
    t_norm = float(decision_step) / float(max(n_decision_steps, 1))
    pilot_frac = mask.float().mean().item()

    snr_t = torch.full((nc,), snr, dtype=torch.float32, device=mag.device)
    t_t = torch.full((nc,), t_norm, dtype=torch.float32, device=mag.device)
    pf_t = torch.full((nc,), pilot_frac, dtype=torch.float32, device=mag.device)

    x = torch.stack(
        [
            mag.float(),
            re.float(),
            im.float(),
            mask.float(),
            snr_t,
            t_t,
            pf_t,
        ],
        dim=0,
    )
    return x


def counterfactual_labels(
    state: RecursiveLMMSEState,
    h_true: torch.Tensor,
    mask: torch.Tensor,
    cfg: TrainConfig,
    *,
    n: int,
    device: torch.device,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    y_label: (Nc,) log(e+eps), loss_mask: (Nc,) 1 on unused subcarriers.
    """
    nc = cfg.n_subcarriers
    na = cfg.n_antennas
    y_label = torch.zeros(nc, device=device, dtype=torch.float32)
    loss_mask = torch.zeros(nc, device=device, dtype=torch.float32)

    unused = [k for k in range(nc) if mask[k].item() < 0.5]
    for k in unused:
        idx_k = vec_indices_for_subcarriers(
            torch.tensor([k], device=device, dtype=torch.long), na, device
        )
        x_k = pilot_matrix_from_indices(n, idx_k, device=device, dtype=cfg.dtype)
        noise = (cfg.sigma2**0.5) * complex_standard_normal(
            na, 1, device=device, dtype=cfg.dtype, generator=generator
        )
        y_k = x_k @ h_true + noise
        st = RecursiveLMMSEState(h=state.h.clone(), P=state.P.clone())
        st_new = recursive_lmmse_step(st, x_k, y_k, cfg.sigma2)
        e = empirical_mse(st_new.h, h_true)
        y_label[k] = torch.log(e + cfg.label_eps)
        loss_mask[k] = 1.0

    return y_label, loss_mask


def pick_rollout_subcarriers(
    policy: RolloutPolicy,
    used_sc: List[int],
    state: RecursiveLMMSEState,
    cfg: TrainConfig,
    *,
    n_new: int,
) -> List[int]:
    nc = cfg.n_subcarriers
    na = cfg.n_antennas
    unused = [k for k in range(nc) if k not in set(used_sc)]
    if n_new <= 0 or not unused:
        return []
    if policy == "random":
        chosen = random.sample(unused, min(n_new, len(unused)))
        return chosen
    scores = [(active_subcarrier_score_J(state.P, k, na, cfg.sigma2), k) for k in unused]
    scores.sort(key=lambda x: -x[0])
    return [scores[i][1] for i in range(min(n_new, len(scores)))]


def measure_and_update(
    state: RecursiveLMMSEState,
    h_true: torch.Tensor,
    cumulative_idx: torch.Tensor,
    cfg: TrainConfig,
    *,
    n: int,
    device: torch.device,
    generator: torch.Generator,
) -> RecursiveLMMSEState:
    x_t = pilot_matrix_from_indices(n, cumulative_idx, device=device, dtype=cfg.dtype)
    noise = (cfg.sigma2**0.5) * complex_standard_normal(
        cumulative_idx.numel(), 1, device=device, dtype=cfg.dtype, generator=generator
    )
    y_t = x_t @ h_true + noise
    return recursive_lmmse_step(state, x_t, y_t, cfg.sigma2)


def collect_snapshots_from_channel(
    cfg: TrainConfig,
    *,
    sigma: torch.Tensor,
    h_true: torch.Tensor,
    device: torch.device,
    grid: SionnaOFDMGrid,
    policy: RolloutPolicy,
    channel_seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """List of (X, y_label, loss_mask) per decision step."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    sched = PilotScheduleConfig(
        n_subcarriers=nc,
        n_antennas=na,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        final_pilot_subcarriers=cfg.final_pilot_subcarriers,
        pilots_added_per_step=cfg.pilots_added_per_step,
        cumulative_pilots=True,
    )
    t_add = num_timesteps_from_pilot_growth(
        cfg.initial_pilot_subcarriers,
        cfg.final_pilot_subcarriers,
        cfg.pilots_added_per_step,
    )
    fixed = FixedPilotSampler(sched, T=t_add + 1, device=device)
    gen = torch.Generator(device=device).manual_seed(channel_seed)

    idx0 = fixed.vec_indices_at_step(0)
    x0 = pilot_matrix_from_indices(n, idx0, device=device, dtype=cfg.dtype)
    noise0 = (cfg.sigma2**0.5) * complex_standard_normal(
        idx0.numel(), 1, device=device, dtype=cfg.dtype, generator=gen
    )
    y0 = x0 @ h_true + noise0
    state = recursive_lmmse_init(sigma, x0, y0, cfg.sigma2)

    used_sc = ordered_subcarriers_from_vec(idx0, na)
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    for k in used_sc:
        mask[k] = 1.0

    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for step in range(1, t_add + 1):
        x_feat = build_features(
            cfg,
            h_hat=state.h,
            mask=mask,
            decision_step=step,
            n_decision_steps=t_add,
        )
        y_lab, lmask = counterfactual_labels(
            state, h_true, mask, cfg, n=n, device=device, generator=gen
        )
        rows.append((x_feat, y_lab, lmask))

        k_prev = min(
            cfg.final_pilot_subcarriers,
            cfg.initial_pilot_subcarriers + (step - 1) * cfg.pilots_added_per_step,
        )
        k_cur = min(
            cfg.final_pilot_subcarriers,
            cfg.initial_pilot_subcarriers + step * cfg.pilots_added_per_step,
        )
        n_new = k_cur - k_prev
        new_sc = pick_rollout_subcarriers(policy, used_sc, state, cfg, n_new=n_new)
        used_sc = used_sc + new_sc
        for k in new_sc:
            mask[k] = 1.0

        idx_t = vec_indices_for_subcarriers(
            torch.tensor(used_sc, device=device, dtype=torch.long), na, device
        )
        state = measure_and_update(
            state, h_true, idx_t, cfg, n=n, device=device, generator=gen
        )

    return rows


def build_minibatch(
    cfg: TrainConfig,
    *,
    batch_size: int,
    sigma: torch.Tensor,
    device: torch.device,
    grid: SionnaOFDMGrid,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ms: List[torch.Tensor] = []
    channel_seed = seed
    while len(xs) < batch_size:
        policy: RolloutPolicy = "random" if (channel_seed % 2 == 0) else "active"
        h_true = sample_tdl_a_channel(cfg, grid=grid, device=device, seed=channel_seed)
        snapshots = collect_snapshots_from_channel(
            cfg,
            sigma=sigma,
            h_true=h_true,
            device=device,
            grid=grid,
            policy=policy,
            channel_seed=channel_seed + 10_000,
        )
        for row in snapshots:
            xs.append(row[0])
            ys.append(row[1])
            ms.append(row[2])
            if len(xs) >= batch_size:
                break
        channel_seed += 1

    x = torch.stack(xs[:batch_size], dim=0)
    y = torch.stack(ys[:batch_size], dim=0)
    m = torch.stack(ms[:batch_size], dim=0)
    return x, y, m


def group_norm_groups(width: int) -> int:
    """Pick num_groups for GroupNorm so channels divide evenly."""
    for g in (8, 4, 2, 1):
        if width % g == 0:
            return g
    return 1


def _conv_gn_gelu(
    in_ch: int, out_ch: int, kernel_size: int, *, num_groups: int
) -> nn.Sequential:
    pad = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, bias=False),
        nn.GroupNorm(num_groups, out_ch),
        nn.GELU(),
    )


class PilotScorerModelA(nn.Module):
    """1D CNN over subcarriers: (B, C_feat, Nc) -> (B, Nc)."""

    def __init__(
        self,
        n_subcarriers: int = 32,
        n_feature_channels: int = NUM_FEATURE_CHANNELS,
        width: int = DEFAULT_MODEL_WIDTH,
        depth: int = DEFAULT_MODEL_DEPTH,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"depth must be >= 2, got {depth}")
        self.n_subcarriers = n_subcarriers
        self.n_feature_channels = n_feature_channels
        self.width = width
        self.depth = depth
        gn_groups = group_norm_groups(width)

        kernel_sizes = [5, 5] + [3] * max(0, depth - 2)
        layers: List[nn.Module] = []
        in_ch = n_feature_channels
        for k in kernel_sizes:
            layers.append(_conv_gn_gelu(in_ch, width, k, num_groups=gn_groups))
            in_ch = width
        layers.extend(
            [
                nn.Conv1d(width, 32, kernel_size=1),
                nn.GELU(),
                nn.Conv1d(32, 1, kernel_size=1),
            ]
        )
        self.net = nn.Sequential(*layers)

    def arch_config(self) -> ModelArchConfig:
        return ModelArchConfig(
            width=self.width,
            depth=self.depth,
            n_feature_channels=self.n_feature_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_feature_channels:
            raise ValueError(
                f"Expected (B, {self.n_feature_channels}, Nc), got {tuple(x.shape)}."
            )
        out = self.net(x)
        return out.squeeze(1)


def masked_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Mean Huber over batch, averaged over masked subcarriers per sample."""
    diff = pred - target
    abs_diff = diff.abs()
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin = abs_diff - quad
    huber = 0.5 * quad.pow(2) + delta * lin
    mask = loss_mask > 0.5
    denom = mask.sum(dim=1).clamp(min=1.0)
    per_sample = (huber * mask.float()).sum(dim=1) / denom
    return per_sample.mean()


def config_to_meta(cfg: TrainConfig) -> Dict[str, Any]:
    return {
        "n_antennas": cfg.n_antennas,
        "n_subcarriers": cfg.n_subcarriers,
        "rho_space": cfg.rho_space,
        "sigma2": cfg.sigma2,
        "initial_pilot_subcarriers": cfg.initial_pilot_subcarriers,
        "final_pilot_subcarriers": cfg.final_pilot_subcarriers,
        "pilots_added_per_step": cfg.pilots_added_per_step,
        "n_cov_mc": cfg.n_cov_mc,
        "seed": cfg.seed,
        "dtype": str(cfg.dtype),
        "huber_delta": cfg.huber_delta,
        "label_eps": cfg.label_eps,
    }


def n_decision_steps(cfg: TrainConfig) -> int:
    return num_timesteps_from_pilot_growth(
        cfg.initial_pilot_subcarriers,
        cfg.final_pilot_subcarriers,
        cfg.pilots_added_per_step,
    )


def dataset_meta(
    cfg: TrainConfig,
    split: SplitName,
    *,
    n_channels: int,
) -> Dict[str, Any]:
    meta = config_to_meta(cfg)
    meta["split"] = split
    meta["n_channels"] = n_channels
    meta["n_snapshots"] = n_channels * n_decision_steps(cfg)
    meta["n_feature_channels"] = NUM_FEATURE_CHANNELS
    return meta


# Fields that define cached X / y_label / loss_mask. Excludes training-only knobs
# (e.g. huber_delta) that do not affect dataset generation.
CACHE_DATASET_META_KEYS = (
    "n_antennas",
    "n_subcarriers",
    "rho_space",
    "sigma2",
    "initial_pilot_subcarriers",
    "final_pilot_subcarriers",
    "pilots_added_per_step",
    "n_cov_mc",
    "seed",
    "dtype",
    "label_eps",
    "split",
    "n_channels",
    "n_snapshots",
    "n_feature_channels",
)


def meta_matches(cache_meta: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    return all(cache_meta.get(k) == expected.get(k) for k in CACHE_DATASET_META_KEYS)


def dataset_path(split: SplitName) -> Path:
    return DATA_DIR / f"{split}.pt"


def generate_split_dataset(
    cfg: TrainConfig,
    split: SplitName,
    *,
    n_channels: int,
    sigma: torch.Tensor,
    device: torch.device,
    verbose: bool = True,
) -> Dict[str, Any]:
    nc = cfg.n_subcarriers
    t_add = n_decision_steps(cfg)
    n_snap = n_channels * t_add
    base_seed = cfg.seed if split == "train" else cfg.seed + VAL_SEED_OFFSET

    X = torch.zeros((n_snap, NUM_FEATURE_CHANNELS, nc), dtype=torch.float32)
    y_label = torch.zeros((n_snap, nc), dtype=torch.float32)
    loss_mask = torch.zeros((n_snap, nc), dtype=torch.float32)

    grid = SionnaOFDMGrid(fft_size=nc)
    t0 = time.perf_counter()
    write_idx = 0

    for k in range(n_channels):
        channel_seed = base_seed + k
        policy: RolloutPolicy = "random" if (channel_seed % 2 == 0) else "active"
        h_true = sample_tdl_a_channel(cfg, grid=grid, device=device, seed=channel_seed)
        rows = collect_snapshots_from_channel(
            cfg,
            sigma=sigma,
            h_true=h_true,
            device=device,
            grid=grid,
            policy=policy,
            channel_seed=channel_seed + 10_000,
        )
        for x_feat, y_lab, lmask in rows:
            X[write_idx] = x_feat.detach().cpu()
            y_label[write_idx] = y_lab.detach().cpu()
            loss_mask[write_idx] = lmask.detach().cpu()
            write_idx += 1

        if verbose and ((k + 1) % GEN_LOG_INTERVAL == 0 or k + 1 == n_channels):
            elapsed = time.perf_counter() - t0
            print(
                f"gen {split}: {k + 1}/{n_channels} channels  "
                f"({write_idx} snapshots, elapsed {elapsed / 60.0:.1f} min)",
                flush=True,
            )

    if write_idx != n_snap:
        raise RuntimeError(f"Expected {n_snap} snapshots for {split}, got {write_idx}.")

    elapsed_min = (time.perf_counter() - t0) / 60.0
    if verbose:
        print(
            f"gen {split} done: {n_snap} snapshots in {elapsed_min:.1f} min",
            flush=True,
        )

    meta = dataset_meta(cfg, split, n_channels=n_channels)
    meta["created_unix"] = time.time()
    return {"X": X, "y_label": y_label, "loss_mask": loss_mask, "meta": meta}


def save_dataset(split: SplitName, bundle: Dict[str, Any]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = dataset_path(split)
    torch.save(bundle, path)
    return path


def load_dataset(split: SplitName) -> Dict[str, Any]:
    path = dataset_path(split)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset cache missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_datasets(
    cfg: TrainConfig,
    *,
    sigma: torch.Tensor,
    device: torch.device,
    train_channels: int,
    val_channels: int,
    force_regen: bool = False,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_expected = dataset_meta(cfg, "train", n_channels=train_channels)
    val_expected = dataset_meta(cfg, "val", n_channels=val_channels)

    train_hit = False
    val_hit = False
    train_bundle: Optional[Dict[str, Any]] = None
    val_bundle: Optional[Dict[str, Any]] = None

    if not force_regen:
        path = dataset_path("train")
        if path.is_file():
            bundle = load_dataset("train")
            if meta_matches(bundle["meta"], train_expected):
                train_bundle = bundle
                train_hit = True
            elif verbose:
                print(f"cache: {path} (stale meta, will rebuild)", flush=True)
        path = dataset_path("val")
        if path.is_file():
            bundle = load_dataset("val")
            if meta_matches(bundle["meta"], val_expected):
                val_bundle = bundle
                val_hit = True
            elif verbose:
                print(f"cache: {path} (stale meta, will rebuild)", flush=True)

    if verbose:
        print(
            f"cache: {dataset_path('train')} ({'hit' if train_hit else 'miss'})",
            flush=True,
        )
        print(
            f"cache: {dataset_path('val')} ({'hit' if val_hit else 'miss'})",
            flush=True,
        )

    if train_bundle is None:
        if verbose:
            print("Building train dataset...", flush=True)
        train_bundle = generate_split_dataset(
            cfg, "train", n_channels=train_channels, sigma=sigma, device=device, verbose=verbose
        )
        save_dataset("train", train_bundle)
    if val_bundle is None:
        if verbose:
            print("Building val dataset...", flush=True)
        val_bundle = generate_split_dataset(
            cfg, "val", n_channels=val_channels, sigma=sigma, device=device, verbose=verbose
        )
        save_dataset("val", val_bundle)

    return train_bundle, val_bundle


def load_cached_datasets(
    cfg: TrainConfig,
    *,
    train_channels: int = TRAIN_CHANNELS,
    val_channels: int = VAL_CHANNELS,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load train/val caches only; raise if missing or dataset meta mismatch.

    Training-only cfg fields (e.g. huber_delta) may differ from cache meta.
    """
    train_expected = dataset_meta(cfg, "train", n_channels=train_channels)
    val_expected = dataset_meta(cfg, "val", n_channels=val_channels)
    train_bundle = load_dataset("train")
    val_bundle = load_dataset("val")
    if not meta_matches(train_bundle["meta"], train_expected):
        raise RuntimeError(
            f"Train cache meta mismatch at {dataset_path('train')}. "
            "Rebuild with Phase 1 train or fix hyperparameters."
        )
    if not meta_matches(val_bundle["meta"], val_expected):
        raise RuntimeError(
            f"Val cache meta mismatch at {dataset_path('val')}. "
            "Rebuild with Phase 1 train or fix hyperparameters."
        )
    if verbose:
        print(f"cache: {dataset_path('train')} (hit)", flush=True)
        print(f"cache: {dataset_path('val')} (hit)", flush=True)
    return train_bundle, val_bundle


def make_snapshot_loaders(
    train_bundle: Dict[str, Any],
    val_bundle: Dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = PilotSnapshotDataset(
        train_bundle["X"], train_bundle["y_label"], train_bundle["loss_mask"]
    )
    val_ds = PilotSnapshotDataset(
        val_bundle["X"], val_bundle["y_label"], val_bundle["loss_mask"]
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, val_loader


class PilotSnapshotDataset(Dataset):
    def __init__(self, X: torch.Tensor, y_label: torch.Tensor, loss_mask: torch.Tensor) -> None:
        if X.shape[0] != y_label.shape[0] or X.shape[0] != loss_mask.shape[0]:
            raise ValueError("X, y_label, loss_mask must have the same batch dimension.")
        self.X = X
        self.y_label = y_label
        self.loss_mask = loss_mask

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_label[idx], self.loss_mask[idx]


def subsample_train_label_mask(
    loss_mask: torch.Tensor,
    max_train_label_sc: Optional[int],
) -> torch.Tensor:
    """Random subset of unused subcarriers for training loss only."""
    if max_train_label_sc is None:
        return loss_mask
    out = torch.zeros_like(loss_mask)
    for b in range(loss_mask.shape[0]):
        unused = (loss_mask[b] > 0.5).nonzero(as_tuple=True)[0]
        if unused.numel() == 0:
            continue
        n_pick = min(int(max_train_label_sc), int(unused.numel()))
        perm = unused[torch.randperm(unused.numel(), device=unused.device)[:n_pick]]
        out[b, perm] = 1.0
    return out


@torch.no_grad()
def masked_top1_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    """Fraction of samples where argmin pred == argmin target over masked SCs."""
    mask = loss_mask > 0.5
    pred_m = pred.masked_fill(~mask, float("inf"))
    target_m = target.masked_fill(~mask, float("inf"))
    pred_k = pred_m.argmin(dim=1)
    true_k = target_m.argmin(dim=1)
    valid = mask.any(dim=1)
    if not valid.any():
        return 0.0
    return (pred_k[valid] == true_k[valid]).float().mean().item()


def run_loader_epoch(
    model: PilotScorerModelA,
    loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device,
    *,
    train: bool,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_train_label_sc: Optional[int] = None,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_top1 = 0.0
    n_batches = 0

    for x, y, m in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        m = m.to(device, non_blocking=True)
        m_loss = subsample_train_label_mask(m, max_train_label_sc) if train else m

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = masked_huber_loss(pred, y, m_loss, cfg.huber_delta)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {'train' if train else 'val'} loss: {loss.item()}")

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()

        top1 = masked_top1_accuracy(pred, y, m)
        total_loss += loss.item()
        total_top1 += top1
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_top1 / n_batches


def save_checkpoint(
    path: Path,
    *,
    model: PilotScorerModelA,
    cfg: TrainConfig,
    best_epoch: int,
    best_val_huber: float,
    best_val_top1: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": config_to_meta(cfg),
            "model_arch": model.arch_config().to_dict(),
            "best_epoch": best_epoch,
            "best_val_huber": best_val_huber,
            "best_val_top1": best_val_top1,
        },
        path,
    )


def train_cfg_from_meta(cfg_dict: Dict[str, Any]) -> TrainConfig:
    return TrainConfig(
        n_antennas=cfg_dict["n_antennas"],
        n_subcarriers=cfg_dict["n_subcarriers"],
        rho_space=cfg_dict["rho_space"],
        sigma2=cfg_dict["sigma2"],
        initial_pilot_subcarriers=cfg_dict["initial_pilot_subcarriers"],
        final_pilot_subcarriers=cfg_dict["final_pilot_subcarriers"],
        pilots_added_per_step=cfg_dict["pilots_added_per_step"],
        n_cov_mc=cfg_dict["n_cov_mc"],
        seed=cfg_dict["seed"],
        dtype=torch.complex64,
        huber_delta=cfg_dict["huber_delta"],
        label_eps=cfg_dict["label_eps"],
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[PilotScorerModelA, TrainConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = payload["cfg"]
    cfg = train_cfg_from_meta(cfg_dict)
    arch = ModelArchConfig.from_dict(payload.get("model_arch", {}))
    model = PilotScorerModelA(
        n_subcarriers=cfg.n_subcarriers,
        n_feature_channels=arch.n_feature_channels,
        width=arch.width,
        depth=arch.depth,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, cfg, payload


class CNNPilotSampler:
    """Model A deploy: argmin predicted log-MSE over unused subcarriers."""

    def __init__(
        self,
        model: PilotScorerModelA,
        cfg: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def scores(
        self,
        h_hat: torch.Tensor,
        mask: torch.Tensor,
        decision_step: int,
        n_decision_steps: int,
    ) -> torch.Tensor:
        x = build_features(
            self.cfg,
            h_hat=h_hat,
            mask=mask,
            decision_step=decision_step,
            n_decision_steps=n_decision_steps,
        )
        xb = x.unsqueeze(0).to(self.device)
        out = self.model(xb).squeeze(0)
        unused = mask < 0.5
        out = out.clone()
        out[~unused] = float("inf")
        return out

    def select_subcarrier(
        self,
        h_hat: torch.Tensor,
        mask: torch.Tensor,
        decision_step: int,
        n_decision_steps: int,
    ) -> int:
        s = self.scores(h_hat, mask, decision_step, n_decision_steps)
        return int(torch.argmin(s).item())


def train_loop(
    model: PilotScorerModelA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    phase1: Phase1TrainConfig,
    device: torch.device,
    *,
    checkpoint_path: Path,
    log_prefix: str = "epoch",
    metrics_path: Optional[Path] = None,
    metrics_extra: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Shared training loop for Phase 1 and sweep runs."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase1.lr,
        weight_decay=phase1.weight_decay,
    )

    best_val_huber = float("inf")
    best_val_top1 = 0.0
    best_epoch = 0
    epochs_without_improve = 0
    metrics_history: List[Dict[str, Any]] = []
    val_huber_epoch1: Optional[float] = None
    t_run0 = time.perf_counter()

    for epoch in range(1, phase1.max_epochs + 1):
        t0 = time.perf_counter()
        train_huber, _ = run_loader_epoch(
            model,
            train_loader,
            cfg,
            device,
            train=True,
            optimizer=optimizer,
            max_train_label_sc=phase1.max_train_label_sc,
        )
        val_huber, val_top1 = run_loader_epoch(
            model, val_loader, cfg, device, train=False, optimizer=None
        )
        elapsed = time.perf_counter() - t0

        if epoch == 1:
            val_huber_epoch1 = val_huber

        improved = val_huber < best_val_huber
        if improved:
            best_val_huber = val_huber
            best_val_top1 = val_top1
            best_epoch = epoch
            epochs_without_improve = 0
            save_checkpoint(
                checkpoint_path,
                model=model,
                cfg=cfg,
                best_epoch=best_epoch,
                best_val_huber=best_val_huber,
                best_val_top1=best_val_top1,
            )
        else:
            epochs_without_improve += 1

        star = " *" if improved else ""
        print(
            f"{log_prefix} {epoch:03d}/{phase1.max_epochs:03d}  "
            f"train_huber={train_huber:.4f}  val_huber={val_huber:.4f}  "
            f"val_top1={val_top1:.2f}  lr={phase1.lr:.0e}  {elapsed:.1f}s{star}",
            flush=True,
        )

        metrics_history.append(
            {
                "epoch": epoch,
                "train_huber": train_huber,
                "val_huber": val_huber,
                "val_top1": val_top1,
                "lr": phase1.lr,
                "sec": elapsed,
                "best": improved,
            }
        )

        if verbose:
            print(f"           epochs_without_improve={epochs_without_improve}", flush=True)

        if epoch >= phase1.min_epochs and epochs_without_improve >= phase1.early_stop_patience:
            print(
                f"early stop at {log_prefix} {epoch} "
                f"(no val Huber improvement for {phase1.early_stop_patience} epochs)",
                flush=True,
            )
            break

    wall_sec = time.perf_counter() - t_run0

    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "cfg": config_to_meta(cfg),
            "phase1": asdict(phase1),
            "model_arch": model.arch_config().to_dict(),
            "best_epoch": best_epoch,
            "best_val_huber": best_val_huber,
            "best_val_top1": best_val_top1,
            "epochs": metrics_history,
            "wall_sec": wall_sec,
        }
        if metrics_extra:
            payload.update(metrics_extra)
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_huber": best_val_huber,
        "best_val_top1": best_val_top1,
        "wall_sec": wall_sec,
        "checkpoint_path": str(checkpoint_path),
        "val_huber_epoch1": val_huber_epoch1,
    }


def train(
    cfg: Optional[TrainConfig] = None,
    phase1: Optional[Phase1TrainConfig] = None,
    *,
    force_regen: bool = False,
    load_cache_only: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Phase 1: build or load cached TDL-A dataset, train PilotScorerModelA on CUDA.
    """
    cfg = cfg or TrainConfig()
    phase1 = phase1 or Phase1TrainConfig()
    device = resolve_device(cfg.device)

    n_train = phase1.train_channels * n_decision_steps(cfg)
    n_val = phase1.val_channels * n_decision_steps(cfg)

    print(
        f"phase1: device={device}  Sigma_hat n_cov_mc={cfg.n_cov_mc}  "
        f"train={n_train} val={n_val}  batch={phase1.batch_size}  "
        f"width={phase1.width} depth={phase1.depth}",
        flush=True,
    )

    if load_cache_only:
        train_bundle, val_bundle = load_cached_datasets(
            cfg,
            train_channels=phase1.train_channels,
            val_channels=phase1.val_channels,
        )
    else:
        print(f"Estimating Sigma_hat from TDL-A (n_cov_mc={cfg.n_cov_mc})...", flush=True)
        sigma = estimate_sigma_hat_tdl_a(cfg, device)
        train_bundle, val_bundle = ensure_datasets(
            cfg,
            sigma=sigma,
            device=device,
            train_channels=phase1.train_channels,
            val_channels=phase1.val_channels,
            force_regen=force_regen,
            verbose=True,
        )

    train_loader, val_loader = make_snapshot_loaders(
        train_bundle, val_bundle, phase1.batch_size, device
    )

    model = PilotScorerModelA(
        n_subcarriers=cfg.n_subcarriers,
        width=phase1.width,
        depth=phase1.depth,
    ).to(device)

    result = train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        phase1,
        device,
        checkpoint_path=BEST_CHECKPOINT_PATH,
        log_prefix="epoch",
        metrics_path=METRICS_PATH,
        verbose=verbose,
    )

    print(
        f"phase1 done: best_val_huber={result['best_val_huber']:.4f} @ epoch {result['best_epoch']}  "
        f"checkpoint={BEST_CHECKPOINT_PATH}",
        flush=True,
    )

    v1 = result.get("val_huber_epoch1")
    if v1 is not None and result["best_val_huber"] >= v1:
        print(
            "warning: best val Huber did not improve vs epoch 1 "
            f"({v1:.4f} -> {result['best_val_huber']:.4f})",
            flush=True,
        )

    result["metrics_path"] = str(METRICS_PATH)
    return result


def load_sweep_config_line(config_path: Path, index: int) -> SweepRunSpec:
    lines = [
        ln.strip()
        for ln in config_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    if index < 0 or index >= len(lines):
        raise IndexError(
            f"Sweep index {index} out of range for {config_path} ({len(lines)} runs)."
        )
    return SweepRunSpec.from_dict(json.loads(lines[index]))


def append_sweep_results_csv(row: Dict[str, Any]) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "stage",
        "lr",
        "weight_decay",
        "batch_size",
        "width",
        "depth",
        "huber_delta",
        "max_train_label_sc",
        "best_epoch",
        "best_val_huber",
        "best_val_top1",
        "wall_sec",
        "checkpoint_path",
    ]
    write_header = not SWEEP_RESULTS_CSV.is_file()
    with SWEEP_RESULTS_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_sweep(
    config_path: Path,
    index: int,
    *,
    device_str: str = "cuda",
    sweep_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """One hyperparameter run (SLURM array task). Loads cache only."""
    spec = load_sweep_config_line(config_path, index)
    cfg = spec.to_train_cfg(device_str)
    phase1 = spec.to_phase1()
    device = resolve_device(cfg.device)
    out_dir = (sweep_dir or SWEEP_DIR) / spec.run_id
    ckpt_path = out_dir / "best.pt"
    metrics_path = out_dir / "metrics.json"

    print(
        f"sweep {spec.run_id} (stage {spec.stage}): device={device}  "
        f"lr={spec.lr:g} wd={spec.weight_decay:g} batch={spec.batch_size}  "
        f"width={spec.width} depth={spec.depth} huber={spec.huber_delta}  "
        f"max_train_label_sc={spec.max_train_label_sc}",
        flush=True,
    )

    train_bundle, val_bundle = load_cached_datasets(cfg)
    train_loader, val_loader = make_snapshot_loaders(
        train_bundle, val_bundle, phase1.batch_size, device
    )

    model = PilotScorerModelA(
        n_subcarriers=cfg.n_subcarriers,
        width=phase1.width,
        depth=phase1.depth,
    ).to(device)

    result = train_loop(
        model,
        train_loader,
        val_loader,
        cfg,
        phase1,
        device,
        checkpoint_path=ckpt_path,
        log_prefix=f"{spec.run_id}",
        metrics_path=metrics_path,
        metrics_extra={"run_id": spec.run_id, "stage": spec.stage, "spec": asdict(spec)},
        verbose=False,
    )

    row = {
        "run_id": spec.run_id,
        "stage": spec.stage,
        "lr": spec.lr,
        "weight_decay": spec.weight_decay,
        "batch_size": spec.batch_size,
        "width": spec.width,
        "depth": spec.depth,
        "huber_delta": spec.huber_delta,
        "max_train_label_sc": spec.max_train_label_sc,
        "best_epoch": result["best_epoch"],
        "best_val_huber": result["best_val_huber"],
        "best_val_top1": result["best_val_top1"],
        "wall_sec": result["wall_sec"],
        "checkpoint_path": str(ckpt_path),
    }
    append_sweep_results_csv(row)

    print(
        f"sweep {spec.run_id} done: best_val_huber={result['best_val_huber']:.4f}  "
        f"val_top1={result['best_val_top1']:.2f} @ epoch {result['best_epoch']}  "
        f"checkpoint={ckpt_path}",
        flush=True,
    )
    return result


def _read_sweep_results() -> List[Dict[str, Any]]:
    if not SWEEP_RESULTS_CSV.is_file():
        return []
    with SWEEP_RESULTS_CSV.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pick_sweep_winner(
    stage: str,
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Best run for a stage: lowest val Huber; tie-break higher val top-1."""
    rows = [r for r in _read_sweep_results() if r.get("stage", "").upper() == stage.upper()]
    if not rows:
        raise RuntimeError(
            f"No results for stage {stage!r} in {SWEEP_RESULTS_CSV}. Run the stage first."
        )

    def score(r: Dict[str, Any]) -> Tuple[float, float]:
        huber = float(r["best_val_huber"])
        top1 = float(r["best_val_top1"])
        return (huber, -top1)

    rows_sorted = sorted(rows, key=score)
    best = rows_sorted[0]
    print(
        f"stage {stage} winner: {best['run_id']}  "
        f"val_huber={float(best['best_val_huber']):.4f}  "
        f"val_top1={float(best['best_val_top1']):.2f}",
        flush=True,
    )
    if len(rows_sorted) > 1:
        print("  top runs:", flush=True)
        for r in rows_sorted[:top_k]:
            print(
                f"    {r['run_id']}: huber={float(r['best_val_huber']):.4f}  "
                f"top1={float(r['best_val_top1']):.2f}",
                flush=True,
            )
    return best


def update_winners_yaml(stage: str, winner_row: Dict[str, Any]) -> Dict[str, Any]:
    """Merge stage winner into checkpoints/sweep/winners.yaml."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    winners: Dict[str, Any] = {}
    if SWEEP_WINNERS_YAML.is_file():
        text = SWEEP_WINNERS_YAML.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v in ("null", "~", ""):
                    winners[k] = None
                else:
                    try:
                        winners[k] = json.loads(v)
                    except json.JSONDecodeError:
                        winners[k] = v

    keys = (
        "lr",
        "weight_decay",
        "batch_size",
        "width",
        "depth",
        "huber_delta",
        "max_train_label_sc",
    )
    for k in keys:
        if k in winner_row and winner_row[k] not in ("", None):
            val = winner_row[k]
            if k == "max_train_label_sc" and (val == "" or val == "None"):
                winners[k] = None
            elif k in ("lr", "weight_decay", "huber_delta", "best_val_huber", "best_val_top1"):
                winners[k] = float(val)
            elif k in ("batch_size", "width", "depth", "best_epoch"):
                winners[k] = int(float(val))
            elif k == "max_train_label_sc":
                winners[k] = int(float(val))
            else:
                winners[k] = val
    winners[f"stage_{stage.lower()}_run_id"] = winner_row.get("run_id", "")

    lines = ["# Sweep winners — updated by sweep-pick after each stage\n"]
    for k, v in winners.items():
        lines.append(f"{k}: {json.dumps(v)}\n")
    SWEEP_WINNERS_YAML.write_text("".join(lines), encoding="utf-8")
    return winners


def regenerate_stage_jsonl(stage: str, winners: Dict[str, Any]) -> Path:
    """Write next-stage jsonl from winners.yaml (after pick)."""
    lr = float(winners.get("lr", 1e-3))
    wd = float(winners.get("weight_decay", 1e-4))
    batch = int(winners.get("batch_size", 128))
    width = int(winners.get("width", 64))
    depth = int(winners.get("depth", 3))
    huber = float(winners.get("huber_delta", 1.0))
    max_sc = winners.get("max_train_label_sc")

    runs: List[Dict[str, Any]] = []
    st = stage.upper()

    if st == "B":
        for i, bs in enumerate((64, 128, 256)):
            runs.append(
                {
                    "run_id": f"B{i}",
                    "stage": "B",
                    "lr": lr,
                    "weight_decay": wd,
                    "batch_size": bs,
                    "width": 64,
                    "depth": 3,
                    "huber_delta": 1.0,
                    "max_train_label_sc": None,
                }
            )
    elif st == "C":
        idx = 0
        for w in (32, 64, 128):
            for d in (2, 3, 4):
                runs.append(
                    {
                        "run_id": f"C{idx}",
                        "stage": "C",
                        "lr": lr,
                        "weight_decay": wd,
                        "batch_size": batch,
                        "width": w,
                        "depth": d,
                        "huber_delta": 1.0,
                        "max_train_label_sc": None,
                    }
                )
                idx += 1
    elif st == "D":
        runs = [
            {
                "run_id": "D0",
                "stage": "D",
                "lr": lr,
                "weight_decay": wd,
                "batch_size": batch,
                "width": width,
                "depth": depth,
                "huber_delta": huber,
                "max_train_label_sc": None,
            },
            {
                "run_id": "D1",
                "stage": "D",
                "lr": lr,
                "weight_decay": wd,
                "batch_size": batch,
                "width": width,
                "depth": depth,
                "huber_delta": huber,
                "max_train_label_sc": 16,
            },
        ]
    elif st == "E":
        for i, hd in enumerate((0.5, 1.0, 2.0)):
            runs.append(
                {
                    "run_id": f"E{i}",
                    "stage": "E",
                    "lr": lr,
                    "weight_decay": wd,
                    "batch_size": batch,
                    "width": width,
                    "depth": depth,
                    "huber_delta": hd,
                    "max_train_label_sc": max_sc,
                }
            )
    else:
        raise ValueError(f"regenerate_stage_jsonl does not support stage {stage!r}")

    path = SWEEP_DIR / f"stage_{stage.lower()}.jsonl"
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(runs)} runs)", flush=True)
    return path


def sweep_pick_and_regen(
    stage: str,
    *,
    next_stage: Optional[str] = None,
) -> Dict[str, Any]:
    winner = pick_sweep_winner(stage)
    winners = update_winners_yaml(stage, winner)
    if next_stage:
        regenerate_stage_jsonl(next_stage, winners)
    return winners


def sanity_check(
    cfg: Optional[TrainConfig] = None,
    *,
    batch_size: int = 12,
    n_epochs: int = 10,
    lr: float = 1e-3,
    verbose: bool = True,
) -> None:
    """
    Overfit a single minibatch: loss should decrease; gradients must be non-zero.
    """
    cfg = cfg or TrainConfig()
    device = resolve_device(cfg.device)

    if verbose:
        print(f"sanity_check: device={device}, batch_size={batch_size}, epochs={n_epochs}")

    sanity_cfg = replace(cfg, n_cov_mc=min(cfg.n_cov_mc, 64))
    if verbose:
        print(f"Estimating Sigma_hat from TDL-A (n_cov_mc={sanity_cfg.n_cov_mc})...")
    sigma = estimate_sigma_hat_tdl_a(sanity_cfg, device)
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)

    if verbose:
        print("Building one minibatch from TDL-A rollouts...")
    x, y, loss_mask = build_minibatch(
        cfg, batch_size=batch_size, sigma=sigma, device=device, grid=grid, seed=cfg.seed
    )
    x = x.to(device)
    y = y.to(device)
    loss_mask = loss_mask.to(device)

    model = PilotScorerModelA(n_subcarriers=cfg.n_subcarriers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on one minibatch for {n_epochs} epochs (features={NUM_FEATURE_CHANNELS} ch, no innovation):")
    losses: List[float] = []
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = masked_huber_loss(pred, y, loss_mask, cfg.huber_delta)
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
        print(f"epoch {epoch + 1}/{n_epochs}  loss={lv:.6f}", flush=True)
        if verbose:
            print(f"           grad_norm={grad_norm:.4e}", flush=True)

    if losses[-1] >= losses[0]:
        raise RuntimeError(
            f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        )
    if len(losses) >= 2 and abs(losses[0] - losses[1]) < 1e-12:
        raise RuntimeError("Loss barely changed by epoch 1; check graph connectivity.")

    if verbose:
        print(
            f"sanity_check PASSED: loss {losses[0]:.6f} -> {losses[-1]:.6f} "
            f"({100.0 * (1.0 - losses[-1] / losses[0]):.1f}% reduction)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN pilot scorer Model A training.")
    parser.add_argument(
        "command",
        nargs="?",
        default="train",
        choices=("train", "sanity", "sweep", "sweep-pick"),
        help="train | sanity | sweep (one HP run) | sweep-pick (winner + regen next jsonl)",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Rebuild data/cnn_pilot_scorer/{train,val}.pt even if cache exists",
    )
    parser.add_argument(
        "--load-cache-only",
        action="store_true",
        help="train: skip Sigma_hat / dataset generation; require cached .pt files",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--width", type=int, default=DEFAULT_MODEL_WIDTH, help="CNN width")
    parser.add_argument("--depth", type=int, default=DEFAULT_MODEL_DEPTH, help="CNN depth (>=2)")
    parser.add_argument(
        "--max-train-label-sc",
        type=int,
        default=None,
        help="Train on random subset of unused SC labels per snapshot (val uses all)",
    )
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber loss delta")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--train-channels",
        type=int,
        default=TRAIN_CHANNELS,
        help="TDL-A train channels (snapshots = 6 x channels)",
    )
    parser.add_argument(
        "--val-channels",
        type=int,
        default=VAL_CHANNELS,
        help="TDL-A val channels",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SWEEP_DIR / "stage_a.jsonl"),
        help="Sweep: path to stage JSONL config",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sweep: line index in JSONL (SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=str(SWEEP_DIR),
        help="Sweep: directory for run outputs",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="A",
        help="sweep-pick: completed stage letter (A, B, ...)",
    )
    parser.add_argument(
        "--regen-next",
        type=str,
        default=None,
        help="sweep-pick: regenerate stage_X.jsonl for this letter (e.g. B after A)",
    )
    parser.add_argument("--verbose", action="store_true", help="Extra per-epoch logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sanity":
        cfg = TrainConfig(device=args.device)
        sanity_check(cfg)
        return

    if args.command == "sweep":
        run_sweep(
            Path(args.config),
            args.index,
            device_str=args.device,
            sweep_dir=Path(args.sweep_dir),
        )
        return

    if args.command == "sweep-pick":
        sweep_pick_and_regen(
            args.stage,
            next_stage=args.regen_next,
        )
        return

    cfg = TrainConfig(device=args.device, huber_delta=args.huber_delta)
    phase1 = Phase1TrainConfig(
        train_channels=args.train_channels,
        val_channels=args.val_channels,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        width=args.width,
        depth=args.depth,
        max_train_label_sc=args.max_train_label_sc,
    )
    train(
        cfg,
        phase1,
        force_regen=args.force_regen,
        load_cache_only=args.load_cache_only,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
