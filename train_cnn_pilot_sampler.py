"""
Model A: CNN pilot scorer training (TDL-A channels).

Phase 0: sanity_check() — overfit one minibatch, verify loss decreases and gradients flow.
Phase 1: full dataset cache + train() on CUDA.
"""

from __future__ import annotations

import argparse
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


class PilotScorerModelA(nn.Module):
    """1D CNN over subcarriers: (B, 7, Nc) -> (B, Nc)."""

    def __init__(
        self,
        n_subcarriers: int = 32,
        n_feature_channels: int = NUM_FEATURE_CHANNELS,
    ) -> None:
        super().__init__()
        self.n_subcarriers = n_subcarriers
        self.n_feature_channels = n_feature_channels
        self.net = nn.Sequential(
            nn.Conv1d(n_feature_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
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


def meta_matches(cache_meta: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    keys = (
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
        "huber_delta",
        "label_eps",
        "split",
        "n_channels",
        "n_snapshots",
        "n_feature_channels",
    )
    return all(cache_meta.get(k) == expected.get(k) for k in keys)


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

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        loss = masked_huber_loss(pred, y, m, cfg.huber_delta)
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
            "best_epoch": best_epoch,
            "best_val_huber": best_val_huber,
            "best_val_top1": best_val_top1,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[PilotScorerModelA, TrainConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = payload["cfg"]
    cfg = TrainConfig(
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
    model = PilotScorerModelA(n_subcarriers=cfg.n_subcarriers).to(device)
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


def train(
    cfg: Optional[TrainConfig] = None,
    phase1: Optional[Phase1TrainConfig] = None,
    *,
    force_regen: bool = False,
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
        f"train={n_train} val={n_val}  batch={phase1.batch_size}",
        flush=True,
    )

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

    train_ds = PilotSnapshotDataset(
        train_bundle["X"], train_bundle["y_label"], train_bundle["loss_mask"]
    )
    val_ds = PilotSnapshotDataset(
        val_bundle["X"], val_bundle["y_label"], val_bundle["loss_mask"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=phase1.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=phase1.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = PilotScorerModelA(n_subcarriers=cfg.n_subcarriers).to(device)
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

    for epoch in range(1, phase1.max_epochs + 1):
        t0 = time.perf_counter()
        train_huber, _ = run_loader_epoch(
            model, train_loader, cfg, device, train=True, optimizer=optimizer
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
                BEST_CHECKPOINT_PATH,
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
            f"epoch {epoch:03d}/{phase1.max_epochs:03d}  "
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
                f"early stop at epoch {epoch} "
                f"(no val Huber improvement for {phase1.early_stop_patience} epochs)",
                flush=True,
            )
            break

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": config_to_meta(cfg),
                "phase1": asdict(phase1),
                "best_epoch": best_epoch,
                "best_val_huber": best_val_huber,
                "best_val_top1": best_val_top1,
                "epochs": metrics_history,
            },
            f,
            indent=2,
        )

    print(
        f"phase1 done: best_val_huber={best_val_huber:.4f} @ epoch {best_epoch}  "
        f"checkpoint={BEST_CHECKPOINT_PATH}",
        flush=True,
    )

    if val_huber_epoch1 is not None and best_val_huber >= val_huber_epoch1:
        print(
            "warning: best val Huber did not improve vs epoch 1 "
            f"({val_huber_epoch1:.4f} -> {best_val_huber:.4f})",
            flush=True,
        )

    return {
        "best_epoch": best_epoch,
        "best_val_huber": best_val_huber,
        "best_val_top1": best_val_top1,
        "metrics_path": str(METRICS_PATH),
        "checkpoint_path": str(BEST_CHECKPOINT_PATH),
    }


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
        choices=("train", "sanity"),
        help="train: Phase 1 full dataset + train; sanity: overfit one minibatch",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="Rebuild data/cnn_pilot_scorer/{train,val}.pt even if cache exists",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Minibatch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
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
    parser.add_argument("--verbose", action="store_true", help="Extra per-epoch logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "sanity":
        cfg = TrainConfig(device=args.device)
        sanity_check(cfg)
        return

    cfg = TrainConfig(device=args.device)
    phase1 = Phase1TrainConfig(
        train_channels=args.train_channels,
        val_channels=args.val_channels,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lr=args.lr,
    )
    train(cfg, phase1, force_regen=args.force_regen, verbose=args.verbose)


if __name__ == "__main__":
    main()
