"""
exp14: animate per-RE true error evolution for active vs cnn vs rl pilot policies.

Single held-out time_freq TDL channel (one (model,speed) bucket), same channel/prior/first-pilot
for all three arms. Each frame = after one more pilot is placed; each panel is a heatmap of the
true per-RE error |H_hat(t,k) - H(t,k)|^2 over the (n_symbols, n_freq_subcarriers) grid, with
pilots placed so far marked and the newest pilot highlighted -- answers "is the policy's pilot
landing where the error actually is".

Read-only reuse of error_estimators/cnn/train.py and error_estimators/rl/train.py (checkpoint
loaders, CnnPilotPolicy/RlPilotPolicy, Sigma-bucket/channel-sampling helpers, seed-offset
conventions) -- no edits to either module. Follows the exp11_cnn_allocator.py pattern: a new,
additive new/experiments/exp*.py file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.lmmse import EstimatorState, lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import sample_tdl_ofdm_channel, vec_from_h
from channel_generators.time_freq import flat_index_to_kt
from error_estimators.cnn.train import (
    EVAL_CHANNEL_OFFSET,
    EVAL_CNN_NOISE_OFFSET,
    EVAL_RL_NOISE_OFFSET,
    NOISE_OFFSET,
    TDL_CHANNEL_OFFSET,
    CnnPilotPolicy,
    CnnTrainConfig,
    h_hat_to_H,
    load_cnn_checkpoint,
    load_sigma_buckets,
    make_grid,
    random_first_subcarrier,
    tdl_channel_seed,
)
from error_estimators.cnn.train import _check_domain_config
from error_estimators.rl.train import RlPilotPolicy, load_rl_checkpoint
from pilot_policy.active import ActivePolicy
from pilot_policy.base import PilotPolicy
from utils import measure_subcarrier, resolve_device, set_seed

DEFAULT_CNN_CHECKPOINT = _NEW_ROOT / "error_estimators" / "cnn" / "checkpoints" / "phase1_tf_cf_best.pt"
DEFAULT_RL_CHECKPOINT = _NEW_ROOT / "error_estimators" / "rl" / "checkpoints" / "reinforce_bandit_tf_reinforce_best.pt"
OUT_PATH = _NEW_ROOT / "figures" / "exp14_error_evolution.gif"

ARM_ORDER = ("active", "cnn", "rl")
ARM_TITLES = {"active": "active (greedy J(k))", "cnn": "cnn (counterfactual)", "rl": "rl (REINFORCE)"}
LOG_FLOOR = 1e-6


@dataclass
class Exp14Config:
    tdl_model: str = "B"
    speed: float = 120.0
    channel_idx: int = 2  # mc within the held-out range for this bucket; picked as representative
    # of the bucket's reported averages (active=2.76e-3, cnn=6.48e-3, rl=2.66e-3) after scanning
    # mc=0..9 with the production checkpoints -- see conversation for the scan.
    seed: int = 1
    device: str = "cuda"
    fps: int = 2


class _FixedFirstPilot:
    """Wrap a policy so every arm bootstraps from the same subcarrier (comparison fairness) --
    mirrors error_estimators/cnn/train.py's _FixedInitPilotPolicy, reimplemented locally rather
    than importing that module-private name."""

    def __init__(self, k_init: int, inner: PilotPolicy) -> None:
        self._k_init = k_init
        self._inner = inner

    def reset(self) -> None:
        self._inner.reset()

    def initial_subcarriers(self) -> List[int]:
        return [self._k_init]

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        return self._inner.next_subcarrier(state, used)


def default_cnn_cfg(exp_cfg: Exp14Config) -> CnnTrainConfig:
    """Production time_freq config both checkpoints were trained under (see cnn-error-estimator
    skill's "Time-freq cluster-scale tuning" section) -- all other fields (n_freq_subcarriers=32,
    n_symbols=16, tdl_models, speeds, n_cov_mc_per_bucket=256, sigma2) are already the defaults."""
    cfg = CnnTrainConfig(
        domain="time_freq",
        kernel_hw=(7, 3),
        label_mode="counterfactual",
        seed=exp_cfg.seed,
        device=exp_cfg.device,
    )
    cfg.n_antennas = 1
    cfg.n_subcarriers = cfg.n_symbols * cfg.n_freq_subcarriers
    _check_domain_config(cfg)
    return cfg


def bucket_channel_mc(cfg: CnnTrainConfig, *, tdl_model: str, speed: float, mc: int) -> int:
    """Same held-out channel-id formula as eval_phase1b_time_freq's per-bucket loop."""
    bi = cfg.tdl_models.index(tdl_model)
    si = cfg.speeds.index(speed)
    n_speeds = len(cfg.speeds)
    return EVAL_CHANNEL_OFFSET + (bi * n_speeds + si) * 1_000_000 + mc


def sample_channel(cfg: CnnTrainConfig, *, tdl_model: str, speed: float, channel_mc: int, device: torch.device) -> torch.Tensor:
    grid = make_grid(cfg)
    h_tf = sample_tdl_ofdm_channel(
        model=tdl_model,
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


def build_arm_policy(
    name: str,
    cfg: CnnTrainConfig,
    *,
    device: torch.device,
    k_init: int,
    cnn_model,
    rl_model,
) -> Tuple[PilotPolicy, int]:
    """Output: (wrapped policy with fixed first pilot, noise_seed) -- noise_seed follows the
    exact per-arm offsets run_closed_loop_mse_curve uses, keyed off this channel's channel_mc
    (passed in via closure below)."""
    if name == "active":
        inner: PilotPolicy = ActivePolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.k0,
            max_pilots=cfg.max_pilots,
            sigma2=cfg.sigma2,
            device=device,
        )
        offset = NOISE_OFFSET["active"]
    elif name == "cnn":
        inner = CnnPilotPolicy(cnn_model, cfg=cfg, device=device, select_mode="argmin")
        offset = EVAL_CNN_NOISE_OFFSET
    elif name == "rl":
        inner = RlPilotPolicy(rl_model, cfg=cfg, device=device)
        offset = EVAL_RL_NOISE_OFFSET
    else:
        raise ValueError(f"unknown arm {name!r}")
    return _FixedFirstPilot(k_init, inner), offset


def rollout_with_snapshots(
    policy: PilotPolicy,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    cfg: CnnTrainConfig,
    device: torch.device,
    noise_seed: int,
) -> Tuple[List[int], List[torch.Tensor]]:
    """Re-implements the incremental LMMSE loop directly (run_until_threshold/RunTrace only keep
    scalar nmse_true/nmse_hat per step, not h_hat) so every pilot's full h_hat snapshot is kept.
    Output: (pilots in placement order, h_hat snapshots, snapshots[i] = state after pilots[:i+1])."""
    na, nc, n = cfg.n_antennas, cfg.n_subcarriers, cfg.n_total
    gen = torch.Generator(device=device).manual_seed(noise_seed)
    policy.reset()
    k0 = policy.initial_subcarriers()[0]
    x0, y0 = measure_subcarrier(
        k0, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
    )
    state = lmmse_initial_update(sigma, x0, y0, cfg.sigma2)
    used = [k0]
    snapshots = [state.h_hat.clone()]
    while len(used) < cfg.max_pilots:
        k = policy.next_subcarrier(state, used)
        x, y = measure_subcarrier(
            k, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
        )
        state = lmmse_incremental_update(state, x, y, cfg.sigma2)
        used.append(k)
        snapshots.append(state.h_hat.clone())
    return used, snapshots


def error_grids(h_true: torch.Tensor, snapshots: List[torch.Tensor], *, grid_na: int, grid_nc: int) -> np.ndarray:
    """Output: (T, n_symbols, n_freq_subcarriers) true per-RE squared error over time."""
    h_true_grid = h_hat_to_H(h_true, grid_na, grid_nc)
    out = np.empty((len(snapshots), grid_na, grid_nc), dtype=np.float64)
    for i, h_hat in enumerate(snapshots):
        h_hat_grid = h_hat_to_H(h_hat, grid_na, grid_nc)
        out[i] = (h_hat_grid - h_true_grid).abs().pow(2).cpu().numpy()
    return out


def pilot_rc(s: int, n_symbols: int) -> Tuple[int, int]:
    """Flat RE index -> (row, col) = (symbol t, subcarrier k), matching h_hat_to_H's grid layout."""
    k, t = flat_index_to_kt(s, n_symbols)
    return t, k


def run_exp14(exp_cfg: Exp14Config, *, out_path: Path = OUT_PATH) -> None:
    device = resolve_device(exp_cfg.device)
    set_seed(exp_cfg.seed, device)
    cfg = default_cnn_cfg(exp_cfg)

    print(f"=== exp14 error evolution: TDL-{exp_cfg.tdl_model} speed={exp_cfg.speed:g}m/s, mc={exp_cfg.channel_idx} ===")
    channel_mc = bucket_channel_mc(cfg, tdl_model=exp_cfg.tdl_model, speed=exp_cfg.speed, mc=exp_cfg.channel_idx)
    h_true = sample_channel(cfg, tdl_model=exp_cfg.tdl_model, speed=exp_cfg.speed, channel_mc=channel_mc, device=device)
    sigma = load_sigma_buckets(cfg, device)[(exp_cfg.tdl_model, exp_cfg.speed)]
    k_init = random_first_subcarrier(tdl_channel_seed(cfg, channel_mc), cfg.n_subcarriers)

    cnn_model = load_cnn_checkpoint(DEFAULT_CNN_CHECKPOINT, device)
    rl_model = load_rl_checkpoint(DEFAULT_RL_CHECKPOINT, device)

    grid_na, grid_nc = cfg.n_symbols, cfg.n_freq_subcarriers  # feature_grid_shape(cfg) for domain="time_freq"
    pilots_by_arm: dict[str, List[int]] = {}
    grids_by_arm: dict[str, np.ndarray] = {}
    for name in ARM_ORDER:
        print(f"  rolling out {name} ...")
        policy, offset = build_arm_policy(name, cfg, device=device, k_init=k_init, cnn_model=cnn_model, rl_model=rl_model)
        noise_seed = cfg.seed + offset + channel_mc
        pilots, snapshots = rollout_with_snapshots(policy, h_true=h_true, sigma=sigma, cfg=cfg, device=device, noise_seed=noise_seed)
        pilots_by_arm[name] = pilots
        grids_by_arm[name] = error_grids(h_true, snapshots, grid_na=grid_na, grid_nc=grid_nc)
        print(f"    final (pilots={len(pilots)}): mean true MSE={grids_by_arm[name][-1].mean():.4e}")

    vmax = max(g.max() for g in grids_by_arm.values())
    vmin = max(min(g.min() for g in grids_by_arm.values()), LOG_FLOOR)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    n_frames = cfg.max_pilots
    fig, axes = plt.subplots(1, len(ARM_ORDER), figsize=(5.2 * len(ARM_ORDER), 4.6))
    ims = []
    for ax, name in zip(axes, ARM_ORDER):
        im = ax.imshow(grids_by_arm[name][0], norm=norm, cmap="magma", aspect="auto", origin="lower")
        ax.set_xlabel("subcarrier k")
        ax.set_ylabel("OFDM symbol t")
        ims.append(im)
    fig.colorbar(ims[-1], ax=axes, shrink=0.85, label="true per-RE error |H_hat-H|^2")
    suptitle = fig.suptitle("", y=0.99)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.78, wspace=0.35)

    def update(frame: int):
        for ax, im, name in zip(axes, ims, ARM_ORDER):
            im.set_data(grids_by_arm[name][frame])
            for coll in list(ax.collections):
                coll.remove()
            pilots_so_far = pilots_by_arm[name][: frame + 1]
            prior = pilots_so_far[:-1]
            newest = pilots_so_far[-1]
            if prior:
                rc = [pilot_rc(s, cfg.n_symbols) for s in prior]
                ax.scatter([c for _, c in rc], [r for r, _ in rc], marker="x", c="white", s=28, linewidths=1.3)
            r, c = pilot_rc(newest, cfg.n_symbols)
            ax.scatter([c], [r], marker="*", c="red", s=140, edgecolors="white", linewidths=0.6)
            ax.set_title(f"{ARM_TITLES[name]}\nmean true MSE={grids_by_arm[name][frame].mean():.3e}")
        suptitle.set_text(
            f"exp14: true per-RE error evolution -- TDL-{exp_cfg.tdl_model} {exp_cfg.speed:g} m/s, "
            f"pilots={frame + 1}/{n_frames}"
        )
        return ims

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=exp_cfg.fps))
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    run_exp14(Exp14Config())


if __name__ == "__main__":
    main()
