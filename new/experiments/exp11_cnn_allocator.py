from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from config import ExpRunConfig
from error_estimators.cnn.train import CNNErrorEstimator, CnnTrainConfig, load_cnn_checkpoint, load_empirical_sigma, sample_tdl_channel
from experiments.common import make_policy, mean_curve, run_mc_mean_full_true_lmmse_mse
from pilot_policy.cnn import CnnPolicy
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

DEFAULT_CHECKPOINT = _NEW_ROOT / "error_estimators" / "cnn" / "checkpoints" / "phase1_cf_best.pt"
EXP11_CHANNEL_OFFSET = 6_000_000  # disjoint from every offset used inside error_estimators/cnn/train.py
NOISE_OFFSET = {"fixed": 0, "active": 20_000, "cnn": 30_000}
FULL_LMMSE_NOISE_OFFSET = 99_999


@dataclass
class Exp11Config(ExpRunConfig):
    """fixed vs active vs cnn-allocator, TDL-A truth, matched empirical Sigma (same prior used for CNN training)."""

    target_pilots: Optional[int] = 32
    initial_pilot_subcarriers: int = 1
    n_mc: int = 50
    n_cov_mc: int = 512


def default_config() -> Exp11Config:
    return Exp11Config()


def _plot_slice(curve: np.ndarray, cfg: Exp11Config) -> tuple[np.ndarray, np.ndarray]:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("exp11 requires target_pilots.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)
    y = curve[k0 - 1 : target]
    if len(y) != len(x):
        raise RuntimeError(f"curve length mismatch (y={len(y)}, x={len(x)}).")
    return x, y


def _sample_h_list(
    cfg: Exp11Config, *, device: torch.device, train_cfg: CnnTrainConfig, grid
) -> List[torch.Tensor]:
    return [
        sample_tdl_channel(train_cfg, channel_mc=EXP11_CHANNEL_OFFSET + mc, device=device, grid=grid)
        for mc in range(cfg.n_mc)
    ]


def _run_mc_baseline(
    cfg: Exp11Config, *, policy_name: str, h_list: List[torch.Tensor], sigma: torch.Tensor, device: torch.device
) -> List[RunTrace]:
    offset = NOISE_OFFSET[policy_name]
    traces: List[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        noise_seed = cfg.seed + offset + mc
        gen = torch.Generator(device=device).manual_seed(noise_seed)
        policy = make_policy(policy_name, cfg, device, seed=noise_seed)
        traces.append(run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen))
    return traces


def _run_mc_cnn(
    cfg: Exp11Config,
    *,
    h_list: List[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
    model: CNNErrorEstimator,
) -> List[RunTrace]:
    offset = NOISE_OFFSET["cnn"]
    traces: List[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        noise_seed = cfg.seed + offset + mc
        gen = torch.Generator(device=device).manual_seed(noise_seed)
        policy = CnnPolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
            max_pilots=cfg.max_pilots,
            sigma2=cfg.sigma2,
            device=device,
            model=model,
        )
        traces.append(run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen))
    return traces


def print_gap_table(
    *, cfg: Exp11Config, curves_true: dict[str, np.ndarray]
) -> None:
    x, cnn = _plot_slice(curves_true["cnn"], cfg)
    _, active = _plot_slice(curves_true["active"], cfg)
    _, fixed = _plot_slice(curves_true["fixed"], cfg)
    gap_vs_active = active - cnn
    gap_vs_fixed = fixed - cnn

    print(f"\n=== exp11 cnn allocator vs fixed/active  n_mc={cfg.n_mc} ===")
    print(f"{'pilots':>7}  {'fixed':>10}  {'active':>10}  {'cnn':>10}  {'gap(act-cnn)':>13}  {'gap(fix-cnn)':>13}")
    for i, m in enumerate(x):
        print(
            f"{int(m):7d}  {fixed[i]:10.4e}  {active[i]:10.4e}  {cnn[i]:10.4e}  "
            f"{gap_vs_active[i]:13.4e}  {gap_vs_fixed[i]:13.4e}"
        )
    print(
        f"  final (pilots={int(x[-1])}): fixed={fixed[-1]:.4e} active={active[-1]:.4e} cnn={cnn[-1]:.4e}\n"
        f"  max gap vs active: {gap_vs_active.max():.4e} at pilots={int(x[int(gap_vs_active.argmax())])}; "
        f"max gap vs fixed: {gap_vs_fixed.max():.4e} at pilots={int(x[int(gap_vs_fixed.argmax())])}"
    )


def save_figure(
    *, cfg: Exp11Config, curves_true: dict[str, np.ndarray], full_lmmse_mse: float, out_path: Path
) -> None:
    x, fixed = _plot_slice(curves_true["fixed"], cfg)
    _, active = _plot_slice(curves_true["active"], cfg)
    _, cnn = _plot_slice(curves_true["cnn"], cfg)
    mark = max(1, len(x) // 8)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.semilogy(x, fixed, "C0-", marker="o", markersize=3, markevery=mark, label="fixed")
    ax.semilogy(x, active, "C1-", marker="s", markersize=3, markevery=mark, label="active (greedy J(k))")
    ax.semilogy(x, cnn, "C2-", marker="^", markersize=3, markevery=mark, label="cnn (counterfactual)")
    ax.axhline(full_lmmse_mse, color="red", linestyle=":", linewidth=1.2, label="full LMMSE (all Nc)")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean true MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title(
        f"exp11 TDL-A  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"k0={cfg.initial_pilot_subcarriers}, pilots→{cfg.target_pilots}"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp11(cfg: Exp11Config, *, checkpoint_path: Path) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp11 requires target_pilots (default 32).")

    # Reuse the exact training-time config/Sigma/channel-sampling convention from
    # error_estimators/cnn/train.py so the comparison uses the same matched prior the
    # CNN was trained under. These are read-only calls into cnn/train.py, not edits.
    train_cfg = CnnTrainConfig(
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        rho_space=cfg.rho_space,
        sigma2=cfg.sigma2,
        seed=cfg.seed,
        n_cov_mc=cfg.n_cov_mc,
    )
    from error_estimators.cnn.train import SionnaOFDMGrid

    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    sigma = load_empirical_sigma(train_cfg, device)
    eig = torch.linalg.eigvalsh(sigma).real
    print(
        f"  Sigma: n_cov_mc={cfg.n_cov_mc}  cond={(eig.max()/eig.min()).item():.3e}  "
        f"#eig@floor={(eig < 1.0001 * train_cfg.reg_empirical).sum().item()}/{eig.numel()}  "
        f"#eig<1%max={(eig < 0.01 * eig.max()).sum().item()}/{eig.numel()}"
    )
    model = load_cnn_checkpoint(checkpoint_path, device)

    print(f"=== exp11 cnn allocator  N={cfg.n_antennas * cfg.n_subcarriers}, checkpoint={checkpoint_path} ===")
    h_list = _sample_h_list(cfg, device=device, train_cfg=train_cfg, grid=grid)

    print("  running fixed ...")
    traces_fixed = _run_mc_baseline(cfg, policy_name="fixed", h_list=h_list, sigma=sigma, device=device)
    print("  running active ...")
    traces_active = _run_mc_baseline(cfg, policy_name="active", h_list=h_list, sigma=sigma, device=device)
    print("  running cnn ...")
    traces_cnn = _run_mc_cnn(cfg, h_list=h_list, sigma=sigma, device=device, model=model)

    curves_true = {
        "fixed": mean_curve(traces_fixed, field="nmse_true"),
        "active": mean_curve(traces_active, field="nmse_true"),
        "cnn": mean_curve(traces_cnn, field="nmse_true"),
    }

    full_lmmse_mse = run_mc_mean_full_true_lmmse_mse(
        cfg, h_list, sigma, device=device, noise_seed_offset=FULL_LMMSE_NOISE_OFFSET
    )

    print_gap_table(cfg=cfg, curves_true=curves_true)
    suffix = "" if cfg.n_cov_mc == 512 else f"_ncovmc{cfg.n_cov_mc}"
    save_figure(
        cfg=cfg,
        curves_true=curves_true,
        full_lmmse_mse=full_lmmse_mse,
        out_path=_NEW_ROOT / "figures" / f"exp11_cnn_allocator{suffix}.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp11: fixed vs active vs cnn-allocator pilot policies (TDL-A).")
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--n-cov-mc", type=int, default=None, help="Override empirical Sigma MC sample count.")
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.target_pilots is not None:
        cfg.target_pilots = None if args.target_pilots == 0 else args.target_pilots
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.n_cov_mc is not None:
        cfg.n_cov_mc = args.n_cov_mc
    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else DEFAULT_CHECKPOINT
    run_exp11(cfg, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main()
