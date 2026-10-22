from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.block_freq import build_sigma_block_kron, sample_block_kron_h
from config import Exp9Config
from experiments.common import make_policy, mean_curve
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

NOISE_OFFSET_FIXED = 0
NOISE_OFFSET_ACTIVE = 10_000


def default_config() -> Exp9Config:
    return Exp9Config()


def _plot_slice(curve: np.ndarray, cfg: Exp9Config) -> tuple[np.ndarray, np.ndarray]:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("target_pilots is required.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)
    y = curve[k0 - 1 : target]
    if len(y) != len(x):
        raise RuntimeError(f"curve length mismatch (y={len(y)}, x={len(x)}).")
    return x, y


def _sample_h_list(
    cfg: Exp9Config,
    *,
    device: torch.device,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
) -> list[torch.Tensor]:
    h_list: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        h_list.append(
            sample_block_kron_h(
                cfg.n_antennas,
                cfg.n_subcarriers,
                l_space,
                l_freq,
                device=device,
                dtype=cfg.dtype,
                generator=gen_ch,
            )
        )
    return h_list


def _run_mc_policy(
    cfg: Exp9Config,
    *,
    policy_name: str,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
    noise_offset: int,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = make_policy(policy_name, cfg, device, seed=cfg.seed + noise_offset + mc)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def print_gap_table(
    *,
    cfg: Exp9Config,
    est_fixed: np.ndarray,
    est_active: np.ndarray,
    true_fixed: np.ndarray,
    true_active: np.ndarray,
) -> None:
    x, ef = _plot_slice(est_fixed, cfg)
    _, ea = _plot_slice(est_active, cfg)
    _, tf = _plot_slice(true_fixed, cfg)
    _, ta = _plot_slice(true_active, cfg)
    gap_est = ef - ea
    gap_true = tf - ta

    print(
        f"\n=== exp9 active vs fixed gaps  "
        f"rho_intra={cfg.rho_intra} rho_inter={cfg.rho_inter} n_mc={cfg.n_mc} ==="
    )
    print(
        f"{'m':>4}  {'est_fix':>10}  {'est_act':>10}  {'gap_est':>10}  "
        f"{'true_fix':>10}  {'true_act':>10}  {'gap_true':>10}"
    )
    for i, m in enumerate(x):
        print(
            f"{int(m):4d}  {ef[i]:10.4e}  {ea[i]:10.4e}  {gap_est[i]:10.4e}  "
            f"{tf[i]:10.4e}  {ta[i]:10.4e}  {gap_true[i]:10.4e}"
        )
    print(
        f"  max gap_est: {gap_est.max():.4e} at m={int(x[int(gap_est.argmax())])}; "
        f"max gap_true: {gap_true.max():.4e} at m={int(x[int(gap_true.argmax())])}"
    )


def save_figure(
    *,
    cfg: Exp9Config,
    est_fixed: np.ndarray,
    est_active: np.ndarray,
    true_fixed: np.ndarray,
    true_active: np.ndarray,
    out_path: Path,
) -> None:
    x, ef = _plot_slice(est_fixed, cfg)
    _, ea = _plot_slice(est_active, cfg)
    _, tf = _plot_slice(true_fixed, cfg)
    _, ta = _plot_slice(true_active, cfg)
    gap_est = ef - ea
    gap_true = tf - ta
    mark = max(1, len(x) // 8)
    half = cfg.n_subcarriers // 2

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.semilogy(x, ef, "C0-", marker="o", markersize=3, markevery=mark, label="fixed est")
    ax.semilogy(x, ea, "C1-", marker="s", markersize=3, markevery=mark, label="active est")
    ax.semilogy(x, tf, "C0--", marker="o", markersize=3, markevery=mark, label="fixed true")
    ax.semilogy(x, ta, "C1--", marker="s", markersize=3, markevery=mark, label="active true")
    ax.axvline(half, color="gray", linestyle=":", alpha=0.6, label="cluster boundary")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Matched LMMSE: fixed vs active")

    ax = axes[1]
    ax.plot(x, gap_est, "C2-", marker="o", markersize=3, markevery=mark, label="gap est (fix−act)")
    ax.plot(x, gap_true, "C3--", marker="s", markersize=3, markevery=mark, label="gap true (fix−act)")
    ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    ax.axvline(half, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE gap (fixed − active)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    ax.set_title("Gap at every pilot count")

    fig.suptitle(
        f"exp9 block-cluster  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"ρ_intra={cfg.rho_intra}, ρ_inter={cfg.rho_inter}, "
        f"k0={cfg.initial_pilot_subcarriers}, pilots→{cfg.target_pilots}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp9(cfg: Exp9Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp9 requires target_pilots (default 32).")

    sigma, l_space, l_freq = build_sigma_block_kron(
        cfg.n_antennas,
        cfg.n_subcarriers,
        cfg.rho_space,
        rho_intra=cfg.rho_intra,
        rho_inter=cfg.rho_inter,
        n_clusters=cfg.n_clusters,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )
    min_eig = torch.linalg.eigvalsh(sigma).real.min().item()
    print(
        f"=== exp9 block-cluster  N={cfg.n_antennas * cfg.n_subcarriers}, "
        f"rho_intra={cfg.rho_intra}, rho_inter={cfg.rho_inter}, "
        f"min_eig={min_eig:.3e} ==="
    )

    h_list = _sample_h_list(cfg, device=device, l_space=l_space, l_freq=l_freq)

    print("  running matched fixed ...")
    traces_fixed = _run_mc_policy(
        cfg,
        policy_name="fixed",
        h_list=h_list,
        sigma=sigma,
        device=device,
        noise_offset=NOISE_OFFSET_FIXED,
    )
    print("  running matched active ...")
    traces_active = _run_mc_policy(
        cfg,
        policy_name="active",
        h_list=h_list,
        sigma=sigma,
        device=device,
        noise_offset=NOISE_OFFSET_ACTIVE,
    )

    est_fixed = mean_curve(traces_fixed, field="nmse_hat")
    est_active = mean_curve(traces_active, field="nmse_hat")
    true_fixed = mean_curve(traces_fixed, field="nmse_true")
    true_active = mean_curve(traces_active, field="nmse_true")

    print_gap_table(
        cfg=cfg,
        est_fixed=est_fixed,
        est_active=est_active,
        true_fixed=true_fixed,
        true_active=true_active,
    )

    save_figure(
        cfg=cfg,
        est_fixed=est_fixed,
        est_active=est_active,
        true_fixed=true_fixed,
        true_active=true_active,
        out_path=_NEW_ROOT / "figures" / "exp9_block_cluster.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp9: block-cluster matched LMMSE active vs fixed pilot policies."
    )
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rho-intra", type=float, default=None)
    p.add_argument("--rho-inter", type=float, default=None)
    p.add_argument("--rho-space", type=float, default=None)
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
    if args.rho_intra is not None:
        cfg.rho_intra = args.rho_intra
    if args.rho_inter is not None:
        cfg.rho_inter = args.rho_inter
    if args.rho_space is not None:
        cfg.rho_space = args.rho_space
    run_exp9(cfg)


if __name__ == "__main__":
    main()
