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

from channel_generators.low_rank import build_sigma_low_rank, sample_low_rank_h
from config import Exp8Config
from experiments.common import mean_curve
from pilot_policy.active import ActivePolicy
from pilot_policy.fixed import FixedPolicy
from simulation import RunTrace, run_until_threshold, run_until_threshold_genie
from utils import resolve_device, set_seed

NOISE_OFFSET_MATCHED_GENIE = 0
NOISE_OFFSET_IDENTITY_ACTIVE = 10_000
NOISE_OFFSET_IDENTITY_FIXED = 20_000
NOISE_OFFSET_MATCHED_FIXED = 30_000
B_FACTOR_SEED_OFFSET = 3_000_000
CHANNEL_SEED_OFFSET = 500_000


def default_config() -> Exp8Config:
    return Exp8Config()


def _make_active_policy(cfg: Exp8Config, device: torch.device) -> ActivePolicy:
    return ActivePolicy(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        max_pilots=cfg.max_pilots,
        sigma2=cfg.sigma2,
        device=device,
        initial_indices=[cfg.initial_subcarrier],
    )


def _make_fixed_policy(cfg: Exp8Config, device: torch.device) -> FixedPolicy:
    return FixedPolicy(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        max_pilots=cfg.max_pilots,
        device=device,
        initial_indices=[cfg.initial_subcarrier],
    )


def _run_mc_standard(
    cfg: Exp8Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
    noise_offset: int,
    policy_factory,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = policy_factory()
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _run_mc_genie(
    cfg: Exp8Config,
    *,
    h_list: list[torch.Tensor],
    sigma_est: torch.Tensor,
    sigma_score: torch.Tensor,
    device: torch.device,
    noise_offset: int,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = _make_active_policy(cfg, device)
        traces.append(
            run_until_threshold_genie(
                h_true,
                sigma_est,
                sigma_score,
                cfg,
                policy,
                device=device,
                generator=gen_noise,
            )
        )
    return traces


def _plot_slice(curve: np.ndarray, cfg: Exp8Config) -> tuple[np.ndarray, np.ndarray]:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("target_pilots is required for plotting.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)
    y = curve[k0 - 1 : target]
    if len(y) != len(x):
        raise RuntimeError(f"curve length mismatch (y={len(y)}, x={len(x)}).")
    return x, y


def save_figure1(
    *,
    curve_matched_est: np.ndarray,
    curve_id_active_est: np.ndarray,
    curve_genie_est: np.ndarray,
    curve_genie_true: np.ndarray,
    cfg: Exp8Config,
    out_path: Path,
) -> None:
    x, y_m = _plot_slice(curve_matched_est, cfg)
    _, y_ia = _plot_slice(curve_id_active_est, cfg)
    _, y_ge = _plot_slice(curve_genie_est, cfg)
    _, y_gt = _plot_slice(curve_genie_true, cfg)
    mark = max(1, len(x) // 8)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    ax.semilogy(x, y_m, color="C2", linestyle="-", marker="s", markersize=4,
                markevery=mark, linewidth=1.6, label="LMMSE matched (est)")
    ax.semilogy(x, y_ia, color="C0", linestyle="-", marker="o", markersize=4,
                markevery=mark, linewidth=1.6, label="Subopt identity+active (est)")
    ax.semilogy(x, y_ge, color="C1", linestyle="-", marker="^", markersize=4,
                markevery=mark, linewidth=1.6, label="Genie identity+matched J(k) (est)")
    ax.semilogy(x, y_gt, color="C3", linestyle="--", marker="x", markersize=4,
                markevery=mark, linewidth=1.6, label="Genie (true MSE)")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    tick_step = 5 if len(x) > 10 else 1
    tick_end = cfg.target_pilots + (tick_step - cfg.target_pilots % tick_step) % tick_step
    ax.set_xticks(np.arange(cfg.initial_pilot_subcarriers, tick_end + 1, tick_step))
    fig.suptitle(
        f"exp8 low-rank genie  "
        f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"pilots={cfg.initial_pilot_subcarriers}..{cfg.target_pilots}, "
        f"init SC={cfg.initial_subcarrier}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_figure2(
    *,
    curve_genie_est: np.ndarray,
    curve_genie_true: np.ndarray,
    curve_fixed_est: np.ndarray,
    curve_fixed_true: np.ndarray,
    cfg: Exp8Config,
    out_path: Path,
) -> None:
    x, y_ge = _plot_slice(curve_genie_est, cfg)
    _, y_gt = _plot_slice(curve_genie_true, cfg)
    _, y_fe = _plot_slice(curve_fixed_est, cfg)
    _, y_ft = _plot_slice(curve_fixed_true, cfg)
    mark = max(1, len(x) // 8)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    ax.semilogy(x, y_ge, color="C1", linestyle="-", marker="^", markersize=4,
                markevery=mark, linewidth=1.6, label="Genie active (est)")
    ax.semilogy(x, y_gt, color="C3", linestyle="--", marker="x", markersize=4,
                markevery=mark, linewidth=1.6, label="Genie active (true)")
    ax.semilogy(x, y_fe, color="C0", linestyle="-", marker="o", markersize=4,
                markevery=mark, linewidth=1.6, label="Identity+fixed (est)")
    ax.semilogy(x, y_ft, color="C4", linestyle="--", marker="D", markersize=4,
                markevery=mark, linewidth=1.6, label="Identity+fixed (true)")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    tick_step = 5 if len(x) > 10 else 1
    tick_end = cfg.target_pilots + (tick_step - cfg.target_pilots % tick_step) % tick_step
    ax.set_xticks(np.arange(cfg.initial_pilot_subcarriers, tick_end + 1, tick_step))
    fig.suptitle(
        f"exp8 genie active vs fixed (identity est)  "
        f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"pilots={cfg.initial_pilot_subcarriers}..{cfg.target_pilots}, "
        f"init SC={cfg.initial_subcarrier}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_figure3(
    *,
    curve_active_est: np.ndarray,
    curve_active_true: np.ndarray,
    curve_fixed_est: np.ndarray,
    cfg: Exp8Config,
    out_path: Path,
) -> None:
    x, y_ae = _plot_slice(curve_active_est, cfg)
    _, y_at = _plot_slice(curve_active_true, cfg)
    _, y_fe = _plot_slice(curve_fixed_est, cfg)
    mark = max(1, len(x) // 8)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
    ax.semilogy(x, y_ae, color="C2", linestyle="-", marker="s", markersize=4,
                markevery=mark, linewidth=1.6, label="Matched+active (est)")
    ax.semilogy(x, y_fe, color="C0", linestyle="-", marker="o", markersize=4,
                markevery=mark, linewidth=1.6, label="Matched+fixed (est)")
    ax.semilogy(x, y_at, color="C3", linestyle="--", marker="x", markersize=4,
                markevery=mark, linewidth=1.6, label="Matched+active (true)")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)
    tick_step = 5 if len(x) > 10 else 1
    tick_end = cfg.target_pilots + (tick_step - cfg.target_pilots % tick_step) % tick_step
    ax.set_xticks(np.arange(cfg.initial_pilot_subcarriers, tick_end + 1, tick_step))
    fig.suptitle(
        f"exp8 matched LMMSE active vs fixed  "
        f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"pilots={cfg.initial_pilot_subcarriers}..{cfg.target_pilots}, "
        f"init SC={cfg.initial_subcarrier}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp8(cfg: Exp8Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp8 requires target_pilots (default 32).")
    if cfg.initial_pilot_subcarriers != 1:
        raise ValueError("exp8 expects initial_pilot_subcarriers=1.")
    if not (0 <= cfg.initial_subcarrier < cfg.n_subcarriers):
        raise ValueError("initial_subcarrier out of range.")

    n = cfg.n_antennas * cfg.n_subcarriers
    dtype = cfg.dtype
    sigma_identity = torch.eye(n, device=device, dtype=dtype)

    gen_b = torch.Generator(device=device).manual_seed(cfg.seed + B_FACTOR_SEED_OFFSET)
    sigma_matched, b_lr = build_sigma_low_rank(
        n,
        cfg.low_rank_r,
        cfg.low_rank_eps,
        device=device,
        dtype=dtype,
        generator=gen_b,
    )
    mean_pow = float(sigma_matched.diag().real.mean().item())
    print(
        f"=== low_rank: N={n}, mean diag(Sigma)={mean_pow:.6f}, "
        f"min_eig={torch.linalg.eigvalsh(sigma_matched).real.min().item():.3e}"
    )

    h_list = []
    for mc in range(cfg.n_mc):
        g = torch.Generator(device=device).manual_seed(cfg.seed + CHANNEL_SEED_OFFSET + mc)
        h_list.append(
            sample_low_rank_h(
                b_lr, cfg.low_rank_eps, device=device, dtype=dtype, generator=g
            )
        )

    print("  running matched LMMSE (active) ...")
    traces_matched = _run_mc_standard(
        cfg,
        h_list=h_list,
        sigma=sigma_matched,
        device=device,
        noise_offset=NOISE_OFFSET_MATCHED_GENIE,
        policy_factory=lambda: _make_active_policy(cfg, device),
    )

    print("  running identity + active ...")
    traces_id_active = _run_mc_standard(
        cfg,
        h_list=h_list,
        sigma=sigma_identity,
        device=device,
        noise_offset=NOISE_OFFSET_IDENTITY_ACTIVE,
        policy_factory=lambda: _make_active_policy(cfg, device),
    )

    print("  running genie (identity est, matched J(k)) ...")
    traces_genie = _run_mc_genie(
        cfg,
        h_list=h_list,
        sigma_est=sigma_identity,
        sigma_score=sigma_matched,
        device=device,
        noise_offset=NOISE_OFFSET_MATCHED_GENIE,
    )

    print("  running identity + fixed ...")
    traces_id_fixed = _run_mc_standard(
        cfg,
        h_list=h_list,
        sigma=sigma_identity,
        device=device,
        noise_offset=NOISE_OFFSET_IDENTITY_FIXED,
        policy_factory=lambda: _make_fixed_policy(cfg, device),
    )

    print("  running matched LMMSE (fixed) ...")
    traces_matched_fixed = _run_mc_standard(
        cfg,
        h_list=h_list,
        sigma=sigma_matched,
        device=device,
        noise_offset=NOISE_OFFSET_MATCHED_FIXED,
        policy_factory=lambda: _make_fixed_policy(cfg, device),
    )

    # Sanity: matched and genie share noise+shadow => same pilot sequence
    n_check = min(5, len(traces_matched))
    for i in range(n_check):
        if traces_matched[i].subcarriers != traces_genie[i].subcarriers:
            raise RuntimeError(
                f"matched/genie pilot mismatch on mc={i}: "
                f"{traces_matched[i].subcarriers} vs {traces_genie[i].subcarriers}"
            )
    print(f"  verified matched==genie pilot sequences on first {n_check} trials")

    curve_m_est = mean_curve(traces_matched, field="nmse_hat")
    curve_m_true = mean_curve(traces_matched, field="nmse_true")
    curve_mf_est = mean_curve(traces_matched_fixed, field="nmse_hat")
    curve_mf_true = mean_curve(traces_matched_fixed, field="nmse_true")
    curve_ia_est = mean_curve(traces_id_active, field="nmse_hat")
    curve_g_est = mean_curve(traces_genie, field="nmse_hat")
    curve_g_true = mean_curve(traces_genie, field="nmse_true")
    curve_f_est = mean_curve(traces_id_fixed, field="nmse_hat")
    curve_f_true = mean_curve(traces_id_fixed, field="nmse_true")

    idx = cfg.target_pilots - 1
    print(
        f"  at {cfg.target_pilots} pilots: "
        f"matched_active_est={curve_m_est[idx]:.6e}, "
        f"matched_active_true={curve_m_true[idx]:.6e}, "
        f"matched_fixed_est={curve_mf_est[idx]:.6e}, "
        f"matched_fixed_true={curve_mf_true[idx]:.6e}, "
        f"id_active_est={curve_ia_est[idx]:.6e}, "
        f"genie_est={curve_g_est[idx]:.6e}, "
        f"genie_true={curve_g_true[idx]:.6e}, "
        f"id_fixed_est={curve_f_est[idx]:.6e}, "
        f"id_fixed_true={curve_f_true[idx]:.6e}"
    )

    out_dir = _NEW_ROOT / "figures" / "exp8"
    save_figure1(
        curve_matched_est=curve_m_est,
        curve_id_active_est=curve_ia_est,
        curve_genie_est=curve_g_est,
        curve_genie_true=curve_g_true,
        cfg=cfg,
        out_path=out_dir / "exp8_low_rank_genie.png",
    )
    save_figure2(
        curve_genie_est=curve_g_est,
        curve_genie_true=curve_g_true,
        curve_fixed_est=curve_f_est,
        curve_fixed_true=curve_f_true,
        cfg=cfg,
        out_path=out_dir / "exp8_genie_vs_fixed.png",
    )
    save_figure3(
        curve_active_est=curve_m_est,
        curve_active_true=curve_m_true,
        curve_fixed_est=curve_mf_est,
        cfg=cfg,
        out_path=out_dir / "exp8_matched_active_vs_fixed.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "exp8: low-rank identity estimator with genie matched-J(k) active "
            "vs matched LMMSE / identity-active / identity-fixed."
        )
    )
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--low-rank-r", type=int, default=None)
    p.add_argument("--low-rank-eps", type=float, default=None)
    p.add_argument("--initial-subcarrier", type=int, default=None)
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
    if args.low_rank_r is not None:
        cfg.low_rank_r = args.low_rank_r
    if args.low_rank_eps is not None:
        cfg.low_rank_eps = args.low_rank_eps
    if args.initial_subcarrier is not None:
        cfg.initial_subcarrier = args.initial_subcarrier
    run_exp8(cfg)


if __name__ == "__main__":
    main()
