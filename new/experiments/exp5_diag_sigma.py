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

from channel_generators.sigma_diag import assert_sigma_diag, build_sigma_diag_from_tdl
from channel_generators.sionna import SionnaOFDMGrid, sample_tdl_a_channel, vec_from_h
from config import Exp5Config
from experiments.common import (
    make_policy,
    mean_curve,
    mean_pilots_to_threshold,
    plot_len_for_threshold,
    run_mc_mean_full_true_lmmse_mse,
)
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

POLICIES = ("fixed", "active")
NOISE_SEED_OFFSET = {"fixed": 0, "active": 10_000}

LEFT_STYLES = {
    "tdl (fixed)": ("C0", "-", "o"),
    "tdl (active)": ("C1", "--", "s"),
}

RIGHT_STYLES = {
    "active true": ("C1", "-", "s"),
    "active est": ("C1", (0, (1, 2)), None),
}


def default_config() -> Exp5Config:
    return Exp5Config()


def _sample_mc_tdl_channels(
    cfg: Exp5Config,
    *,
    grid: SionnaOFDMGrid,
    device: torch.device,
) -> list[torch.Tensor]:
    h_tdl: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        h_mat = sample_tdl_a_channel(
            n_antennas=cfg.n_antennas,
            n_subcarriers=cfg.n_subcarriers,
            rho_space=cfg.rho_space,
            grid=grid,
            device=device,
            dtype=cfg.dtype,
            seed=cfg.seed + 100_000 + mc,
        )
        h_tdl.append(vec_from_h(h_mat))
    return h_tdl


def _run_mc_policy(
    cfg: Exp5Config,
    *,
    policy_name: str,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    noise_offset = NOISE_SEED_OFFSET[policy_name]
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = make_policy(policy_name, cfg, device, seed=cfg.seed + noise_offset + mc)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _run_policies(
    cfg: Exp5Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> dict[str, list[RunTrace]]:
    return {
        policy: _run_mc_policy(cfg, policy_name=policy, h_list=h_list, sigma=sigma, device=device)
        for policy in POLICIES
    }


def _build_true_policy_curves(traces_by_policy: dict[str, list[RunTrace]]) -> dict[str, np.ndarray]:
    return {
        f"tdl ({policy})": mean_curve(traces_by_policy[policy], field="nmse_true")
        for policy in POLICIES
    }


def _build_active_true_est_curves(traces_by_policy: dict[str, list[RunTrace]]) -> dict[str, np.ndarray]:
    traces = traces_by_policy["active"]
    return {
        "active true": mean_curve(traces, field="nmse_true"),
        "active est": mean_curve(traces, field="nmse_hat"),
    }


def _plot_truncated_semilogy(
    ax,
    curves: dict[str, np.ndarray],
    *,
    styles: dict[str, tuple],
    plot_len: int,
    threshold: float | None = None,
    extra_hlines: list[tuple[float, str, str, str]] | None = None,
) -> None:
    steps = np.arange(plot_len)
    for name, curve in curves.items():
        color, ls, mk = styles[name]
        y = np.full(plot_len, np.nan, dtype=np.float64)
        n_copy = min(len(curve), plot_len)
        y[:n_copy] = curve[:n_copy]
        plot_kwargs = dict(linestyle=ls, linewidth=1.6, color=color, label=name)
        if mk is not None:
            plot_kwargs["marker"] = mk
        ax.semilogy(steps, y, **plot_kwargs)

    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2, label="threshold (est.)")
    if extra_hlines:
        for value, color, ls, label in extra_hlines:
            ax.axhline(value, color=color, linestyle=ls, linewidth=1.2, label=label)

    tick_step = 5 if plot_len > 10 else 1
    tick_end = plot_len + (tick_step - plot_len % tick_step) % tick_step
    ax.set_xticks(np.arange(0, tick_end, tick_step))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Pilot step")
    ax.legend(fontsize=8)


def save_exp5_figure(
    true_curves: dict[str, np.ndarray],
    active_curves: dict[str, np.ndarray],
    *,
    cfg: Exp5Config,
    sigma_h2: float,
    plot_len: int,
    full_lmmse_true_mse: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    _plot_truncated_semilogy(
        axes[0],
        true_curves,
        styles=LEFT_STYLES,
        plot_len=plot_len,
        threshold=cfg.nmse_threshold,
    )
    axes[0].set_ylabel("Mean true MSE")
    axes[0].set_title("True MSE vs pilot step")

    _plot_truncated_semilogy(
        axes[1],
        active_curves,
        styles=RIGHT_STYLES,
        plot_len=plot_len,
        extra_hlines=[
            (
                full_lmmse_true_mse,
                "red",
                ":",
                f"full LMMSE true (Nc={cfg.n_subcarriers})",
            ),
        ],
    )
    axes[1].set_ylabel("Mean MSE  true or tr(P)/N")
    axes[1].set_title("Active: true vs estimated MSE")

    suptitle = (
        f"exp5 TDL-A + Sigma_diag  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, stop est. thresh={cfg.nmse_threshold}, sigma_H2={sigma_h2:.3f}"
    )
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp5(cfg: Exp5Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is not None:
        raise ValueError("exp5 requires target_pilots=None (threshold stop on estimated MSE).")

    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    sigma, sigma_h2, tap_powers = build_sigma_diag_from_tdl(
        cfg.n_antennas,
        cfg.n_subcarriers,
        model=cfg.tdl_model,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )
    assert_sigma_diag(sigma, sigma_h2, reg=cfg.reg_kron)
    print(
        f"Sigma_diag: model=TDL-{cfg.tdl_model}, L={tap_powers.numel()}, "
        f"sigma_H2={sigma_h2:.6f}, reg={cfg.reg_kron:g}"
    )

    h_tdl = _sample_mc_tdl_channels(cfg, grid=grid, device=device)
    traces_by_policy = _run_policies(cfg, h_list=h_tdl, sigma=sigma, device=device)

    for policy in POLICIES:
        mean_pilots = mean_pilots_to_threshold(traces_by_policy[policy], cfg.nmse_threshold)
        print(f"tdl ({policy}): mean pilots to est. threshold = {mean_pilots:.2f}")

    true_curves = _build_true_policy_curves(traces_by_policy)
    active_curves = _build_active_true_est_curves(traces_by_policy)
    est_curves = {policy: mean_curve(traces_by_policy[policy], field="nmse_hat") for policy in POLICIES}
    plot_len = plot_len_for_threshold(est_curves, cfg.nmse_threshold)

    full_lmmse_true_mse = run_mc_mean_full_true_lmmse_mse(
        cfg,
        h_tdl,
        sigma,
        device=device,
        noise_seed_offset=NOISE_SEED_OFFSET["active"],
    )
    print(
        f"full LMMSE true MSE (active noise, Nc={cfg.n_subcarriers}): "
        f"{full_lmmse_true_mse:.6e}"
    )

    for policy in POLICIES:
        true_s = mean_curve(traces_by_policy[policy], field="nmse_true")
        est_s = mean_curve(traces_by_policy[policy], field="nmse_hat")
        n = min(plot_len, len(true_s), len(est_s))
        if n > 0:
            print(
                f"tdl ({policy}) at step {n - 1}: true={true_s[n - 1]:.6e}, "
                f"est={est_s[n - 1]:.6e}, gap={true_s[n - 1] - est_s[n - 1]:.6e}"
            )

    out_path = _NEW_ROOT / "figures" / "exp5_diag_sigma.png"
    save_exp5_figure(
        true_curves,
        active_curves,
        cfg=cfg,
        sigma_h2=sigma_h2,
        plot_len=plot_len,
        full_lmmse_true_mse=full_lmmse_true_mse,
        out_path=out_path,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp5: TDL-A with diagonal Sigma prior from tap-power sum")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--nmse-threshold", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Sigma_diag (default 1e-9).")
    p.add_argument("--tdl-model", type=str, default=None, help="TDL model for tap powers (default A).")
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_antennas is not None:
        cfg.n_antennas = args.n_antennas
    if args.n_subcarriers is not None:
        cfg.n_subcarriers = args.n_subcarriers
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.nmse_threshold is not None:
        cfg.nmse_threshold = args.nmse_threshold
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    if args.tdl_model is not None:
        cfg.tdl_model = args.tdl_model
    run_exp5(cfg)


if __name__ == "__main__":
    main()
