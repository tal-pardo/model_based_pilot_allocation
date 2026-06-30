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
from channel_generators.sionna import SionnaOFDMGrid
from config import Exp6Config
from experiments.common import EXP3_STYLES, make_policy, mean_curve
from simulation import RunTrace, run_until_threshold
from utils import complex_standard_normal, exponential_covariance, resolve_device, set_seed

POLICY = "fixed"
NOISE_SEED_OFFSET = 0


def default_config() -> Exp6Config:
    return Exp6Config()


def build_sigma_toeplitz(
    n: int,
    rho: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Input: N, rho. Output: (N,N) Hermitian Toeplitz Sigma with [Sigma]_ij = rho^|i-j|."""
    sigma = exponential_covariance(n, rho, device=device, dtype=dtype)
    return 0.5 * (sigma + sigma.mH)


def sample_toeplitz_h(
    chol: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Input: Cholesky L of Sigma_toep. Output: h (N,1) ~ CN(0, Sigma)."""
    n = chol.shape[0]
    z = complex_standard_normal(n, 1, device=device, dtype=dtype, generator=generator)
    return chol @ z


def _sample_mc_toeplitz_channels(
    cfg: Exp6Config,
    *,
    chol: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        channels.append(
            sample_toeplitz_h(chol, device=device, dtype=cfg.dtype, generator=gen_ch)
        )
    return channels


def _run_mc_fixed(
    cfg: Exp6Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + NOISE_SEED_OFFSET + mc)
        policy = make_policy(POLICY, cfg, device, seed=cfg.seed + NOISE_SEED_OFFSET + mc)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _build_fixed_curves(traces: list[RunTrace]) -> dict[str, np.ndarray]:
    return {
        "fixed true": mean_curve(traces, field="nmse_true"),
        "fixed est": mean_curve(traces, field="nmse_hat"),
    }


def _slice_pilot_axis(
    curve: np.ndarray,
    *,
    k0: int,
    target_pilots: int,
) -> np.ndarray:
    """Input: per-trace-index curve. Output: values at pilot counts k0..target_pilots."""
    start = k0 - 1
    end = target_pilots
    return curve[start:end]


def save_exp6_figure(
    curves: dict[str, np.ndarray],
    *,
    cfg: Exp6Config,
    sigma_h2: float,
    out_path: Path,
) -> None:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("target_pilots is required for plotting.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)
    plot_len = len(x)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)
    styles = {"fixed true": EXP3_STYLES["fixed true"], "fixed est": EXP3_STYLES["fixed est"]}

    for name, curve in curves.items():
        color, ls, mk = styles[name]
        y = _slice_pilot_axis(curve, k0=k0, target_pilots=target)
        plot_kwargs = dict(linestyle=ls, linewidth=1.6, color=color, label=name)
        if mk is not None:
            plot_kwargs["marker"] = mk
        ax.semilogy(x, y, **plot_kwargs)

    tick_step = 5 if plot_len > 10 else 1
    tick_end = target + (tick_step - target % tick_step) % tick_step
    ax.set_xticks(np.arange(k0, tick_end + 1, tick_step))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE  true or tr(P)/N")
    ax.legend(fontsize=9)

    suptitle = (
        f"exp6 Toeplitz rho={cfg.rho_toeplitz} + Sigma_diag (fixed)  "
        f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"pilots={k0}..{target}, sigma_H2={sigma_h2:.3f}"
    )
    fig.suptitle(suptitle, fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp6(cfg: Exp6Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp6 requires target_pilots (default 32).")

    n = cfg.n_antennas * cfg.n_subcarriers
    if not (0.0 <= cfg.rho_toeplitz < 1.0):
        raise ValueError("rho_toeplitz must satisfy 0 <= rho < 1.")

    sigma_toep = build_sigma_toeplitz(n, cfg.rho_toeplitz, device=device, dtype=cfg.dtype)
    min_eig = torch.linalg.eigvalsh(sigma_toep).real.min().item()
    if min_eig <= 0:
        raise ValueError(f"Sigma_toeplitz is not PD: min eigenvalue = {min_eig:g}.")
    chol = torch.linalg.cholesky(sigma_toep)
    print(f"Sigma_toeplitz: N={n}, rho={cfg.rho_toeplitz}, min_eig={min_eig:.6e}")

    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    sigma_diag, sigma_h2, tap_powers = build_sigma_diag_from_tdl(
        cfg.n_antennas,
        cfg.n_subcarriers,
        model=cfg.tdl_model,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )
    assert_sigma_diag(sigma_diag, sigma_h2, reg=cfg.reg_kron)
    print(
        f"Sigma_diag (LMMSE prior): model=TDL-{cfg.tdl_model}, L={tap_powers.numel()}, "
        f"sigma_H2={sigma_h2:.6f}, reg={cfg.reg_kron:g}"
    )

    h_list = _sample_mc_toeplitz_channels(cfg, chol=chol, device=device)
    traces = _run_mc_fixed(cfg, h_list=h_list, sigma=sigma_diag, device=device)

    target = cfg.target_pilots
    k0 = cfg.initial_pilot_subcarriers
    curves = _build_fixed_curves(traces)
    true_s = curves["fixed true"]
    est_s = curves["fixed est"]
    n_plot = min(target - k0 + 1, len(true_s) - (k0 - 1), len(est_s) - (k0 - 1))
    if n_plot > 0:
        idx = k0 - 1 + n_plot - 1
        print(
            f"Toeplitz (fixed) at {target} pilots: true={true_s[idx]:.6e}, "
            f"est={est_s[idx]:.6e}, gap={true_s[idx] - est_s[idx]:.6e}"
        )
    save_exp6_figure(
        curves,
        cfg=cfg,
        sigma_h2=sigma_h2,
        out_path=_NEW_ROOT / "figures" / "exp6_gaussian_with_diag.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp6: Toeplitz Gaussian truth + diagonal Sigma prior (true vs est MSE)"
    )
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None, help="Fixed pilot count (default 32).")
    p.add_argument("--rho-toeplitz", type=float, default=None, help="Toeplitz correlation (default 0.8).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Sigma_diag (default 0).")
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
    if args.target_pilots is not None:
        cfg.target_pilots = None if args.target_pilots == 0 else args.target_pilots
    if args.rho_toeplitz is not None:
        cfg.rho_toeplitz = args.rho_toeplitz
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    if args.tdl_model is not None:
        cfg.tdl_model = args.tdl_model
    run_exp6(cfg)


if __name__ == "__main__":
    main()
