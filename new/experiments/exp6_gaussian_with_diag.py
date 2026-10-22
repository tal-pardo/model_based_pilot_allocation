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
from experiments.common import make_policy, mean_curve
from simulation import RunTrace, run_until_threshold
from utils import complex_standard_normal, exponential_covariance, resolve_device, set_seed

# (policy, prior_label) — only combinations plotted
RUN_SPECS = (
    ("active", "diag"),
    ("active", "toep"),
    ("fixed", "toep"),
)
NOISE_SEED_OFFSET = {"fixed": 0, "active": 10_000}

EXP6_STYLES: dict[str, tuple[str, str | tuple, str | None]] = {
    "active diag true": ("C0", "-", "o"),
    "active diag est": ("C0", (0, (1, 2)), None),
    "active toep true": ("C2", "-", "s"),
    "active toep est": ("C2", (0, (1, 2)), None),
    "fixed toep true": ("C1", "-", "^"),
    "fixed toep est": ("C1", (0, (1, 2)), None),
}
CURVE_ORDER = tuple(EXP6_STYLES.keys())


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


def _run_mc_policy(
    cfg: Exp6Config,
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


def _curve_name(policy: str, prior: str, field: str) -> str:
    return f"{policy} {prior} {field}"


def _build_curves(
    traces_by_spec: dict[tuple[str, str], list[RunTrace]],
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for policy, prior in RUN_SPECS:
        traces = traces_by_spec[(policy, prior)]
        curves[_curve_name(policy, prior, "true")] = mean_curve(traces, field="nmse_true")
        curves[_curve_name(policy, prior, "est")] = mean_curve(traces, field="nmse_hat")
    return curves


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

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0), constrained_layout=True)
    for name in CURVE_ORDER:
        color, ls, mk = EXP6_STYLES[name]
        y = _slice_pilot_axis(curves[name], k0=k0, target_pilots=target)
        plot_kwargs = dict(linestyle=ls, linewidth=1.6, color=color, label=name)
        if mk is not None:
            plot_kwargs["marker"] = mk
            plot_kwargs["markersize"] = 4
            plot_kwargs["markevery"] = max(1, plot_len // 8)
        ax.semilogy(x, y, **plot_kwargs)

    tick_step = 5 if plot_len > 10 else 1
    tick_end = target + (tick_step - target % tick_step) % tick_step
    ax.set_xticks(np.arange(k0, tick_end + 1, tick_step))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean MSE  true or tr(P)/N")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"exp6 Toeplitz truth: active diag vs active/fixed toep  "
        f"rho={cfg.rho_toeplitz}, Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, pilots={k0}..{target}, sigma_H2={sigma_h2:.3f}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def _print_summary(curves: dict[str, np.ndarray], *, target: int) -> None:
    idx = target - 1
    if idx < 0:
        return
    parts: list[str] = []
    for policy, prior in RUN_SPECS:
        true_name = _curve_name(policy, prior, "true")
        est_name = _curve_name(policy, prior, "est")
        if idx >= len(curves[true_name]):
            return
        t = curves[true_name][idx]
        e = curves[est_name][idx]
        parts.append(f"{policy}+{prior}: true={t:.6e}, est={e:.6e}, gap={t - e:.6e}")
    ad = curves[_curve_name("active", "diag", "true")][idx]
    at = curves[_curve_name("active", "toep", "true")][idx]
    ft = curves[_curve_name("fixed", "toep", "true")][idx]
    parts.append(f"true gaps: active(diag-toep)={ad - at:.6e}, fixed_toep-active_toep={ft - at:.6e}")
    print(f"at {target} pilots: " + "; ".join(parts))


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

    sigmas = {"diag": sigma_diag, "toep": sigma_toep}
    h_list = _sample_mc_toeplitz_channels(cfg, chol=chol, device=device)

    traces_by_spec: dict[tuple[str, str], list[RunTrace]] = {}
    for policy, prior in RUN_SPECS:
        print(f"Running policy={policy}, prior={prior} ...")
        traces_by_spec[(policy, prior)] = _run_mc_policy(
            cfg,
            policy_name=policy,
            h_list=h_list,
            sigma=sigmas[prior],
            device=device,
        )

    curves = _build_curves(traces_by_spec)
    _print_summary(curves, target=cfg.target_pilots)
    save_exp6_figure(
        curves,
        cfg=cfg,
        sigma_h2=sigma_h2,
        out_path=_NEW_ROOT / "figures" / "exp6_gaussian_with_diag.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "exp6: Toeplitz truth; one figure — active+diag, active+toep, fixed+toep "
            "(true+est each)"
        )
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
