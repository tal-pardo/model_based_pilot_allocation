from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from channel_generators.jakes import build_sigma_jakes_kron
from channel_generators.low_rank import build_sigma_low_rank, sample_low_rank_h
from config import Exp7Config
from experiments.common import mean_curve
from pilot_policy.active import ActivePolicy
from simulation import RunTrace, run_until_threshold
from utils import complex_standard_normal, exponential_covariance, resolve_device, set_seed

NOISE_OFFSET_MATCHED = 0
NOISE_OFFSET_IDENTITY = 10_000
B_FACTOR_SEED_OFFSET = 3_000_000

SCENARIO_ORDER = ("toeplitz", "kronecker", "jakes", "low_rank")
SCENARIO_TITLES = {
    "toeplitz": "Toeplitz Gaussian",
    "kronecker": "Kronecker Gaussian",
    "jakes": "Jakes spatial + AR(1) freq",
    "low_rank": "Low-rank factor Gaussian",
}
FIG_NAMES = {
    "toeplitz": "exp7_toeplitz.png",
    "kronecker": "exp7_kronecker.png",
    "jakes": "exp7_jakes.png",
    "low_rank": "exp7_low_rank.png",
}


def default_config() -> Exp7Config:
    return Exp7Config()


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


def sample_chol_h(
    chol: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Input: Cholesky L of Sigma. Output: h (N,1) ~ CN(0, Sigma)."""
    n = chol.shape[0]
    z = complex_standard_normal(n, 1, device=device, dtype=dtype, generator=generator)
    return chol @ z


def _make_active_policy(cfg: Exp7Config, device: torch.device) -> ActivePolicy:
    return ActivePolicy(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        max_pilots=cfg.max_pilots,
        sigma2=cfg.sigma2,
        device=device,
        initial_indices=[cfg.initial_subcarrier],
    )


def _run_mc(
    cfg: Exp7Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
    noise_offset: int,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = _make_active_policy(cfg, device)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _mean_power(sigma: torch.Tensor) -> float:
    n = sigma.shape[0]
    return float(sigma.diag().real.mean().item()) if n else 0.0


def _prepare_scenarios(
    cfg: Exp7Config,
    *,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, Callable[[int], torch.Tensor]]]:
    """
    Output: name -> (Sigma_matched, sample_fn(mc) -> h).
    sample_fn uses seed+mc (and scenario offsets where needed).
    """
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype

    sigma_toep = build_sigma_toeplitz(n, cfg.rho_toeplitz, device=device, dtype=dtype)
    chol_toep = torch.linalg.cholesky(sigma_toep)

    sigma_kron, l_space, l_freq = build_sigma_kron(
        na,
        nc,
        cfg.rho_space,
        cfg.rho_freq,
        device=device,
        dtype=dtype,
        reg=cfg.reg_kron,
    )

    sigma_jakes, l_j_space, l_j_freq = build_sigma_jakes_kron(
        na,
        nc,
        cfg.d_over_lambda,
        cfg.rho_freq,
        device=device,
        dtype=dtype,
        reg=cfg.reg_kron,
    )

    gen_b = torch.Generator(device=device).manual_seed(cfg.seed + B_FACTOR_SEED_OFFSET)
    sigma_lr, b_lr = build_sigma_low_rank(
        n,
        cfg.low_rank_r,
        cfg.low_rank_eps,
        device=device,
        dtype=dtype,
        generator=gen_b,
    )

    def sample_toep(mc: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        return sample_chol_h(chol_toep, device=device, dtype=dtype, generator=g)

    def sample_kron(mc: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        return sample_gaussian_h(na, nc, l_space, l_freq, device=device, dtype=dtype, generator=g)

    def sample_jakes(mc: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(cfg.seed + 400_000 + mc)
        return sample_gaussian_h(
            na, nc, l_j_space, l_j_freq, device=device, dtype=dtype, generator=g
        )

    def sample_lr(mc: int) -> torch.Tensor:
        g = torch.Generator(device=device).manual_seed(cfg.seed + 500_000 + mc)
        return sample_low_rank_h(
            b_lr, cfg.low_rank_eps, device=device, dtype=dtype, generator=g
        )

    return {
        "toeplitz": (sigma_toep, sample_toep),
        "kronecker": (sigma_kron, sample_kron),
        "jakes": (sigma_jakes, sample_jakes),
        "low_rank": (sigma_lr, sample_lr),
    }


def save_scenario_figure(
    *,
    scenario: str,
    curve_matched: np.ndarray,
    curve_identity: np.ndarray,
    cfg: Exp7Config,
    out_path: Path,
) -> None:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("target_pilots is required for plotting.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)
    # Trace index 0 = after 1st pilot; slice length target - k0 + 1
    y_m = curve_matched[k0 - 1 : target]
    y_i = curve_identity[k0 - 1 : target]
    if len(y_m) != len(x) or len(y_i) != len(x):
        raise RuntimeError(
            f"{scenario}: curve length mismatch "
            f"(matched={len(y_m)}, identity={len(y_i)}, x={len(x)})."
        )

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8), constrained_layout=True)
    ax.semilogy(x, y_m, color="C2", linestyle="-", marker="s", markersize=4,
                markevery=max(1, len(x) // 8), linewidth=1.6, label="LMMSE (matched)")
    ax.semilogy(x, y_i, color="C0", linestyle="-", marker="o", markersize=4,
                markevery=max(1, len(x) // 8), linewidth=1.6, label="Suboptimal (identity)")
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean estimated MSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    tick_step = 5 if len(x) > 10 else 1
    tick_end = target + (tick_step - target % tick_step) % tick_step
    ax.set_xticks(np.arange(k0, tick_end + 1, tick_step))
    fig.suptitle(
        f"exp7 {SCENARIO_TITLES[scenario]}  "
        f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}, "
        f"pilots={k0}..{target}, active, init SC={cfg.initial_subcarrier}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp7(cfg: Exp7Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp7 requires target_pilots (default 32).")
    if cfg.initial_pilot_subcarriers != 1:
        raise ValueError("exp7 expects initial_pilot_subcarriers=1.")
    if not (0 <= cfg.initial_subcarrier < cfg.n_subcarriers):
        raise ValueError("initial_subcarrier out of range.")

    n = cfg.n_antennas * cfg.n_subcarriers
    sigma_identity = torch.eye(n, device=device, dtype=cfg.dtype)
    scenarios = _prepare_scenarios(cfg, device=device)

    out_dir = _NEW_ROOT / "figures" / "exp7"
    for name in SCENARIO_ORDER:
        sigma_matched, sample_fn = scenarios[name]
        mean_pow = _mean_power(sigma_matched)
        print(
            f"=== {name}: N={n}, mean diag(Sigma)={mean_pow:.6f}, "
            f"min_eig={torch.linalg.eigvalsh(sigma_matched).real.min().item():.3e}"
        )
        h_list = [sample_fn(mc) for mc in range(cfg.n_mc)]

        print(f"  running matched LMMSE ...")
        traces_m = _run_mc(
            cfg,
            h_list=h_list,
            sigma=sigma_matched,
            device=device,
            noise_offset=NOISE_OFFSET_MATCHED,
        )
        print(f"  running suboptimal identity ...")
        traces_i = _run_mc(
            cfg,
            h_list=h_list,
            sigma=sigma_identity,
            device=device,
            noise_offset=NOISE_OFFSET_IDENTITY,
        )

        curve_m = mean_curve(traces_m, field="nmse_hat")
        curve_i = mean_curve(traces_i, field="nmse_hat")
        idx = cfg.target_pilots - 1
        gap = float(curve_i[idx] - curve_m[idx])
        print(
            f"  at {cfg.target_pilots} pilots: matched={curve_m[idx]:.6e}, "
            f"identity={curve_i[idx]:.6e}, gap(id-matched)={gap:.6e}"
        )
        save_scenario_figure(
            scenario=name,
            curve_matched=curve_m,
            curve_identity=curve_i,
            cfg=cfg,
            out_path=out_dir / FIG_NAMES[name],
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "exp7: matched LMMSE vs identity prior — estimated MSE for "
            "Toeplitz, Kronecker, Jakes+Rf, low-rank (active, k0=1)."
        )
    )
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rho-toeplitz", type=float, default=None)
    p.add_argument("--rho-space", type=float, default=None)
    p.add_argument("--rho-freq", type=float, default=None)
    p.add_argument("--d-over-lambda", type=float, default=None)
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
    if args.rho_toeplitz is not None:
        cfg.rho_toeplitz = args.rho_toeplitz
    if args.rho_space is not None:
        cfg.rho_space = args.rho_space
    if args.rho_freq is not None:
        cfg.rho_freq = args.rho_freq
    if args.d_over_lambda is not None:
        cfg.d_over_lambda = args.d_over_lambda
    if args.low_rank_r is not None:
        cfg.low_rank_r = args.low_rank_r
    if args.low_rank_eps is not None:
        cfg.low_rank_eps = args.low_rank_eps
    if args.initial_subcarrier is not None:
        cfg.initial_subcarrier = args.initial_subcarrier
    run_exp7(cfg)


if __name__ == "__main__":
    main()
