from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.lmmse import lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import SionnaOFDMGrid, sample_cdl_channel, sample_tdl_ofdm_channel, vec_from_h
from config import Exp10Config
from error_estimators.trace_min import estimate_nmse
from experiments.common import (
    FULL_LMMSE_COLOR,
    FULL_LMMSE_LINESTYLE,
    POLICY_STYLES,
    mean_curve,
    run_mc_mean_full_true_lmmse_mse,
)
from pilot_policy.active import ActivePolicy
from pilot_policy.fixed import FixedPolicy
from pilot_policy.random import RandomPolicy
from simulation import RunTrace, run_until_threshold
from utils import empirical_covariance, empirical_nmse, measure_subcarrier, resolve_device, set_seed

# Seed-offset regions (disjoint by construction; see exp2/3's 1_000_000-style convention).
CHANNEL_OFFSET = 0  # seed + mc -> h_true shared across all arms for a given trial
NOISE_OFFSET_FIXED = 10_000
NOISE_OFFSET_EMPIRICAL = 20_000
NOISE_OFFSET_RANDOM = 40_000
NOISE_OFFSET_FULL_LMMSE = 50_000
ORACLE_TRIAL_OFFSET = 4_000_000  # seed + ORACLE_TRIAL_OFFSET + mc*100_000 + step*1_000 + k
GENERIC_COV_OFFSET = 3_000_000  # seed + GENERIC_COV_OFFSET + k

ConfigTuple = Tuple[str, str, float]  # (family, model, delay_spread)


def default_config() -> Exp10Config:
    return Exp10Config()


def build_sweep(cfg: Exp10Config) -> List[ConfigTuple]:
    """Input: Exp10Config. Output: list of (family, model, delay_spread), TDL+CDL x A/B/C x delay_spreads."""
    configs: List[ConfigTuple] = []
    for ds in cfg.delay_spreads:
        for model in cfg.tdl_models:
            configs.append(("TDL", model, ds))
        for model in cfg.cdl_models:
            configs.append(("CDL", model, ds))
    return configs


def sample_channel(
    config: ConfigTuple,
    *,
    cfg: Exp10Config,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Input: (family, model, delay_spread), config, device, seed. Output: h (N,1) vec(H)."""
    family, model, delay_spread = config
    grid = SionnaOFDMGrid(
        fft_size=cfg.n_subcarriers,
        subcarrier_spacing=cfg.subcarrier_spacing,
        delay_spread=delay_spread,
    )
    if family == "TDL":
        h_mat = sample_tdl_ofdm_channel(
            model=model,
            n_ofdm_symbols=1,
            n_subcarriers=cfg.n_subcarriers,
            grid=grid,
            device=device,
            n_antennas=cfg.n_antennas,
            rho_space=cfg.rho_space,
            dtype=cfg.dtype,
            seed=seed,
        )
    elif family == "CDL":
        h_mat = sample_cdl_channel(
            model=model,
            n_antennas=cfg.n_antennas,
            n_subcarriers=cfg.n_subcarriers,
            grid=grid,
            device=device,
            dtype=cfg.dtype,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown channel family: {family!r}")
    return vec_from_h(h_mat)


def estimate_generic_sigma(
    configs: List[ConfigTuple],
    *,
    cfg: Exp10Config,
    device: torch.device,
) -> torch.Tensor:
    """Input: full sweep, config, device.
    Output: (N,N) empirical covariance pooled across configs (round-robin draws), i.e. the
    Sigma a naive estimator would recover from a representative deployment stream without
    knowing samples come from different configs (baseline 2's Sigma)."""
    n = cfg.n_antennas * cfg.n_subcarriers
    n_configs = len(configs)
    samples = torch.zeros((cfg.n_cov_mc, n), device=device, dtype=cfg.dtype)
    for k in range(cfg.n_cov_mc):
        config = configs[k % n_configs]
        seed = cfg.seed + GENERIC_COV_OFFSET + k
        h = sample_channel(config, cfg=cfg, device=device, seed=seed)
        samples[k] = h.squeeze(-1)
    return empirical_covariance(samples, reg=cfg.reg_empirical, device=device, dtype=cfg.dtype)


def _make_policy(policy_cls, cfg: Exp10Config, device: torch.device, seed: int):
    if policy_cls is RandomPolicy:
        return RandomPolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
            max_pilots=cfg.max_pilots,
            device=device,
            seed=seed,
            initial_indices=[cfg.initial_subcarrier],
        )
    if policy_cls is ActivePolicy:
        return ActivePolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
            max_pilots=cfg.max_pilots,
            sigma2=cfg.sigma2,
            device=device,
            initial_indices=[cfg.initial_subcarrier],
        )
    return FixedPolicy(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        max_pilots=cfg.max_pilots,
        device=device,
        initial_indices=[cfg.initial_subcarrier],
    )


def run_episodes(
    cfg: Exp10Config,
    *,
    policy_cls,
    sigma_list: List[torch.Tensor],
    h_list: List[torch.Tensor],
    device: torch.device,
    noise_offset: int,
) -> List[RunTrace]:
    """Input: policy class (Active/Fixed/RandomPolicy), per-episode Sigma list, h_true list.
    Output: list of RunTrace, one per episode, each using its own paired Sigma."""
    traces: List[RunTrace] = []
    for mc, (h_true, sigma) in enumerate(zip(h_list, sigma_list)):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_offset + mc)
        policy = _make_policy(policy_cls, cfg, device, seed=cfg.seed + noise_offset + mc)
        traces.append(run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise))
    return traces


def run_oracle(
    cfg: Exp10Config,
    *,
    sigma_list: List[torch.Tensor],
    h_list: List[torch.Tensor],
    device: torch.device,
) -> List[RunTrace]:
    """Brute-force oracle: fixed initial pilot (same as the other arms), then at each
    step exhaustively try every unused subcarrier with its own deterministic trial-noise
    draw, actually compute the resulting true NMSE via h_true, and commit the arg-min
    candidate (its already-drawn measurement becomes the real one, never redrawn).

    Uses the same Sigma as the active arm for its LMMSE belief update, so relative to that
    arm the only thing that changes is the selection rule (brute-force true-NMSE search vs.
    greedy J(k))."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2
    traces: List[RunTrace] = []

    for mc, (h_true, sigma) in enumerate(zip(h_list, sigma_list)):
        trace = RunTrace()
        used: List[int] = []

        k0 = cfg.initial_subcarrier
        gen_init = torch.Generator(device=device).manual_seed(cfg.seed + ORACLE_TRIAL_OFFSET + mc * 100_000)
        x0, y0 = measure_subcarrier(
            k0, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_init
        )
        state = lmmse_initial_update(sigma, x0, y0, sigma2)
        used.append(k0)
        trace.subcarriers.append(k0)
        trace.n_pilots = 1
        trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
        trace.nmse_hat.append(estimate_nmse(state, n))

        step = 1
        while trace.n_pilots < cfg.max_pilots:
            if cfg.target_pilots is not None and trace.n_pilots >= cfg.target_pilots:
                break
            unused = [k for k in range(nc) if k not in used]
            if not unused:
                raise RuntimeError("Oracle: no unused subcarriers remain.")

            best_k = None
            best_err = None
            best_state = None
            for k in unused:
                gen_trial = torch.Generator(device=device).manual_seed(
                    cfg.seed + ORACLE_TRIAL_OFFSET + mc * 100_000 + step * 1_000 + k
                )
                x_k, y_k = measure_subcarrier(
                    k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_trial
                )
                candidate_state = lmmse_incremental_update(state, x_k, y_k, sigma2)
                err = empirical_nmse(candidate_state.h_hat, h_true)
                if best_err is None or err < best_err:
                    best_err = err
                    best_k = k
                    best_state = candidate_state

            state = best_state
            used.append(best_k)
            trace.subcarriers.append(best_k)
            trace.n_pilots += 1
            trace.nmse_true.append(best_err)
            trace.nmse_hat.append(estimate_nmse(state, n))
            step += 1

        trace.stopped_reason = (
            "target_pilots"
            if cfg.target_pilots is not None and trace.n_pilots >= cfg.target_pilots
            else "max_pilots"
        )
        traces.append(trace)
    return traces


def save_figure(
    curves: Dict[str, np.ndarray], cfg: Exp10Config, out_path: Path, *, full_lmmse_mse: float
) -> None:
    target = cfg.target_pilots
    if target is None:
        raise ValueError("target_pilots is required for plotting.")
    k0 = cfg.initial_pilot_subcarriers
    x = np.arange(k0, target + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0), constrained_layout=True)
    styles = {
        "fixed": (*POLICY_STYLES["fixed"], "fixed"),
        "random": (*POLICY_STYLES["random"], "random"),
        "active": (*POLICY_STYLES["active"], "active"),
        "oracle": ("C4", "-", "P", "oracle (brute-force)"),
    }
    for key, (color, ls, mk, label) in styles.items():
        y = curves[key][k0 - 1 : target]
        if len(y) != len(x):
            raise RuntimeError(f"{key}: curve length mismatch (got {len(y)}, expected {len(x)}).")
        ax.semilogy(
            x, y, color=color, linestyle=ls, marker=mk, label=label,
            markersize=5, markevery=max(1, len(x) // 8), linewidth=1.6,
        )
    ax.axhline(
        full_lmmse_mse,
        color=FULL_LMMSE_COLOR,
        linestyle=FULL_LMMSE_LINESTYLE,
        linewidth=1.2,
        label=f"full LMMSE (Nc={cfg.n_subcarriers})",
    )
    # Label decades plus the 2x / 5x minor ticks; denser labels collide over two decades.
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2g"))
    ax.yaxis.set_minor_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.2g}" if f"{v:.1e}"[0] in "25" else "")
    )
    ax.set_xlabel("Number of pilots")
    ax.set_ylabel("Mean true NMSE")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.set_title(
        f"exp10 generic Sigma  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, n_mc={cfg.n_mc}\n"
        f"18 configs (TDL+CDL x A/B/C x {len(cfg.delay_spreads)} delay spreads), "
        f"init SC={cfg.initial_subcarrier}",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp10(cfg: Exp10Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp10 requires target_pilots (default 32).")
    if cfg.initial_pilot_subcarriers != 1:
        raise ValueError("exp10 expects initial_pilot_subcarriers=1.")
    if not (0 <= cfg.initial_subcarrier < cfg.n_subcarriers):
        raise ValueError("initial_subcarrier out of range.")

    configs = build_sweep(cfg)
    n_configs = len(configs)
    print(
        f"Sweep: {n_configs} configs ({len(cfg.tdl_models)} TDL + {len(cfg.cdl_models)} CDL models "
        f"x {len(cfg.delay_spreads)} delay spreads)"
    )

    print(f"Precomputing generic/pooled Sigma (empirical, n_cov_mc={cfg.n_cov_mc} total)...")
    sigma_generic = estimate_generic_sigma(configs, cfg=cfg, device=device)

    print(f"Generating {cfg.n_mc} test episodes (round-robin over {n_configs} configs)...")
    h_list: List[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        seed = cfg.seed + CHANNEL_OFFSET + mc
        h_list.append(sample_channel(configs[mc % n_configs], cfg=cfg, device=device, seed=seed))
    sigma_per_mc = [sigma_generic] * cfg.n_mc

    traces: Dict[str, List[RunTrace]] = {}
    for name, policy_cls, offset in (
        ("fixed", FixedPolicy, NOISE_OFFSET_FIXED),
        ("random", RandomPolicy, NOISE_OFFSET_RANDOM),
        ("active", ActivePolicy, NOISE_OFFSET_EMPIRICAL),
    ):
        print(f"Running {name} + generic Sigma ...")
        traces[name] = run_episodes(
            cfg, policy_cls=policy_cls, sigma_list=sigma_per_mc, h_list=h_list, device=device, noise_offset=offset
        )
    print("Running oracle (brute-force) + generic Sigma ...")
    traces["oracle"] = run_oracle(cfg, sigma_list=sigma_per_mc, h_list=h_list, device=device)

    curves = {name: mean_curve(t, field="nmse_true") for name, t in traces.items()}
    full_lmmse_mse = run_mc_mean_full_true_lmmse_mse(
        cfg, h_list, sigma_generic, device=device, noise_seed_offset=NOISE_OFFSET_FULL_LMMSE
    )

    idx = cfg.target_pilots - 1
    print(f"\nAt {cfg.target_pilots} pilots (mean true NMSE, generic Sigma):")
    for name, curve in curves.items():
        print(f"  {name:<8} = {float(curve[idx]):.6e}")
    print(f"  full LMMSE (batch, Nc={cfg.n_subcarriers}) = {full_lmmse_mse:.6e}")

    out_dir = _NEW_ROOT / "figures" / "exp10"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "exp10_generic_sigma_curves.npz", full_lmmse_mse=full_lmmse_mse, **curves)
    save_figure(curves, cfg, out_dir / "exp10_generic_sigma.png", full_lmmse_mse=full_lmmse_mse)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "exp10: fixed / random / active / oracle brute-force under one generic "
            "(pooled) Sigma, TDL+CDL x A/B/C x delay-spread sweep."
        )
    )
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rho-space", type=float, default=None)
    p.add_argument("--n-cov-mc", type=int, default=None)
    p.add_argument("--initial-subcarrier", type=int, default=None)
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.rho_space is not None:
        cfg.rho_space = args.rho_space
    if args.n_cov_mc is not None:
        cfg.n_cov_mc = args.n_cov_mc
    if args.initial_subcarrier is not None:
        cfg.initial_subcarrier = args.initial_subcarrier
    run_exp10(cfg)


if __name__ == "__main__":
    main()
