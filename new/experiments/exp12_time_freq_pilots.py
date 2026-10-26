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

from channel_generators.sionna import SionnaOFDMGrid, sample_tdl_a_time_freq_channel, vec_from_h
from channel_generators.time_freq import build_sigma_time_freq, flat_index_to_kt
from config import Exp12Config
from experiments.common import (
    FULL_LMMSE_COLOR,
    FULL_LMMSE_LINESTYLE,
    POLICY_STYLES,
    full_subcarrier_lmmse_nmse,
    make_policy,
    mean_curve,
    mean_pilots_to_threshold,
    plot_threshold_curves,
)
from pilot_policy.dmrs_comb import DmrsCombPolicy
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

# exp12-only seed offsets (noise). Fixed=+0 reuses the near-universal convention
# (every experiment's fixed-policy noise offset is +0); the rest are a fresh block
# not used elsewhere in the skill's seed-offset table.
NOISE_SEED_OFFSET = {"fixed": 0, "random": 1_210_000, "active": 1_220_000}
FULL_LMMSE_NOISE_SEED_OFFSET = 1_230_000
# TDL channel draw reuses the established "TDL test" convention (seed + 100_000 + mc),
# shared across exp2/3/5/10 and this experiment alike -- not experiment-specific.
TDL_CHANNEL_SEED_OFFSET = 100_000

POLICIES = ("fixed", "random", "active")


def default_config() -> Exp12Config:
    return Exp12Config()


def _check_config(cfg: Exp12Config) -> None:
    if cfg.n_antennas != 1:
        raise ValueError("exp12 requires n_antennas=1 (single antenna; pilot = one (k,t) RE).")
    expected_n = cfg.n_symbols * cfg.n_freq_subcarriers
    if cfg.n_subcarriers != expected_n:
        raise ValueError(
            f"cfg.n_subcarriers ({cfg.n_subcarriers}) must equal "
            f"n_symbols*n_freq_subcarriers ({cfg.n_symbols}*{cfg.n_freq_subcarriers}={expected_n})."
        )


def _sample_mc_tdl_time_freq_channels(
    cfg: Exp12Config,
    *,
    grid: SionnaOFDMGrid,
    device: torch.device,
) -> list[torch.Tensor]:
    """Input: config, OFDM grid, device. Output: list of h_true (Nt*Nc,1), TDL-A time-freq truth."""
    h_list: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        h_tf = sample_tdl_a_time_freq_channel(
            n_symbols=cfg.n_symbols,
            n_subcarriers=cfg.n_freq_subcarriers,
            grid=grid,
            device=device,
            speed=cfg.speed,
            dtype=cfg.dtype,
            seed=cfg.seed + TDL_CHANNEL_SEED_OFFSET + mc,
        )
        h_list.append(vec_from_h(h_tf))
    return h_list


def _make_exp12_policy(name: str, cfg: Exp12Config, device: torch.device, seed: int):
    """Input: policy name, config, device, seed. Output: PilotPolicy instance.

    'fixed' is a DMRS Type-1-style comb (comb_spacing in frequency, repeated at
    cfg.dmrs_pilot_symbols) instead of common.make_policy's 1D largest-gap bisection, which
    doesn't respect the 2D (k,t) grid (see exp12 section of the pilot-selection-simulator
    skill). 'random'/'active' are unaffected by the flattening and reuse make_policy as-is.
    """
    if name == "fixed":
        return DmrsCombPolicy(
            n_subcarriers=cfg.n_subcarriers,
            n_antennas=cfg.n_antennas,
            initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
            max_pilots=cfg.max_pilots,
            device=device,
            n_freq_subcarriers=cfg.n_freq_subcarriers,
            n_symbols=cfg.n_symbols,
            comb_spacing=cfg.dmrs_comb_spacing,
            pilot_symbols=cfg.dmrs_pilot_symbols,
            comb_offset=cfg.dmrs_comb_offset,
        )
    return make_policy(name, cfg, device, seed=seed)


def _run_mc_policy(
    name: str,
    cfg: Exp12Config,
    *,
    device: torch.device,
    sigma: torch.Tensor,
    h_list: list[torch.Tensor],
    noise_seed_offset: int,
) -> list[RunTrace]:
    traces: list[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_seed_offset + mc)
        policy = _make_exp12_policy(name, cfg, device, seed=cfg.seed + noise_seed_offset + mc)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _run_mc_full_lmmse_nmse(
    cfg: Exp12Config,
    *,
    device: torch.device,
    sigma: torch.Tensor,
    h_list: list[torch.Tensor],
) -> float:
    values: list[float] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(
            cfg.seed + FULL_LMMSE_NOISE_SEED_OFFSET + mc
        )
        values.append(
            full_subcarrier_lmmse_nmse(h_true, sigma, cfg, device=device, generator=gen_noise)
        )
    return float(np.mean(values))


def plot_exp12(
    curves: dict[str, np.ndarray],
    cfg: Exp12Config,
    out_path: Path,
    *,
    full_lmmse_nmse: float,
) -> None:
    """Input: name -> mean est. curve, config, path, full-LMMSE value. Output: saves PNG (no threshold line)."""
    title = (
        f"exp12 time-freq pilot policy  Nt={cfg.n_symbols}, Nc={cfg.n_freq_subcarriers} "
        f"(N={cfg.n_subcarriers} REs), speed={cfg.speed:g} m/s, n_mc={cfg.n_mc}, "
        f"pilots={cfg.target_pilots}  [fixed=DMRS comb-{cfg.dmrs_comb_spacing} "
        f"@ symbols {cfg.dmrs_pilot_symbols}]"
    )
    plot_threshold_curves(
        curves,
        plot_len=max(len(c) for c in curves.values()),
        out_path=out_path,
        title=title,
        styles=POLICY_STYLES,
        ylabel="Mean estimated MSE  tr(P)/N",
        xlabel="Number of pilots (resource elements)",
        extra_hlines=[
            (
                full_lmmse_nmse,
                FULL_LMMSE_COLOR,
                FULL_LMMSE_LINESTYLE,
                f"full LMMSE (N={cfg.n_subcarriers} REs)",
            ),
        ],
    )


def plot_pilot_pattern(
    traces_by_policy: dict[str, list[RunTrace]],
    cfg: Exp12Config,
    out_path: Path,
    *,
    mc_index: int = 0,
) -> None:
    """
    Input: policy -> MC traces, config, path, which MC trial to visualize.
    Output: saves a 1x3 PNG scattering each policy's pilot (k,t) positions for one trial,
    colored by selection order. Diagnostic only: shows whether a policy's pilots respect
    the 2D (subcarrier, symbol) grid structure or were chosen as if it were a flat 1D line.
    """
    fig, axes = plt.subplots(1, len(POLICIES), figsize=(4.2 * len(POLICIES), 4.2), squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, POLICIES):
        trace = traces_by_policy[name][mc_index]
        kt = [flat_index_to_kt(s, cfg.n_symbols) for s in trace.subcarriers]
        ks = [p[0] for p in kt]
        ts = [p[1] for p in kt]
        order = np.arange(len(kt))
        sc = ax.scatter(ks, ts, c=order, cmap="viridis", s=28, edgecolors="black", linewidths=0.3)
        ax.set_title(f"{name} (mc={mc_index})")
        ax.set_xlabel("subcarrier k")
        ax.set_xlim(-0.5, cfg.n_freq_subcarriers - 0.5)
        ax.set_ylim(-0.5, cfg.n_symbols - 0.5)
        if ax is axes[0]:
            ax.set_ylabel("OFDM symbol t")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02, label="selection order")
    fig.suptitle(
        f"exp12 pilot placement in (k,t) grid  Nt={cfg.n_symbols}, Nc={cfg.n_freq_subcarriers}, "
        f"pilots={cfg.target_pilots}",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp12(cfg: Exp12Config) -> None:
    _check_config(cfg)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)

    grid = SionnaOFDMGrid(fft_size=cfg.n_freq_subcarriers)
    sigma, _l_time, _l_freq = build_sigma_time_freq(
        cfg.n_symbols,
        cfg.n_freq_subcarriers,
        cfg.rho_time,
        cfg.rho_freq,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )

    h_list = _sample_mc_tdl_time_freq_channels(cfg, grid=grid, device=device)

    traces_by_policy = {
        name: _run_mc_policy(
            name,
            cfg,
            device=device,
            sigma=sigma,
            h_list=h_list,
            noise_seed_offset=NOISE_SEED_OFFSET[name],
        )
        for name in POLICIES
    }

    curves = {name: mean_curve(traces) for name, traces in traces_by_policy.items()}
    for name, traces in traces_by_policy.items():
        mean_pilots = mean_pilots_to_threshold(traces, cfg.nmse_threshold)
        print(
            f"{name}: mean pilots to est. MSE <= {cfg.nmse_threshold} = {mean_pilots:.2f}, "
            f"est. MSE at {len(curves[name])} pilots = {curves[name][-1]:.4e}"
        )

    full_lmmse_nmse = _run_mc_full_lmmse_nmse(cfg, device=device, sigma=sigma, h_list=h_list)
    print(f"full LMMSE (N={cfg.n_subcarriers} REs): mean estimated NMSE = {full_lmmse_nmse:.6f}")

    out_path = _NEW_ROOT / "figures" / "exp12_time_freq_pilots.png"
    plot_exp12(curves, cfg, out_path, full_lmmse_nmse=full_lmmse_nmse)

    pattern_path = _NEW_ROOT / "figures" / "exp12_pilot_pattern.png"
    plot_pilot_pattern(traces_by_policy, cfg, pattern_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="exp12: fixed vs random vs active pilot policies over a time-freq (OFDM symbol x "
        "subcarrier) grid, single antenna, TDL-A truth"
    )
    p.add_argument("--n-symbols", type=int, default=None, help="Nt, OFDM symbols per slot (default 16).")
    p.add_argument(
        "--n-freq-subcarriers", type=int, default=None, help="Nc, frequency subcarriers (default 32)."
    )
    p.add_argument("--speed", type=float, default=None, help="UE speed m/s driving Doppler (default 30.0).")
    p.add_argument("--rho-time", type=float, default=None, help="AR(1) time correlation (default 0.85).")
    p.add_argument("--rho-freq", type=float, default=None, help="AR(1) freq correlation (default 0.85).")
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None, help="Pilots per run (default 64).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Kronecker Sigma (default 1e-9).")
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_symbols is not None:
        cfg.n_symbols = args.n_symbols
    if args.n_freq_subcarriers is not None:
        cfg.n_freq_subcarriers = args.n_freq_subcarriers
    cfg.n_subcarriers = cfg.n_symbols * cfg.n_freq_subcarriers
    if args.speed is not None:
        cfg.speed = args.speed
    if args.rho_time is not None:
        cfg.rho_time = args.rho_time
    if args.rho_freq is not None:
        cfg.rho_freq = args.rho_freq
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots
        cfg.max_pilots = max(cfg.max_pilots, args.target_pilots)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    run_exp12(cfg)


if __name__ == "__main__":
    main()
