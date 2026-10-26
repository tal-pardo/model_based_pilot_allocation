from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.cov_update import (
    damp_sigma_update,
    inflate_covariance,
    ledoit_wolf_shrinkage,
    rebuild_state_from_sigma,
    sbl_alpha_update,
)
from channel_estimators.lmmse import lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import SionnaOFDMGrid, sample_tdl_a_time_freq_channel, vec_from_h
from channel_generators.time_freq import build_delay_doppler_transform
from config import Exp13Config
from error_estimators.trace_min import estimate_nmse
from experiments.common import (
    FULL_LMMSE_COLOR,
    FULL_LMMSE_LINESTYLE,
    make_policy,
    mean_curve,
    run_mc_mean_full_true_lmmse_mse,
)
from simulation import RunTrace, run_until_threshold
from utils import empirical_covariance, empirical_nmse, measure_subcarrier, resolve_device, set_seed

# exp13-only seed offsets (new disjoint block; channel draws reuse the standard "TDL test"
# convention shared across exp2/3/5/10/12).
CALIB_COV_OFFSET = 5_000_000       # seed + CALIB_COV_OFFSET + k         (Sigma_hat_raw draws)
TDL_CHANNEL_SEED_OFFSET = 100_000  # seed + TDL_CHANNEL_SEED_OFFSET + mc (truth, shared per speed)
NOISE_OFFSET_SHRINK = 0
NOISE_OFFSET_INFLATION = 10_000
NOISE_OFFSET_SBL = 20_000
FULL_LMMSE_NOISE_OFFSET = 30_000

ARM_STYLES: dict[str, tuple[str, str, str]] = {
    "shrink": ("C3", "-", "D"),
    "inflation": ("C4", "--", "v"),
    "sbl": ("C5", "-.", "*"),
}


def default_config() -> Exp13Config:
    return Exp13Config()


def _check_config(cfg: Exp13Config) -> None:
    if cfg.n_antennas != 1:
        raise ValueError("exp13 requires n_antennas=1 (single antenna; pilot = one (k,t) RE).")
    expected_n = cfg.n_symbols * cfg.n_freq_subcarriers
    if cfg.n_subcarriers != expected_n:
        raise ValueError(
            f"cfg.n_subcarriers ({cfg.n_subcarriers}) must equal "
            f"n_symbols*n_freq_subcarriers ({cfg.n_symbols}*{cfg.n_freq_subcarriers}={expected_n})."
        )


def _sample_time_freq_channel(
    cfg: Exp13Config, *, grid: SionnaOFDMGrid, device: torch.device, speed: float, seed: int
) -> torch.Tensor:
    h_tf = sample_tdl_a_time_freq_channel(
        n_symbols=cfg.n_symbols,
        n_subcarriers=cfg.n_freq_subcarriers,
        grid=grid,
        device=device,
        speed=speed,
        dtype=cfg.dtype,
        seed=seed,
    )
    return vec_from_h(h_tf)


def build_calibration_sigma(
    cfg: Exp13Config, *, grid: SionnaOFDMGrid, device: torch.device, speed: float
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Output: (Sigma_hat_raw, Sigma_shrink, delta) from cfg.n_cov_mc_shrink iid draws at this
    speed -- Sigma_hat_raw is deliberately built from K << N samples (the realistic, cheap
    deployment calibration artifact all three arms start from)."""
    n = cfg.n_subcarriers
    samples = torch.zeros((cfg.n_cov_mc_shrink, n), device=device, dtype=cfg.dtype)
    for k in range(cfg.n_cov_mc_shrink):
        h = _sample_time_freq_channel(
            cfg, grid=grid, device=device, speed=speed, seed=cfg.seed + CALIB_COV_OFFSET + k
        )
        samples[k] = h.squeeze(-1)
    sigma_hat_raw = empirical_covariance(samples, reg=cfg.reg_empirical, device=device, dtype=cfg.dtype)
    sigma_shrink, delta = ledoit_wolf_shrinkage(
        samples, reg=cfg.reg_empirical, device=device, dtype=cfg.dtype
    )
    return sigma_hat_raw, sigma_shrink, delta


def _sample_mc_h_list(
    cfg: Exp13Config, *, grid: SionnaOFDMGrid, device: torch.device, speed: float
) -> List[torch.Tensor]:
    return [
        _sample_time_freq_channel(
            cfg, grid=grid, device=device, speed=speed, seed=cfg.seed + TDL_CHANNEL_SEED_OFFSET + mc
        )
        for mc in range(cfg.n_mc)
    ]


def run_shrink_arm(
    cfg: Exp13Config, *, h_list: List[torch.Tensor], sigma_shrink: torch.Tensor, device: torch.device
) -> List[RunTrace]:
    """Arm A (floor): frozen Sigma_shrink through the generic driver, unmodified."""
    traces: List[RunTrace] = []
    for mc, h_true in enumerate(h_list):
        seed = cfg.seed + NOISE_OFFSET_SHRINK + mc
        gen_noise = torch.Generator(device=device).manual_seed(seed)
        policy = make_policy("active", cfg, device, seed=seed)
        traces.append(
            run_until_threshold(h_true, sigma_shrink, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def run_inflation_arm(
    cfg: Exp13Config, *, h_list: List[torch.Tensor], sigma_init: torch.Tensor, device: torch.device
) -> List[RunTrace]:
    """Arm B: custom inline loop (exp10 run_oracle template). Inflates P before each reactive
    policy call so ActivePolicy scores against the hedged belief."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2
    traces: List[RunTrace] = []

    for mc, h_true in enumerate(h_list):
        seed = cfg.seed + NOISE_OFFSET_INFLATION + mc
        gen_noise = torch.Generator(device=device).manual_seed(seed)
        policy = make_policy("active", cfg, device, seed=seed)
        trace = RunTrace()
        used: List[int] = []

        policy.reset()
        initial = policy.initial_subcarriers()
        k0 = initial[0]
        x0, y0 = measure_subcarrier(
            k0, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
        )
        state = lmmse_initial_update(sigma_init, x0, y0, sigma2)
        used.append(k0)
        trace.subcarriers.append(k0)
        trace.n_pilots = 1
        trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
        trace.nmse_hat.append(estimate_nmse(state, n))

        for k in initial[1:]:
            x, y = measure_subcarrier(
                k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
            )
            state = lmmse_incremental_update(state, x, y, sigma2)
            used.append(k)
            trace.subcarriers.append(k)
            trace.n_pilots += 1
            trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
            trace.nmse_hat.append(estimate_nmse(state, n))

        estimate_err = trace.nmse_hat[-1]
        while trace.n_pilots < cfg.max_pilots:
            if cfg.target_pilots is not None:
                if trace.n_pilots >= cfg.target_pilots:
                    break
            elif estimate_err <= cfg.nmse_threshold:
                break

            state.P = inflate_covariance(
                state.P,
                n_pilots=trace.n_pilots,
                mode=cfg.inflation_mode,
                factor0=cfg.inflation_factor0,
                decay=cfg.inflation_decay,
                eps=cfg.inflation_eps,
            )
            k = policy.next_subcarrier(state, used)
            x, y = measure_subcarrier(
                k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
            )
            state = lmmse_incremental_update(state, x, y, sigma2)
            used.append(k)
            trace.subcarriers.append(k)
            trace.n_pilots += 1
            trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
            estimate_err = estimate_nmse(state, n)
            trace.nmse_hat.append(estimate_err)

        trace.stopped_reason = (
            "target_pilots"
            if cfg.target_pilots is not None and trace.n_pilots >= cfg.target_pilots
            else ("threshold" if estimate_err <= cfg.nmse_threshold else "max_pilots")
        )
        traces.append(trace)
    return traces


def run_sbl_arm(
    cfg: Exp13Config,
    *,
    h_list: List[torch.Tensor],
    sigma_init: torch.Tensor,
    sigma_hat_raw: torch.Tensor,
    transform: torch.Tensor,
    device: torch.device,
) -> List[RunTrace]:
    """Arm C: custom inline loop. Maintains running sufficient statistics (j_data, b_data) so
    Sigma can be rebuilt mid-run at O(N^3) per rebuild event rather than per pilot; applies the
    SBL fixed point in the delay-Doppler domain, then transforms back to subcarrier-major vec(H)
    basis (load-bearing for lmmse_incremental_update/active_subcarrier_score)."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2
    alpha_init = 1.0 / torch.diagonal(transform @ sigma_hat_raw @ transform.mH).real.clamp(
        min=cfg.sbl_eps
    )
    traces: List[RunTrace] = []

    for mc, h_true in enumerate(h_list):
        seed = cfg.seed + NOISE_OFFSET_SBL + mc
        gen_noise = torch.Generator(device=device).manual_seed(seed)
        policy = make_policy("active", cfg, device, seed=seed)
        trace = RunTrace()
        used: List[int] = []
        alpha = alpha_init.clone()
        sigma_current = sigma_init
        j_data = torch.zeros((n, n), device=device, dtype=dtype)
        b_data = torch.zeros((n, 1), device=device, dtype=dtype)

        policy.reset()
        initial = policy.initial_subcarriers()
        k0 = initial[0]
        x0, y0 = measure_subcarrier(
            k0, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
        )
        state = lmmse_initial_update(sigma_init, x0, y0, sigma2)
        j_data = j_data + (1.0 / sigma2) * (x0.mH @ x0)
        b_data = b_data + (1.0 / sigma2) * (x0.mH @ y0)
        used.append(k0)
        trace.subcarriers.append(k0)
        trace.n_pilots = 1
        trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
        trace.nmse_hat.append(estimate_nmse(state, n))

        for k in initial[1:]:
            x, y = measure_subcarrier(
                k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
            )
            state = lmmse_incremental_update(state, x, y, sigma2)
            j_data = j_data + (1.0 / sigma2) * (x.mH @ x)
            b_data = b_data + (1.0 / sigma2) * (x.mH @ y)
            used.append(k)
            trace.subcarriers.append(k)
            trace.n_pilots += 1
            trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
            trace.nmse_hat.append(estimate_nmse(state, n))

        estimate_err = trace.nmse_hat[-1]
        while trace.n_pilots < cfg.max_pilots:
            if cfg.target_pilots is not None:
                if trace.n_pilots >= cfg.target_pilots:
                    break
            elif estimate_err <= cfg.nmse_threshold:
                break

            if (
                trace.n_pilots >= cfg.sbl_burn_in_pilots
                and (trace.n_pilots - cfg.sbl_burn_in_pilots) % cfg.sbl_update_every == 0
            ):
                a_hat = transform @ state.h_hat
                p_delay_diag = torch.diagonal(transform @ state.P @ transform.mH).real
                alpha = sbl_alpha_update(
                    a_hat,
                    p_delay_diag,
                    alpha,
                    alpha_min=cfg.sbl_alpha_min,
                    alpha_max=cfg.sbl_alpha_max,
                    eps=cfg.sbl_eps,
                )
                sigma_candidate = transform.mH @ torch.diag(1.0 / alpha).to(dtype) @ transform
                sigma_candidate = 0.5 * (sigma_candidate + sigma_candidate.mH)
                sigma_current = damp_sigma_update(
                    sigma_current, sigma_candidate, damping=cfg.sbl_sigma_damping
                )
                state = rebuild_state_from_sigma(sigma_current, j_data, b_data)

            k = policy.next_subcarrier(state, used)
            x, y = measure_subcarrier(
                k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=gen_noise
            )
            state = lmmse_incremental_update(state, x, y, sigma2)
            j_data = j_data + (1.0 / sigma2) * (x.mH @ x)
            b_data = b_data + (1.0 / sigma2) * (x.mH @ y)
            used.append(k)
            trace.subcarriers.append(k)
            trace.n_pilots += 1
            trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
            estimate_err = estimate_nmse(state, n)
            trace.nmse_hat.append(estimate_err)

        trace.stopped_reason = (
            "target_pilots"
            if cfg.target_pilots is not None and trace.n_pilots >= cfg.target_pilots
            else ("threshold" if estimate_err <= cfg.nmse_threshold else "max_pilots")
        )
        traces.append(trace)
    return traces


def mean_pilots_to_threshold_true(traces: List[RunTrace], threshold: float) -> float:
    """Like experiments.common.mean_pilots_to_threshold but keyed on nmse_true: arm B's P is
    deliberately inflated and arm C's P is periodically rebuilt under a changing Sigma, so their
    self-reported nmse_hat is not comparable across arms (see exp13 plan risk #1)."""
    counts = []
    for t in traces:
        hit = next((i + 1 for i, v in enumerate(t.nmse_true) if v <= threshold), t.n_pilots)
        counts.append(hit)
    return float(np.mean(counts))


def save_figure(
    curves_by_speed: Dict[float, Dict[str, np.ndarray]],
    cfg: Exp13Config,
    out_path: Path,
    *,
    full_lmmse_by_speed: Dict[float, float],
) -> None:
    speeds = list(curves_by_speed.keys())
    target = cfg.target_pilots or cfg.max_pilots
    fig, axes = plt.subplots(1, len(speeds), figsize=(5.5 * len(speeds), 4.8), constrained_layout=True)
    if len(speeds) == 1:
        axes = [axes]
    for ax, speed in zip(axes, speeds):
        curves = curves_by_speed[speed]
        x = np.arange(1, target + 1)
        for name, (color, ls, mk) in ARM_STYLES.items():
            y = curves[name][:target]
            ax.semilogy(
                x[: len(y)],
                y,
                color=color,
                linestyle=ls,
                marker=mk,
                label=name,
                markersize=5,
                markevery=max(1, len(y) // 8),
                linewidth=1.6,
            )
        ax.axhline(
            full_lmmse_by_speed[speed],
            color=FULL_LMMSE_COLOR,
            linestyle=FULL_LMMSE_LINESTYLE,
            linewidth=1.2,
            label=f"full LMMSE (N={cfg.n_subcarriers})",
        )
        ax.set_title(f"speed={speed:g} m/s")
        ax.set_xlabel("Number of pilots (resource elements)")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean true NMSE")
    fig.suptitle(
        f"exp13 unknown Sigma (time-freq)  Nt={cfg.n_symbols}, Nc={cfg.n_freq_subcarriers} "
        f"(N={cfg.n_subcarriers} REs), n_mc={cfg.n_mc}, pilots={target}  [ActivePolicy, reactive]",
        fontsize=11,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def run_exp13(cfg: Exp13Config) -> None:
    _check_config(cfg)
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp13 requires target_pilots.")

    grid = SionnaOFDMGrid(fft_size=cfg.n_freq_subcarriers)

    curves_by_speed: Dict[float, Dict[str, np.ndarray]] = {}
    full_lmmse_by_speed: Dict[float, float] = {}

    for speed in cfg.speeds:
        print(f"\n=== speed={speed:g} m/s ===")
        print(f"Calibrating Sigma_hat_raw from {cfg.n_cov_mc_shrink} draws (N={cfg.n_subcarriers}) ...")
        sigma_hat_raw, sigma_shrink, delta = build_calibration_sigma(
            cfg, grid=grid, device=device, speed=speed
        )
        print(f"Ledoit-Wolf delta={delta:.4f}")

        transform = build_delay_doppler_transform(
            cfg.n_freq_subcarriers, cfg.n_symbols, device=device, dtype=cfg.dtype
        )

        h_list = _sample_mc_h_list(cfg, grid=grid, device=device, speed=speed)

        traces: Dict[str, List[RunTrace]] = {}
        print("Running shrink arm ...")
        traces["shrink"] = run_shrink_arm(cfg, h_list=h_list, sigma_shrink=sigma_shrink, device=device)
        print("Running inflation arm ...")
        traces["inflation"] = run_inflation_arm(cfg, h_list=h_list, sigma_init=sigma_shrink, device=device)
        print("Running SBL arm ...")
        traces["sbl"] = run_sbl_arm(
            cfg,
            h_list=h_list,
            sigma_init=sigma_shrink,
            sigma_hat_raw=sigma_hat_raw,
            transform=transform,
            device=device,
        )

        curves = {name: mean_curve(t, field="nmse_true") for name, t in traces.items()}
        curves_by_speed[speed] = curves

        full_lmmse = run_mc_mean_full_true_lmmse_mse(
            cfg, h_list, sigma_shrink, device=device, noise_seed_offset=FULL_LMMSE_NOISE_OFFSET
        )
        full_lmmse_by_speed[speed] = full_lmmse

        idx = cfg.target_pilots - 1
        print(f"At {cfg.target_pilots} pilots (mean true NMSE):")
        for name, curve in curves.items():
            mean_pilots = mean_pilots_to_threshold_true(traces[name], cfg.nmse_threshold)
            print(
                f"  {name:<10} = {float(curve[idx]):.6e}  "
                f"(mean pilots to true NMSE <= {cfg.nmse_threshold} = {mean_pilots:.2f})"
            )
        print(f"  full LMMSE (batch, N={cfg.n_subcarriers}) = {full_lmmse:.6e}")

    out_dir = _NEW_ROOT / "figures" / "exp13"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "exp13_unknown_sigma_time_freq_curves.npz",
        **{
            f"speed_{speed:g}_{name}": curve
            for speed, curves in curves_by_speed.items()
            for name, curve in curves.items()
        },
    )
    save_figure(
        curves_by_speed,
        cfg,
        out_dir / "exp13_unknown_sigma_time_freq.png",
        full_lmmse_by_speed=full_lmmse_by_speed,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "exp13: online Sigma-update methods (static Ledoit-Wolf shrinkage vs covariance "
            "inflation vs delay-Doppler SBL) under unknown true Sigma, TDL-A time-freq truth, "
            "ActivePolicy reacting to each arm's own evolving belief, swept across speeds."
        )
    )
    p.add_argument(
        "--speed", type=float, default=None, help="Single speed override (m/s); replaces the default sweep."
    )
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None)
    p.add_argument("--n-cov-mc-shrink", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.speed is not None:
        cfg.speeds = (args.speed,)
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots
        cfg.max_pilots = max(cfg.max_pilots, args.target_pilots)
    if args.n_cov_mc_shrink is not None:
        cfg.n_cov_mc_shrink = args.n_cov_mc_shrink
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    run_exp13(cfg)


if __name__ == "__main__":
    main()
