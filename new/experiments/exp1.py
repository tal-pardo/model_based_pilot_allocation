from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from config import Exp1Config
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
from simulation import run_until_threshold
from utils import resolve_device, set_seed


def default_config() -> Exp1Config:
    return Exp1Config()


def _run_mc_policy(
    name: str,
    cfg: Exp1Config,
    *,
    device: torch.device,
    sigma: torch.Tensor,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
    noise_seed_offset: int,
):
    traces = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        h_true = sample_gaussian_h(
            cfg.n_antennas,
            cfg.n_subcarriers,
            l_space,
            l_freq,
            device=device,
            dtype=cfg.dtype,
            generator=gen_ch,
        )
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_seed_offset + mc)
        policy = make_policy(name, cfg, device, seed=cfg.seed + noise_seed_offset + mc)
        traces.append(
            run_until_threshold(h_true, sigma, cfg, policy, device=device, generator=gen_noise)
        )
    return traces


def _run_mc_full_lmmse_nmse(
    cfg: Exp1Config,
    *,
    device: torch.device,
    sigma: torch.Tensor,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
    noise_seed_offset: int = 30_000,
) -> float:
    values: list[float] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        h_true = sample_gaussian_h(
            cfg.n_antennas,
            cfg.n_subcarriers,
            l_space,
            l_freq,
            device=device,
            dtype=cfg.dtype,
            generator=gen_ch,
        )
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_seed_offset + mc)
        values.append(
            full_subcarrier_lmmse_nmse(h_true, sigma, cfg, device=device, generator=gen_noise)
        )
    return float(np.mean(values))


def plot_exp1(
    curves: dict[str, np.ndarray],
    cfg: Exp1Config,
    out_path: Path,
    *,
    full_lmmse_nmse: float,
    title: str | None = None,
    extra_styles: dict[str, tuple[str, str, str]] | None = None,
) -> None:
    """Input: name -> mean est. curve, config, path, full-LMMSE value. Output: saves PNG (no threshold line)."""
    styles = {**POLICY_STYLES, **(extra_styles or {})}
    if title is None:
        title = (
            f"exp1 pilot policy  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
            f"n_mc={cfg.n_mc}, pilots={cfg.target_pilots}"
        )
    plot_threshold_curves(
        curves,
        plot_len=max(len(c) for c in curves.values()),
        out_path=out_path,
        title=title,
        styles=styles,
        ylabel="Mean estimated MSE  tr(P)/N",
        extra_hlines=[
            (full_lmmse_nmse, FULL_LMMSE_COLOR, FULL_LMMSE_LINESTYLE, f"full LMMSE (Nc={cfg.n_subcarriers})"),
        ],
    )


NOISE_SEED_OFFSET = {"fixed": 0, "random": 10_000, "active": 20_000}


def compute_exp1_curves(
    cfg: Exp1Config, noise_seed_offsets: dict[str, int]
) -> tuple[dict[str, np.ndarray], float]:
    """Input: config, policy name -> noise seed offset. Output: name -> mean est. curve, full-LMMSE est. NMSE."""
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    sigma, l_space, l_freq = build_sigma_kron(
        cfg.n_antennas,
        cfg.n_subcarriers,
        cfg.rho_space,
        cfg.rho_freq,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )

    policies = {
        name: _run_mc_policy(
            name, cfg, device=device, sigma=sigma, l_space=l_space, l_freq=l_freq, noise_seed_offset=offset
        )
        for name, offset in noise_seed_offsets.items()
    }

    curves = {name: mean_curve(traces) for name, traces in policies.items()}
    for name, traces in policies.items():
        mean_pilots = mean_pilots_to_threshold(traces, cfg.nmse_threshold)
        print(
            f"{name}: mean pilots to est. MSE <= {cfg.nmse_threshold} = {mean_pilots:.2f}, "
            f"est. MSE at {len(curves[name])} pilots = {curves[name][-1]:.4e}"
        )

    full_lmmse_nmse = _run_mc_full_lmmse_nmse(
        cfg, device=device, sigma=sigma, l_space=l_space, l_freq=l_freq
    )
    print(f"full LMMSE (Nc={cfg.n_subcarriers}): mean estimated NMSE = {full_lmmse_nmse:.6f}")
    return curves, full_lmmse_nmse


def run_exp1(cfg: Exp1Config) -> None:
    curves, full_lmmse_nmse = compute_exp1_curves(cfg, NOISE_SEED_OFFSET)
    out_path = _NEW_ROOT / "figures" / "exp1_pilot_policy.png"
    plot_exp1(curves, cfg, out_path, full_lmmse_nmse=full_lmmse_nmse)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp1: fixed vs random vs active pilot policies (Gaussian)")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--nmse-threshold", type=float, default=None)
    p.add_argument("--target-pilots", type=int, default=None, help="Pilots per run (default 32; 0 -> stop on threshold).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Kronecker Sigma (default 1e-9).")
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
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots or None
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    run_exp1(cfg)


if __name__ == "__main__":
    main()
