from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.compound_gaussian import sample_compound_gaussian_h
from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from config import Exp4Config
from experiments.common import (
    build_exp3_curves,
    make_policy,
    print_exp3_final_gaps,
    run_mc_mean_full_true_baselines_by_policy,
    save_exp_multi_panel_err_figure,
)
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

POLICIES = ("fixed", "active")
NOISE_SEED_OFFSET = {"fixed": 0, "active": 10_000}
COMPOUND_SEED_OFFSET = 200_000


def default_config() -> Exp4Config:
    return Exp4Config()


def _sample_mc_gaussian_channels(
    cfg: Exp4Config,
    *,
    device: torch.device,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        channels.append(
            sample_gaussian_h(
                cfg.n_antennas,
                cfg.n_subcarriers,
                l_space,
                l_freq,
                device=device,
                dtype=cfg.dtype,
                generator=gen_ch,
            )
        )
    return channels


def _sample_mc_compound_gaussian_channels(
    cfg: Exp4Config,
    *,
    device: torch.device,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + COMPOUND_SEED_OFFSET + mc)
        channels.append(
            sample_compound_gaussian_h(
                cfg.n_antennas,
                cfg.n_subcarriers,
                l_space,
                l_freq,
                texture_alpha=cfg.texture_alpha,
                device=device,
                dtype=cfg.dtype,
                generator=gen_ch,
            )
        )
    return channels


def _run_mc_policy(
    cfg: Exp4Config,
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
    cfg: Exp4Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> dict[str, list[RunTrace]]:
    return {
        policy: _run_mc_policy(cfg, policy_name=policy, h_list=h_list, sigma=sigma, device=device)
        for policy in POLICIES
    }


def _run_family_panel(
    cfg: Exp4Config,
    *,
    label: str,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    traces = _run_policies(cfg, h_list=h_list, sigma=sigma, device=device)
    curves = build_exp3_curves(traces)
    baselines = run_mc_mean_full_true_baselines_by_policy(
        cfg, h_list, sigma, device=device, noise_seed_offset_by_policy=NOISE_SEED_OFFSET
    )
    for policy, mse in baselines.items():
        print(
            f"{label} full batch true MSE ({policy}, Nc={cfg.n_subcarriers}, "
            f"seed+{NOISE_SEED_OFFSET[policy]}): {mse:.6e}"
        )
    print_exp3_final_gaps(curves, cfg, label=label)
    return curves, baselines


def run_exp4(cfg: Exp4Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp4 requires target_pilots (default 16).")

    sigma, l_space, l_freq = build_sigma_kron(
        cfg.n_antennas,
        cfg.n_subcarriers,
        cfg.rho_space,
        cfg.rho_freq,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )

    h_gaussian = _sample_mc_gaussian_channels(cfg, device=device, l_space=l_space, l_freq=l_freq)
    h_compound = _sample_mc_compound_gaussian_channels(
        cfg, device=device, l_space=l_space, l_freq=l_freq
    )

    gaussian_curves, gaussian_baselines = _run_family_panel(
        cfg,
        label="Gaussian Kronecker",
        h_list=h_gaussian,
        sigma=sigma,
        device=device,
    )
    compound_curves, compound_baselines = _run_family_panel(
        cfg,
        label=f"Compound-Gaussian alpha={cfg.texture_alpha:g}",
        h_list=h_compound,
        sigma=sigma,
        device=device,
    )

    target = cfg.target_pilots or cfg.max_pilots
    suptitle = (
        f"exp4 known Sigma  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, k0={cfg.initial_pilot_subcarriers}, pilots={target}"
    )
    panels = [
        ("Gaussian (control)", gaussian_curves, gaussian_baselines),
        (
            f"Compound-Gaussian (alpha={cfg.texture_alpha:g})",
            compound_curves,
            compound_baselines,
        ),
    ]
    save_exp_multi_panel_err_figure(
        panels,
        cfg=cfg,
        out_path=_NEW_ROOT / "figures" / "exp4_known_sigma.png",
        suptitle=suptitle,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp4: true vs estimated MSE with known matched Sigma")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--target-pilots", type=int, default=None, help="Fixed pilot count per run (default 16).")
    p.add_argument("--texture-alpha", type=float, default=None, help="Gamma texture shape (default 1.0).")
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
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots
    if args.texture_alpha is not None:
        cfg.texture_alpha = args.texture_alpha
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    run_exp4(cfg)


if __name__ == "__main__":
    main()
