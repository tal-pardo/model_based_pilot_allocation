from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from channel_generators.sionna import (
    SionnaOFDMGrid,
    estimate_empirical_sigma_tdl_a,
    sample_tdl_a_channel,
    vec_from_h,
)
from config import Exp3Config
from experiments.common import (
    build_exp3_curves,
    make_policy,
    print_exp3_final_gaps,
    run_mc_mean_full_true_baselines_by_policy,
    save_exp3_gaussian_figure,
    save_exp3_tdl_figure,
)
from simulation import RunTrace, run_until_threshold
from utils import resolve_device, set_seed

POLICIES = ("fixed", "active")
NOISE_SEED_OFFSET = {"fixed": 0, "active": 10_000}
COV_SEED_OFFSET = 1_000_000


def default_config() -> Exp3Config:
    return Exp3Config()


def _sample_mc_channels(
    cfg: Exp3Config,
    *,
    device: torch.device,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
    grid: SionnaOFDMGrid,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    h_gaussian: list[torch.Tensor] = []
    h_tdl: list[torch.Tensor] = []
    for mc in range(cfg.n_mc):
        gen_ch = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        h_gaussian.append(
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
    return h_gaussian, h_tdl


def _run_mc_policy(
    cfg: Exp3Config,
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
    cfg: Exp3Config,
    *,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    device: torch.device,
) -> dict[str, list[RunTrace]]:
    return {
        policy: _run_mc_policy(cfg, policy_name=policy, h_list=h_list, sigma=sigma, device=device)
        for policy in POLICIES
    }


def run_exp3(cfg: Exp3Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.target_pilots is None:
        raise ValueError("exp3 requires target_pilots (default 16).")

    sigma_kron, l_space, l_freq = build_sigma_kron(
        cfg.n_antennas,
        cfg.n_subcarriers,
        cfg.rho_space,
        cfg.rho_freq,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    h_gaussian, h_tdl = _sample_mc_channels(
        cfg, device=device, l_space=l_space, l_freq=l_freq, grid=grid
    )

    figures_dir = _NEW_ROOT / "figures"

    gaussian_traces = _run_policies(cfg, h_list=h_gaussian, sigma=sigma_kron, device=device)
    gaussian_curves = build_exp3_curves(gaussian_traces)
    gaussian_baselines = run_mc_mean_full_true_baselines_by_policy(
        cfg, h_gaussian, sigma_kron, device=device, noise_seed_offset_by_policy=NOISE_SEED_OFFSET
    )
    for policy, mse in gaussian_baselines.items():
        print(
            f"Gaussian Kronecker full batch true MSE ({policy}, Nc={cfg.n_subcarriers}, "
            f"seed+{NOISE_SEED_OFFSET[policy]}): {mse:.6e}"
        )
    print_exp3_final_gaps(gaussian_curves, cfg, label="Gaussian Kronecker")
    save_exp3_gaussian_figure(
        gaussian_curves,
        cfg=cfg,
        full_lmmse_baselines=gaussian_baselines,
        out_path=figures_dir / "exp3_gaussian_err.png",
    )

    print(
        f"Estimating empirical Sigma from {cfg.n_cov_mc} TDL-A draws "
        f"(seed offset {COV_SEED_OFFSET}, reg_empirical={cfg.reg_empirical:g})..."
    )
    sigma_hat_tdl = estimate_empirical_sigma_tdl_a(
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        rho_space=cfg.rho_space,
        n_cov_mc=cfg.n_cov_mc,
        seed=cfg.seed,
        seed_offset=COV_SEED_OFFSET,
        reg_empirical=cfg.reg_empirical,
        grid=grid,
        device=device,
        dtype=cfg.dtype,
    )

    tdl_panels: list[tuple[str, dict, dict[str, float]]] = []
    for panel_title, sigma in (
        ("Kronecker Sigma", sigma_kron),
        (f"Empirical Sigma (n={cfg.n_cov_mc}, reg={cfg.reg_empirical:g})", sigma_hat_tdl),
    ):
        traces = _run_policies(cfg, h_list=h_tdl, sigma=sigma, device=device)
        curves = build_exp3_curves(traces)
        baselines = run_mc_mean_full_true_baselines_by_policy(
            cfg, h_tdl, sigma, device=device, noise_seed_offset_by_policy=NOISE_SEED_OFFSET
        )
        for policy, mse in baselines.items():
            print(
                f"TDL-A {panel_title} full batch true MSE ({policy}, Nc={cfg.n_subcarriers}, "
                f"seed+{NOISE_SEED_OFFSET[policy]}): {mse:.6e}"
            )
        print_exp3_final_gaps(curves, cfg, label=f"TDL-A {panel_title}")
        tdl_panels.append((panel_title, curves, baselines))

    save_exp3_tdl_figure(tdl_panels, cfg=cfg, out_path=figures_dir / "exp3_tdl_err.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp3: true vs estimated MSE (Gaussian + TDL-A)")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--n-cov-mc", type=int, default=None, help="TDL-A samples for empirical Sigma (default 300).")
    p.add_argument("--target-pilots", type=int, default=None, help="Fixed pilot count per run (default 16).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Kronecker Sigma (default 1e-9).")
    p.add_argument(
        "--reg-empirical",
        type=float,
        default=None,
        help="Ridge on empirical Sigma_hat (default 1e-3).",
    )
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
    if args.n_cov_mc is not None:
        cfg.n_cov_mc = args.n_cov_mc
    if args.target_pilots is not None:
        cfg.target_pilots = args.target_pilots
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    if args.reg_empirical is not None:
        cfg.reg_empirical = args.reg_empirical
    run_exp3(cfg)


if __name__ == "__main__":
    main()
