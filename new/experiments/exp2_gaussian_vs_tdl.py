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
from channel_generators.sionna import SionnaOFDMGrid, sample_tdl_a_channel, vec_from_h
from config import Exp2Config
from experiments.common import (
    make_policy,
    mean_curve,
    mean_pilots_to_threshold,
    plot_threshold_curves,
    run_mc_mean_full_true_lmmse_mse,
)
from simulation import RunTrace, run_until_threshold
from utils import empirical_covariance, resolve_device, set_seed

FAMILIES = ("gaussian", "tdl")
POLICIES = ("fixed", "active")
NOISE_SEED_OFFSET = {"fixed": 0, "active": 10_000}

COV_SEED_OFFSET_FIG2 = 1_000_000
COV_SEED_OFFSET_TDL_COMPARE = (2_000_000, 3_000_000)

EXP2_STYLES = {
    "gaussian (fixed)": ("C0", "-", "o"),
    "gaussian (active)": ("C0", "--", "s"),
    "tdl (fixed)": ("C1", "-", "o"),
    "tdl (active)": ("C1", "--", "s"),
}

TDL_SIGMA_COLORS = ("C0", "C1", "C2")


def default_config() -> Exp2Config:
    return Exp2Config()


def _sample_mc_channels(
    cfg: Exp2Config,
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


def _estimate_sigma_hat_tdl_a(
    cfg: Exp2Config,
    *,
    n_cov_mc: int,
    seed_offset: int,
    grid: SionnaOFDMGrid,
    device: torch.device,
) -> torch.Tensor:
    """Empirical Sigma from hold-out TDL-A draws (disjoint seed block per estimate)."""
    n = cfg.n_antennas * cfg.n_subcarriers
    samples = torch.zeros((n_cov_mc, n), device=device, dtype=cfg.dtype)
    for k in range(n_cov_mc):
        h_mat = sample_tdl_a_channel(
            n_antennas=cfg.n_antennas,
            n_subcarriers=cfg.n_subcarriers,
            rho_space=cfg.rho_space,
            grid=grid,
            device=device,
            dtype=cfg.dtype,
            seed=cfg.seed + seed_offset + k,
        )
        samples[k] = vec_from_h(h_mat).squeeze(-1)
    return empirical_covariance(
        samples, reg=cfg.reg_empirical, device=device, dtype=cfg.dtype
    )


def _run_mc_family_policy(
    cfg: Exp2Config,
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


def _build_exp2_curves(traces_by_key: dict[tuple[str, str], list[RunTrace]]) -> dict[str, np.ndarray]:
    """Four true-MSE curves: family x policy."""
    curves: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        for policy in POLICIES:
            key = f"{family} ({policy})"
            curves[key] = mean_curve(traces_by_key[(family, policy)], field="nmse_true")
    return curves


def _build_tdl_sigma_curves(
    traces_by_key: dict[tuple[str, str], list[RunTrace]],
    sigma_labels: tuple[str, ...],
) -> dict[str, np.ndarray]:
    curves: dict[str, np.ndarray] = {}
    for sigma_label in sigma_labels:
        for policy in POLICIES:
            key = f"{sigma_label} ({policy})"
            curves[key] = mean_curve(traces_by_key[(sigma_label, policy)], field="nmse_true")
    return curves


def _stop_label(cfg: Exp2Config) -> str:
    if cfg.target_pilots is not None:
        return f"target_pilots={cfg.target_pilots}"
    return f"stop thresh={cfg.nmse_threshold} (est.)"


def _plot_len(cfg: Exp2Config) -> int | None:
    return cfg.target_pilots


def _full_true_lmmse_hlines(
    cfg: Exp2Config,
    *,
    h_by_key: dict[str, list[torch.Tensor]],
    sigma_by_key: dict[str, torch.Tensor],
    device: torch.device,
    color_by_key: dict[str, str] | None = None,
) -> list[tuple[float, str, str, str]]:
    hlines: list[tuple[float, str, str, str]] = []
    for key in h_by_key:
        mse = run_mc_mean_full_true_lmmse_mse(
            cfg, h_by_key[key], sigma_by_key[key], device=device
        )
        color = color_by_key[key] if color_by_key else "red"
        hlines.append((mse, color, ":", f"full true LMMSE ({key}, Nc={cfg.n_subcarriers})"))
    return hlines


def _run_figure(
    cfg: Exp2Config,
    *,
    figure_name: str,
    prior_label: str,
    sigma_by_family: dict[str, torch.Tensor],
    h_by_family: dict[str, list[torch.Tensor]],
    device: torch.device,
    out_path: Path,
) -> None:
    traces_by_key: dict[tuple[str, str], list[RunTrace]] = {}
    for family in FAMILIES:
        sigma = sigma_by_family[family]
        for policy in POLICIES:
            traces = _run_mc_family_policy(
                cfg,
                policy_name=policy,
                h_list=h_by_family[family],
                sigma=sigma,
                device=device,
            )
            traces_by_key[(family, policy)] = traces
            est_pilots = mean_pilots_to_threshold(traces, cfg.nmse_threshold)
            print(
                f"{figure_name} {family} ({policy}): "
                f"mean pilots to est. threshold = {est_pilots:.2f}"
            )

    curves = _build_exp2_curves(traces_by_key)
    title = (
        f"exp2 {figure_name}  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, {_stop_label(cfg)}  prior: {prior_label}"
    )
    full_hlines = _full_true_lmmse_hlines(
        cfg,
        h_by_key={f: h_by_family[f] for f in FAMILIES},
        sigma_by_key={f: sigma_by_family[f] for f in FAMILIES},
        device=device,
    )
    plot_threshold_curves(
        curves,
        threshold=None,
        plot_len=_plot_len(cfg),
        out_path=out_path,
        title=title,
        styles=EXP2_STYLES,
        ylabel="Mean true MSE  (1/N)||h-hat - h||^2",
        extra_hlines=full_hlines,
    )


def _tdl_sigma_compare_styles(sigma_labels: tuple[str, ...]) -> dict[str, tuple[str, str, str]]:
    styles: dict[str, tuple[str, str, str]] = {}
    for i, label in enumerate(sigma_labels):
        color = TDL_SIGMA_COLORS[i % len(TDL_SIGMA_COLORS)]
        styles[f"{label} (fixed)"] = (color, "-", "o")
        styles[f"{label} (active)"] = (color, "--", "s")
    return styles


def _run_tdl_sigma_compare_figure(
    cfg: Exp2Config,
    *,
    h_tdl: list[torch.Tensor],
    sigma_kron: torch.Tensor,
    sigma_by_label: dict[str, torch.Tensor],
    device: torch.device,
    out_path: Path,
) -> None:
    sigma_modes = {"Kronecker": sigma_kron, **sigma_by_label}
    sigma_labels = tuple(sigma_modes.keys())
    traces_by_key: dict[tuple[str, str], list[RunTrace]] = {}

    for label, sigma in sigma_modes.items():
        for policy in POLICIES:
            traces = _run_mc_family_policy(
                cfg,
                policy_name=policy,
                h_list=h_tdl,
                sigma=sigma,
                device=device,
            )
            traces_by_key[(label, policy)] = traces
            est_pilots = mean_pilots_to_threshold(traces, cfg.nmse_threshold)
            print(
                f"tdl_sigma_compare {label} ({policy}): "
                f"mean pilots to est. threshold = {est_pilots:.2f}"
            )

    curves = _build_tdl_sigma_curves(traces_by_key, sigma_labels)
    emp_sizes = ", ".join(str(n) for n in cfg.tdl_empirical_cov_sizes)
    title = (
        f"exp2 tdl_sigma_compare  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, {_stop_label(cfg)}  TDL-A; empirical n_cov in ({emp_sizes})"
    )
    color_by_key = {
        label: TDL_SIGMA_COLORS[i % len(TDL_SIGMA_COLORS)]
        for i, label in enumerate(sigma_labels)
    }
    full_hlines = _full_true_lmmse_hlines(
        cfg,
        h_by_key={label: h_tdl for label in sigma_labels},
        sigma_by_key=sigma_modes,
        device=device,
        color_by_key=color_by_key,
    )
    plot_threshold_curves(
        curves,
        threshold=None,
        plot_len=_plot_len(cfg),
        out_path=out_path,
        title=title,
        styles=_tdl_sigma_compare_styles(sigma_labels),
        ylabel="Mean true MSE  (1/N)||h-hat - h||^2",
        x_tick_step=5,
        extra_hlines=full_hlines,
    )


def run_exp2(cfg: Exp2Config) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
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
    h_gaussian, h_tdl = _sample_mc_channels(cfg, device=device, l_space=l_space, l_freq=l_freq, grid=grid)

    print(f"Estimating empirical Sigma from {cfg.n_cov_mc} TDL-A draws (seed offset {COV_SEED_OFFSET_FIG2})...")
    sigma_hat_tdl = _estimate_sigma_hat_tdl_a(
        cfg,
        n_cov_mc=cfg.n_cov_mc,
        seed_offset=COV_SEED_OFFSET_FIG2,
        grid=grid,
        device=device,
    )

    h_by_family = {"gaussian": h_gaussian, "tdl": h_tdl}
    figures_dir = _NEW_ROOT / "figures"

    sigma_kron_all = {"gaussian": sigma_kron, "tdl": sigma_kron}
    _run_figure(
        cfg,
        figure_name="gaussian_sigma",
        prior_label="Kronecker Sigma for all",
        sigma_by_family=sigma_kron_all,
        h_by_family=h_by_family,
        device=device,
        out_path=figures_dir / "exp2_gaussian_sigma.png",
    )

    sigma_emp = {"gaussian": sigma_kron, "tdl": sigma_hat_tdl}
    _run_figure(
        cfg,
        figure_name="empirical_sigma",
        prior_label=f"Kronecker (Gaussian, reg={cfg.reg_kron:g}), "
        f"empirical Sigma (TDL-A, n_cov={cfg.n_cov_mc}, reg={cfg.reg_empirical:g})",
        sigma_by_family=sigma_emp,
        h_by_family=h_by_family,
        device=device,
        out_path=figures_dir / "exp2_empirical_sigma.png",
    )

    if len(cfg.tdl_empirical_cov_sizes) != 2:
        raise ValueError(
            f"tdl_empirical_cov_sizes must have length 2 for tdl_sigma_compare figure; "
            f"got {cfg.tdl_empirical_cov_sizes}."
        )
    sigma_by_label: dict[str, torch.Tensor] = {}
    for n_cov, seed_offset in zip(cfg.tdl_empirical_cov_sizes, COV_SEED_OFFSET_TDL_COMPARE):
        print(f"Estimating empirical Sigma from {n_cov} TDL-A draws (seed offset {seed_offset})...")
        sigma_by_label[f"emp n={n_cov}"] = _estimate_sigma_hat_tdl_a(
            cfg,
            n_cov_mc=n_cov,
            seed_offset=seed_offset,
            grid=grid,
            device=device,
        )

    _run_tdl_sigma_compare_figure(
        cfg,
        h_tdl=h_tdl,
        sigma_kron=sigma_kron,
        sigma_by_label=sigma_by_label,
        device=device,
        out_path=figures_dir / "exp2_tdl_sigma_compare.png",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp2: Gaussian vs TDL-A, fixed vs active, two Sigma modes")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument(
        "--n-cov-mc",
        type=int,
        default=None,
        help="TDL-A samples for empirical Sigma_hat in fig2 (default 300, hold-out from eval MC).",
    )
    p.add_argument(
        "--tdl-empirical-cov-sizes",
        type=str,
        default=None,
        help="Comma-separated n_cov for fig3 TDL Sigma compare (default 250,500).",
    )
    p.add_argument("--nmse-threshold", type=float, default=None)
    p.add_argument(
        "--target-pilots",
        type=int,
        default=None,
        help="Fixed pilot count per run (exp2 default 16). Pass 0 to use estimated-NMSE threshold stop.",
    )
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
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.n_cov_mc is not None:
        cfg.n_cov_mc = args.n_cov_mc
    if args.tdl_empirical_cov_sizes is not None:
        parts = [int(x.strip()) for x in args.tdl_empirical_cov_sizes.split(",")]
        cfg.tdl_empirical_cov_sizes = tuple(parts)
    if args.n_subcarriers is not None:
        cfg.n_subcarriers = args.n_subcarriers
    if args.nmse_threshold is not None:
        cfg.nmse_threshold = args.nmse_threshold
    if args.target_pilots is not None:
        cfg.target_pilots = None if args.target_pilots == 0 else args.target_pilots
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    if args.reg_empirical is not None:
        cfg.reg_empirical = args.reg_empirical
    run_exp2(cfg)


if __name__ == "__main__":
    main()
