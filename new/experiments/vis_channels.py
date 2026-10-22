from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from channel_generators.sionna import (
    SionnaOFDMGrid,
    sample_tdl_a_channel,
    sample_tdl_a_time_freq_channel,
    sample_tdl_c_channel,
    sample_tdl_c_time_freq_channel,
)
from config import SimConfig
from utils import resolve_device, set_seed

TDL_SEED_OFFSET = 100_000

CHANNEL_ALIASES: dict[str, str] = {
    "tdl": "tdl_a",
}

CHANNEL_TITLES: dict[str, str] = {
    "gaussian": "Gaussian (Kronecker prior draw)",
    "tdl_a": "TDL-A (Sionna, power-normalized)",
    "tdl_c": "TDL-C (Sionna, power-normalized)",
}


@dataclass
class ChannelVisConfig(SimConfig):
    n_subcarriers: int = 1024
    n_realizations: int = 5
    out_dir: Path = _NEW_ROOT / "figures"
    n_symbols: int = 16
    speed: float = 30.0


def h_mat_from_vec(h: torch.Tensor, n_antennas: int, n_subcarriers: int) -> torch.Tensor:
    """Input: vec(H) (N,1). Output: H (Na,Nc) column-stacked by subcarrier."""
    return h.squeeze(-1).view(n_subcarriers, n_antennas).T


def to_angle_delay_domain(h: torch.Tensor) -> torch.Tensor:
    """Input: H (Na,Nc) space-frequency. Output: H_ad (Na,Nc) angle-delay.
    Antenna axis -> angle via orthonormal FFT (fftshift to center angle 0).
    Subcarrier axis -> delay via orthonormal IFFT (causal, no shift)."""
    h_angle = torch.fft.fftshift(torch.fft.fft(h, dim=0, norm="ortho"), dim=0)
    return torch.fft.ifft(h_angle, dim=1, norm="ortho")


def _sample_gaussian_realizations(
    cfg: ChannelVisConfig,
    *,
    device: torch.device,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_realizations):
        gen = torch.Generator(device=device).manual_seed(cfg.seed + mc)
        h_vec = sample_gaussian_h(
            cfg.n_antennas,
            cfg.n_subcarriers,
            l_space,
            l_freq,
            device=device,
            dtype=cfg.dtype,
            generator=gen,
        )
        channels.append(h_mat_from_vec(h_vec, cfg.n_antennas, cfg.n_subcarriers))
    return channels


def _sample_tdl_realizations(
    cfg: ChannelVisConfig,
    *,
    device: torch.device,
    grid: SionnaOFDMGrid,
    sampler: Callable[..., torch.Tensor],
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_realizations):
        h_mat = sampler(
            n_antennas=cfg.n_antennas,
            n_subcarriers=cfg.n_subcarriers,
            rho_space=cfg.rho_space,
            grid=grid,
            device=device,
            dtype=cfg.dtype,
            seed=cfg.seed + TDL_SEED_OFFSET + mc,
        )
        channels.append(h_mat)
    return channels


TIME_FREQ_SAMPLERS: dict[str, Callable[..., torch.Tensor]] = {
    "tdl_a": sample_tdl_a_time_freq_channel,
    "tdl_c": sample_tdl_c_time_freq_channel,
}


def _sample_tdl_time_freq_realizations(
    cfg: ChannelVisConfig,
    *,
    device: torch.device,
    grid: SionnaOFDMGrid,
    sampler: Callable[..., torch.Tensor],
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(cfg.n_realizations):
        h_tf = sampler(
            n_symbols=cfg.n_symbols,
            n_subcarriers=cfg.n_subcarriers,
            grid=grid,
            device=device,
            speed=cfg.speed,
            dtype=cfg.dtype,
            seed=cfg.seed + TDL_SEED_OFFSET + mc,
        )
        channels.append(h_tf)
    return channels


def resolve_channel_name(name: str) -> str:
    """Input: CLI channel name. Output: canonical registry key."""
    key = name.lower().strip()
    return CHANNEL_ALIASES.get(key, key)


def list_channel_types() -> tuple[str, ...]:
    return tuple(CHANNEL_TITLES.keys())


def sample_channel_realizations(
    channel: str,
    cfg: ChannelVisConfig,
    *,
    device: torch.device,
    l_space: torch.Tensor | None = None,
    l_freq: torch.Tensor | None = None,
    grid: SionnaOFDMGrid | None = None,
) -> list[torch.Tensor]:
    """
    Input: canonical channel type, vis config, device, optional prebuilt factors/grid.
    Output: list of H (Na,Nc) matrices, same sampling as exp1/exp2.
    """
    if channel == "gaussian":
        if l_space is None or l_freq is None:
            raise ValueError("Gaussian sampling requires l_space and l_freq.")
        return _sample_gaussian_realizations(cfg, device=device, l_space=l_space, l_freq=l_freq)
    if channel == "tdl_a":
        if grid is None:
            raise ValueError("TDL sampling requires grid.")
        return _sample_tdl_realizations(cfg, device=device, grid=grid, sampler=sample_tdl_a_channel)
    if channel == "tdl_c":
        if grid is None:
            raise ValueError("TDL sampling requires grid.")
        return _sample_tdl_realizations(cfg, device=device, grid=grid, sampler=sample_tdl_c_channel)
    raise ValueError(
        f"Unknown channel type {channel!r}. Choose from: {', '.join(list_channel_types())} "
        f"(alias: tdl -> tdl_a)."
    )


DOMAIN_CHOICES: tuple[str, ...] = ("space-freq", "angle-delay", "both", "freq-time")

DOMAIN_SPECS: dict[str, dict[str, str]] = {
    "space-freq": {
        "y_label": "antenna i",
        "x_label": "subcarrier k",
        "cbar_label": r"$|H_{i,k}|$",
    },
    "angle-delay": {
        "y_label": "angle bin",
        "x_label": "delay tap",
        "cbar_label": r"$|H^{ad}_{i,k}|$ (dB)",
    },
    "freq-time": {
        "y_label": "OFDM symbol l",
        "x_label": "subcarrier k",
        "cbar_label": r"$|H_{l,k}|$",
    },
}


def resolve_domains(domain: str) -> list[str]:
    """Input: CLI --domain value. Output: ordered list of domain rows to render."""
    if domain == "both":
        return ["space-freq", "angle-delay"]
    if domain not in DOMAIN_SPECS:
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {', '.join(DOMAIN_CHOICES)}.")
    return [domain]


def _space_freq_magnitudes(channels: list[torch.Tensor]) -> list[np.ndarray]:
    return [h.abs().detach().cpu().to(torch.float32).numpy() for h in channels]


def _angle_delay_db_magnitudes(channels: list[torch.Tensor], *, db_floor: float) -> list[np.ndarray]:
    mags = [to_angle_delay_domain(h).abs().detach().cpu().to(torch.float32).numpy() for h in channels]
    global_max = max(float(m.max()) for m in mags)
    if global_max <= 0.0:
        raise ValueError("Angle-delay magnitude is all-zero; cannot convert to dB.")
    floor_ratio = 10.0 ** (db_floor / 20.0)
    return [20.0 * np.log10(np.maximum(m / global_max, floor_ratio)) for m in mags]


def plot_domain_panels(
    *,
    title: str,
    channels: list[torch.Tensor],
    domains: list[str],
    out_path: Path,
    panel_title_fmt: str = "realization {idx}",
    db_floor: float = -40.0,
) -> None:
    """Input: title, list of H (Na,Nc), domain rows to render, output path.
    Output: saves one PNG with one row of magnitude heatmaps per domain."""
    if not channels:
        raise ValueError("channels must be non-empty.")
    if not domains:
        raise ValueError("domains must be non-empty.")

    n_cols = len(channels)
    n_rows = len(domains)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 3.4 * n_rows), squeeze=False)

    for row, domain in enumerate(domains):
        spec = DOMAIN_SPECS[domain]
        if domain in ("space-freq", "freq-time"):
            magnitudes = _space_freq_magnitudes(channels)
            vmin = min(float(m.min()) for m in magnitudes)
            vmax = max(float(m.max()) for m in magnitudes)
        else:
            magnitudes = _angle_delay_db_magnitudes(channels, db_floor=db_floor)
            vmin, vmax = db_floor, 0.0
        if vmin == vmax:
            vmax = vmin + 1.0

        im = None
        for col, mag in enumerate(magnitudes):
            ax = axes[row, col]
            im = ax.imshow(mag, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(panel_title_fmt.format(idx=col + 1))
            ax.set_xlabel(spec["x_label"])
            if col == 0:
                ax.set_ylabel(spec["y_label"])
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.03, pad=0.02, label=spec["cbar_label"])

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.88 if n_rows == 1 else 0.92, wspace=0.28, hspace=0.45)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_channel(
    channel: str,
    cfg: ChannelVisConfig,
    *,
    device: torch.device,
    l_space: torch.Tensor | None = None,
    l_freq: torch.Tensor | None = None,
    grid: SionnaOFDMGrid | None = None,
    domain: str = "space-freq",
) -> Path:
    """Input: channel type, config, device, domain(s) to render. Output: path to saved figure."""
    domains = resolve_domains(domain)
    label = CHANNEL_TITLES[channel]

    if domains == ["freq-time"]:
        if channel not in TIME_FREQ_SAMPLERS:
            raise ValueError(
                f"freq-time domain only supports {', '.join(TIME_FREQ_SAMPLERS)}, got {channel!r}."
            )
        if grid is None:
            raise ValueError("TDL sampling requires grid.")
        channels = _sample_tdl_time_freq_realizations(
            cfg, device=device, grid=grid, sampler=TIME_FREQ_SAMPLERS[channel]
        )
        subtitle = f"L={cfg.n_symbols} symbols, Nc={cfg.n_subcarriers}, speed={cfg.speed} m/s"
        out_path = cfg.out_dir / f"{channel}_freq_time_vis.png"
    else:
        channels = sample_channel_realizations(
            channel,
            cfg,
            device=device,
            l_space=l_space,
            l_freq=l_freq,
            grid=grid,
        )
        na, nc = cfg.n_antennas, cfg.n_subcarriers
        if channel == "gaussian":
            subtitle = f"Na={na}, Nc={nc}, rho_space={cfg.rho_space}, rho_freq={cfg.rho_freq}"
        else:
            subtitle = f"Na={na}, Nc={nc}, rho_space={cfg.rho_space}"
        out_path = cfg.out_dir / f"{channel}_vis.png"

    plot_domain_panels(
        title=f"{label}  {subtitle}",
        channels=channels,
        domains=domains,
        out_path=out_path,
    )
    return out_path


def run_vis(cfg: ChannelVisConfig, channels: list[str], *, domain: str = "space-freq") -> list[Path]:
    """Input: config, list of canonical channel types. Output: saved figure paths."""
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    if cfg.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    _, l_space, l_freq = build_sigma_kron(
        cfg.n_antennas,
        cfg.n_subcarriers,
        cfg.rho_space,
        cfg.rho_freq,
        device=device,
        dtype=cfg.dtype,
        reg=cfg.reg_kron,
    )
    grid = SionnaOFDMGrid(
        fft_size=cfg.n_subcarriers,
        subcarrier_spacing=15e3,
        carrier_frequency=3.5e9,
        delay_spread=300e-9,
    )

    out_paths: list[Path] = []
    for channel in channels:
        path = visualize_channel(
            channel,
            cfg,
            device=device,
            l_space=l_space,
            l_freq=l_freq,
            grid=grid,
            domain=domain,
        )
        print(f"Saved {path}")
        out_paths.append(path)
    return out_paths


def default_config() -> ChannelVisConfig:
    return ChannelVisConfig()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize channel magnitude samples (same generators as simulation)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--channel",
        type=str,
        help=f"Channel type: {', '.join(list_channel_types())} (alias: tdl -> tdl_a)",
    )
    g.add_argument("--all", action="store_true", help="Generate figures for all supported channel types.")
    p.add_argument("--n-antennas", type=int, default=None)
    p.add_argument("--n-subcarriers", type=int, default=None)
    p.add_argument("--n-realizations", type=int, default=None, help="Number of random samples (default 5).")
    p.add_argument("--rho-space", type=float, default=None)
    p.add_argument("--rho-freq", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--reg-kron", type=float, default=None, help="Ridge on Kronecker Sigma (default 1e-9).")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default new/figures/).")
    p.add_argument(
        "--domain",
        type=str,
        default="space-freq",
        choices=list(DOMAIN_CHOICES),
        help=(
            "Domain row(s) to render: space-freq (default), angle-delay, both, or "
            "freq-time (single-antenna OFDM slot over symbols x subcarriers; TDL only)."
        ),
    )
    p.add_argument(
        "--n-symbols",
        type=int,
        default=None,
        help="freq-time domain only: number of OFDM symbols L (default 16).",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "freq-time domain only: UE speed in m/s driving Doppler (default 30.0, i.e. "
            "~108 km/h vehicular -- pedestrian speeds show negligible decorrelation over "
            "16 symbols/~1ms; 0 = static)."
        ),
    )
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_antennas is not None:
        cfg.n_antennas = args.n_antennas
    if args.n_subcarriers is not None:
        cfg.n_subcarriers = args.n_subcarriers
    elif args.domain == "freq-time":
        # freq-time mirrors the sim grid (Nc=32) by default, unlike the 1024-subcarrier
        # default used for space-freq/angle-delay frequency resolution.
        cfg.n_subcarriers = SimConfig().n_subcarriers
    if args.n_realizations is not None:
        cfg.n_realizations = args.n_realizations
    if args.rho_space is not None:
        cfg.rho_space = args.rho_space
    if args.rho_freq is not None:
        cfg.rho_freq = args.rho_freq
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.reg_kron is not None:
        cfg.reg_kron = args.reg_kron
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.n_symbols is not None:
        cfg.n_symbols = args.n_symbols
    if args.speed is not None:
        cfg.speed = args.speed

    if args.all:
        channels = list(TIME_FREQ_SAMPLERS.keys()) if args.domain == "freq-time" else list(list_channel_types())
    else:
        channels = [resolve_channel_name(args.channel)]

    run_vis(cfg, channels, domain=args.domain)


if __name__ == "__main__":
    main()
