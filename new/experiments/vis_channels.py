from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h
from channel_generators.sionna import SionnaOFDMGrid, sample_tdl_a_channel, sample_tdl_c_channel
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
    n_realizations: int = 5
    out_dir: Path = _NEW_ROOT / "figures"


def h_mat_from_vec(h: torch.Tensor, n_antennas: int, n_subcarriers: int) -> torch.Tensor:
    """Input: vec(H) (N,1). Output: H (Na,Nc) column-stacked by subcarrier."""
    return h.squeeze(-1).view(n_subcarriers, n_antennas).T


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


def plot_magnitude_panels(
    *,
    title: str,
    channels: list[torch.Tensor],
    out_path: Path,
    y_label: str = "antenna i",
    x_label: str = "subcarrier k",
    panel_title_fmt: str = "realization {idx}",
) -> None:
    """Input: title, list of H (Na,Nc), output path. Output: saves magnitude heatmap PNG."""
    if not channels:
        raise ValueError("channels must be non-empty.")

    magnitudes = [h.abs().detach().cpu().to(torch.float32).numpy() for h in channels]
    vmin = min(float(m.min()) for m in magnitudes)
    vmax = max(float(m.max()) for m in magnitudes)
    if vmin == vmax:
        vmax = vmin + 1.0

    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.4), squeeze=False)
    im = None
    for col, mag in enumerate(magnitudes):
        ax = axes[0, col]
        im = ax.imshow(mag, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title_fmt.format(idx=col + 1))
        ax.set_xlabel(x_label)
        if col == 0:
            ax.set_ylabel(y_label)

    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.82, wspace=0.28)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label=r"$|H_{i,k}|$")
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
) -> Path:
    """Input: channel type, config, device. Output: path to saved figure."""
    channels = sample_channel_realizations(
        channel,
        cfg,
        device=device,
        l_space=l_space,
        l_freq=l_freq,
        grid=grid,
    )
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    label = CHANNEL_TITLES[channel]
    if channel == "gaussian":
        subtitle = f"Na={na}, Nc={nc}, rho_space={cfg.rho_space}, rho_freq={cfg.rho_freq}"
    else:
        subtitle = f"Na={na}, Nc={nc}, rho_space={cfg.rho_space}"
    out_path = cfg.out_dir / f"{channel}_vis.png"
    plot_magnitude_panels(
        title=f"{label}  {subtitle}",
        channels=channels,
        out_path=out_path,
    )
    return out_path


def run_vis(cfg: ChannelVisConfig, channels: list[str]) -> list[Path]:
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
    return p.parse_args()


def main() -> None:
    cfg = default_config()
    args = parse_args()
    if args.n_antennas is not None:
        cfg.n_antennas = args.n_antennas
    if args.n_subcarriers is not None:
        cfg.n_subcarriers = args.n_subcarriers
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

    if args.all:
        channels = list(list_channel_types())
    else:
        channels = [resolve_channel_name(args.channel)]

    run_vis(cfg, channels)


if __name__ == "__main__":
    main()
