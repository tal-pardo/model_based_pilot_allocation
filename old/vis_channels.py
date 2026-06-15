from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

from data_generator import complex_standard_normal, exponential_covariance
from sionna_channels import (
    SionnaOFDMGrid,
    sample_cdl_c_channel,
    sample_tdl_a_block_fading_channels,
    sample_tdl_a_block_fading_slots,
    sample_tdl_c_channel,
)


@dataclass
class TdlAMimoVisConfig:
    n_antennas: int = 8
    n_subcarriers: int = 16
    n_realizations: int = 5
    rho_space: Optional[float] = None
    seed: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    subcarrier_spacing: float = 15e3
    carrier_frequency: float = 3.5e9
    delay_spread: float = 100e-9
    out_dir: Path = Path(__file__).resolve().parent / "figures" / "channel_vis"


@dataclass
class TdlASlotVisConfig:
    n_ofdm_symbols: int = 14
    n_subcarriers: int = 24
    n_slots: int = 5
    seed: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    subcarrier_spacing: float = 15e3
    carrier_frequency: float = 3.5e9
    delay_spread: float = 100e-9
    out_dir: Path = Path(__file__).resolve().parent / "figures" / "channel_vis"


@dataclass
class ChannelVisConfig:
    n_antennas: int = 32
    n_subcarriers: int = 64
    rho_space: float = 0.8
    rho_freq: float = 0.85
    n_realizations: int = 5
    seed: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    out_dir: Path = Path(__file__).resolve().parent / "figures" / "channel_vis"


def _resolve_device(device_str: str) -> torch.device:
    raw = device_str.lower().strip()
    if raw == "gpu":
        raw = "cuda"
    want_cuda = raw == "cuda" or raw.startswith("cuda:")
    if want_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (device=%r) but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and drivers, or set cfg.device='cpu'."
                % (device_str,)
            )
        device = torch.device(raw if raw.startswith("cuda:") else "cuda:0")
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        return device
    if raw != "cpu":
        raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', or 'gpu'; got %r." % (device_str,))
    return torch.device("cpu")


def _build_sigma_kron(
    Na: int,
    Nc: int,
    rho_space: float,
    rho_freq: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    R_space = exponential_covariance(Na, rho_space, device=device, dtype=dtype)
    R_freq = exponential_covariance(Nc, rho_freq, device=device, dtype=dtype)
    R_space = 0.5 * (R_space + R_space.mH)
    R_freq = 0.5 * (R_freq + R_freq.mH)
    L_space = torch.linalg.cholesky(R_space)
    L_freq = torch.linalg.cholesky(R_freq)
    return L_space, L_freq


def _sample_gaussian_H(
    Na: int,
    Nc: int,
    L_space: torch.Tensor,
    L_freq: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    Z = complex_standard_normal(Na, Nc, device=device, dtype=dtype, generator=generator)
    return L_space @ Z @ L_freq.T


def _sample_family_realizations(
    family: str,
    *,
    cfg: ChannelVisConfig,
    device: torch.device,
    L_space: torch.Tensor,
    L_freq: torch.Tensor,
    grid: SionnaOFDMGrid,
) -> list[torch.Tensor]:
    Na, Nc = cfg.n_antennas, cfg.n_subcarriers
    dtype = cfg.dtype
    out: list[torch.Tensor] = []

    for mc in range(cfg.n_realizations):
        if family == "gaussian":
            gen = torch.Generator(device=device).manual_seed(cfg.seed + mc)
            H = _sample_gaussian_H(Na, Nc, L_space, L_freq, device=device, dtype=dtype, generator=gen)
        elif family == "tdl":
            H = sample_tdl_c_channel(
                n_antennas=Na,
                n_subcarriers=Nc,
                rho_space=cfg.rho_space,
                grid=grid,
                device=device,
                dtype=dtype,
                seed=cfg.seed + 100_000 + mc,
            )
        elif family == "cdl":
            H = sample_cdl_c_channel(
                n_antennas=Na,
                n_subcarriers=Nc,
                grid=grid,
                device=device,
                dtype=dtype,
                seed=cfg.seed + 200_000 + mc,
            )
        else:
            raise ValueError("family must be 'gaussian', 'tdl', or 'cdl'; got %r." % (family,))
        out.append(H)
    return out


def _plot_family_magnitude(
    *,
    family: str,
    title: str,
    channels: list[torch.Tensor],
    out_path: Path,
    y_label: str = "antenna i",
    x_label: str = "subcarrier k",
    panel_title_fmt: str = "realization {idx}",
) -> None:
    if not channels:
        raise ValueError("channels must be non-empty.")

    magnitudes = [H.abs().detach().cpu().to(torch.float32).numpy() for H in channels]
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


def visualize_channel_families(cfg: ChannelVisConfig | None = None) -> list[Path]:
    cfg = cfg or ChannelVisConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    dtype = cfg.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    Na, Nc = cfg.n_antennas, cfg.n_subcarriers
    L_space, L_freq = _build_sigma_kron(Na, Nc, cfg.rho_space, cfg.rho_freq, device=device, dtype=dtype)
    grid = SionnaOFDMGrid(fft_size=Nc, subcarrier_spacing=15e3, carrier_frequency=3.5e9, delay_spread=100e-9)

    families = (
        ("gaussian", "Gaussian (Kronecker prior draw)"),
        ("tdl", "TDL-C (Sionna, power-normalized)"),
        ("cdl", "CDL-C (Sionna, power-normalized)"),
    )
    out_paths: list[Path] = []
    for family, label in families:
        channels = _sample_family_realizations(
            family,
            cfg=cfg,
            device=device,
            L_space=L_space,
            L_freq=L_freq,
            grid=grid,
        )
        out_path = cfg.out_dir / f"{family}_magnitude.png"
        _plot_family_magnitude(
            family=family,
            title=f"{label}  Na={Na}, Nc={Nc}, rho_space={cfg.rho_space}, rho_freq={cfg.rho_freq}",
            channels=channels,
            out_path=out_path,
        )
        out_paths.append(out_path)
    return out_paths


def visualize_tdl_a_block_fading_slots(cfg: TdlASlotVisConfig | None = None) -> Path:
    cfg = cfg or TdlASlotVisConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    dtype = cfg.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    grid = SionnaOFDMGrid(
        fft_size=cfg.n_subcarriers,
        subcarrier_spacing=cfg.subcarrier_spacing,
        carrier_frequency=cfg.carrier_frequency,
        delay_spread=cfg.delay_spread,
    )
    slots = sample_tdl_a_block_fading_slots(
        n_slots=cfg.n_slots,
        n_ofdm_symbols=cfg.n_ofdm_symbols,
        n_subcarriers=cfg.n_subcarriers,
        grid=grid,
        device=device,
        dtype=dtype,
        seed=cfg.seed,
    )
    out_path = cfg.out_dir / "tdl_a_block_fading_slots_magnitude.png"
    _plot_family_magnitude(
        family="tdl_a",
        title=(
            "TDL-A block-fading slots (Sionna, power-normalized)  "
            f"L={cfg.n_ofdm_symbols}, K={cfg.n_subcarriers}, v=0"
        ),
        channels=slots,
        out_path=out_path,
        y_label="OFDM symbol l",
        x_label="subcarrier k",
        panel_title_fmt="slot {idx}",
    )
    return out_path


def visualize_tdl_a_block_fading_mimo(cfg: TdlAMimoVisConfig | None = None) -> Path:
    cfg = cfg or TdlAMimoVisConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    dtype = cfg.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    grid = SionnaOFDMGrid(
        fft_size=cfg.n_subcarriers,
        subcarrier_spacing=cfg.subcarrier_spacing,
        carrier_frequency=cfg.carrier_frequency,
        delay_spread=cfg.delay_spread,
    )
    channels = sample_tdl_a_block_fading_channels(
        n_realizations=cfg.n_realizations,
        n_antennas=cfg.n_antennas,
        n_subcarriers=cfg.n_subcarriers,
        grid=grid,
        device=device,
        rho_space=cfg.rho_space,
        dtype=dtype,
        seed=cfg.seed,
    )
    rho_label = "uncorrelated" if cfg.rho_space is None else f"rho_space={cfg.rho_space}"
    out_path = cfg.out_dir / "tdl_a_block_fading_mimo_magnitude.png"
    _plot_family_magnitude(
        family="tdl_a",
        title=(
            "TDL-A block-fading MIMO (Sionna, power-normalized)  "
            f"Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, v=0, {rho_label}"
        ),
        channels=channels,
        out_path=out_path,
        panel_title_fmt="realization {idx}",
    )
    return out_path


if __name__ == "__main__":
    path = visualize_tdl_a_block_fading_mimo()
    print(path)
