from __future__ import annotations

from typing import Tuple

import torch
from sionna.phy.channel.tr38901 import TDL

from channel_generators.sionna import SionnaOFDMGrid


def _sionna_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"cuda:{index}"
    return "cpu"


def tdl_tap_powers(
    *,
    model: str,
    grid: SionnaOFDMGrid,
    device: torch.device,
) -> torch.Tensor:
    """
    Input: TDL model name, OFDM grid, device.
    Output: (L,) linear tap powers P_l from the fixed 3GPP PDP (sum = 1 for normalized profiles).
    """
    tdl = TDL(
        model=model,
        delay_spread=grid.delay_spread,
        carrier_frequency=grid.carrier_frequency,
        num_rx_ant=1,
        num_tx_ant=1,
        min_speed=0.0,
        max_speed=0.0,
        device=_sionna_device(device),
    )
    return tdl.mean_powers.to(device=device, dtype=torch.float32)


def sigma_h_squared(tap_powers: torch.Tensor) -> float:
    """Input: (L,) tap powers. Output: sigma_H^2 = sum_l P_l."""
    return float(tap_powers.sum().item())


def build_sigma_diag(
    n: int,
    sigma_h2: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> torch.Tensor:
    """
    Input: N, scalar sigma_H^2, device, dtype, ridge reg.
    Output: Sigma (N,N) = sigma_H^2 I + reg I.
    """
    if sigma_h2 <= 0:
        raise ValueError("sigma_h2 must be positive.")
    sigma = (sigma_h2 + reg) * torch.eye(n, device=device, dtype=dtype)
    return 0.5 * (sigma + sigma.mH)


def build_sigma_diag_from_tdl(
    n_antennas: int,
    n_subcarriers: int,
    *,
    model: str,
    grid: SionnaOFDMGrid,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    """
    Input: Na, Nc, TDL model, grid, device, dtype, reg.
    Output: Sigma (N,N), sigma_H^2, tap_powers (L,).
    """
    tap_powers = tdl_tap_powers(model=model, grid=grid, device=device)
    sigma_h2 = sigma_h_squared(tap_powers)
    n = n_antennas * n_subcarriers
    sigma = build_sigma_diag(n, sigma_h2, device=device, dtype=dtype, reg=reg)
    return sigma, sigma_h2, tap_powers


def assert_sigma_diag(
    sigma: torch.Tensor,
    sigma_h2: float,
    *,
    reg: float = 1e-9,
    tol: float = 1e-6,
) -> None:
    """Input: Sigma, expected sigma_H^2, reg, tol. Output: raises if not diagonal constant-variance."""
    n = sigma.shape[0]
    off_diag = sigma - torch.diag(torch.diag(sigma))
    if off_diag.abs().max().item() > tol:
        raise ValueError("Sigma is not diagonal within tolerance.")
    expected = sigma_h2 + reg
    diag = sigma.diagonal().real
    if not torch.allclose(diag, torch.full((n,), expected, device=diag.device, dtype=diag.dtype), atol=tol):
        raise ValueError(
            f"Sigma diagonal must be {expected:g}, got min={diag.min().item():.6g}, max={diag.max().item():.6g}."
        )
