from __future__ import annotations

import math
from typing import Tuple

import torch
from scipy.special import j0 as bessel_j0

from utils import exponential_covariance


def jakes_spatial_covariance(
    n_antennas: int,
    d_over_lambda: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> torch.Tensor:
    """
    Input: Na, adjacent spacing in wavelengths d/lambda, device, dtype, ridge.
    Output: (Na,Na) Hermitian R_space with [R]_ij = J0(2 pi |i-j| d/lambda) + reg I.
    """
    if d_over_lambda <= 0:
        raise ValueError("d_over_lambda must be positive.")
    idx = torch.arange(n_antennas, device="cpu")
    lag = (idx[:, None] - idx[None, :]).abs().to(torch.float64)
    arg = (2.0 * math.pi * d_over_lambda * lag).numpy()
    mat = torch.from_numpy(bessel_j0(arg).astype("float64")).to(
        device=device, dtype=torch.float32
    )
    if dtype in (torch.complex64, torch.complex128):
        mat = mat.to(dtype)
    else:
        mat = mat.to(dtype)
    mat = mat + reg * torch.eye(n_antennas, device=device, dtype=mat.dtype)
    return 0.5 * (mat + mat.mH)


def build_sigma_jakes_kron(
    n_antennas: int,
    n_subcarriers: int,
    d_over_lambda: float,
    rho_freq: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input: Na, Nc, d/lambda, rho_freq, device, dtype, ridge.
    Output: Sigma (N,N) = R_freq ⊗ R_jakes (+ ridge), L_space, L_freq.
    """
    r_space = jakes_spatial_covariance(
        n_antennas, d_over_lambda, device=device, dtype=dtype, reg=reg
    )
    r_freq = exponential_covariance(n_subcarriers, rho_freq, device=device, dtype=dtype)
    r_freq = 0.5 * (r_freq + r_freq.mH)
    sigma = torch.kron(r_freq, r_space)
    sigma = 0.5 * (sigma + sigma.mH)
    n = n_antennas * n_subcarriers
    sigma = sigma + reg * torch.eye(n, device=device, dtype=dtype)
    l_space = torch.linalg.cholesky(r_space)
    l_freq = torch.linalg.cholesky(r_freq)
    return sigma, l_space, l_freq
