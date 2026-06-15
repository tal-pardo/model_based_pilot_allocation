from __future__ import annotations

from typing import Tuple

import torch

from utils import complex_standard_normal, exponential_covariance


def build_sigma_kron(
    n_antennas: int,
    n_subcarriers: int,
    rho_space: float,
    rho_freq: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input: Na, Nc, spatial/freq correlation, device, dtype, reg (prior ridge on Sigma).
    Output: Sigma (N,N), L_space, L_freq Cholesky factors.
    """
    r_space = exponential_covariance(n_antennas, rho_space, device=device, dtype=dtype)
    r_freq = exponential_covariance(n_subcarriers, rho_freq, device=device, dtype=dtype)
    r_space = 0.5 * (r_space + r_space.mH)
    r_freq = 0.5 * (r_freq + r_freq.mH)
    sigma = torch.kron(r_freq, r_space)
    sigma = 0.5 * (sigma + sigma.mH)
    n = n_antennas * n_subcarriers
    sigma = sigma + reg * torch.eye(n, device=device, dtype=dtype)
    l_space = torch.linalg.cholesky(r_space)
    l_freq = torch.linalg.cholesky(r_freq)
    return sigma, l_space, l_freq


def sample_gaussian_h(
    n_antennas: int,
    n_subcarriers: int,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Input: Cholesky factors, device, generator.
    Output: h (N,1) column-stacked vec(H), H = L_space Z L_freq^T.
    """
    z = complex_standard_normal(n_antennas, n_subcarriers, device=device, dtype=dtype, generator=generator)
    h_mat = l_space @ z @ l_freq.T
    return h_mat.T.contiguous().view(n_antennas * n_subcarriers, 1)
