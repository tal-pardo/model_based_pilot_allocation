from __future__ import annotations

from typing import Tuple

import torch

from channel_generators.gaussian import sample_gaussian_h
from utils import exponential_covariance

__all__ = [
    "build_block_ar1_freq",
    "build_sigma_block_kron",
    "sample_block_kron_h",
]


def _cluster_id(k: int, n_subcarriers: int, n_clusters: int) -> int:
    size = n_subcarriers // n_clusters
    if n_subcarriers % n_clusters != 0:
        raise ValueError("n_subcarriers must be divisible by n_clusters.")
    return min(k // size, n_clusters - 1)


def build_block_ar1_freq(
    n_subcarriers: int,
    *,
    rho_intra: float,
    rho_inter: float,
    n_clusters: int = 2,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Two (or more) contiguous SC clusters with AR(1) within each cluster and
    constant cross-cluster correlation rho_inter.

    Input: Nc, rho_intra in [0,1), rho_inter in [0,1), n_clusters, device, dtype.
    Output: R_freq (Nc, Nc) Hermitian PSD correlation matrix (unit diagonal).
    """
    if not (0.0 <= rho_intra < 1.0):
        raise ValueError("rho_intra must satisfy 0 <= rho_intra < 1.")
    if not (0.0 <= rho_inter < 1.0):
        raise ValueError("rho_inter must satisfy 0 <= rho_inter < 1.")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    idx = torch.arange(n_subcarriers, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    intra = (rho_intra**dist).to(real_dtype)

    cluster = torch.tensor(
        [_cluster_id(int(k), n_subcarriers, n_clusters) for k in range(n_subcarriers)],
        device=device,
    )
    same = cluster[:, None] == cluster[None, :]
    r_freq = torch.where(same, intra, torch.full_like(intra, rho_inter))
    r_freq = 0.5 * (r_freq + r_freq.mH)
    return r_freq.to(dtype)


def build_sigma_block_kron(
    n_antennas: int,
    n_subcarriers: int,
    rho_space: float,
    *,
    rho_intra: float,
    rho_inter: float,
    n_clusters: int = 2,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block-cluster frequency prior: Sigma = kron(R_freq, R_space) + reg I.

    Input: Na, Nc, rho_space, rho_intra, rho_inter, n_clusters, device, dtype, reg.
    Output: Sigma (N,N), L_space (Na,Na), L_freq (Nc,Nc) Cholesky factors.
    """
    r_space = exponential_covariance(n_antennas, rho_space, device=device, dtype=dtype)
    r_freq = build_block_ar1_freq(
        n_subcarriers,
        rho_intra=rho_intra,
        rho_inter=rho_inter,
        n_clusters=n_clusters,
        device=device,
        dtype=dtype,
    )
    r_space = 0.5 * (r_space + r_space.mH)
    r_freq = 0.5 * (r_freq + r_freq.mH)
    sigma = torch.kron(r_freq, r_space)
    sigma = 0.5 * (sigma + sigma.mH)
    n = n_antennas * n_subcarriers
    sigma = sigma + reg * torch.eye(n, device=device, dtype=dtype)
    l_space = torch.linalg.cholesky(r_space)
    l_freq = torch.linalg.cholesky(r_freq)
    return sigma, l_space, l_freq


def sample_block_kron_h(
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
    Input: Cholesky factors from build_sigma_block_kron, device, generator.
    Output: h (N,1) matched Gaussian sample, H = L_space Z L_freq^T.
    """
    return sample_gaussian_h(
        n_antennas,
        n_subcarriers,
        l_space,
        l_freq,
        device=device,
        dtype=dtype,
        generator=generator,
    )
