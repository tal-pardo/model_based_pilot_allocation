from __future__ import annotations

import math
from typing import Tuple

import torch

from utils import complex_standard_normal


def build_sigma_low_rank(
    n: int,
    rank: int,
    eps: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: N, rank r, residual eps, device, dtype, RNG for random semi-orthogonal B.
    Output: Sigma (N,N) = B B^H + eps I with tr(Sigma)/N = 1, and B (N,r).

    Columns of the unscaled factor are orthonormal; then scaled so
    tr(B B^H) = N (1 - eps), hence average power is 1 with residual eps I.
    """
    if rank <= 0 or rank >= n:
        raise ValueError("rank must satisfy 0 < r < N.")
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must satisfy 0 < eps < 1.")

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    a_r = torch.randn(n, rank, device=device, dtype=real_dtype, generator=generator)
    a_i = torch.randn(n, rank, device=device, dtype=real_dtype, generator=generator)
    a = torch.complex(a_r, a_i).to(dtype)
    q, _ = torch.linalg.qr(a, mode="reduced")
    # beta^2 * r + eps * N = N  =>  beta^2 = N (1 - eps) / r
    beta = math.sqrt(n * (1.0 - eps) / float(rank))
    b = q * beta
    sigma = b @ b.mH + eps * torch.eye(n, device=device, dtype=dtype)
    sigma = 0.5 * (sigma + sigma.mH)
    return sigma, b


def sample_low_rank_h(
    b: torch.Tensor,
    eps: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Input: B (N,r), eps. Output: h (N,1) = B z + sqrt(eps) w."""
    n, rank = b.shape
    z = complex_standard_normal(rank, 1, device=device, dtype=dtype, generator=generator)
    w = complex_standard_normal(n, 1, device=device, dtype=dtype, generator=generator)
    return b @ z + (eps**0.5) * w
