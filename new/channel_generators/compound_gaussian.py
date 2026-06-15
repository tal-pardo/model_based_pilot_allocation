from __future__ import annotations

import torch

from channel_generators.gaussian import sample_gaussian_h


def sample_compound_gaussian_h(
    n_antennas: int,
    n_subcarriers: int,
    l_space: torch.Tensor,
    l_freq: torch.Tensor,
    *,
    texture_alpha: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Input: Cholesky factors, texture shape alpha (E[s]=1), device, generator.
    Output: h (N,1) with h = sqrt(s) * g, g ~ CN(0, Sigma), s ~ Gamma(alpha, alpha).
    Covariance E[hh^H] = Sigma when g uses the Kronecker Sigma from build_sigma_kron.
    """
    if texture_alpha <= 0.0:
        raise ValueError("texture_alpha must be positive.")
    g = sample_gaussian_h(
        n_antennas,
        n_subcarriers,
        l_space,
        l_freq,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    s = _sample_unit_mean_gamma(texture_alpha, device=device, dtype=real_dtype, generator=generator)
    return g * s.sqrt().to(dtype)


def _sample_unit_mean_gamma(
    alpha: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """Input: alpha>0, device, generator. Output: scalar s with E[s]=1, s ~ Gamma(alpha, rate=alpha)."""
    if alpha == 1.0:
        u = torch.rand(1, device=device, dtype=dtype, generator=generator)
        return (-torch.log(u.clamp(min=1e-12))).squeeze(0)
    concentration = torch.tensor(alpha, device=device, dtype=dtype)
    rate = torch.tensor(alpha, device=device, dtype=dtype)
    return torch.distributions.Gamma(concentration, rate).sample()
