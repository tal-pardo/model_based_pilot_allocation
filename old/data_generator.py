from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def complex_standard_normal(
    *shape: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be a complex dtype (torch.complex64/128).")
    if generator is not None and device is not None:
        gen_device = getattr(generator, "device", None)
        if gen_device is not None and gen_device.type != torch.device(device).type:
            raise ValueError(
                "torch.Generator device must match tensor device. "
                f"Got generator.device={gen_device} but device={device}."
            )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    re = torch.randn(*shape, device=device, dtype=real_dtype, generator=generator)
    im = torch.randn(*shape, device=device, dtype=real_dtype, generator=generator)
    return (re + 1j * im).to(dtype) / (2.0**0.5)


def exponential_covariance(
    n_antennas: int,
    rho: float,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1 for a well-conditioned covariance.")
    if n_antennas <= 0:
        raise ValueError("n_antennas must be positive.")

    idx = torch.arange(n_antennas, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    Sigma_real = (rho ** dist).to(torch.float32 if dtype == torch.complex64 else torch.float64)
    return Sigma_real.to(dtype)


def pilot_matrix_from_indices(
    n_antennas: int,
    pilot_indices: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Builds a *selection* matrix X of shape (m, N), where each row selects one pilot entry.
    If pilot_indices = [i0, i1, ...], then X[k, i_k] = 1.
    This yields y_t in C^{m} consisting only of the observed pilot positions.
    """
    if pilot_indices.ndim != 1:
        raise ValueError("pilot_indices must be a 1D tensor of indices.")
    m = pilot_indices.numel()
    X = torch.zeros((m, n_antennas), device=device, dtype=dtype)
    X[torch.arange(m, device=device), pilot_indices.to(device=device)] = torch.ones(
        (m,), device=device, dtype=dtype
    )
    return X


def empirical_covariance(
    h_samples: torch.Tensor,
    *,
    reg: float = 1e-9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    Sample covariance Sigma_hat = (1/K) sum_k h_k h_k^H with Hermitian symmetrization and reg*I.
    h_samples may be (K, N) or (K, N, 1).
    """
    if h_samples.ndim == 3 and h_samples.shape[-1] == 1:
        h_samples = h_samples.squeeze(-1)
    if h_samples.ndim != 2:
        raise ValueError("h_samples must have shape (K, N) or (K, N, 1).")
    if h_samples.shape[0] < 1:
        raise ValueError("h_samples must contain at least one vector.")

    h_samples = h_samples.to(device=device, dtype=dtype)
    K, N = h_samples.shape
    Sigma_hat = (h_samples.mH @ h_samples) / float(K)
    Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.mH)
    Sigma_hat = Sigma_hat + (reg * torch.eye(N, device=Sigma_hat.device, dtype=Sigma_hat.dtype))
    return Sigma_hat


def stack_observations(X_list: List[torch.Tensor], y_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stacks X_t vertically: X_all = [X_0; X_1; ...] in C^{(sum_t m_t) x N}
    and y_all similarly in C^{(sum_t m_t) x 1}.
    """
    if len(X_list) == 0 or len(y_list) == 0 or len(X_list) != len(y_list):
        raise ValueError("X_list and y_list must be non-empty and have equal length.")
    X_all = torch.cat(X_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return X_all, y_all

