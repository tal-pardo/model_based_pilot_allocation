from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def resolve_device(device_str: str) -> torch.device:
    """
    Input: device_str ('cuda', 'cuda:N', 'cpu', 'gpu')
    Output: torch.device; raises if CUDA requested but unavailable
    """
    raw = device_str.lower().strip()
    if raw == "gpu":
        raw = "cuda"
    want_cuda = raw == "cuda" or raw.startswith("cuda:")
    if want_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (device=%r) but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and drivers."
                % (device_str,)
            )
        device = torch.device(raw if raw.startswith("cuda:") else "cuda:0")
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        print(
            f"Using CUDA device: {torch.cuda.get_device_name(cuda_index)} "
            f"(capability {torch.cuda.get_device_capability(cuda_index)})"
        )
        return device
    if raw != "cpu":
        raise ValueError("device must be 'cpu', 'cuda', 'cuda:N', or 'gpu'; got %r." % (device_str,))
    print("Using device: cpu")
    return torch.device("cpu")


def set_seed(seed: int, device: torch.device) -> None:
    """Input: seed int, device. Output: none (sets global RNG state)."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def complex_standard_normal(
    *shape: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Input: shape dims, device, dtype, optional Generator. Output: CN(0,1) tensor."""
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be torch.complex64 or torch.complex128.")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    re = torch.randn(*shape, device=device, dtype=real_dtype, generator=generator)
    im = torch.randn(*shape, device=device, dtype=real_dtype, generator=generator)
    return (re + 1j * im).to(dtype) / (2.0**0.5)


def exponential_covariance(
    n: int,
    rho: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Input: dimension n, correlation rho. Output: (n,n) exponential correlation matrix."""
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must satisfy 0 <= rho < 1.")
    idx = torch.arange(n, device=device)
    dist = (idx[:, None] - idx[None, :]).abs()
    mat = (rho**dist).to(torch.float32 if dtype == torch.complex64 else torch.float64)
    return mat.to(dtype)


def empirical_covariance(
    h_samples: torch.Tensor,
    *,
    reg: float = 1e-9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Input: h_samples (K,N), reg ridge. Output: (N,N) sample covariance + reg*I."""
    if h_samples.ndim == 3 and h_samples.shape[-1] == 1:
        h_samples = h_samples.squeeze(-1)
    h_samples = h_samples.to(device=device, dtype=dtype)
    k, n = h_samples.shape
    sigma_hat = (h_samples.mH @ h_samples) / float(k)
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.mH)
    sigma_hat = sigma_hat + reg * torch.eye(n, device=sigma_hat.device, dtype=sigma_hat.dtype)
    return sigma_hat


def subcarrier_selection_matrix(
    k: int,
    n_antennas: int,
    n_total: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Input: subcarrier index k, Na, N=Na*Nc. Output: X (Na, N) selecting block k."""
    x = torch.zeros((n_antennas, n_total), device=device, dtype=dtype)
    start = k * n_antennas
    rows = torch.arange(n_antennas, device=device)
    cols = torch.arange(start, start + n_antennas, device=device)
    x[rows, cols] = torch.ones(n_antennas, device=device, dtype=dtype)
    return x


def measure_subcarrier(
    k: int,
    h_true: torch.Tensor,
    sigma2: float,
    *,
    n_antennas: int,
    n_total: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Input: subcarrier k, h_true (N,1), sigma2, Na, N, device, generator.
    Output: X (Na,N), y (Na,1) with y = X h + n.
    """
    x = subcarrier_selection_matrix(k, n_antennas, n_total, device=device, dtype=dtype)
    noise = (sigma2**0.5) * complex_standard_normal(
        n_antennas, 1, device=device, dtype=dtype, generator=generator
    )
    y = x @ h_true + noise
    return x, y


def empirical_nmse(h_hat: torch.Tensor, h_true: torch.Tensor) -> float:
    """Input: h_hat, h_true (N,1). Output: (1/N)||h_hat - h_true||^2."""
    err = h_hat - h_true
    return err.abs().pow(2).mean().real.item()


def stack_observations(
    x_list: List[torch.Tensor], y_list: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input: lists of X_t, y_t. Output: vertically stacked X_all, y_all."""
    return torch.cat(x_list, dim=0), torch.cat(y_list, dim=0)
