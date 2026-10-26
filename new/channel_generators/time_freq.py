from __future__ import annotations

import math
from typing import Tuple

import torch

from channel_generators.gaussian import build_sigma_kron, sample_gaussian_h


def build_sigma_time_freq(
    n_symbols: int,
    n_subcarriers: int,
    rho_time: float,
    rho_freq: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    reg: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Input: Nt (OFDM symbols), Nc (frequency subcarriers), rho_time, rho_freq, device, dtype, reg.
    Output: Sigma (Nt*Nc, Nt*Nc) = kron(R_freq, R_time) + reg*I, L_time (Nt,Nt), L_freq (Nc,Nc).

    Thin, clearly-named wrapper around channel_generators.gaussian.build_sigma_kron: the
    'antenna' axis there is relabeled 'time' (OFDM symbol index) here. The resulting block
    layout (one Nt-sized block per subcarrier k) matches
    channel_generators.sionna.sample_tdl_a_time_freq_channel's H_tf (Nt,Nc) convention once
    passed through vec_from_h, so the same flat (Nt*Nc,1) index space can hold either a
    Gaussian draw from this prior (sample_gaussian_h_time_freq) or a TDL-A time-freq truth.
    """
    return build_sigma_kron(
        n_symbols, n_subcarriers, rho_time, rho_freq, device=device, dtype=dtype, reg=reg
    )


def sample_gaussian_h_time_freq(
    n_symbols: int,
    n_subcarriers: int,
    l_time: torch.Tensor,
    l_freq: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Input: Nt, Nc, Cholesky factors L_time (Nt,Nt) / L_freq (Nc,Nc), device, generator.
    Output: h (Nt*Nc,1), vec(H_tf) column-stacked by subcarrier: block s=k*Nt .. k*Nt+Nt-1
    holds subcarrier k across all Nt OFDM symbols. Thin wrapper around
    channel_generators.gaussian.sample_gaussian_h (antenna axis relabeled time).
    """
    return sample_gaussian_h(
        n_symbols, n_subcarriers, l_time, l_freq, device=device, dtype=dtype, generator=generator
    )


def flat_index_to_kt(s: int, n_symbols: int) -> Tuple[int, int]:
    """Input: flat RE index s in [0, Nt*Nc), Nt. Output: (subcarrier k, OFDM symbol t)."""
    return s // n_symbols, s % n_symbols


def kt_to_flat_index(k: int, t: int, n_symbols: int) -> int:
    """Input: subcarrier k, OFDM symbol t, Nt. Output: flat RE index s = k*Nt + t."""
    return k * n_symbols + t


def _unitary_dft(m: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    idx = torch.arange(m, device=device, dtype=torch.float64)
    phase = -2j * math.pi * idx[:, None] * idx[None, :] / m
    return (torch.exp(phase) / (m**0.5)).to(dtype)


def build_delay_doppler_transform(
    n_freq_subcarriers: int,
    n_symbols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Input: Nc (frequency subcarriers), Nt (OFDM symbols), device, dtype.
    Output: T (Nt*Nc, Nt*Nc) = kron(F_freq, F_time), unitary (T.mH @ T == I). F_freq/F_time are
    (Nc,Nc)/(Nt,Nt) unitary DFT matrices.

    Matches build_sigma_time_freq's own kron(R_freq, R_time) factor order and vec_from_h's block
    layout (block s=k*Nt..k*Nt+Nt-1 holds subcarrier k's Nt time samples): T maps the outer
    (subcarrier) axis to delay and the inner (OFDM-symbol) axis to Doppler. A TDL-A time-freq
    channel has only a handful of physical paths, each with one delay and one Doppler shift, so
    it is sparse in this joint delay-Doppler representation (the doubly-selective-channel analog
    of OFDM delay-domain sparsity).
    """
    f_freq = _unitary_dft(n_freq_subcarriers, device=device, dtype=dtype)
    f_time = _unitary_dft(n_symbols, device=device, dtype=dtype)
    return torch.kron(f_freq, f_time)
