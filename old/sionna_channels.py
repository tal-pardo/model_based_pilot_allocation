from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sionna.phy.channel import cir_to_ofdm_channel, exp_corr_mat, subcarrier_frequencies
from sionna.phy.channel.tr38901 import AntennaArray, CDL, TDL


@dataclass(frozen=True)
class SionnaOFDMGrid:
    fft_size: int
    subcarrier_spacing: float = 15e3
    carrier_frequency: float = 3.5e9
    delay_spread: float = 100e-9

    @property
    def bandwidth(self) -> float:
        return float(self.fft_size * self.subcarrier_spacing)


def _normalize_channel_power(H: torch.Tensor) -> torch.Tensor:
    scale = torch.sqrt(torch.mean(H.abs().pow(2)))
    if scale.real.item() <= 0.0:
        raise ValueError("Channel power normalization failed: mean |H|^2 is zero.")
    return H / scale


def _h_freq_to_H(h_freq: torch.Tensor, n_antennas: int, n_subcarriers: int) -> torch.Tensor:
    # h_freq: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_subcarriers]
    H = h_freq[0, 0, :n_antennas, 0, 0, 0, :n_subcarriers]
    return _normalize_channel_power(H)


def _h_freq_to_H_ofdm_slot(
    h_freq: torch.Tensor,
    n_ofdm_symbols: int,
    n_subcarriers: int,
) -> torch.Tensor:
    # Single-antenna block-fading slot H[l, k] over OFDM symbols and subcarriers.
    H = h_freq[0, 0, 0, 0, 0, :n_ofdm_symbols, :n_subcarriers]
    return _normalize_channel_power(H)


def _sionna_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"cuda:{index}"
    return "cpu"


def _set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sample_tdl_ofdm_channel(
    *,
    model: str,
    n_ofdm_symbols: int,
    n_subcarriers: int,
    grid: SionnaOFDMGrid,
    device: torch.device,
    n_antennas: int = 1,
    rho_space: Optional[float] = None,
    dtype: torch.dtype = torch.complex64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        _set_torch_seed(seed, device)

    sionna_device = _sionna_device(device)
    tdl_kwargs: dict = {
        "model": model,
        "delay_spread": grid.delay_spread,
        "carrier_frequency": grid.carrier_frequency,
        "num_rx_ant": n_antennas,
        "num_tx_ant": 1,
        "min_speed": 0.0,
        "max_speed": 0.0,
        "device": sionna_device,
    }
    if rho_space is not None:
        rho = torch.tensor(rho_space, device=sionna_device, dtype=torch.float32)
        tdl_kwargs["rx_corr_mat"] = exp_corr_mat(rho, n_antennas, device=sionna_device).to(dtype)

    tdl = TDL(**tdl_kwargs)
    a, tau = tdl(batch_size=1, num_time_steps=n_ofdm_symbols, sampling_frequency=grid.bandwidth)
    frequencies = subcarrier_frequencies(grid.fft_size, grid.subcarrier_spacing, device=sionna_device)
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    if n_ofdm_symbols == 1 and n_antennas > 1:
        H = _h_freq_to_H(h_freq, n_antennas, n_subcarriers)
    else:
        H = _h_freq_to_H_ofdm_slot(h_freq, n_ofdm_symbols, n_subcarriers)
    return H.to(device=device, dtype=dtype)


def sample_tdl_c_channel(
    *,
    n_antennas: int,
    n_subcarriers: int,
    rho_space: float,
    grid: SionnaOFDMGrid,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    return sample_tdl_ofdm_channel(
        model="C",
        n_ofdm_symbols=1,
        n_subcarriers=n_subcarriers,
        grid=grid,
        device=device,
        n_antennas=n_antennas,
        rho_space=rho_space,
        dtype=dtype,
        seed=seed,
    )


def sample_tdl_a_block_fading_slots(
    *,
    n_slots: int,
    n_ofdm_symbols: int,
    n_subcarriers: int,
    grid: SionnaOFDMGrid,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    seed: Optional[int] = None,
) -> list[torch.Tensor]:
    slots: list[torch.Tensor] = []
    for slot in range(n_slots):
        slot_seed = None if seed is None else seed + slot
        slots.append(
            sample_tdl_ofdm_channel(
                model="A",
                n_ofdm_symbols=n_ofdm_symbols,
                n_subcarriers=n_subcarriers,
                grid=grid,
                device=device,
                dtype=dtype,
                seed=slot_seed,
            )
        )
    return slots


def sample_tdl_a_block_fading_channels(
    *,
    n_realizations: int,
    n_antennas: int,
    n_subcarriers: int,
    grid: SionnaOFDMGrid,
    device: torch.device,
    rho_space: Optional[float] = None,
    dtype: torch.dtype = torch.complex64,
    seed: Optional[int] = None,
) -> list[torch.Tensor]:
    channels: list[torch.Tensor] = []
    for mc in range(n_realizations):
        mc_seed = None if seed is None else seed + mc
        channels.append(
            sample_tdl_ofdm_channel(
                model="A",
                n_ofdm_symbols=1,
                n_subcarriers=n_subcarriers,
                grid=grid,
                device=device,
                n_antennas=n_antennas,
                rho_space=rho_space,
                dtype=dtype,
                seed=mc_seed,
            )
        )
    return channels


def sample_cdl_c_channel(
    *,
    n_antennas: int,
    n_subcarriers: int,
    grid: SionnaOFDMGrid,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    if seed is not None:
        _set_torch_seed(seed, device)

    sionna_device = _sionna_device(device)
    ut_array = AntennaArray(
        num_rows=1,
        num_cols=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=grid.carrier_frequency,
    )
    bs_array = AntennaArray(
        num_rows=1,
        num_cols=n_antennas,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=grid.carrier_frequency,
    )
    cdl = CDL(
        model="C",
        delay_spread=grid.delay_spread,
        carrier_frequency=grid.carrier_frequency,
        ut_array=ut_array,
        bs_array=bs_array,
        direction="uplink",
        min_speed=0.0,
        max_speed=0.0,
        device=sionna_device,
    )
    a, tau = cdl(batch_size=1, num_time_steps=1, sampling_frequency=grid.bandwidth)
    frequencies = subcarrier_frequencies(grid.fft_size, grid.subcarrier_spacing, device=sionna_device)
    h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    H = _h_freq_to_H(h_freq, n_antennas, n_subcarriers).to(device=device, dtype=dtype)
    return H


def vec_from_H(H: torch.Tensor) -> torch.Tensor:
    # Column-stacked vec(H): [H[:,0]; H[:,1]; ...]
    return H.T.contiguous().view(-1, 1)
