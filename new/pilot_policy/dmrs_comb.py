from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from channel_estimators.lmmse import EstimatorState
from channel_generators.time_freq import kt_to_flat_index


def dmrs_comb_pattern(
    n_freq_subcarriers: int,
    n_symbols: int,
    *,
    comb_spacing: int,
    pilot_symbols: Sequence[int],
    comb_offset: int = 0,
) -> List[int]:
    """
    Input: Nc, Nt, frequency comb spacing, OFDM-symbol positions carrying pilots, comb offset.
    Output: flat RE indices (s=k*Nt+t) for a DMRS Type-1-style comb: every `comb_spacing`-th
    subcarrier (starting at `comb_offset`), repeated identically at each symbol in
    `pilot_symbols`. Pattern order is time-major (all comb frequencies of the earliest pilot
    symbol first, then the next symbol's), so an early prefix of the sequence is a
    frequency-spread bootstrap at the front-loaded symbol -- mirrors NR PDSCH DM-RS, where a
    single front-loaded symbol can be used alone before additional symbols are added for
    higher-mobility tracking.
    """
    if comb_spacing <= 0:
        raise ValueError("comb_spacing must be positive.")
    if any(t < 0 or t >= n_symbols for t in pilot_symbols):
        raise ValueError("pilot_symbols out of range.")
    if len(set(pilot_symbols)) != len(pilot_symbols):
        raise ValueError("pilot_symbols must be unique.")
    freqs = list(range(comb_offset % comb_spacing, n_freq_subcarriers, comb_spacing))
    return [kt_to_flat_index(k, t, n_symbols) for t in pilot_symbols for k in freqs]


class DmrsCombPolicy:
    """
    Non-adaptive DMRS-style baseline: a fixed frequency comb repeated at a handful of fixed
    OFDM-symbol positions (NR PDSCH DM-RS Type 1 comb-2, front-loaded + additional symbols
    for Doppler tracking), instead of the 1D largest-gap bisection applied naively to the
    flattened (k,t) index (see pilot_policy.fixed.FixedPolicy).
    """

    def __init__(
        self,
        *,
        n_subcarriers: int,
        n_antennas: int,
        initial_pilot_subcarriers: int,
        max_pilots: int,
        device: torch.device,
        n_freq_subcarriers: int,
        n_symbols: int,
        comb_spacing: int,
        pilot_symbols: Sequence[int],
        comb_offset: int = 0,
        initial_indices: Optional[Sequence[int]] = None,
    ) -> None:
        _ = (n_antennas, device)
        if n_subcarriers != n_symbols * n_freq_subcarriers:
            raise ValueError("n_subcarriers must equal n_symbols*n_freq_subcarriers.")
        pattern = dmrs_comb_pattern(
            n_freq_subcarriers,
            n_symbols,
            comb_spacing=comb_spacing,
            pilot_symbols=pilot_symbols,
            comb_offset=comb_offset,
        )
        if len(pattern) < max_pilots:
            raise ValueError(
                f"DMRS comb pattern has only {len(pattern)} REs, need max_pilots={max_pilots}; "
                "widen comb_spacing or add more pilot_symbols."
            )
        self._sequence = pattern[:max_pilots]
        if initial_indices is not None:
            init = [int(k) for k in initial_indices]
        else:
            init = self._sequence[:initial_pilot_subcarriers]
        if len(init) != initial_pilot_subcarriers:
            raise ValueError("initial_indices length must equal initial_pilot_subcarriers.")
        self._initial = init

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        _ = state
        used_set = set(used)
        for s in self._sequence:
            if s not in used_set:
                return s
        raise RuntimeError("DmrsCombPolicy: pattern exhausted.")
