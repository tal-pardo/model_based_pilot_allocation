from __future__ import annotations

from typing import List

import torch

from channel_estimators.lmmse import EstimatorState
from pilot_policy.base import initial_subcarriers_uniform, next_uniform_subcarrier


class FixedPolicy:
    """Adds pilots one at a time to keep subcarrier spacing as uniform as possible."""

    def __init__(
        self,
        *,
        n_subcarriers: int,
        n_antennas: int,
        initial_pilot_subcarriers: int,
        max_pilots: int,
        device: torch.device,
    ) -> None:
        _ = (n_antennas, max_pilots, device)
        self._nc = n_subcarriers
        self._initial = initial_subcarriers_uniform(initial_pilot_subcarriers, n_subcarriers)

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        _ = state
        return next_uniform_subcarrier(used, self._nc)
