from __future__ import annotations

import random
from typing import List, Optional, Sequence

import torch

from channel_estimators.lmmse import EstimatorState
from pilot_policy.base import initial_subcarriers_uniform


class RandomPolicy:
    """Uniform random choice among unused subcarriers."""

    def __init__(
        self,
        *,
        n_subcarriers: int,
        n_antennas: int,
        initial_pilot_subcarriers: int,
        max_pilots: int,
        device: torch.device,
        seed: int,
        initial_indices: Optional[Sequence[int]] = None,
    ) -> None:
        _ = (n_antennas, max_pilots, device)
        self._nc = n_subcarriers
        if initial_indices is not None:
            init = [int(k) for k in initial_indices]
            if len(init) != initial_pilot_subcarriers:
                raise ValueError("initial_indices length must equal initial_pilot_subcarriers.")
            if any(k < 0 or k >= n_subcarriers for k in init):
                raise ValueError("initial_indices out of range.")
            if len(set(init)) != len(init):
                raise ValueError("initial_indices must be unique.")
            self._initial = init
        else:
            self._initial = initial_subcarriers_uniform(initial_pilot_subcarriers, n_subcarriers)
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        _ = state
        unused = [k for k in range(self._nc) if k not in set(used)]
        if not unused:
            raise RuntimeError("RandomPolicy: no unused subcarriers.")
        return self._rng.choice(unused)
