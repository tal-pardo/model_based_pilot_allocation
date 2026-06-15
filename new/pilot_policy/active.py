from __future__ import annotations

from typing import List

import torch

from channel_estimators.lmmse import EstimatorState
from error_estimators.trace_min import active_subcarrier_score
from pilot_policy.base import initial_subcarriers_uniform


class ActivePolicy:
    """Greedy argmax J(k) using posterior trace score."""

    def __init__(
        self,
        *,
        n_subcarriers: int,
        n_antennas: int,
        initial_pilot_subcarriers: int,
        max_pilots: int,
        sigma2: float,
        device: torch.device,
    ) -> None:
        _ = (max_pilots, device)
        self._nc = n_subcarriers
        self._na = n_antennas
        self._sigma2 = sigma2
        self._initial = initial_subcarriers_uniform(initial_pilot_subcarriers, n_subcarriers)

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        unused = [k for k in range(self._nc) if k not in set(used)]
        if not unused:
            raise RuntimeError("ActivePolicy: no unused subcarriers.")
        best_k = max(
            unused,
            key=lambda k: active_subcarrier_score(state, k, self._na, self._sigma2),
        )
        return best_k
