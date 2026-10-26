from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from channel_estimators.lmmse import EstimatorState
from error_estimators.trace_min import active_scores_all
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
        initial_indices: Optional[Sequence[int]] = None,
    ) -> None:
        _ = (max_pilots, device)
        self._nc = n_subcarriers
        self._na = n_antennas
        self._sigma2 = sigma2
        if initial_indices is not None:
            init = [int(k) for k in initial_indices]
            if len(init) != initial_pilot_subcarriers:
                raise ValueError(
                    "initial_indices length must equal initial_pilot_subcarriers."
                )
            if any(k < 0 or k >= n_subcarriers for k in init):
                raise ValueError("initial_indices out of range.")
            if len(set(init)) != len(init):
                raise ValueError("initial_indices must be unique.")
            self._initial = init
        else:
            self._initial = initial_subcarriers_uniform(
                initial_pilot_subcarriers, n_subcarriers
            )

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        used_set = set(used)
        if len(used_set) >= self._nc:
            raise RuntimeError("ActivePolicy: no unused subcarriers.")
        scores = active_scores_all(state, self._na, self._sigma2).clone()
        if used_set:
            idx = torch.tensor(sorted(used_set), device=scores.device, dtype=torch.long)
            scores[idx] = -float("inf")
        return int(torch.argmax(scores).item())
