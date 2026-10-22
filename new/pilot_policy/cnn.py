from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch

from channel_estimators.lmmse import EstimatorState
from error_estimators.cnn.train import CNNErrorEstimator, build_features, load_cnn_checkpoint
from pilot_policy.base import initial_subcarriers_uniform


class CnnPolicy:
    """Greedy pilot selection via a trained CNNErrorEstimator (counterfactual label paradigm).

    The model predicts, for each unused subcarrier, the log-NMSE that would result from
    hypothetically placing a pilot there (see error_estimators/cnn/train.py::counterfactual_labels).
    Lower predicted score is better, so selection is argmin over unused subcarriers (unlike the
    current-error paradigm's argmax used internally by train.py's own CnnPilotPolicy).

    Exactly one of `checkpoint_path` or `model` must be given.
    """

    def __init__(
        self,
        *,
        n_subcarriers: int,
        n_antennas: int,
        initial_pilot_subcarriers: int,
        max_pilots: int,
        sigma2: float,
        device: torch.device,
        checkpoint_path: Optional[Path] = None,
        model: Optional[CNNErrorEstimator] = None,
        initial_indices: Optional[Sequence[int]] = None,
    ) -> None:
        _ = max_pilots
        if (checkpoint_path is None) == (model is None):
            raise ValueError("Pass exactly one of checkpoint_path or model.")
        self._nc = n_subcarriers
        self._na = n_antennas
        self._sigma2 = sigma2
        self._device = device
        if model is not None:
            self._model = model
        else:
            self._model = load_cnn_checkpoint(Path(checkpoint_path), device)
        self._model.eval()

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

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return list(self._initial)

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        used_set = set(used)
        unused = [k for k in range(self._nc) if k not in used_set]
        if not unused:
            raise RuntimeError("CnnPolicy: no unused subcarriers.")
        mask = torch.zeros(self._nc, device=self._device, dtype=torch.float32)
        for k in used:
            mask[k] = 1.0
        x = build_features(state.h_hat, mask, len(used), self._sigma2, na=self._na, nc=self._nc)
        with torch.no_grad():
            pred = self._model(x.unsqueeze(0)).squeeze(0)
        scores = pred.clone()
        for k in used:
            scores[k] = float("inf")
        return int(scores.argmin().item())
