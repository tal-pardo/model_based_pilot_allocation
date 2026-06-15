from __future__ import annotations

import torch

from channel_estimators.lmmse import EstimatorState


def estimate_nmse(state: EstimatorState, n: int) -> float:
    """
    Input: EstimatorState with P (N,N), channel dimension N.
    Output: tr(P)/N (posterior NMSE under matched prior).
    """
    return (state.P.trace().real / n).item()


def active_subcarrier_score(
    state: EstimatorState,
    k: int,
    n_antennas: int,
    sigma2: float,
) -> float:
    """
    Input: state with P, candidate subcarrier k, Na, sigma2.
    Output: J(k)=tr((sigma2 I + P_k)^{-1} Q_k) for greedy active piloting.
    """
    p = state.P
    s = k * n_antennas
    p_k = p[s : s + n_antennas, s : s + n_antennas]
    q_k = p[s : s + n_antennas, :] @ p[:, s : s + n_antennas]
    eye = torch.eye(n_antennas, device=p.device, dtype=p.dtype)
    tm = torch.linalg.solve(sigma2 * eye + p_k, q_k)
    return tm.trace().real.item()
