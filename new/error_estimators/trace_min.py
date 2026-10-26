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


def active_scores_all(
    state: EstimatorState,
    n_antennas: int,
    sigma2: float,
) -> torch.Tensor:
    """
    Input: state with P (N,N), N=nc*n_antennas, n_antennas, sigma2.
    Output: real (nc,) tensor, scores[k] = tr((sigma2 I + P_k)^{-1} Q_k) for every subcarrier k
    at once -- a vectorized form of active_subcarrier_score across all candidates in a single
    batched solve, instead of one Python-level call (and GPU kernel launch) per candidate.

    Q_k = P[s:s+na,:] @ P[:,s:s+na] is exactly the (k,k) block of P@P (P is Hermitian, so
    column-block k equals (row-block k)^H), so every Q_k comes from one P@P matmul rather than
    nc separate block products.
    """
    p = state.P
    n = p.shape[0]
    nc = n // n_antennas
    p2 = p @ p
    p_blocks = p.reshape(nc, n_antennas, nc, n_antennas)
    p2_blocks = p2.reshape(nc, n_antennas, nc, n_antennas)
    p_k_all = torch.diagonal(p_blocks, dim1=0, dim2=2).permute(2, 0, 1)  # (nc, na, na)
    q_k_all = torch.diagonal(p2_blocks, dim1=0, dim2=2).permute(2, 0, 1)  # (nc, na, na)
    eye = torch.eye(n_antennas, device=p.device, dtype=p.dtype).unsqueeze(0)
    tm = torch.linalg.solve(sigma2 * eye + p_k_all, q_k_all)  # (nc, na, na)
    return torch.diagonal(tm, dim1=-2, dim2=-1).sum(-1).real
