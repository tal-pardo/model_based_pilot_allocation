from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class RecursiveLMMSEState:
    h: torch.Tensor  # (N,1) complex
    P: torch.Tensor  # (N,N) complex Hermitian PSD


def _as_col(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError("Expected a vector of shape (N,) or column (N,1).")


def batch_lmmse(
    Sigma: torch.Tensor,
    X_all: torch.Tensor,
    y_all: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """
    Batch LMMSE for y = X h + n, with h~CN(0,Sigma), n~CN(0,sigma2 I).

    Uses the equivalent form (avoids N x N inversion):
      h_hat = Sigma X^H (X Sigma X^H + sigma2 I)^{-1} y
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    y_all = _as_col(y_all)

    N = Sigma.shape[-1]
    if Sigma.shape != (N, N):
        raise ValueError("Sigma must be (N,N).")
    if X_all.ndim != 2 or X_all.shape[1] != N:
        raise ValueError("X_all must have shape (M,N) with N matching Sigma.")
    if y_all.shape[0] != X_all.shape[0]:
        raise ValueError("y_all and X_all must have matching number of rows.")

    M = X_all.shape[0]
    I_M = torch.eye(M, device=Sigma.device, dtype=Sigma.dtype)

    XS = X_all @ Sigma  # (M,N)
    S = (XS @ X_all.mH) + (sigma2 * I_M)  # (M,M)
    tmp = torch.linalg.solve(S, y_all)  # (M,1)
    h_hat = Sigma @ (X_all.mH @ tmp)  # (N,1)
    return h_hat


def recursive_lmmse_init(
    Sigma: torch.Tensor,
    X0: torch.Tensor,
    y0: torch.Tensor,
    sigma2: float,
) -> RecursiveLMMSEState:
    """
    Initializes with the prior:
      h_0 = 0
      P_0 = Sigma

    Then performs the first measurement update using X0, y0.
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    y0 = _as_col(y0)
    N = Sigma.shape[-1]
    if Sigma.shape != (N, N):
        raise ValueError("Sigma must be (N,N).")
    if X0.ndim != 2 or X0.shape[1] != N:
        raise ValueError("X0 must have shape (m0, N).")
    if y0.shape[0] != X0.shape[0] or y0.shape[1] != 1:
        raise ValueError("y0 must be (m0,1) matching X0 rows.")

    h_init = torch.zeros((N, 1), device=Sigma.device, dtype=Sigma.dtype)
    P_init = 0.5 * (Sigma + Sigma.mH)
    state0 = RecursiveLMMSEState(h=h_init, P=P_init)
    return recursive_lmmse_step(state0, X0, y0, sigma2)


def recursive_lmmse_step(
    state: RecursiveLMMSEState,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    sigma2: float,
) -> RecursiveLMMSEState:
    """
    Kalman-style measurement update for a static channel.

    Gain: K = P X^H (sigma2 I + X P X^H)^{-1}
    Update: h <- h + K (y - X h)
    Cov:   P <- (I - K X) P
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    y_t = _as_col(y_t)
    h_prev, P_prev = state.h, state.P

    N = h_prev.shape[0]
    if h_prev.shape != (N, 1) or P_prev.shape != (N, N):
        raise ValueError("state.h must be (N,1) and state.P must be (N,N).")
    if X_t.ndim != 2 or X_t.shape[1] != N:
        raise ValueError("X_t must have shape (m_t, N).")
    if y_t.shape != (X_t.shape[0], 1):
        raise ValueError("y_t must be (m_t,1) matching X_t rows.")

    m_t = X_t.shape[0]
    I_m = torch.eye(m_t, device=h_prev.device, dtype=h_prev.dtype)

    S = (X_t @ P_prev @ X_t.mH) + (sigma2 * I_m)  # (m_t,m_t)
    K = (P_prev @ X_t.mH) @ torch.linalg.solve(S, I_m)  # (N,m_t)

    innovation = y_t - (X_t @ h_prev)  # (m_t,1)
    h_new = h_prev + (K @ innovation)  # (N,1)
    P_new = P_prev - (K @ X_t @ P_prev)  # (N,N)
    P_new = 0.5 * (P_new + P_new.mH)
    return RecursiveLMMSEState(h=h_new, P=P_new)


def recursive_lmmse(
    Sigma: torch.Tensor,
    X_list: List[torch.Tensor],
    y_list: List[torch.Tensor],
    sigma2: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs recursive LMMSE over t=0..T-1.
    Returns (h_list, P_list).
    """
    if len(X_list) == 0 or len(y_list) == 0 or len(X_list) != len(y_list):
        raise ValueError("X_list and y_list must be non-empty and have equal length.")
    state = recursive_lmmse_init(Sigma, X_list[0], y_list[0], sigma2)
    h_list = [state.h]
    P_list = [state.P]
    for t in range(1, len(X_list)):
        state = recursive_lmmse_step(state, X_list[t], y_list[t], sigma2)
        h_list.append(state.h)
        P_list.append(state.P)
    return h_list, P_list

