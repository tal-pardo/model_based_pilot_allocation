from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EstimatorState:
    h_hat: torch.Tensor  # (N, 1)
    P: torch.Tensor  # (N, N)


def _as_col(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x[:, None]
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    raise ValueError("Expected vector shape (N,) or (N,1).")


def lmmse_initial_update(
    sigma: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    sigma2: float,
) -> EstimatorState:
    """
    Input: Sigma (N,N), X0 (Na,N), y0 (Na,1), sigma2.
    Output: EstimatorState after P0=(Sigma^{-1}+(1/sigma2)X0^H X0)^{-1}, h_hat0=P0 (1/sigma2) X0^H y0.
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    y0 = _as_col(y0)
    n = sigma.shape[0]
    eye_n = torch.eye(n, device=sigma.device, dtype=sigma.dtype)
    sigma_inv = torch.linalg.solve(sigma, eye_n)
    j0 = sigma_inv + (1.0 / sigma2) * (x0.mH @ x0)
    p0 = torch.linalg.solve(j0, eye_n)
    h_hat0 = p0 @ ((1.0 / sigma2) * (x0.mH @ y0))
    p0 = 0.5 * (p0 + p0.mH)
    return EstimatorState(h_hat=h_hat0, P=p0)


def lmmse_incremental_update(
    state: EstimatorState,
    x: torch.Tensor,
    y: torch.Tensor,
    sigma2: float,
) -> EstimatorState:
    """
    Input: state (h_hat, P), X (Na,N), y (Na,1), sigma2.
    Output: updated state via K=PX^H(S)^{-1}, h=(I-KX)h+Ky, P=(I-KX)P.
    """
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    y = _as_col(y)
    h_prev, p_prev = state.h_hat, state.P
    n = h_prev.shape[0]
    na = x.shape[0]
    eye_n = torch.eye(n, device=h_prev.device, dtype=h_prev.dtype)
    eye_na = torch.eye(na, device=h_prev.device, dtype=h_prev.dtype)
    s = sigma2 * eye_na + x @ p_prev @ x.mH
    k_gain = p_prev @ x.mH @ torch.linalg.solve(s, eye_na)
    i_kx = eye_n - k_gain @ x
    h_new = i_kx @ h_prev + k_gain @ y
    p_new = i_kx @ p_prev
    p_new = 0.5 * (p_new + p_new.mH)
    return EstimatorState(h_hat=h_new, P=p_new)
