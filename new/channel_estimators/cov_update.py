from __future__ import annotations

from typing import Optional, Tuple

import torch

from channel_estimators.lmmse import EstimatorState


def ledoit_wolf_shrinkage(
    h_samples: torch.Tensor,
    *,
    reg: float = 1e-9,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.complex64,
) -> Tuple[torch.Tensor, float]:
    """
    Input: h_samples (K,N) iid draws (K << N expected), reg ridge added after shrinkage.
    Output: (Sigma_LW (N,N), delta in [0,1]), shrunk toward target=(tr(Sigma_hat)/N)*I
    (Ledoit-Wolf complex-Hermitian shrinkage intensity).
    """
    if h_samples.ndim == 3 and h_samples.shape[-1] == 1:
        h_samples = h_samples.squeeze(-1)
    h_samples = h_samples.to(device=device, dtype=dtype)
    k, n = h_samples.shape
    sigma_hat = (h_samples.T @ h_samples.conj()) / float(k)
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.mH)

    mu = (sigma_hat.trace().real / n).item()
    eye_n = torch.eye(n, device=sigma_hat.device, dtype=sigma_hat.dtype)
    target = mu * eye_n
    d2 = (sigma_hat - target).abs().pow(2).sum().item()
    if d2 <= 0.0:
        delta = 1.0
    else:
        x = h_samples.unsqueeze(-1)  # (K,N,1)
        outer = x @ x.mH  # (K,N,N)
        diff = outer - sigma_hat.unsqueeze(0)
        b2_bar = diff.abs().pow(2).sum().item() / float(k**2)
        delta = float(min(max(min(b2_bar, d2) / d2, 0.0), 1.0))

    sigma_lw = delta * target + (1.0 - delta) * sigma_hat
    sigma_lw = 0.5 * (sigma_lw + sigma_lw.mH)
    sigma_lw = sigma_lw + reg * eye_n
    return sigma_lw, delta


def inflate_covariance(
    p: torch.Tensor,
    *,
    n_pilots: int,
    mode: str = "decaying",
    factor0: float = 0.05,
    decay: float = 0.9,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Input: current posterior P (N,N), pilot count so far, inflation schedule params.
    Output: P_inflated = (1+factor_t)*P + eps*I, re-Hermitized. factor_t = factor0 if
    mode=="constant" else factor0*(decay**max(n_pilots-1, 0)) if mode=="decaying".
    """
    if mode == "constant":
        factor = factor0
    elif mode == "decaying":
        factor = factor0 * (decay ** max(n_pilots - 1, 0))
    else:
        raise ValueError(f"Unknown inflation mode: {mode!r}")
    n = p.shape[0]
    eye_n = torch.eye(n, device=p.device, dtype=p.dtype)
    p_inflated = (1.0 + factor) * p + eps * eye_n
    return 0.5 * (p_inflated + p_inflated.mH)


def sbl_alpha_update(
    a_hat: torch.Tensor,
    p_delay_diag: torch.Tensor,
    alpha: torch.Tensor,
    *,
    alpha_min: float = 1e-6,
    alpha_max: float = 1e8,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Input: delay-Doppler posterior mean a_hat (N,1), diag(P in that basis) real (N,), current
    alpha (N,) real precisions. Output: updated alpha via Tipping's SBL fixed point:
    gamma_i = clamp(1 - alpha_i*p_delay_diag_i, 0, 1), alpha_i_new = clamp(gamma_i /
    max(|a_hat_i|^2, eps), alpha_min, alpha_max).
    """
    a_flat = a_hat.reshape(-1)
    gamma = (1.0 - alpha * p_delay_diag).clamp(min=0.0, max=1.0)
    power = a_flat.abs().pow(2).clamp(min=eps)
    alpha_new = gamma / power
    return alpha_new.clamp(min=alpha_min, max=alpha_max)


def damp_sigma_update(
    sigma_prev: torch.Tensor,
    sigma_candidate: torch.Tensor,
    *,
    damping: float,
) -> torch.Tensor:
    """
    Input: previous Sigma, newly rebuilt candidate Sigma, damping weight on sigma_prev in [0,1].
    Output: damping*sigma_prev + (1-damping)*sigma_candidate, re-Hermitized -- prevents one
    noisy/under-determined SBL rebuild (too few pilots relative to the number of free
    delay-Doppler coefficients) from fully overwriting the running Sigma belief in one step.
    """
    blended = damping * sigma_prev + (1.0 - damping) * sigma_candidate
    return 0.5 * (blended + blended.mH)


def rebuild_state_from_sigma(
    sigma_new: torch.Tensor,
    j_data: torch.Tensor,
    b_data: torch.Tensor,
) -> EstimatorState:
    """
    Input: new prior Sigma (N,N), running data terms j_data=sum (1/sigma2) x^H x (N,N),
    b_data=sum (1/sigma2) x^H y (N,1). Output: EstimatorState under the new prior via two
    O(N^3) solves (cost per rebuild event, not per pilot): j_new=solve(sigma_new,I)+j_data,
    P=solve(j_new,I), h_hat=P@b_data.
    """
    n = sigma_new.shape[0]
    eye_n = torch.eye(n, device=sigma_new.device, dtype=sigma_new.dtype)
    sigma_inv = torch.linalg.solve(sigma_new, eye_n)
    j_new = sigma_inv + j_data
    p_new = torch.linalg.solve(j_new, eye_n)
    h_new = p_new @ b_data
    p_new = 0.5 * (p_new + p_new.mH)
    return EstimatorState(h_hat=h_new, P=p_new)
