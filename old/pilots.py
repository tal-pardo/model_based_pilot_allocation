from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

from data_generator import complex_standard_normal, pilot_matrix_from_indices
from estimators import recursive_lmmse_init, recursive_lmmse_step


def num_timesteps_from_pilot_growth(
    initial_pilot_subcarriers: int,
    final_pilot_subcarriers: int,
    pilots_added_per_step: int,
) -> int:
    k0, kf, dk = initial_pilot_subcarriers, final_pilot_subcarriers, pilots_added_per_step
    if k0 > kf:
        raise ValueError("initial_pilot_subcarriers must be <= final_pilot_subcarriers.")
    if dk < 0:
        raise ValueError("pilots_added_per_step must be non-negative.")
    if dk == 0:
        if k0 != kf:
            raise ValueError("pilots_added_per_step is 0 but initial != final.")
        return 0
    if k0 == kf:
        return 0
    # T counts *new pilot-addition steps* needed to go from k0 to kf.
    return math.ceil((kf - k0) / dk)


def vec_indices_for_subcarriers(
    sc_idx: torch.Tensor, n_antennas: int, device: torch.device
) -> torch.Tensor:
    blocks = []
    for k in sc_idx.tolist():
        start = k * n_antennas
        blocks.append(torch.arange(start, start + n_antennas, device=device, dtype=torch.long))
    return torch.cat(blocks, dim=0)


def ordered_subcarriers_from_vec(idx: torch.Tensor, n_antennas: int) -> List[int]:
    seen = set()
    out: List[int] = []
    for i in idx.tolist():
        k = i // n_antennas
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


@dataclass
class PilotScheduleConfig:
    n_subcarriers: int
    n_antennas: int
    initial_pilot_subcarriers: int
    final_pilot_subcarriers: int
    pilots_added_per_step: int
    cumulative_pilots: bool


class FixedPilotSampler:
    """Evenly spaced initial subcarriers; cumulative growth along a near-uniform final set."""

    def __init__(self, cfg: PilotScheduleConfig, T: int, device: torch.device) -> None:
        self.cfg = cfg
        self.T = T
        self.device = device
        self._schedule = self._build()

    def _build(self) -> List[torch.Tensor]:
        c = self.cfg
        Nc, Na = c.n_subcarriers, c.n_antennas
        T = self.T
        dev = self.device

        if not c.cumulative_pilots:
            out: List[torch.Tensor] = []
            for t in range(T):
                start = (t * c.initial_pilot_subcarriers) % Nc
                sc = (start + torch.arange(c.initial_pilot_subcarriers, device=dev)) % Nc
                out.append(vec_indices_for_subcarriers(sc, Na, dev))
            return out

        step = Nc / float(c.final_pilot_subcarriers)
        sc_final = [int(round(i * step)) % Nc for i in range(c.final_pilot_subcarriers)]
        sc_final = list(dict.fromkeys(sc_final))
        if len(sc_final) < c.final_pilot_subcarriers:
            remaining = [k for k in range(Nc) if k not in set(sc_final)]
            sc_final.extend(remaining[: (c.final_pilot_subcarriers - len(sc_final))])
        sc_final_t = torch.tensor(sc_final, device=dev, dtype=torch.long)

        sc0 = torch.linspace(0, Nc, steps=c.initial_pilot_subcarriers + 1, device=dev, dtype=torch.float32)[
            :-1
        ].round().to(torch.long) % Nc
        sc0 = torch.unique_consecutive(sc0)
        sc0_set = set(sc0.tolist())
        add_order = [k for k in sc_final_t.tolist() if k not in sc0_set]
        sc0_list = sc0.tolist()

        out = []
        for t in range(T):
            k_sc_t = min(c.final_pilot_subcarriers, c.initial_pilot_subcarriers + t * c.pilots_added_per_step)
            if k_sc_t <= len(sc0_list):
                sc_t_list = sc0_list[:k_sc_t]
            else:
                need = k_sc_t - len(sc0_list)
                sc_t_list = sc0_list + add_order[:need]
            sc_t = torch.tensor(sc_t_list, device=dev, dtype=torch.long)
            out.append(vec_indices_for_subcarriers(sc_t, Na, dev))
        return out

    def vec_indices_at_step(self, t: int) -> torch.Tensor:
        return self._schedule[t]


def active_subcarrier_score_J(P: torch.Tensor, k: int, n_antennas: int, sigma2: float) -> float:
    """Greedy score J(k) = tr( (sigma2 I + P_k)^{-1} Q_k ) for candidate subcarrier k."""
    Na = n_antennas
    s = k * Na
    Pk = P[s : s + Na, s : s + Na]
    Qk = P[s : s + Na, :] @ P[:, s : s + Na]
    I = torch.eye(Na, device=P.device, dtype=P.dtype)
    Tm = torch.linalg.solve(sigma2 * I + Pk, Qk)
    return torch.trace(Tm).real.item()


def active_subcarrier_score_block_variance(P: torch.Tensor, k: int, n_antennas: int, sigma2: float) -> float:
    """Alternative scorer: tr(P_k), the k-th diagonal block of posterior covariance."""
    Na = n_antennas
    s = k * Na
    Pk = P[s : s + Na, s : s + Na]
    return torch.trace(Pk).real.item()


class ActivePilotSampler:
    """
    Same initial layout as fixed; then greedily adds subcarriers maximizing
    tr( (sigma^2 I + P_k)^{-1} Q_k ) with P_k = X P X^H block and Q_k the k-th diagonal block of P^2.
    """

    def __init__(
        self,
        cfg: PilotScheduleConfig,
        T: int,
        device: torch.device,
        fixed: FixedPilotSampler,
        sigma2: float,
        score_fn: Callable[[torch.Tensor, int, int, float], float] = active_subcarrier_score_J,
    ) -> None:
        self.cfg = cfg
        self.T = T
        self.device = device
        self._fixed = fixed
        self._sigma2 = sigma2
        self._score_fn = score_fn
        self._used_sc: List[int] = []

    def reset(self) -> None:
        idx0 = self._fixed.vec_indices_at_step(0)
        self._used_sc = ordered_subcarriers_from_vec(idx0, self.cfg.n_antennas)

    def vec_indices_at_step(self, t: int, P: Optional[torch.Tensor]) -> torch.Tensor:
        c = self.cfg
        Na, Nc = c.n_antennas, c.n_subcarriers
        if t == 0:
            self.reset()
            return self._fixed.vec_indices_at_step(0)

        k_prev = min(c.final_pilot_subcarriers, c.initial_pilot_subcarriers + (t - 1) * c.pilots_added_per_step)
        k_cur = min(c.final_pilot_subcarriers, c.initial_pilot_subcarriers + t * c.pilots_added_per_step)
        n_new = k_cur - k_prev
        if P is None:
            raise ValueError("Active sampler needs P for t >= 1.")

        if n_new <= 0:
            return vec_indices_for_subcarriers(
                torch.tensor(self._used_sc, device=self.device, dtype=torch.long), Na, self.device
            )

        unused = [k for k in range(Nc) if k not in set(self._used_sc)]
        sigma2 = self._sigma2
        score_fn = self._score_fn

        scores: List[Tuple[float, int]] = []
        for k in unused:
            score = score_fn(P, k, Na, sigma2)
            scores.append((score, k))

        scores.sort(key=lambda x: -x[0])
        chosen = [scores[i][1] for i in range(min(n_new, len(scores)))]
        self._used_sc = self._used_sc + chosen
        sc_t = torch.tensor(self._used_sc, device=self.device, dtype=torch.long)
        return vec_indices_for_subcarriers(sc_t, Na, self.device)


def sequential_lmmse_mse_curve(
    Sigma: torch.Tensor,
    h_true: torch.Tensor,
    sigma2: float,
    T: int,
    vec_idx_provider,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    on_step=None,
) -> Tuple[List[float], List[float]]:
    """
    One observation per step; vec_idx_provider(t, P_or_none) -> Long vec indices.

    Returns (empirical_mses, theoretical_mses), each of length (T + 1):
    - index 0: estimate after applying the *initial* observation (initial pilot set)
    - indices 1..T: after each additional observation/update
    """
    N = Sigma.shape[0]
    mses: List[float] = []
    theo_mses: List[float] = []

    def one_step(idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_t = pilot_matrix_from_indices(N, idx, device=device, dtype=dtype)
        n_t = (sigma2**0.5) * complex_standard_normal(
            idx.numel(), 1, device=device, dtype=dtype, generator=generator
        )
        y_t = X_t @ h_true + n_t
        return X_t, y_t

    # t=0: initial observation/update (initial pilot set)
    idx0 = vec_idx_provider(0, None)
    X0, y0 = one_step(idx0)
    state = recursive_lmmse_init(Sigma, X0, y0, sigma2)
    mse0 = _mse(state.h, h_true)
    theo0 = posterior_mse(state.P)
    mses.append(mse0)
    theo_mses.append(theo0)
    if on_step is not None:
        on_step(0, idx0, state.h, state.P, mse0)

    # t=1..T: subsequent observations/updates
    for t in range(1, T + 1):
        idx = vec_idx_provider(t, state.P)
        Xt, yt = one_step(idx)
        state = recursive_lmmse_step(state, Xt, yt, sigma2)
        mse_t = _mse(state.h, h_true)
        theo_t = posterior_mse(state.P)
        mses.append(mse_t)
        theo_mses.append(theo_t)
        if on_step is not None:
            on_step(t, idx, state.h, state.P, mse_t)
    return mses, theo_mses


def posterior_mse(P: torch.Tensor) -> float:
    N = P.shape[0]
    return (P.trace().real / N).item()


def _mse(h_hat: torch.Tensor, h_true: torch.Tensor) -> float:
    err = h_hat - h_true
    return (err.abs().pow(2).mean()).real.item()
