from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

import torch

from channel_estimators.lmmse import EstimatorState, lmmse_incremental_update, lmmse_initial_update
from config import ExpRunConfig
from error_estimators.trace_min import estimate_nmse
from pilot_policy.base import PilotPolicy
from utils import empirical_nmse, measure_subcarrier


@dataclass
class RunTrace:
    nmse_true: List[float] = field(default_factory=list)
    nmse_hat: List[float] = field(default_factory=list)
    subcarriers: List[int] = field(default_factory=list)
    n_pilots: int = 0
    stopped_reason: Literal["threshold", "max_pilots", "target_pilots"] = "max_pilots"


def run_until_threshold(
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    cfg: ExpRunConfig,
    policy: PilotPolicy,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> RunTrace:
    """
    Input: h_true, Sigma, config, pilot policy, device, noise generator.
    Output: RunTrace with per-step true/estimated NMSE until target_pilots, threshold, or max_pilots.
    """
    if cfg.target_pilots is not None and cfg.target_pilots > cfg.max_pilots:
        raise ValueError(
            f"target_pilots ({cfg.target_pilots}) cannot exceed max_pilots ({cfg.max_pilots})."
        )

    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2
    trace = RunTrace()

    policy.reset()
    initial = policy.initial_subcarriers()
    if not initial:
        raise ValueError("initial_pilot_subcarriers must be >= 1.")

    used: List[int] = []

    k0 = initial[0]
    x0, y0 = measure_subcarrier(
        k0, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
    )
    state = lmmse_initial_update(sigma, x0, y0, sigma2)
    used.append(k0)
    trace.subcarriers.append(k0)
    trace.n_pilots = 1
    trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
    trace.nmse_hat.append(estimate_nmse(state, n))

    for k in initial[1:]:
        x, y = measure_subcarrier(
            k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
        )
        state = lmmse_incremental_update(state, x, y, sigma2)
        used.append(k)
        trace.subcarriers.append(k)
        trace.n_pilots += 1
        trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
        trace.nmse_hat.append(estimate_nmse(state, n))

    estimate_err = trace.nmse_hat[-1]
    while trace.n_pilots < cfg.max_pilots:
        if cfg.target_pilots is not None:
            if trace.n_pilots >= cfg.target_pilots:
                break
        elif estimate_err <= cfg.nmse_threshold:
            break

        k = policy.next_subcarrier(state, used)
        x, y = measure_subcarrier(
            k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
        )
        state = lmmse_incremental_update(state, x, y, sigma2)
        used.append(k)
        trace.subcarriers.append(k)
        trace.n_pilots += 1
        trace.nmse_true.append(empirical_nmse(state.h_hat, h_true))
        estimate_err = estimate_nmse(state, n)
        trace.nmse_hat.append(estimate_err)

    if cfg.target_pilots is not None and trace.n_pilots >= cfg.target_pilots:
        trace.stopped_reason = "target_pilots"
    elif estimate_err <= cfg.nmse_threshold:
        trace.stopped_reason = "threshold"
    else:
        trace.stopped_reason = "max_pilots"
    return trace
