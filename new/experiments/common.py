from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from channel_estimators.lmmse import lmmse_incremental_update, lmmse_initial_update
from config import ExpRunConfig
from error_estimators.trace_min import estimate_nmse
from pilot_policy.active import ActivePolicy
from pilot_policy.fixed import FixedPolicy
from pilot_policy.random import RandomPolicy
from simulation import RunTrace
from utils import empirical_nmse, measure_subcarrier, stack_observations

EXP3_STYLES: dict[str, tuple[str, str | tuple, str | None]] = {
    "fixed true": ("C0", "-", "o"),
    "fixed est": ("C0", (0, (1, 2)), None),
    "active true": ("C1", "-", "s"),
    "active est": ("C1", (0, (1, 2)), None),
}


def make_policy(name: str, cfg: ExpRunConfig, device: torch.device, seed: int):
    """Input: policy name, run config, device, seed. Output: PilotPolicy instance."""
    common = dict(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        max_pilots=cfg.max_pilots,
        device=device,
    )
    if name == "fixed":
        return FixedPolicy(**common)
    if name == "random":
        return RandomPolicy(**common, seed=seed)
    if name == "active":
        return ActivePolicy(**common, sigma2=cfg.sigma2)
    raise ValueError(f"Unknown policy: {name}")


def mean_curve(traces: Sequence[RunTrace], *, field: str = "nmse_hat") -> np.ndarray:
    """Input: MC RunTrace list. Output: mean per-step curve (NaN-padded)."""
    max_len = max(len(getattr(t, field)) for t in traces)
    buf = np.full((len(traces), max_len), np.nan, dtype=np.float64)
    for i, t in enumerate(traces):
        values = getattr(t, field)
        buf[i, : len(values)] = values
    return np.nanmean(buf, axis=0)


def first_cross_step(curve: np.ndarray, threshold: float) -> int | None:
    """Input: mean curve, threshold. Output: first step index where curve <= threshold."""
    hits = np.where(curve <= threshold)[0]
    return int(hits[0]) if hits.size else None


def mean_pilots_to_threshold(traces: Sequence[RunTrace], threshold: float) -> float:
    """Input: MC traces, threshold. Output: mean pilot count to reach threshold."""
    counts = []
    for t in traces:
        hit = next((i + 1 for i, v in enumerate(t.nmse_hat) if v <= threshold), t.n_pilots)
        counts.append(hit)
    return float(np.mean(counts))


def plot_len_for_threshold(curves: dict[str, np.ndarray], threshold: float) -> int:
    """Input: name -> mean curve, threshold. Output: x-axis length for threshold plots."""
    cross_steps = []
    for curve in curves.values():
        step = first_cross_step(curve, threshold)
        if step is not None:
            cross_steps.append(step)
    if cross_steps:
        return max(cross_steps) + 1
    return max(len(c) for c in curves.values())


def plot_len_for_curves(
    curves: dict[str, np.ndarray],
    *,
    threshold: float | None = None,
    plot_len: int | None = None,
) -> int:
    """Input: curves, optional threshold and fixed plot_len. Output: x-axis length."""
    if plot_len is not None:
        return plot_len
    if threshold is not None:
        return plot_len_for_threshold(curves, threshold)
    return max(len(c) for c in curves.values())


def plot_threshold_curves(
    curves: dict[str, np.ndarray],
    *,
    threshold: float | None = None,
    plot_len: int | None = None,
    out_path: Path,
    title: str,
    styles: dict[str, tuple[str, str, str]],
    ylabel: str = "Mean estimated MSE  tr(P)/N",
    xlabel: str = "Time step",
    x_tick_step: int = 5,
    extra_hlines: Sequence[tuple[float, str, str, str]] | None = None,
) -> None:
    """
    Input: name -> mean curve, styles name -> (color, linestyle, marker).
    Output: saves semilogy PNG to out_path.
    extra_hlines: optional (value, color, linestyle, label) tuples.
    """
    n_steps = plot_len_for_curves(curves, threshold=threshold, plot_len=plot_len)
    plt.figure(figsize=(8.4, 4.8))
    ax = plt.gca()
    steps = np.arange(n_steps)
    for name, curve in curves.items():
        color, ls, mk = styles[name]
        y = np.full(n_steps, np.nan, dtype=np.float64)
        n_copy = min(len(curve), n_steps)
        y[:n_copy] = curve[:n_copy]
        ax.semilogy(steps, y, marker=mk, linestyle=ls, linewidth=1.6, color=color, label=name)

    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2, label="threshold")
    if extra_hlines:
        for value, color, ls, label in extra_hlines:
            ax.axhline(value, color=color, linestyle=ls, linewidth=1.2, label=label)

    tick_end = n_steps + (x_tick_step - n_steps % x_tick_step) % x_tick_step
    ax.set_xticks(np.arange(0, tick_end, x_tick_step))
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved figure: {out_path}")


def full_subcarrier_lmmse_nmse(
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    cfg: ExpRunConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Input: h_true, Sigma, config, device, noise generator. Output: tr(P)/N after all Nc subcarriers."""
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2

    x0, y0 = measure_subcarrier(
        0, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
    )
    state = lmmse_initial_update(sigma, x0, y0, sigma2)
    for k in range(1, nc):
        x, y = measure_subcarrier(
            k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
        )
        state = lmmse_incremental_update(state, x, y, sigma2)
    return estimate_nmse(state, n)


def full_subcarrier_batch_true_mse(
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    cfg: ExpRunConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
    subcarriers: Sequence[int] | None = None,
) -> float:
    """
    Input: h_true, Sigma, config, device, noise generator, optional subcarrier list.
    Output: true MSE after one-shot LMMSE on stacked pilots (equivalent to incremental chain).
    """
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    dtype = cfg.dtype
    sigma2 = cfg.sigma2
    if subcarriers is None:
        subcarriers = range(nc)

    x_list: list[torch.Tensor] = []
    y_list: list[torch.Tensor] = []
    for k in subcarriers:
        x, y = measure_subcarrier(
            k, h_true, sigma2, n_antennas=na, n_total=n, device=device, dtype=dtype, generator=generator
        )
        x_list.append(x)
        y_list.append(y)
    x_all, y_all = stack_observations(x_list, y_list)
    state = lmmse_initial_update(sigma, x_all, y_all, sigma2)
    return empirical_nmse(state.h_hat, h_true)


def full_subcarrier_true_mse(
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    cfg: ExpRunConfig,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    """Input: h_true, Sigma, config, device, noise generator. Output: true MSE after all Nc subcarriers."""
    return full_subcarrier_batch_true_mse(
        h_true, sigma, cfg, device=device, generator=generator, subcarriers=None
    )


def run_mc_mean_full_true_lmmse_mse(
    cfg: ExpRunConfig,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    *,
    device: torch.device,
    noise_seed_offset: int = 30_000,
) -> float:
    """Input: MC channel list, Sigma, noise seed offset. Output: mean batch true MSE (all Nc subcarriers)."""
    values: list[float] = []
    for mc, h_true in enumerate(h_list):
        gen_noise = torch.Generator(device=device).manual_seed(cfg.seed + noise_seed_offset + mc)
        values.append(
            full_subcarrier_batch_true_mse(h_true, sigma, cfg, device=device, generator=gen_noise)
        )
    return float(np.mean(values))


def run_mc_mean_full_true_baselines_by_policy(
    cfg: ExpRunConfig,
    h_list: list[torch.Tensor],
    sigma: torch.Tensor,
    *,
    device: torch.device,
    noise_seed_offset_by_policy: dict[str, int],
) -> dict[str, float]:
    """Input: MC channels, Sigma, policy->seed offsets. Output: mean batch true MSE per policy seed."""
    return {
        policy: run_mc_mean_full_true_lmmse_mse(
            cfg, h_list, sigma, device=device, noise_seed_offset=offset
        )
        for policy, offset in noise_seed_offset_by_policy.items()
    }


def full_lmmse_nmse_for_sigma(
    sigma: torch.Tensor,
    cfg: ExpRunConfig,
    *,
    device: torch.device,
    h_true: torch.Tensor | None = None,
) -> float:
    """Input: Sigma, config, device, optional h_true. Output: tr(P)/N after measuring subcarriers 0..Nc-1."""
    if h_true is None:
        n = cfg.n_antennas * cfg.n_subcarriers
        h_true = torch.zeros((n, 1), device=device, dtype=cfg.dtype)
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 99_999)
    return full_subcarrier_lmmse_nmse(h_true, sigma, cfg, device=device, generator=gen)


def build_exp3_curves(
    traces_by_policy: dict[str, list[RunTrace]],
) -> dict[str, np.ndarray]:
    """Input: policy -> MC traces. Output: fixed/active x true/est mean curves."""
    curves: dict[str, np.ndarray] = {}
    for policy in ("fixed", "active"):
        traces = traces_by_policy[policy]
        curves[f"{policy} true"] = mean_curve(traces, field="nmse_true")
        curves[f"{policy} est"] = mean_curve(traces, field="nmse_hat")
    return curves


def print_exp3_final_gaps(curves: dict[str, np.ndarray], cfg: ExpRunConfig, *, label: str) -> None:
    """Input: exp3 curve dict, config, panel label. Output: prints true vs est gaps at t=T."""
    _, trace_slice = exp3_trace_slice(cfg)
    t_final = exp3_time_steps(cfg)[-1]
    for policy in ("fixed", "active"):
        true_sliced = curves[f"{policy} true"][trace_slice]
        est_sliced = curves[f"{policy} est"][trace_slice]
        true_final = float(true_sliced[-1])
        est_final = float(est_sliced[-1])
        print(
            f"exp3 {label} {policy}: t={t_final} true={true_final:.6e}, "
            f"est={est_final:.6e}, gap={true_final - est_final:.6e}"
        )


def exp3_time_steps(cfg: ExpRunConfig) -> np.ndarray:
    """
    Time steps t=0..T with k0 initial pilots at t=0 (before policy adds) and target_pilots at t=T.
    Example: k0=2, target=16 -> t=0..14 (14 policy additions after initialization).
    """
    k0 = cfg.initial_pilot_subcarriers
    target = cfg.target_pilots or cfg.max_pilots
    if target < k0:
        raise ValueError(f"target_pilots ({target}) must be >= initial_pilot_subcarriers ({k0}).")
    return np.arange(target - k0 + 1)


def exp3_trace_slice(cfg: ExpRunConfig) -> tuple[np.ndarray, slice]:
    """
    Input: exp3 run config. Output: time steps t=0..T and slice into RunTrace curves.
    RunTrace index 0 is after the first init subcarrier; index k0-1 is t=0 (k0 pilots, pre-policy).
    """
    steps = exp3_time_steps(cfg)
    k0 = cfg.initial_pilot_subcarriers
    target = cfg.target_pilots or cfg.max_pilots
    return steps, slice(k0 - 1, target)


def plot_true_vs_est_overlay(
    ax,
    curves: dict[str, np.ndarray],
    *,
    title: str,
    n_subcarriers: int,
    cfg: ExpRunConfig,
    x_tick_step: int = 4,
    full_lmmse_baselines: dict[str, float] | None = None,
) -> None:
    """Input: axes, name->curve, title, per-policy full-LMMSE baselines. Output: semilogy overlay on ax."""
    steps, trace_slice = exp3_trace_slice(cfg)
    t_max = int(steps[-1])
    for name, curve in curves.items():
        color, ls, mk = EXP3_STYLES[name]
        y = np.asarray(curve[trace_slice], dtype=np.float64)
        if len(y) != len(steps):
            raise ValueError(
                f"exp3 curve '{name}' has {len(y)} points after slice {trace_slice}, "
                f"expected {len(steps)} (t=0..{t_max})."
            )
        plot_kwargs = dict(linestyle=ls, linewidth=1.6, color=color, label=name)
        if mk is not None:
            plot_kwargs["marker"] = mk
        ax.semilogy(steps, y, **plot_kwargs)

    if full_lmmse_baselines:
        baseline_values = [full_lmmse_baselines[policy] for policy in ("fixed", "active")]
        ax.axhline(
            float(np.mean(baseline_values)),
            color="red",
            linestyle=":",
            linewidth=1.2,
            label=f"full LMMSE true (Nc={n_subcarriers})",
        )
    ticks = list(np.arange(0, t_max + 1, x_tick_step))
    if t_max not in ticks:
        ticks.append(t_max)
    ax.set_xticks(ticks)
    ax.set_xlim(-0.5, t_max + 0.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step t")
    ax.set_ylabel("Mean MSE  true or tr(P)/N")
    ax.legend(fontsize=8)


def save_exp3_gaussian_figure(
    curves: dict[str, np.ndarray],
    *,
    cfg: ExpRunConfig,
    full_lmmse_baselines: dict[str, float],
    out_path: Path,
) -> None:
    """Input: exp3 curves, config, per-policy baselines, path. Output: saves single-panel PNG."""
    target = cfg.target_pilots or cfg.max_pilots
    title = (
        f"exp3 Gaussian  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, k0={cfg.initial_pilot_subcarriers}, pilots={target}, Kronecker Sigma"
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    plot_true_vs_est_overlay(
        ax,
        curves,
        title=title,
        n_subcarriers=cfg.n_subcarriers,
        cfg=cfg,
        full_lmmse_baselines=full_lmmse_baselines,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")


def save_exp3_tdl_figure(
    panels: list[tuple[str, dict[str, np.ndarray], dict[str, float]]],
    *,
    cfg: ExpRunConfig,
    out_path: Path,
) -> None:
    """Input: list of (panel_title, curves, baselines_by_policy), config, path. Output: 1x2 PNG."""
    target = cfg.target_pilots or cfg.max_pilots
    suptitle = (
        f"exp3 TDL-A  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, k0={cfg.initial_pilot_subcarriers}, pilots={target}"
    )
    save_exp_multi_panel_err_figure(panels, cfg=cfg, out_path=out_path, suptitle=suptitle)


def save_exp_multi_panel_err_figure(
    panels: list[tuple[str, dict[str, np.ndarray], dict[str, float]]],
    *,
    cfg: ExpRunConfig,
    out_path: Path,
    suptitle: str,
) -> None:
    """Input: list of (panel_title, curves, baselines_by_policy), config, path, suptitle. Output: 1xN PNG."""
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.8), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (panel_title, curves, full_lmmse_baselines) in zip(axes, panels):
        plot_true_vs_est_overlay(
            ax,
            curves,
            title=panel_title,
            n_subcarriers=cfg.n_subcarriers,
            cfg=cfg,
            full_lmmse_baselines=full_lmmse_baselines,
        )
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out_path}")
