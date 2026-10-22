"""
Single-channel diagnostic for the trained counterfactual-label CNN allocator.

Two questions this answers:
  1. How close is the CNN's predicted per-subcarrier score to the real oracle
     (counterfactual_labels, computed from h_true) it was trained to imitate?
  2. Once the policy places a pilot at subcarrier k, does the TRUE current error
     at k actually drop in the very next step?

Not an experiment (no MC averaging, no figures/ comparison claim) -- a one-sample
rollout inspection tool. Lives entirely under error_estimators/cnn/, consistent
with the cnn/-only implementation constraint; only imports from pilot_policy/cnn.py
(read-only) for reference, and otherwise reuses train.py's existing functions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

_CNN_DIR = Path(__file__).resolve().parent
_NEW_ROOT = _CNN_DIR.parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.lmmse import lmmse_incremental_update, lmmse_initial_update
from channel_generators.sionna import SionnaOFDMGrid
from error_estimators.cnn.train import (
    CnnTrainConfig,
    block_error_labels,
    build_features,
    counterfactual_labels,
    load_cnn_checkpoint,
    load_empirical_sigma,
    sample_tdl_channel,
)
from utils import measure_subcarrier, resolve_device

DEFAULT_CHECKPOINT = _CNN_DIR / "checkpoints" / "phase1_cf_best.pt"
FIGURES_DIR = _CNN_DIR / "figures"
INSPECT_NOISE_OFFSET = 50_000  # disjoint from build/eval/phase2/exp11 noise blocks
MC_ORACLE_SEED_BASE = 88_000_000  # dedicated, never touches the main rollout's generator stream


def mc_averaged_oracle(
    state,
    h_true: torch.Tensor,
    used: List[int],
    na: int,
    nc: int,
    sigma2: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    label_eps: float,
    n_draws: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average counterfactual_labels over n_draws independent noise realizations at a fixed state.

    Returns (mean_log_label, std_log_label), each (Nc,) with used SCs = nan. The mean approximates
    what a Huber-trained regressor converges to (the per-sample log-label is the actual SGD target,
    not the log of the mean raw error), and the std quantifies single-draw label noise per candidate.
    """
    draws = np.zeros((n_draws, nc), dtype=np.float64)
    for m in range(n_draws):
        gen_m = torch.Generator(device=device).manual_seed(seed + m)
        draws[m] = counterfactual_labels(
            state, h_true, used, na, nc, sigma2, gen_m, label_eps=label_eps
        ).cpu().numpy()
    used_set = set(used)
    mean_log = draws.mean(axis=0)
    std_log = draws.std(axis=0)
    for k in used_set:
        mean_log[k] = np.nan
        std_log[k] = np.nan
    return mean_log, std_log


def run_one_rollout(
    cfg: CnnTrainConfig,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    model,
    device: torch.device,
    channel_mc: int,
    mc_oracle_draws: int = 0,
) -> Dict[str, Any]:
    na, nc, n = cfg.n_antennas, cfg.n_subcarriers, cfg.n_total
    noise_seed = cfg.seed + INSPECT_NOISE_OFFSET + channel_mc
    gen = torch.Generator(device=device).manual_seed(noise_seed)

    k_init = 0
    x0, y0 = measure_subcarrier(
        k_init, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
    )
    state = lmmse_initial_update(sigma, x0, y0, cfg.sigma2)
    used: List[int] = [k_init]
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    mask[k_init] = 1.0
    n_pilots = 1

    model.eval()
    true_err_hist: List[np.ndarray] = []
    pred_hist: List[np.ndarray] = []
    oracle_hist: List[np.ndarray] = []
    oracle_mc_hist: List[np.ndarray] = []
    oracle_mc_std_hist: List[np.ndarray] = []
    picks: List[Tuple[int, int]] = []

    step = 0
    while n_pilots < cfg.max_pilots:
        # block_error_labels returns log(e_k + eps); recover raw e_k for the before/after plots.
        true_err_log = block_error_labels(state.h_hat, h_true, na, nc, label_eps=cfg.label_eps)
        true_err = (torch.exp(true_err_log) - cfg.label_eps).clamp(min=0.0)
        true_err_hist.append(true_err.cpu().numpy())

        x_feat = build_features(state.h_hat, mask, n_pilots, cfg.sigma2, na=na, nc=nc)
        with torch.no_grad():
            pred = model(x_feat.unsqueeze(0)).squeeze(0)
        oracle = counterfactual_labels(state, h_true, used, na, nc, cfg.sigma2, gen, label_eps=cfg.label_eps)

        pred_np = pred.cpu().numpy().copy()
        oracle_np = oracle.cpu().numpy().copy()
        for k in used:
            pred_np[k] = np.nan
            oracle_np[k] = np.nan
        pred_hist.append(pred_np)
        oracle_hist.append(oracle_np)

        if mc_oracle_draws > 0:
            mean_log, std_log = mc_averaged_oracle(
                state, h_true, used, na, nc, cfg.sigma2,
                device=device, dtype=cfg.dtype, label_eps=cfg.label_eps,
                n_draws=mc_oracle_draws, seed=MC_ORACLE_SEED_BASE + step * 10_000,
            )
            oracle_mc_hist.append(mean_log)
            oracle_mc_std_hist.append(std_log)

        scores = pred.clone()
        for k in used:
            scores[k] = float("inf")
        k_next = int(scores.argmin().item())
        picks.append((step, k_next))

        x, y = measure_subcarrier(
            k_next, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
        )
        state = lmmse_incremental_update(state, x, y, cfg.sigma2)
        used.append(k_next)
        mask[k_next] = 1.0
        n_pilots += 1
        step += 1

    final_err_log = block_error_labels(state.h_hat, h_true, na, nc, label_eps=cfg.label_eps)
    final_err = (torch.exp(final_err_log) - cfg.label_eps).clamp(min=0.0)
    true_err_hist.append(final_err.cpu().numpy())

    return {
        "true_err_hist": np.stack(true_err_hist),  # (T+1, Nc)
        "pred_hist": pred_hist,  # length T, each (Nc,) with used SCs = nan
        "oracle_hist": oracle_hist,  # length T, each (Nc,) with used SCs = nan (single noisy draw)
        "oracle_mc_hist": oracle_mc_hist,  # length T (if mc_oracle_draws>0), MC-averaged log label
        "oracle_mc_std_hist": oracle_mc_std_hist,  # length T, per-candidate std across MC draws
        "picks": picks,  # length T, (step, k)
        "k_init": k_init,
    }


def plot_error_per_step(rollout: Dict[str, Any], *, cfg: CnnTrainConfig, out_path: Path) -> None:
    true_err = rollout["true_err_hist"]  # (T+1, Nc)
    picks = rollout["picks"]
    nc = cfg.n_subcarriers

    fig, (ax_heat, ax_ba) = plt.subplots(2, 1, figsize=(9.5, 8.5), height_ratios=[2.2, 1.0])

    log_err = np.log10(true_err + cfg.label_eps)
    im = ax_heat.imshow(
        log_err.T, aspect="auto", origin="lower", cmap="viridis",
        extent=(0.5, true_err.shape[0] - 0.5, -0.5, nc - 0.5),
    )
    for step, k in picks:
        ax_heat.plot(step + 1.5, k, marker="x", color="red", markersize=6, markeredgewidth=1.6)
    ax_heat.set_xlabel("n_pilots (step index on x; red x = subcarrier just piloted at that step)")
    ax_heat.set_ylabel("Subcarrier index")
    ax_heat.set_title("True per-subcarrier block error log10(e_k) over the rollout (one channel)")
    fig.colorbar(im, ax=ax_heat, label="log10(true block error)")

    before = np.array([true_err[step, k] for step, k in picks])
    after = np.array([true_err[step + 1, k] for step, k in picks])
    x = np.arange(1, len(picks) + 1)
    width = 0.38
    ax_ba.bar(x - width / 2, before, width, label="error before placing pilot there", color="C3")
    ax_ba.bar(x + width / 2, after, width, label="error after placing pilot there", color="C2")
    ax_ba.set_yscale("log")
    ax_ba.set_xlabel("Step (pilot placement event, in order)")
    ax_ba.set_ylabel("True block error at the picked SC")
    ax_ba.set_title("Error at the just-picked subcarrier: immediately before vs. after")
    ax_ba.legend(fontsize=8)
    ax_ba.grid(True, which="both", linestyle="--", alpha=0.4)

    n_reduced = int((after < before).sum())
    fig.suptitle(
        f"error reduced at {n_reduced}/{len(picks)} pilot placements "
        f"(median reduction factor {np.median(before / np.clip(after, 1e-12, None)):.1f}x)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  error reduced immediately after placement: {n_reduced}/{len(picks)} steps")
    print(f"  median before={np.median(before):.4e}  median after={np.median(after):.4e}  "
          f"median reduction factor={np.median(before / np.clip(after, 1e-12, None)):.2f}x")


def plot_pred_vs_oracle(rollout: Dict[str, Any], *, cfg: CnnTrainConfig, steps: List[int], out_path: Path) -> None:
    pred_hist = rollout["pred_hist"]
    oracle_hist = rollout["oracle_hist"]
    n_steps_total = len(pred_hist)
    steps = [s for s in steps if 0 <= s < n_steps_total]

    n_panels = len(steps)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    overall_pred: List[float] = []
    overall_oracle: List[float] = []
    overall_match = 0

    for ax, step in zip(axes, steps):
        pred = pred_hist[step]
        oracle = oracle_hist[step]
        valid = ~np.isnan(pred)
        p, o = pred[valid], oracle[valid]
        overall_pred.extend(p.tolist())
        overall_oracle.extend(o.tolist())

        r = np.corrcoef(p, o)[0, 1] if len(p) > 1 else float("nan")
        cnn_pick = int(np.nanargmin(pred))
        oracle_pick = int(np.nanargmin(oracle))
        match = cnn_pick == oracle_pick
        overall_match += int(match)

        ax.scatter(o, p, s=18, alpha=0.7, color="C0", label="unused SCs")
        lo, hi = min(o.min(), p.min()), max(o.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
        ax.scatter([oracle[oracle_pick]], [pred[oracle_pick]], marker="*", s=180, color="gold",
                   edgecolor="k", zorder=5, label="oracle's best (argmin)")
        ax.scatter([oracle[cnn_pick]], [pred[cnn_pick]], marker="D", s=70, color="C3",
                   edgecolor="k", zorder=5, label="cnn's pick (argmin pred)")
        ax.set_xlabel("Oracle counterfactual log-NMSE")
        ax.set_ylabel("CNN predicted log-NMSE")
        ax.set_title(f"n_pilots={step + 1}  r={r:.2f}  top1_match={match}")
        ax.grid(True, linestyle="--", alpha=0.4)
        if step == steps[0]:
            ax.legend(fontsize=7)

    overall_pred_a = np.array(overall_pred)
    overall_oracle_a = np.array(overall_oracle)
    overall_r = np.corrcoef(overall_pred_a, overall_oracle_a)[0, 1]
    # top-1 match across every step of the rollout (not just the plotted ones)
    full_match = 0
    for step in range(n_steps_total):
        pred = pred_hist[step]
        oracle = oracle_hist[step]
        if np.nanargmin(pred) == np.nanargmin(oracle):
            full_match += 1

    fig.suptitle(
        f"CNN predicted vs oracle counterfactual log-NMSE  "
        f"(pooled r over shown steps={overall_r:.2f}; "
        f"top-1 match over ALL {n_steps_total} steps of this rollout: {full_match}/{n_steps_total})",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  pooled Pearson r (shown steps): {overall_r:.3f}")
    print(f"  top-1 match (cnn argmin == oracle argmin) over all {n_steps_total} steps: {full_match}/{n_steps_total}")


def plot_correlation_vs_step(rollout: Dict[str, Any], *, cfg: CnnTrainConfig, out_path: Path) -> None:
    """Per-step CNN-vs-oracle correlation (single noisy draw vs MC-averaged oracle), alongside
    the between-candidate signal spread vs within-candidate noise spread that explains it."""
    pred_hist = rollout["pred_hist"]
    oracle_hist = rollout["oracle_hist"]
    oracle_mc_hist = rollout["oracle_mc_hist"]
    oracle_mc_std_hist = rollout["oracle_mc_std_hist"]
    n_steps = len(pred_hist)
    if not oracle_mc_hist:
        raise ValueError("rollout has no MC-averaged oracle data; run with mc_oracle_draws > 0.")

    n_pilots = np.arange(1, n_steps + 1)
    r_single = np.full(n_steps, np.nan)
    r_mc = np.full(n_steps, np.nan)
    top1_single = np.zeros(n_steps, dtype=bool)
    top1_mc = np.zeros(n_steps, dtype=bool)
    signal_std = np.full(n_steps, np.nan)
    noise_std = np.full(n_steps, np.nan)

    for step in range(n_steps):
        pred = pred_hist[step]
        single = oracle_hist[step]
        mc_mean = oracle_mc_hist[step]
        mc_std = oracle_mc_std_hist[step]
        valid = ~np.isnan(pred)
        if valid.sum() > 1:
            r_single[step] = np.corrcoef(pred[valid], single[valid])[0, 1]
            r_mc[step] = np.corrcoef(pred[valid], mc_mean[valid])[0, 1]
        top1_single[step] = np.nanargmin(pred) == np.nanargmin(single)
        top1_mc[step] = np.nanargmin(pred) == np.nanargmin(mc_mean)
        signal_std[step] = np.nanstd(mc_mean[valid])
        noise_std[step] = np.nanmean(mc_std[valid])

    fig, (ax_r, ax_std) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    ax_r.plot(n_pilots, r_single, "-", color="C3", linewidth=1.2, label="r(pred, single-draw oracle)")
    ax_r.scatter(n_pilots[top1_single], r_single[top1_single], marker="o", color="C3", s=40, zorder=5,
                 label="top1 match (single-draw)")
    ax_r.scatter(n_pilots[~top1_single], r_single[~top1_single], marker="x", color="C3", s=30, zorder=5)
    ax_r.plot(n_pilots, r_mc, "-", color="C0", linewidth=1.2, label="r(pred, MC-averaged oracle)")
    ax_r.scatter(n_pilots[top1_mc], r_mc[top1_mc], marker="o", color="C0", s=40, zorder=5,
                 label="top1 match (MC-averaged)")
    ax_r.scatter(n_pilots[~top1_mc], r_mc[~top1_mc], marker="x", color="C0", s=30, zorder=5)
    ax_r.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
    ax_r.set_ylabel("Pearson r")
    ax_r.set_title("CNN prediction vs. oracle correlation, per step (o = top-1 match, x = mismatch)")
    ax_r.grid(True, linestyle="--", alpha=0.4)
    ax_r.legend(fontsize=8, ncol=2)

    ax_std.plot(n_pilots, signal_std, "o-", color="C2", markersize=4,
                label="between-candidate signal std (MC-averaged oracle)")
    ax_std.plot(n_pilots, noise_std, "o-", color="C1", markersize=4,
                label="within-candidate noise std (single-draw oracle)")
    ax_std.set_xlabel("n_pilots")
    ax_std.set_ylabel("std (log-NMSE units)")
    ax_std.set_title("Signal (across candidates) vs. noise (single draw) in the oracle label")
    ax_std.grid(True, linestyle="--", alpha=0.4)
    ax_std.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    crossover = np.where(noise_std > signal_std)[0]
    first_crossover = int(n_pilots[crossover[0]]) if len(crossover) else None
    print(f"Saved {out_path}")
    print(f"  mean r (single-draw oracle): {np.nanmean(r_single):.3f}   mean r (MC-averaged oracle): {np.nanmean(r_mc):.3f}")
    print(f"  top-1 match vs single-draw oracle: {int(top1_single.sum())}/{n_steps}")
    print(f"  top-1 match vs MC-averaged oracle: {int(top1_mc.sum())}/{n_steps}")
    if first_crossover is not None:
        print(f"  noise std first exceeds signal std at n_pilots={first_crossover}")
    else:
        print("  noise std never exceeds signal std over this rollout")


def main() -> None:
    p = argparse.ArgumentParser(description="One-channel diagnostic: cnn vs oracle, error before/after pilot placement.")
    p.add_argument("--channel-mc", type=int, default=9_000_000, help="Fresh channel offset, disjoint from training/eval.")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--steps", type=int, nargs="+", default=[0, 7, 15, 23])
    p.add_argument(
        "--mc-oracle-draws", type=int, default=100,
        help="MC draws per step for the denoised oracle correlation plot (0 to skip it).",
    )
    args = p.parse_args()

    cfg = CnnTrainConfig()
    if args.device is not None:
        cfg.device = args.device
    device = resolve_device(cfg.device)

    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    sigma = load_empirical_sigma(cfg, device)
    model = load_cnn_checkpoint(Path(args.checkpoint), device)
    h_true = sample_tdl_channel(cfg, channel_mc=args.channel_mc, device=device, grid=grid)

    print(
        f"Running one-channel rollout: channel_mc={args.channel_mc}, checkpoint={args.checkpoint}, "
        f"mc_oracle_draws={args.mc_oracle_draws}"
    )
    rollout = run_one_rollout(
        cfg, h_true=h_true, sigma=sigma, model=model, device=device, channel_mc=args.channel_mc,
        mc_oracle_draws=args.mc_oracle_draws,
    )

    plot_error_per_step(rollout, cfg=cfg, out_path=FIGURES_DIR / "inspect_rollout_error_per_step.png")
    plot_pred_vs_oracle(rollout, cfg=cfg, steps=args.steps, out_path=FIGURES_DIR / "inspect_rollout_pred_vs_oracle.png")
    if args.mc_oracle_draws > 0:
        plot_correlation_vs_step(rollout, cfg=cfg, out_path=FIGURES_DIR / "inspect_rollout_correlation_vs_step.png")


if __name__ == "__main__":
    main()
