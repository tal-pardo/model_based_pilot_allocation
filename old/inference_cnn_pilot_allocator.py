"""
Closed-loop TDL-A MSE comparison: fixed vs active vs CNN pilot allocation.

Loads a trained checkpoint (TrainConfig embedded in .pt), estimates Sigma_hat from TDL-A,
and runs Monte Carlo sequential LMMSE curves. Outputs PNG + JSON under figures/inference/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch

from data_generator import complex_standard_normal, pilot_matrix_from_indices
from estimators import recursive_lmmse_init, recursive_lmmse_step
from pilots import (
    ActivePilotSampler,
    FixedPilotSampler,
    PilotScheduleConfig,
    num_timesteps_from_pilot_growth,
    ordered_subcarriers_from_vec,
    sequential_lmmse_mse_curve,
    vec_indices_for_subcarriers,
)
from sionna_channels import SionnaOFDMGrid

from train_cnn_pilot_allocator import (
    BEST_CHECKPOINT_PATH,
    CNNPilotSampler,
    TrainConfig,
    estimate_sigma_hat_tdl_a,
    load_checkpoint,
    n_decision_steps,
    resolve_device,
    sample_tdl_a_channel,
)

EVAL_CHANNEL_SEED_OFFSET = 300_000
NOISE_SEED_OFFSET_FIXED = 0
NOISE_SEED_OFFSET_ACTIVE = 10_000
NOISE_SEED_OFFSET_CNN_E0 = 20_000
NOISE_SEED_OFFSET_CNN_D2 = 30_000
DEFAULT_OUTPUT_DIR = Path("figures/inference")
SWEEP_CHECKPOINT_E0 = Path("checkpoints/E0_best.pt")
SWEEP_CHECKPOINT_D2 = Path("checkpoints/D2b_best.pt")
SWEEP_COMPARISON_PNG = DEFAULT_OUTPUT_DIR / "models_comparison_after_sweep.png"
SWEEP_COMPARISON_JSON = DEFAULT_OUTPUT_DIR / "models_comparison_after_sweep.json"


@dataclass
class InferenceConfig:
    checkpoint: Path = BEST_CHECKPOINT_PATH
    checkpoint_e0: Path = SWEEP_CHECKPOINT_E0
    checkpoint_d2: Path = SWEEP_CHECKPOINT_D2
    n_mc: int = 100
    n_cov_mc: Optional[int] = None
    seed: Optional[int] = None
    sigma2: Optional[float] = None
    device: str = "cuda"
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_name: Optional[str] = None
    save_json: bool = True
    show_plot: bool = True


def _empirical_mse(h_hat: torch.Tensor, h_true: torch.Tensor) -> float:
    err = h_hat - h_true
    return (err.abs().pow(2).mean()).real.item()


def _pilot_schedule(cfg: TrainConfig, device: torch.device) -> Tuple[PilotScheduleConfig, FixedPilotSampler, int]:
    sched = PilotScheduleConfig(
        n_subcarriers=cfg.n_subcarriers,
        n_antennas=cfg.n_antennas,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        final_pilot_subcarriers=cfg.final_pilot_subcarriers,
        pilots_added_per_step=cfg.pilots_added_per_step,
        cumulative_pilots=True,
    )
    t_add = n_decision_steps(cfg)
    fixed = FixedPilotSampler(sched, T=t_add + 1, device=device)
    return sched, fixed, t_add


def sequential_mse_curve_cnn(
    sigma: torch.Tensor,
    h_true: torch.Tensor,
    cfg: TrainConfig,
    cnn: CNNPilotSampler,
    fixed: FixedPilotSampler,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> List[float]:
    """
    Sequential LMMSE MSE curve using CNNPilotSampler for t>=1 pilot additions.
    Returns empirical MSE list of length T+1 (matches sequential_lmmse_mse_curve).
    """
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = na * nc
    sigma2 = cfg.sigma2
    dtype = cfg.dtype
    t_add = n_decision_steps(cfg)

    def one_step(idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = pilot_matrix_from_indices(n, idx, device=device, dtype=dtype)
        noise = (sigma2**0.5) * complex_standard_normal(
            idx.numel(), 1, device=device, dtype=dtype, generator=generator
        )
        y_t = x_t @ h_true + noise
        return x_t, y_t

    idx0 = fixed.vec_indices_at_step(0)
    x0, y0 = one_step(idx0)
    state = recursive_lmmse_init(sigma, x0, y0, sigma2)
    mses: List[float] = [_empirical_mse(state.h, h_true)]

    used_sc = ordered_subcarriers_from_vec(idx0, na)
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    for k in used_sc:
        mask[k] = 1.0

    for t in range(1, t_add + 1):
        k_prev = min(
            cfg.final_pilot_subcarriers,
            cfg.initial_pilot_subcarriers + (t - 1) * cfg.pilots_added_per_step,
        )
        k_cur = min(
            cfg.final_pilot_subcarriers,
            cfg.initial_pilot_subcarriers + t * cfg.pilots_added_per_step,
        )
        n_new = k_cur - k_prev
        for _ in range(n_new):
            k_new = cnn.select_subcarrier(
                state.h, mask, decision_step=t, n_decision_steps=t_add
            )
            used_sc.append(k_new)
            mask[k_new] = 1.0

        idx_t = vec_indices_for_subcarriers(
            torch.tensor(used_sc, device=device, dtype=torch.long), na, device
        )
        x_t, y_t = one_step(idx_t)
        state = recursive_lmmse_step(state, x_t, y_t, sigma2)
        mses.append(_empirical_mse(state.h, h_true))

    return mses


def load_cnn_policy(
    checkpoint: Path,
    device: torch.device,
) -> Tuple[CNNPilotSampler, TrainConfig, Dict[str, Any]]:
    model, cfg, payload = load_checkpoint(checkpoint, device)
    cnn = CNNPilotSampler(model, cfg, device)
    return cnn, cfg, payload


def _apply_inf_overrides(train_cfg: TrainConfig, inf_cfg: InferenceConfig) -> None:
    if inf_cfg.sigma2 is not None:
        train_cfg.sigma2 = inf_cfg.sigma2
    if inf_cfg.seed is not None:
        train_cfg.seed = inf_cfg.seed
    if inf_cfg.n_cov_mc is not None:
        train_cfg.n_cov_mc = inf_cfg.n_cov_mc


def _assert_compatible_cfg(base: TrainConfig, other: TrainConfig, other_name: str) -> None:
    keys = (
        "n_antennas",
        "n_subcarriers",
        "initial_pilot_subcarriers",
        "final_pilot_subcarriers",
        "pilots_added_per_step",
    )
    for k in keys:
        if getattr(base, k) != getattr(other, k):
            raise ValueError(
                f"Checkpoint {other_name} cfg.{k}={getattr(other, k)!r} "
                f"!= reference {getattr(base, k)!r}."
            )


def run_sweep_models_comparison(inf_cfg: InferenceConfig) -> Dict[str, Any]:
    """
    Compare fixed, active, CNN E0, and CNN D2b on TDL-A (post hyperparameter sweep).
    Saves figures/inference/models_comparison_after_sweep.png (+ JSON).
    """
    device = resolve_device(inf_cfg.device)
    ckpt_e0 = Path(inf_cfg.checkpoint_e0)
    ckpt_d2 = Path(inf_cfg.checkpoint_d2)

    cnn_e0, cfg_e0, payload_e0 = load_cnn_policy(ckpt_e0, device)
    cnn_d2, cfg_d2, payload_d2 = load_cnn_policy(ckpt_d2, device)
    _assert_compatible_cfg(cfg_e0, cfg_d2, "D2b")

    train_cfg = cfg_e0
    _apply_inf_overrides(train_cfg, inf_cfg)

    eval_seed = train_cfg.seed
    n_mc = inf_cfg.n_mc
    t_add = n_decision_steps(train_cfg)
    t_steps = t_add + 1

    print(
        f"sweep comparison: E0={ckpt_e0}  D2b={ckpt_d2}  device={device}  n_mc={n_mc}  "
        f"T={t_add}  Sigma_hat n_cov_mc={train_cfg.n_cov_mc}",
        flush=True,
    )

    print("Estimating Sigma_hat from TDL-A...", flush=True)
    sigma_hat = estimate_sigma_hat_tdl_a(train_cfg, device)
    grid = SionnaOFDMGrid(fft_size=train_cfg.n_subcarriers)

    sched, fixed, _ = _pilot_schedule(train_cfg, device)
    active = ActivePilotSampler(
        sched, T=t_add + 1, device=device, fixed=fixed, sigma2=train_cfg.sigma2
    )

    policy_names = ("fixed", "active", "cnn_e0", "cnn_d2")
    mse_all = {p: torch.zeros((n_mc, t_steps), dtype=torch.float64) for p in policy_names}

    cnn_policies = (
        ("cnn_e0", cnn_e0, NOISE_SEED_OFFSET_CNN_E0),
        ("cnn_d2", cnn_d2, NOISE_SEED_OFFSET_CNN_D2),
    )

    for mc in range(n_mc):
        channel_seed = eval_seed + EVAL_CHANNEL_SEED_OFFSET + mc
        h_true = sample_tdl_a_channel(
            train_cfg, grid=grid, device=device, seed=channel_seed
        )

        gen_fixed = torch.Generator(device=device).manual_seed(
            eval_seed + NOISE_SEED_OFFSET_FIXED + mc
        )
        emp_fixed, _ = sequential_lmmse_mse_curve(
            sigma_hat,
            h_true,
            train_cfg.sigma2,
            t_add,
            lambda t, _P, _f=fixed: _f.vec_indices_at_step(t),
            device=device,
            dtype=train_cfg.dtype,
            generator=gen_fixed,
        )
        mse_all["fixed"][mc] = torch.tensor(emp_fixed, dtype=torch.float64)

        active.reset()
        gen_active = torch.Generator(device=device).manual_seed(
            eval_seed + NOISE_SEED_OFFSET_ACTIVE + mc
        )
        emp_active, _ = sequential_lmmse_mse_curve(
            sigma_hat,
            h_true,
            train_cfg.sigma2,
            t_add,
            active.vec_indices_at_step,
            device=device,
            dtype=train_cfg.dtype,
            generator=gen_active,
        )
        mse_all["active"][mc] = torch.tensor(emp_active, dtype=torch.float64)

        for pname, cnn, noise_off in cnn_policies:
            gen_cnn = torch.Generator(device=device).manual_seed(eval_seed + noise_off + mc)
            emp_cnn = sequential_mse_curve_cnn(
                sigma_hat,
                h_true,
                train_cfg,
                cnn,
                fixed,
                device=device,
                generator=gen_cnn,
            )
            mse_all[pname][mc] = torch.tensor(emp_cnn, dtype=torch.float64)

        if (mc + 1) % max(1, n_mc // 10) == 0 or mc + 1 == n_mc:
            print(f"  mc {mc + 1}/{n_mc}", flush=True)

    curves = {p: mse_all[p].mean(dim=0) for p in policy_names}
    curves_std = {p: mse_all[p].std(dim=0, unbiased=False) for p in policy_names}

    t_axis = torch.arange(t_steps, dtype=torch.float64)
    final = {p: curves[p][-1].item() for p in policy_names}
    print(
        f"final-step MSE: fixed={final['fixed']:.6f}  active={final['active']:.6f}  "
        f"cnn_e0={final['cnn_e0']:.6f}  cnn_d2={final['cnn_d2']:.6f}",
        flush=True,
    )

    out_dir = Path(inf_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "models_comparison_after_sweep.png"
    json_path = out_dir / "models_comparison_after_sweep.json"

    plot_mse_comparison(
        t_axis,
        curves,
        title="TDL-A MSE: fixed vs active vs CNN E0 vs CNN D2b (after sweep)",
        out_path=png_path,
        show=inf_cfg.show_plot,
        curve_styles={
            "fixed": {"linestyle": "-", "marker": "o", "color": "C0", "label": "fixed"},
            "active": {"linestyle": "--", "marker": "s", "color": "C1", "label": "active"},
            "cnn_e0": {"linestyle": "-.", "marker": "^", "color": "C2", "label": "cnn E0"},
            "cnn_d2": {"linestyle": ":", "marker": "D", "color": "C3", "label": "cnn D2b"},
        },
    )

    results: Dict[str, Any] = {
        "comparison": "sweep_models",
        "checkpoints": {
            "e0": str(ckpt_e0.resolve()),
            "d2b": str(ckpt_d2.resolve()),
        },
        "n_mc": n_mc,
        "eval_seed": eval_seed,
        "train_cfg": {
            "n_antennas": train_cfg.n_antennas,
            "n_subcarriers": train_cfg.n_subcarriers,
            "sigma2": train_cfg.sigma2,
            "n_cov_mc": train_cfg.n_cov_mc,
            "seed": train_cfg.seed,
        },
        "checkpoint_meta": {
            "e0": {
                "best_epoch": payload_e0.get("best_epoch"),
                "best_val_huber": payload_e0.get("best_val_huber"),
                "best_val_top1": payload_e0.get("best_val_top1"),
            },
            "d2b": {
                "best_epoch": payload_d2.get("best_epoch"),
                "best_val_huber": payload_d2.get("best_val_huber"),
                "best_val_top1": payload_d2.get("best_val_top1"),
            },
        },
        "t_steps": t_steps,
        "mse_mean": {p: curves[p].tolist() for p in policy_names},
        "mse_std": {p: curves_std[p].tolist() for p in policy_names},
        "final_mse": final,
        "figure_path": str(png_path.resolve()),
    }

    if inf_cfg.save_json:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved: {png_path}", flush=True)
        print(f"saved: {json_path}", flush=True)

    return results


def run_tdl_a_comparison(inf_cfg: InferenceConfig) -> Dict[str, Any]:
    device = resolve_device(inf_cfg.device)
    checkpoint = Path(inf_cfg.checkpoint)
    cnn, train_cfg, payload = load_cnn_policy(checkpoint, device)

    _apply_inf_overrides(train_cfg, inf_cfg)

    eval_seed = train_cfg.seed
    n_mc = inf_cfg.n_mc
    t_add = n_decision_steps(train_cfg)
    t_steps = t_add + 1

    print(
        f"inference: checkpoint={checkpoint}  device={device}  n_mc={n_mc}  "
        f"T={t_add}  Sigma_hat n_cov_mc={train_cfg.n_cov_mc}",
        flush=True,
    )

    print("Estimating Sigma_hat from TDL-A...", flush=True)
    sigma_hat = estimate_sigma_hat_tdl_a(train_cfg, device)
    grid = SionnaOFDMGrid(fft_size=train_cfg.n_subcarriers)

    sched, fixed, _ = _pilot_schedule(train_cfg, device)
    t_total = t_add + 1
    active = ActivePilotSampler(
        sched, T=t_total, device=device, fixed=fixed, sigma2=train_cfg.sigma2
    )

    mse_fixed = torch.zeros((n_mc, t_steps), dtype=torch.float64)
    mse_active = torch.zeros((n_mc, t_steps), dtype=torch.float64)
    mse_cnn = torch.zeros((n_mc, t_steps), dtype=torch.float64)

    for mc in range(n_mc):
        channel_seed = eval_seed + EVAL_CHANNEL_SEED_OFFSET + mc
        h_true = sample_tdl_a_channel(
            train_cfg, grid=grid, device=device, seed=channel_seed
        )

        gen_fixed = torch.Generator(device=device).manual_seed(
            eval_seed + NOISE_SEED_OFFSET_FIXED + mc
        )
        emp_fixed, _ = sequential_lmmse_mse_curve(
            sigma_hat,
            h_true,
            train_cfg.sigma2,
            t_add,
            lambda t, _P, _f=fixed: _f.vec_indices_at_step(t),
            device=device,
            dtype=train_cfg.dtype,
            generator=gen_fixed,
        )
        mse_fixed[mc] = torch.tensor(emp_fixed, dtype=torch.float64)

        active.reset()
        gen_active = torch.Generator(device=device).manual_seed(
            eval_seed + NOISE_SEED_OFFSET_ACTIVE + mc
        )
        emp_active, _ = sequential_lmmse_mse_curve(
            sigma_hat,
            h_true,
            train_cfg.sigma2,
            t_add,
            active.vec_indices_at_step,
            device=device,
            dtype=train_cfg.dtype,
            generator=gen_active,
        )
        mse_active[mc] = torch.tensor(emp_active, dtype=torch.float64)

        gen_cnn = torch.Generator(device=device).manual_seed(
            eval_seed + NOISE_SEED_OFFSET_CNN_E0 + mc
        )
        emp_cnn = sequential_mse_curve_cnn(
            sigma_hat,
            h_true,
            train_cfg,
            cnn,
            fixed,
            device=device,
            generator=gen_cnn,
        )
        mse_cnn[mc] = torch.tensor(emp_cnn, dtype=torch.float64)

        if (mc + 1) % max(1, n_mc // 10) == 0 or mc + 1 == n_mc:
            print(f"  mc {mc + 1}/{n_mc}", flush=True)

    curves = {
        "fixed": mse_fixed.mean(dim=0),
        "active": mse_active.mean(dim=0),
        "cnn": mse_cnn.mean(dim=0),
    }
    curves_std = {
        "fixed": mse_fixed.std(dim=0, unbiased=False),
        "active": mse_active.std(dim=0, unbiased=False),
        "cnn": mse_cnn.std(dim=0, unbiased=False),
    }

    t_axis = torch.arange(t_steps, dtype=torch.float64)
    final = {p: curves[p][-1].item() for p in curves}
    print(
        f"final-step MSE: fixed={final['fixed']:.6f}  "
        f"active={final['active']:.6f}  cnn={final['cnn']:.6f}",
        flush=True,
    )
    if final["cnn"] < final["fixed"]:
        print("  -> CNN below fixed (learned policy beats static pilots)", flush=True)
    elif final["cnn"] < final["active"]:
        print("  -> CNN between fixed and active (promising)", flush=True)
    else:
        print("  -> CNN at or above fixed (review training / ranking)", flush=True)

    run_name = inf_cfg.run_name or checkpoint.stem
    out_dir = Path(inf_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"cnn_pilot_tdl_a_{run_name}.png"
    json_path = out_dir / f"cnn_pilot_tdl_a_{run_name}.json"

    plot_mse_comparison(
        t_axis,
        curves,
        title=f"TDL-A MSE: fixed vs active vs CNN ({run_name})",
        out_path=png_path,
        show=inf_cfg.show_plot,
    )

    results: Dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "run_name": run_name,
        "n_mc": n_mc,
        "eval_seed": eval_seed,
        "eval_channel_seed_offset": EVAL_CHANNEL_SEED_OFFSET,
        "train_cfg": {
            "n_antennas": train_cfg.n_antennas,
            "n_subcarriers": train_cfg.n_subcarriers,
            "rho_space": train_cfg.rho_space,
            "sigma2": train_cfg.sigma2,
            "initial_pilot_subcarriers": train_cfg.initial_pilot_subcarriers,
            "final_pilot_subcarriers": train_cfg.final_pilot_subcarriers,
            "pilots_added_per_step": train_cfg.pilots_added_per_step,
            "n_cov_mc": train_cfg.n_cov_mc,
            "seed": train_cfg.seed,
        },
        "checkpoint_meta": {
            "best_epoch": payload.get("best_epoch"),
            "best_val_huber": payload.get("best_val_huber"),
            "best_val_top1": payload.get("best_val_top1"),
        },
        "t_steps": t_steps,
        "mse_mean": {p: curves[p].tolist() for p in curves},
        "mse_std": {p: curves_std[p].tolist() for p in curves_std},
        "final_mse": final,
        "figure_path": str(png_path.resolve()),
    }

    if inf_cfg.save_json:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"saved: {png_path}", flush=True)
        print(f"saved: {json_path}", flush=True)

    return results


def plot_mse_comparison(
    t: torch.Tensor,
    curves: Dict[str, torch.Tensor],
    *,
    title: str,
    out_path: Path,
    show: bool = True,
    curve_styles: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    default_styles = {
        "fixed": {"linestyle": "-", "marker": "o", "color": "C0", "label": "fixed"},
        "active": {"linestyle": "--", "marker": "s", "color": "C1", "label": "active"},
        "cnn": {"linestyle": "-.", "marker": "^", "color": "C2", "label": "cnn"},
        "cnn_e0": {"linestyle": "-.", "marker": "^", "color": "C2", "label": "cnn E0"},
        "cnn_d2": {"linestyle": ":", "marker": "D", "color": "C3", "label": "cnn D2b"},
    }
    styles = curve_styles or default_styles
    plt.figure(figsize=(8.4, 4.8))
    for name, mse_mean in curves.items():
        st = styles.get(name, {"linestyle": "-", "marker": "o", "color": None, "label": name})
        plt.semilogy(
            t.numpy(),
            mse_mean.numpy(),
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=1.6,
            color=st["color"],
            label=st["label"],
        )
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Time step t")
    plt.ylabel("Mean MSE over MC  (1/N)||h_hat - h||^2")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TDL-A closed-loop MSE: fixed vs active vs CNN pilot allocation."
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Single CNN checkpoint vs fixed/active (default: sweep E0 + D2b comparison)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BEST_CHECKPOINT_PATH,
        help="Path to single CNN checkpoint .pt (with --single)",
    )
    parser.add_argument(
        "--checkpoint-e0",
        type=Path,
        default=SWEEP_CHECKPOINT_E0,
        help="E0 checkpoint for --sweep-comparison",
    )
    parser.add_argument(
        "--checkpoint-d2",
        type=Path,
        default=SWEEP_CHECKPOINT_D2,
        help="D2b checkpoint for --sweep-comparison",
    )
    parser.add_argument("--n-mc", type=int, default=50, help="Monte Carlo channel trials")
    parser.add_argument(
        "--n-cov-mc",
        type=int,
        default=None,
        help="Override n_cov_mc for Sigma_hat (default: from checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override eval seed base")
    parser.add_argument("--sigma2", type=float, default=None, help="Override noise variance")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and JSON outputs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output file stem (default: checkpoint stem)",
    )
    parser.add_argument("--no-json", action="store_true", help="Skip writing JSON results")
    parser.add_argument("--no-show", action="store_true", help="Save PNG without plt.show()")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inf_cfg = InferenceConfig(
        checkpoint=args.checkpoint,
        checkpoint_e0=args.checkpoint_e0,
        checkpoint_d2=args.checkpoint_d2,
        n_mc=args.n_mc,
        n_cov_mc=args.n_cov_mc,
        seed=args.seed,
        sigma2=args.sigma2,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        save_json=not args.no_json,
        show_plot=not args.no_show,
    )
    if args.single:
        run_tdl_a_comparison(inf_cfg)
    else:
        run_sweep_models_comparison(inf_cfg)


if __name__ == "__main__":
    main()
