"""
Validate Phase-1 CNN build smoke caches (train.pt / val.pt).

Run from repo root (cluster sinteractive + GPU):
  conda activate model_based_pilot_allocation
  python new/error_estimators/cnn/smoke_validation.py

Cached-data checks cover full tensor layout and Ĥ feature channels (ch0–2).
Re-run rollouts compare only deterministic channels (ch3–5: mask, SNR, pilot
fraction) for random/fixed policies — not ch0–2 (Sionna/LMMSE not bit-reproducible
across processes) or active-policy masks (depend on ĥ).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import torch

_CNN_DIR = Path(__file__).resolve().parent
if str(_CNN_DIR) not in sys.path:
    sys.path.insert(0, str(_CNN_DIR))

from train import (  # noqa: E402
    CnnTrainConfig,
    DATA_DIR,
    NUM_FEATURE_CHANNELS,
    POLICY_NAMES,
    _channel_mc_for_split,
    collect_snapshots_one_channel,
    load_dataset,
    load_empirical_sigma,
    rollout_policy_name,
    sample_tdl_channel,
    stack_snapshots,
)
from channel_generators.sionna import SionnaOFDMGrid  # noqa: E402
from utils import resolve_device  # noqa: E402

DET_ATOL = 1e-6
ZSCORE_MEAN_ATOL = 1e-4
ZSCORE_STD_ATOL = 1e-4
DEFAULT_SMOKE_N_CHANNELS = 10


class CheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail


def cfg_from_meta(meta: Dict[str, Any]) -> CnnTrainConfig:
    """Rebuild CnnTrainConfig from cache meta (dtype stored as string)."""
    valid = {f.name for f in fields(CnnTrainConfig)}
    kwargs: Dict[str, Any] = {}
    for key in valid:
        if key not in meta:
            continue
        if key == "dtype":
            raw = meta[key]
            kwargs[key] = torch.complex64 if raw == "torch.complex64" else torch.complex64
        else:
            kwargs[key] = meta[key]
    return CnnTrainConfig(**kwargs)


def rerun_channel_x(
    cfg: CnnTrainConfig,
    sigma: torch.Tensor,
    split: Literal["train", "val"],
    channel_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Re-run one channel rollout; return stacked X on CPU."""
    grid = SionnaOFDMGrid(fft_size=cfg.n_subcarriers)
    mc = _channel_mc_for_split(split, channel_idx, cfg)
    h_true = sample_tdl_channel(cfg, channel_mc=mc, device=device, grid=grid)
    rows = collect_snapshots_one_channel(
        cfg,
        h_true=h_true,
        sigma=sigma,
        channel_mc=mc,
        channel_idx=channel_idx,
        device=device,
    )
    x, _, _ = stack_snapshots(rows)
    return x.cpu()


def check_smoke_guard(
    meta: Dict[str, Any],
    *,
    expected_n_channels: int,
    skip: bool = False,
) -> CheckResult:
    n_ch = int(meta["n_channels"])
    if skip:
        return CheckResult("smoke_guard", True, f"skipped (--allow-full-build); n_channels={n_ch}")
    ok = n_ch == expected_n_channels
    detail = f"n_channels={n_ch} (expected {expected_n_channels} for smoke)"
    return CheckResult("smoke_guard", ok, detail)


def check_dimensions(
    x: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    meta: Dict[str, Any],
) -> CheckResult:
    na, nc = int(meta["n_antennas"]), int(meta["n_subcarriers"])
    k0, max_pilots = int(meta["k0"]), int(meta["max_pilots"])
    n_ch = int(meta["n_channels"])
    snaps_per_ch = max_pilots - k0
    expected_d = n_ch * snaps_per_ch

    ok = (
        x.shape == (expected_d, NUM_FEATURE_CHANNELS, na, nc)
        and y.shape == (expected_d, nc)
        and m.shape == (expected_d, nc)
        and int(meta["n_snapshots"]) == expected_d
        and int(meta["feature_channels"]) == NUM_FEATURE_CHANNELS
    )
    detail = (
        f"X={tuple(x.shape)} y={tuple(y.shape)} loss_mask={tuple(m.shape)} "
        f"(expected D={expected_d}, [D,6,{na},{nc}])"
    )
    return CheckResult("dimensions", ok, detail)


def check_cached_channels_012(x: torch.Tensor, meta: Dict[str, Any]) -> List[CheckResult]:
    """Structural checks on z-scored Ĥ channels (cached data only)."""
    na, nc = int(meta["n_antennas"]), int(meta["n_subcarriers"])
    results: List[CheckResult] = []

    block = x[:, 0:3]
    ok_shape = block.shape == (x.shape[0], 3, na, nc)
    results.append(
        CheckResult(
            "ch0-2_shape",
            ok_shape,
            f"got {tuple(block.shape)}, expected ({x.shape[0]}, 3, {na}, {nc})",
        )
    )

    ok_finite = torch.isfinite(block).all().item()
    results.append(CheckResult("ch0-2_finite", ok_finite, f"shape={tuple(block.shape)}"))

    max_mean_abs = 0.0
    max_std_err = 0.0
    for c in range(3):
        plane = x[:, c]
        max_mean_abs = max(max_mean_abs, plane.mean(dim=2).abs().max().item())
        std_err = (plane.std(dim=2, unbiased=False) - 1.0).abs().max().item()
        max_std_err = max(max_std_err, std_err)

    ok_zscore = max_mean_abs <= ZSCORE_MEAN_ATOL and max_std_err <= ZSCORE_STD_ATOL
    results.append(
        CheckResult(
            "ch0-2_zscore",
            ok_zscore,
            f"max|mean|={max_mean_abs:.2e} (tol {ZSCORE_MEAN_ATOL}), "
            f"max|std-1|={max_std_err:.2e} (tol {ZSCORE_STD_ATOL})",
        )
    )

    return results


def check_cached_channels_345(x: torch.Tensor, meta: Dict[str, Any]) -> List[CheckResult]:
    """Internal consistency of mask / SNR / pilot-frac channels (cached data only)."""
    nc = int(meta["n_subcarriers"])
    sigma2 = float(meta["sigma2"])
    expected_snr = math.log10(1.0 / sigma2)
    results: List[CheckResult] = []

    mask_plane = x[0, 3]
    ok_mask_shape = mask_plane.shape == (x.shape[2], nc)
    row0 = mask_plane[0]
    ok_broadcast = torch.allclose(mask_plane, row0.unsqueeze(0).expand_as(mask_plane))
    ok_binary = torch.all((mask_plane == 0) | (mask_plane == 1))
    pilot_frac = x[0, 5, 0, 0].item()
    mask_frac = row0.sum().item() / nc
    ok_frac = abs(pilot_frac - mask_frac) < DET_ATOL and abs(pilot_frac - 1.0 / nc) < DET_ATOL
    results.append(
        CheckResult(
            "ch3_mask_sample0",
            ok_mask_shape and ok_broadcast and ok_binary and ok_frac,
            f"shape={tuple(mask_plane.shape)}, binary={bool(ok_binary)}, "
            f"broadcast={bool(ok_broadcast)}, frac={pilot_frac:.6g} (expect 1/{nc})",
        )
    )

    snr_vals = x[:, 4, 0, 0]
    unique_snr = torch.unique(snr_vals)
    ok_snr = unique_snr.numel() == 1 and torch.allclose(
        unique_snr[0], torch.tensor(expected_snr), atol=DET_ATOL
    )
    results.append(
        CheckResult(
            "ch4_snr_constant",
            ok_snr,
            f"SNR={unique_snr[0].item():.6g} (log10(1/sigma2)), sigma2={sigma2}",
        )
    )

    snaps_per_ch = int(meta["max_pilots"]) - int(meta["k0"])
    block = x[0:snaps_per_ch]
    ok_pf = True
    prev = -1.0
    for t in range(snaps_per_ch):
        expected = (t + 1) / nc
        pf = block[t, 5, 0, 0].item()
        mask_frac_t = block[t, 3, 0].sum().item() / nc
        if abs(pf - expected) > DET_ATOL or abs(mask_frac_t - expected) > DET_ATOL or pf <= prev:
            ok_pf = False
            break
        prev = pf
    results.append(
        CheckResult(
            "ch5_pilot_frac_rollout_ch0",
            ok_pf,
            f"channel 0: {1/nc:.6g} .. {snaps_per_ch/nc:.6g} step 1/{nc}",
        )
    )

    ch4_broadcast = torch.allclose(x[:, 4], x[:, 4, 0:1, 0:1].expand_as(x[:, 4]), atol=0)
    ch5_broadcast = torch.allclose(x[:, 5], x[:, 5, 0:1, 0:1].expand_as(x[:, 5]), atol=0)
    results.append(
        CheckResult(
            "ch4-5_broadcast",
            ch4_broadcast and ch5_broadcast,
            f"ch4={bool(ch4_broadcast)} ch5={bool(ch5_broadcast)}",
        )
    )

    return results


def check_cheap_sanity(
    y: torch.Tensor,
    _m: torch.Tensor,
    meta: Dict[str, Any],
) -> List[CheckResult]:
    results: List[CheckResult] = []

    ok_finite = torch.isfinite(y).all().item()
    results.append(CheckResult("y_label_finite", ok_finite, f"shape={tuple(y.shape)}"))

    fc = int(meta.get("feature_channels", -1))
    ok_fc = fc == NUM_FEATURE_CHANNELS
    results.append(CheckResult("meta_feature_channels", ok_fc, f"feature_channels={fc}"))

    return results


def check_regenerate_channels_345(
    cached: torch.Tensor,
    regenerated: torch.Tensor,
    *,
    label: str,
) -> CheckResult:
    """Exact match on mask, SNR, and pilot-frac planes (policy-deterministic)."""
    if cached.shape != regenerated.shape:
        return CheckResult(
            f"regenerate_ch3-5_{label}",
            False,
            f"shape mismatch cached={tuple(cached.shape)} regen={tuple(regenerated.shape)}",
        )

    c = cached[:, 3:6]
    r = regenerated[:, 3:6]
    ok = torch.allclose(c, r, rtol=0.0, atol=DET_ATOL)
    if ok:
        detail = f"{label}: ch3-5 match ({tuple(c.shape)})"
    else:
        diff = (c - r).abs()
        per_ch = [diff[:, i].max().item() for i in range(3)]
        detail = f"{label}: max_abs_diff ch3/4/5={per_ch[0]:.3e}/{per_ch[1]:.3e}/{per_ch[2]:.3e}"
    return CheckResult(f"regenerate_ch3-5_{label}", ok, detail)


def check_policy_mix_train(
    x: torch.Tensor,
    cfg: CnnTrainConfig,
    sigma: torch.Tensor,
    device: torch.device,
    meta: Dict[str, Any],
) -> List[CheckResult]:
    snaps_per_ch = int(meta["max_pilots"]) - int(meta["k0"])
    results: List[CheckResult] = []

    for ch in range(3):
        expected_policy = POLICY_NAMES[ch]
        actual_policy = rollout_policy_name(ch)
        results.append(
            CheckResult(
                f"policy_name_ch{ch}",
                actual_policy == expected_policy,
                f"expected={expected_policy}, got={actual_policy}",
            )
        )

        if expected_policy == "active":
            results.append(
                CheckResult(
                    f"regenerate_ch3-5_train_ch{ch}_active",
                    True,
                    "skipped: active policy depends on ĥ (non-deterministic across Sionna runs)",
                )
            )
            continue

        cached_block = x[ch * snaps_per_ch : (ch + 1) * snaps_per_ch]
        regen = rerun_channel_x(cfg, sigma, "train", ch, device)
        results.append(
            check_regenerate_channels_345(
                cached_block, regen, label=f"train_ch{ch}_{expected_policy}"
            )
        )

    return results


def validate_split(
    path: Path,
    split: Literal["train", "val"],
    *,
    cfg: CnnTrainConfig,
    sigma: torch.Tensor,
    device: torch.device,
    expected_n_channels: int,
    check_policies: bool,
    skip_smoke_guard: bool = False,
) -> Tuple[List[CheckResult], bool]:
    print(f"\n=== {path.name} ===")
    x, y, m, meta = load_dataset(path)
    results: List[CheckResult] = []

    results.append(
        check_smoke_guard(meta, expected_n_channels=expected_n_channels, skip=skip_smoke_guard)
    )
    results.append(check_dimensions(x, y, m, meta))
    results.extend(check_cached_channels_012(x, meta))
    results.extend(check_cached_channels_345(x, meta))
    results.extend(check_cheap_sanity(y, m, meta))

    if not check_policies:
        snaps_per_ch = int(meta["max_pilots"]) - int(meta["k0"])
        cached_ch0 = x[0:snaps_per_ch]
        policy = rollout_policy_name(0)
        if policy == "active":
            results.append(
                CheckResult(
                    f"regenerate_ch3-5_{split}_ch0",
                    True,
                    "skipped: active policy on ch0",
                )
            )
        else:
            regen_ch0 = rerun_channel_x(cfg, sigma, split, 0, device)
            results.append(
                check_regenerate_channels_345(cached_ch0, regen_ch0, label=f"{split}_ch0_{policy}")
            )

    if check_policies:
        results.extend(check_policy_mix_train(x, cfg, sigma, device, meta))

    all_ok = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        suffix = f" — {r.detail}" if r.detail else ""
        print(f"{status} {r.name}{suffix}")
        if not r.passed:
            all_ok = False

    return results, all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CNN build smoke caches.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train.pt and val.pt",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:0, cpu")
    parser.add_argument(
        "--smoke-n-channels",
        type=int,
        default=DEFAULT_SMOKE_N_CHANNELS,
        help="Expected meta['n_channels'] (smoke build uses 10)",
    )
    parser.add_argument(
        "--allow-full-build",
        action="store_true",
        help="Skip smoke-only guard (accept any n_channels matching D = n_channels * 31)",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "train.pt"
    val_path = args.data_dir / "val.pt"
    for p in (train_path, val_path):
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(1)

    device = resolve_device(args.device)
    print(f"Device: {device}")

    _, _, _, meta0 = load_dataset(train_path)
    cfg = cfg_from_meta(meta0)
    print("Loading empirical Sigma for rollout replay (random/fixed policies)...")
    sigma = load_empirical_sigma(cfg, device)

    expected_n = args.smoke_n_channels

    _, train_ok = validate_split(
        train_path,
        "train",
        cfg=cfg,
        sigma=sigma,
        device=device,
        expected_n_channels=expected_n,
        check_policies=True,
        skip_smoke_guard=args.allow_full_build,
    )
    _, val_ok = validate_split(
        val_path,
        "val",
        cfg=cfg,
        sigma=sigma,
        device=device,
        expected_n_channels=expected_n,
        check_policies=False,
        skip_smoke_guard=args.allow_full_build,
    )

    if train_ok and val_ok:
        print("\nALL CHECKS PASSED")
        sys.exit(0)

    print("\nVALIDATION FAILED", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
