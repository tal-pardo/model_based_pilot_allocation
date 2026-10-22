---
name: cnn-error-estimator
description: >-
  Supervised 2D CNN per-subcarrier error estimator under new/error_estimators/cnn/
  (CNNErrorEstimator, train.py phases, TDL-A snapshots, current vs counterfactual
  label modes). Use when editing cnn/train.py, smoke_validation.py,
  cluster_environment.yml, Phase 1–2 training/eval, unused-SC Huber / Phase 1b
  curves, or the pilot_policy/cnn.py + exp11_cnn_allocator.py consumers. For
  vec(H)/LMMSE/incremental loop use pilot-selection-simulator. Not the legacy
  old/ Model A (cnn_allocator).
---

# CNN error estimator (Layer C candidate)

## Scope

Learned **per-subcarrier** predictor for the incremental simulator in `new/`. Intended future Layer C replacement / supplement for `error_estimators/trace_min.py` (stopping via `mean(ê_k)`, pilot pick via `argmax`/`argmin ê` on unused SCs). Two label modes now coexist (see below) — `--label-mode` selects between them and keeps their artifacts (cache/checkpoint/figures) separately named.

**Companion skill:** [pilot-selection-simulator](../pilot-selection-simulator/SKILL.md) — vec(H) column-stacking, incremental LMMSE, pilot policies, seed offsets, `measure_subcarrier`. Do not change those conventions here without updating both skills.

**Not this skill:** legacy cumulative Model A in `old/` (`PilotScorerModelA`, `cnn_allocator`). Incompatible features/labels/rollouts — do not reuse `old/data/cnn_pilot_scorer/*.pt` or Model A checkpoints. (Its *counterfactual-label idea* is what inspired the `counterfactual` label mode below, reimplemented from scratch against `new/`'s incremental update — no code or cache is shared.)

**Source plan:** `.cursor/plans/cnn_supervised_training_8ef4d544.plan.md` (full design + phase console contracts, current-error paradigm only).

**Hard repo-layout constraint:** all implementation for this CNN lives inside `new/error_estimators/cnn/`. The only two files outside this folder that are allowed to exist/change for this work are `new/pilot_policy/cnn.py` (first-class `PilotPolicy` wrapping a trained checkpoint) and `new/experiments/exp11_cnn_allocator.py` (fixed vs active vs cnn comparison). Nothing else in `new/` (simulator, other pilot policies, `channel_estimators/`, `channel_generators/`, `trace_min.py`, `experiments/common.py`, other `exp*.py`) may be edited to support this CNN — those two consumer files call existing functions (e.g. `experiments.common.make_policy` for the fixed/active arms) read-only.

## Role in `new/` (Layer C)

| Today (`trace_min`) | This CNN |
|---------------------|----------|
| `estimate_nmse` = `tr(P)/N` | Predict a per-SC score `ŷ_k` |
| `active_subcarrier_score` → `J(k)` | Deploy: pick best unused SC by `ŷ_k` |
| Wired in `simulation.py` | **Not wired** into `run_until_threshold`/`pilot_policy` directly — only via the standalone `pilot_policy/cnn.py` wrapper |

## Label modes

| `--label-mode` | Label | Selection | Status |
|---|---|---|---|
| `current` (default) | `e_k = (1/Na)\|\|ĥ_k − h_k\|\|²` at the *current* snapshot (`block_error_labels`) | `argmax ŷ` on unused (pilot the worst-currently-estimated SC) | Original baseline. Phase 1b v1 (all-SC Huber) did **not** clearly beat fixed/active; unused-SC Huber fix landed but was never re-evaluated. |
| `counterfactual` | Full-vector `e = (1/N)\|\|ĥ_new − h\|\|²` after *hypothetically* placing one pilot at SC k (`counterfactual_labels`), fresh noise, one incremental LMMSE step | `argmin ŷ` on unused (pilot wherever the resulting NMSE is predicted lowest) | New paradigm, inspired by `old/`'s Model A (which beat active there). This is the mode actually wired into `pilot_policy/cnn.py` / `exp11_cnn_allocator.py`. |

Both modes share the same `(6,Na,Nc)` features, `CNNErrorEstimator` architecture, and masked-Huber-on-unused-SCs training loop — only the label source and the greedy sign convention differ. `CnnPilotPolicy.pick_next` (and `CnnPolicy` in `pilot_policy/cnn.py`) take a `select_mode` (`"argmax"`/`"argmin"`) to match whichever mode trained the loaded checkpoint.

`counterfactual_labels` exploits that a single-subcarrier pilot is a pure block-selector: the Kalman gain/update is computed directly from `P`'s block columns (`O(N·Na²)` per candidate) instead of a generic dense `(N,N)` update, and is vectorized across all unused candidates per snapshot via batched `(Na,Na)` solves — this is what makes counterfactual-label dataset generation tractable (verified against the generic `lmmse_incremental_update` on a held-out candidate; matches to high precision modulo the independent noise draw).

Phase 1b / training rollouts (both modes): **no threshold stop** — run to `max_pilots = Nc = 32`.

## Entry points

| Role | Path |
|------|------|
| Sole training/eval CLI | `new/error_estimators/cnn/train.py` |
| Build-cache smoke checks (current-mode cache only; not yet extended to `_cf`) | `new/error_estimators/cnn/smoke_validation.py` |
| Cluster conda env | `new/error_estimators/cnn/cluster_environment.yml` → `model_based_pilot_allocation` |
| First-class pilot policy (counterfactual mode, `argmin`) | `new/pilot_policy/cnn.py` (`CnnPolicy`) — one of the two sanctioned exceptions to the cnn/-only rule |
| fixed vs active vs cnn comparison | `new/experiments/exp11_cnn_allocator.py` — the other sanctioned exception |

**Imports:** `new/` only (`sys.path` → repo `new/`). No `old/`.

## Essentials

**Features** `(6, Na, Nc)` float32: z-scored Re/Im/|Ĥ| across SCs; pilot mask; `log10(1/σ²)`; `n_pilots/Nc`. No `P`, no innovations. Same for both label modes.

**Model** `CNNErrorEstimator`: `(B,6,16,32)` → 3× Conv2d(+GN+GELU) → mean over `Na` → Conv1d head → `(B, Nc)` (~33k params). Same for both label modes.

**Loss:** masked Huber on `log(e_k+ε)` (whichever `e_k` the label mode defines). Mask = **unused SCs only** (`unused_mask_from_features` from feature ch3); per-sample mean then batch mean. Checkpoint = lowest **val Huber** (unused-only). `cfg.early_stop_patience` (optional, `None` by default = no early stop) added for the reduced-scale counterfactual run; current-mode default stays fixed `max_epochs=40`, no patience.

**Data (Phase 1, current mode):** TDL-A truth; empirical Σ̂ (`estimate_empirical_sigma_tdl_a`, `n_cov_mc=512`); 12k train / 2k val; policy mix `ch % 3` → random/fixed/active; `k0=1` + random first SC; 31 snapshots/channel.

**Data (Phase 1, counterfactual mode, this session's run):** same rollout/policy-mixture machinery, reduced scale for a local 2GB-GPU run: `train_channels=3000`, `val_channels=600`, overridable via `--train-channels`/`--val-channels`. HPs borrowed from `old/`'s Model A sweep winners (no sweep budget here): `batch_size=64`, `lr=3e-4`, `weight_decay=1e-3`, `huber_delta=0.5`, `max_epochs=25`, `early_stop_patience=8`.

Phase 2 (on-policy fine-tune, `build-finetune`/`finetune`): current-mode-only scaffold, out of scope for the counterfactual work this session.

**Phase 1b eval (`--phase eval`):** mean **average MSE** `(1/N)||ĥ−h||²` vs `n_pilots` for cnn / fixed / active + full-LMMSE hline, via `run_closed_loop_mse_curve` (uses `CnnPilotPolicy` with `select_mode` matched to `cfg.label_mode`). This is a diagnostic curve inside `error_estimators/cnn/figures/`; the final deliverable comparison figure is `new/figures/exp11_cnn_allocator.png` (via `exp11_cnn_allocator.py`, true MSE not average-MSE, fresh held-out channels).

## CLI

```bash
# Local (sanity only)
conda activate gaussian_san_check
python new/error_estimators/cnn/train.py --phase sanity

# Current-error mode (original baseline, cluster-scale defaults)
conda activate model_based_pilot_allocation
python new/error_estimators/cnn/train.py --phase build --n-channels 10   # optional smoke
python new/error_estimators/cnn/train.py --phase build --force-regen     # full 12k/2k
python new/error_estimators/cnn/smoke_validation.py                      # after smoke
python new/error_estimators/cnn/train.py --phase train
python new/error_estimators/cnn/train.py --phase eval
python new/error_estimators/cnn/train.py --phase build-finetune
python new/error_estimators/cnn/train.py --phase finetune

# Counterfactual mode (this session, reduced local scale)
python new/error_estimators/cnn/train.py --phase build --label-mode counterfactual \
    --train-channels 3000 --val-channels 600 --force-regen
python new/error_estimators/cnn/train.py --phase train --label-mode counterfactual \
    --batch-size 64 --lr 3e-4 --weight-decay 1e-3 --huber-delta 0.5 --max-epochs 25 --early-stop-patience 8
python new/error_estimators/cnn/train.py --phase eval --label-mode counterfactual --eval-n-mc 50
```

| `--phase` | `--label-mode current` output | `--label-mode counterfactual` output |
|-----------|-------------------------------|----------------------------------------|
| `sanity` | no save | no save (label mode not threaded into sanity path) |
| `build` | `data/train.pt`, `data/val.pt` | `data/train_cf.pt`, `data/val_cf.pt` |
| `train` | `checkpoints/phase1_best.pt`, `phase1_metrics.json` | `checkpoints/phase1_cf_best.pt`, `phase1_cf_metrics.json` |
| `eval` | `figures/phase1b_average_mse_vs_pilots.{png,json}` | `figures/phase1b_average_mse_vs_pilots_cf.{png,json}` |
| `build-finetune` | `data/phase2_{train,val}.pt` | n/a (Phase 2 not extended to counterfactual mode) |
| `finetune` | `checkpoints/phase2_best.pt`, `phase2_metrics.json` | n/a |

After smoke build, always `--force-regen` before production train (same cache paths per mode).

## Artifacts (gitignored)

Under `new/error_estimators/cnn/`: `data/`, `checkpoints/`, `figures/`. Not committed. Current-mode artifacts live on the cluster only; counterfactual-mode artifacts for this session are produced locally (`gaussian_san_check`, NVIDIA MX330).

## Seeds (subset)

| Use | Formula |
|-----|---------|
| TDL | `seed + 100_000 + mc` |
| Empirical Σ̂ | `seed + 1_000_000 + k` |
| First pilot SC | `Random(channel_seed + 777_777)` |
| Phase 1 train/val `mc` | `ch` / `500_000+ch` |
| Phase 1b eval `mc` | `2_000_000 + mc` |
| Phase 2 train/val `mc` | `3_000_000+ch` / `3_500_000+ch` |
| Noise (build) fixed/random/active | `+0` / `+10_000` / `+20_000` |
| Noise Phase 2 / eval CNN | `+30_000` / `+40_000` |
| `exp11_cnn_allocator.py` channel offset | `6_000_000 + mc` (fresh, held out from every offset above) |
| `exp11_cnn_allocator.py` noise fixed/active/cnn | `+0` / `+20_000` / `+30_000`; full-LMMSE `+99_999` |

## Non-negotiables

- Inherit vec(H) and incremental one-SC-per-step from **pilot-selection-simulator**.
- Class name **`CNNErrorEstimator`** (not `PilotScorerModelA`).
- No `old/` imports; no legacy cache reuse (the counterfactual *idea* is reimplemented from scratch against `new/`'s incremental update, not ported code).
- Do not wire the CNN into `simulation.py` itself, `pilot_policy/base.py`, `pilot_policy/active.py`, `pilot_policy/fixed.py`, `error_estimators/trace_min.py`, or `experiments/common.py`. The only sanctioned consumers outside this folder are `new/pilot_policy/cnn.py` and `new/experiments/exp11_cnn_allocator.py`, and even those must call existing shared helpers (e.g. `make_policy`) read-only rather than edit them.
- Terminology: plot/log **average MSE** `(1/N)||ĥ−h||²` in Phase 1b (not NMSE unless explicitly normalized); `exp11_cnn_allocator.py` plots **true MSE** per `RunTrace.nmse_true`, matching the rest of `new/experiments/`'s convention.

## Progress

Update this section when phases complete or design changes land.

| Milestone | Status | Notes |
|-----------|--------|-------|
| `train.py` scaffold + all `--phase`s (current mode) | **Done** | Single file; sections 1–14; `new/`-only imports |
| `CNNErrorEstimator` + features + labels + Huber (current mode) | **Done** | Full 2D arch; `build_features` 6ch |
| Local `--phase sanity` | **Done** | Passed (`gaussian_san_check`) |
| `smoke_validation.py` | **Done** (current mode only) | Validates smoke `train.pt` / `val.pt`; not extended to `_cf` cache |
| Cluster smoke `build --n-channels 10` (current mode) | **Done** (cluster) | Then full build with `--force-regen` |
| Full Phase 1 `build` (12k/2k, current mode) | **Done** (cluster) | Artifacts on cluster only (gitignored) |
| Phase 1 `train` (all-SC Huber v1, current mode) | **Done** (cluster) | `phase1_best.pt` |
| Phase 1b `eval` v1 (current mode) | **Done** (cluster) | Saved as `phase1b_average_mse_vs_pilots_v1`; CNN did **not** clearly beat fixed/active |
| Improvement #1: unused-SC Huber in code (current mode) | **Done** | `unused_mask_from_features`; no cache regen |
| Re-`train` + re-`eval` after unused-SC loss (current mode) | **Not started** | Deprioritized in favor of the counterfactual mode below |
| `counterfactual_labels` + `--label-mode` plumbing | **Done** | Vectorized batched-solve implementation; correctness spot-checked against generic `lmmse_incremental_update` |
| Counterfactual `build` (3k/600, local) | **Done** | Local MX330; 93,000 train / 18,600 val snapshots; ~0.55-0.6s/channel actual compute rate |
| Counterfactual `train` | **Done** | `checkpoints/phase1_cf_best.pt`; best val Huber 0.0263 @ epoch 14; early-stopped @ epoch 22 |
| Counterfactual `eval` (Phase 1b diagnostic) | **Done** | `figures/phase1b_average_mse_vs_pilots_cf.{png,json}`; cnn clearly beats active for pilots ~4-27 |
| `pilot_policy/cnn.py` (`CnnPolicy`, argmin) | **Done** | First-class `PilotPolicy`, sanctioned exception file |
| `exp11_cnn_allocator.py` (fixed/active/cnn figure) | **Done** | `new/figures/exp11_cnn_allocator.png`, 50 fresh held-out TDL-A channels; confirms Phase 1b finding — see conclusion delivered in chat |
| Phase 2 `build-finetune` / `finetune` | **Not started** | Current-mode-only scaffold; out of scope this session |
| Phase 3 HP sweep | **Deferred** | |

**Resume:** this session's counterfactual-mode CNN allocator is trained and evaluated; `new/pilot_policy/cnn.py` + `new/experiments/exp11_cnn_allocator.py` are the reusable entry points. Follow-ups if picked back up: re-`train`/`eval` the current-error mode under unused-SC loss (still pending from before this session), or extend counterfactual mode with Phase 2 on-policy fine-tuning / a cluster-scale (12k/2k) rebuild for tighter MC noise.
