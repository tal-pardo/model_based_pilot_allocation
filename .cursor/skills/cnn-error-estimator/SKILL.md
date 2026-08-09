---
name: cnn-error-estimator
description: >-
  Supervised 2D CNN per-subcarrier error estimator under new/error_estimators/cnn/
  (CNNErrorEstimator, train.py phases, TDL-A snapshots). Use when editing cnn/
  train.py, smoke_validation.py, cluster_environment.yml, Phase 1–2 training/eval,
  or unused-SC Huber / Phase 1b curves. For vec(H)/LMMSE/incremental loop use
  pilot-selection-simulator. Not the legacy old/ Model A (cnn_allocator).
---

# CNN error estimator (Layer C candidate)

## Scope

Learned **per-subcarrier block-error** predictor for the incremental simulator in `new/`. Intended future Layer C replacement / supplement for `error_estimators/trace_min.py` (stopping via `mean(ê_k)`, pilot pick via `argmax ê` on unused SCs).

**Companion skill:** [pilot-selection-simulator](../pilot-selection-simulator/SKILL.md) — vec(H) column-stacking, incremental LMMSE, pilot policies, seed offsets, `measure_subcarrier`. Do not change those conventions here without updating both skills.

**Not this skill:** legacy cumulative Model A in `old/` (`PilotScorerModelA`, `cnn_allocator`). Incompatible features/labels/rollouts — do not reuse `old/data/cnn_pilot_scorer/*.pt` or Model A checkpoints.

**Source plan:** `.cursor/plans/cnn_supervised_training_8ef4d544.plan.md` (full design + phase console contracts).

## Role in `new/` (Layer C)

| Today (`trace_min`) | This CNN |
|---------------------|----------|
| `estimate_nmse` = `tr(P)/N` | Predict `ŷ_k ≈ log(e_k + ε)` per SC |
| `active_subcarrier_score` → `J(k)` | Deploy: `argmax_{k∉used} ŷ_k` |
| Wired in `simulation.py` | **Not wired** into `run_until_threshold` yet |

Labels (oracle, train only): `e_k = (1/Na)||ĥ_k − h_k||²`, `y_k = log(e_k + 1e-8)`.

Phase 1b / training rollouts: **no threshold stop** — run to `max_pilots = Nc = 32`.

## Entry points

| Role | Path |
|------|------|
| Sole training/eval CLI | `new/error_estimators/cnn/train.py` |
| Build-cache smoke checks | `new/error_estimators/cnn/smoke_validation.py` |
| Cluster conda env | `new/error_estimators/cnn/cluster_environment.yml` → `model_based_pilot_allocation` |

**Imports:** `new/` only (`sys.path` → repo `new/`). No `old/`.

## Essentials

**Features** `(6, Na, Nc)` float32: z-scored Re/Im/|Ĥ| across SCs; pilot mask; `log10(1/σ²)`; `n_pilots/Nc`. No `P`, no innovations.

**Model** `CNNErrorEstimator`: `(B,6,16,32)` → 3× Conv2d(+GN+GELU) → mean over `Na` → Conv1d head → `(B, Nc)` (~33k params).

**Loss:** masked Huber (`δ=1.0`) on `log(e_k+ε)`. **Current code:** mask = **unused SCs only** (`unused_mask_from_features` from feature ch3); per-sample mean then batch mean. Cached `.pt` `loss_mask` may still be all-ones — ignored at train time. Checkpoint = lowest **val Huber** (unused-only); fixed `max_epochs=40`, no patience early-stop.

**Data (Phase 1):** TDL-A truth; empirical Σ̂ (`estimate_empirical_sigma_tdl_a`, `n_cov_mc=512`); 12k train / 2k val; policy mix `ch % 3` → random/fixed/active; `k0=1` + random first SC; 31 snapshots/channel. Phase 2: separate on-policy cache (5k/1k), 75% `k0=1` / 25% `k0~U{1..8}`.

**Phase 1b eval:** mean **average MSE** `(1/N)||ĥ−h||²` vs `n_pilots` for cnn / fixed / active + full-LMMSE hline. CNN policy = greedy argmax ŷ on unused.

## CLI

```bash
# Local (sanity only)
conda activate gaussian_san_check
python new/error_estimators/cnn/train.py --phase sanity

# Cluster
conda activate model_based_pilot_allocation
python new/error_estimators/cnn/train.py --phase build --n-channels 10   # optional smoke
python new/error_estimators/cnn/train.py --phase build --force-regen     # full 12k/2k
python new/error_estimators/cnn/smoke_validation.py                      # after smoke
python new/error_estimators/cnn/train.py --phase train
python new/error_estimators/cnn/train.py --phase eval
python new/error_estimators/cnn/train.py --phase build-finetune
python new/error_estimators/cnn/train.py --phase finetune
```

| `--phase` | Where | Output |
|-----------|-------|--------|
| `sanity` | local | no save |
| `build` | cluster | `data/train.pt`, `data/val.pt` |
| `train` | cluster | `checkpoints/phase1_best.pt`, `phase1_metrics.json` |
| `eval` | cluster | `figures/phase1b_average_mse_vs_pilots.{png,json}` |
| `build-finetune` | cluster | `data/phase2_{train,val}.pt` |
| `finetune` | cluster | `checkpoints/phase2_best.pt`, `phase2_metrics.json` |

After smoke build, always `--force-regen` before production train (same cache paths).

## Artifacts (gitignored)

Under `new/error_estimators/cnn/`: `data/`, `checkpoints/`, `figures/`. Not committed; live on cluster.

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

## Non-negotiables

- Inherit vec(H) and incremental one-SC-per-step from **pilot-selection-simulator**.
- Class name **`CNNErrorEstimator`** (not `PilotScorerModelA`).
- No `old/` imports; no legacy cache reuse.
- Do not wire into `simulation.py` until Phase 1b looks competitive (plan out-of-scope).
- Terminology: plot/log **average MSE** `(1/N)||ĥ−h||²` in Phase 1b (not NMSE unless explicitly normalized).

## Progress

Update this section when phases complete or design changes land.

| Milestone | Status | Notes |
|-----------|--------|-------|
| `train.py` scaffold + all `--phase`s | **Done** | Single file; sections 1–14; `new/`-only imports |
| `CNNErrorEstimator` + features + labels + Huber | **Done** | Full 2D arch; `build_features` 6ch |
| Local `--phase sanity` | **Done** | Passed (`gaussian_san_check`) |
| `smoke_validation.py` | **Done** | Validates smoke `train.pt` / `val.pt` |
| Cluster smoke `build --n-channels 10` | **Done** (cluster) | Then full build with `--force-regen` |
| Full Phase 1 `build` (12k/2k) | **Done** (cluster) | Artifacts on cluster only (gitignored) |
| Phase 1 `train` (all-SC Huber v1) | **Done** (cluster) | `phase1_best.pt` |
| Phase 1b `eval` v1 | **Done** (cluster) | Saved as `phase1b_average_mse_vs_pilots_v1`; CNN did **not** clearly beat fixed/active |
| Improvement #1: unused-SC Huber in code | **Done** | `unused_mask_from_features`; no cache regen |
| Re-`train` + re-`eval` after unused-SC loss | **Next** | Compare to v1 figure; pairwise hinge deferred |
| Phase 2 `build-finetune` / `finetune` | **Not started** | Needs solid Phase 1b first |
| Wire into `simulation.py` / `pilot_policy` | **Not started** | Follow-up after 1b |
| Phase 3 HP sweep | **Deferred** | |

**Resume:** on cluster, with existing Phase 1 `data/*.pt`, run `--phase train` then `--phase eval` under unused-SC loss; compare curves to v1.
