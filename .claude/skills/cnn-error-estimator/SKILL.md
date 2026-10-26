---
name: cnn-error-estimator
description: >-
  Supervised 2D CNN per-subcarrier error estimator under new/error_estimators/cnn/
  (CNNErrorEstimator, train.py phases, TDL-A snapshots, current vs counterfactual
  label modes, antenna_snapshot vs time_freq domains). Use when editing
  cnn/train.py, smoke_validation.py, cluster_environment.yml, Phase 1–2
  training/eval, unused-SC Huber / Phase 1b curves, the domain="time_freq"
  TDL-A/B/C x speed-bucket mode, or the pilot_policy/cnn.py +
  exp11_cnn_allocator.py consumers. For vec(H)/LMMSE/incremental loop use
  pilot-selection-simulator. Not the legacy old/ Model A (cnn_allocator).
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

## Domains

`CnnTrainConfig.domain: Literal["antenna_snapshot", "time_freq"]` (default `"antenna_snapshot"`, preserves all prior behavior exactly). Added to diagnose/test whether the antenna-domain Phase 1b finding (`cnn`/`fixed`/`active` nearly overlapping, see Progress) was a domain artifact rather than a bug — `exp12_time_freq_pilots.py` shows a much clearer active>fixed gap on a time-freq grid with a realistic DMRS-comb "fixed" baseline, so this mode reproduces that domain/baseline pairing inside the CNN pipeline.

| | `antenna_snapshot` (default) | `time_freq` |
|---|---|---|
| Grid | static snapshot, 1 OFDM symbol, `(Na=16, Nc=32)` | single antenna, joint `(n_symbols=16, n_freq_subcarriers=32)` grid, flat index `s=k*n_symbols+t` (subcarrier-major, matches `channel_generators/time_freq.py::kt_to_flat_index`) |
| Channel model | TDL-A only, `speed=0.0` (no Doppler) | `tdl_models=("A","B","C") x speeds=(3.0,30.0,120.0)` m/s = 9 buckets, round-robined by `channel_mc % 9` (`bucket_for_channel`) |
| Sigma | one shared empirical Sigma (`estimate_empirical_sigma_tdl_a`, `n_cov_mc=512`) | one Ledoit-Wolf-shrunk Sigma **per bucket** (`load_sigma_buckets`, `n_cov_mc_per_bucket=256`) — a single shared Sigma would be statistically mismatched once channels vary in TDL model/speed |
| "fixed" rollout policy | generic `FixedPolicy` (1D largest-gap bisection) | `DmrsCombPolicy` (`pilot_policy/dmrs_comb.py`) — the generic bisection doesn't respect the 2D `(k,t)` grid; `comb_offset = channel_mc % dmrs_comb_spacing` varies it per channel since the policy itself has no seed |
| CNN head | `CNNErrorEstimator(pool_mode="mean")`: mean-pool the antenna axis, one score per subcarrier, `(B,Nc)` | `pool_mode="per_cell"`: no pooling (every `(symbol,subcarrier)` cell is its own candidate), per-cell `Conv2d` head, flattened `permute(0,2,1).reshape(...)` to match `s=k*n_symbols+t` |
| `build_features`/`zscore_across_nc` | `zscore_mode="per_row"` (per-antenna-row, antennas assumed exchangeable); mask broadcast `(1,Nc)→(Na,Nc)` (one pilot reveals all antennas) | `zscore_mode="global"` (both axes carry sample-varying model/speed-dependent structure); mask genuinely 2D via `mask_to_grid` (one pilot reveals exactly one cell) |
| `h_hat_to_H`/`feature_grid_shape` reshape | `na=n_antennas, nc=n_subcarriers` (grid IS the flat candidate space) | `na=n_symbols, nc=n_freq_subcarriers` — **distinct** from the flat candidate space (`n_antennas=1, n_subcarriers=n_symbols*n_freq_subcarriers`) used by labels/LMMSE/policies; `build_features`'s `n_candidates` param decouples the two so the pilot-fraction feature uses the true flat count, not the grid width |
| Artifact suffix | `label_suffix` → `""`/`"_cf"` | `label_suffix` → `"_tf"`/`"_tf_cf"` |
| Phase 1b eval | `eval_phase1b`: one pooled cnn/fixed/active/full-LMMSE curve | `eval_phase1b_time_freq`: a `len(tdl_models) x len(speeds)` grid of panels, one curve set per bucket (a pooled curve would hide whether the CNN generalizes evenly across buckets) |
| CLI | (no `--domain` flag needed) | `--domain time_freq --tdl-models A B C --speeds 3 30 120 --n-cov-mc-per-bucket 256 --n-freq-subcarriers 32 --n-symbols 16`; `--label-mode` auto-defaults to `counterfactual` unless passed explicitly (current mode didn't clearly beat fixed/active, no reason to carry that default into a pricier new-domain run) |

Everything else (label functions at `na=1` block size, LMMSE primitives, `ActivePolicy`/`RandomPolicy`, `CnnPilotPolicy`'s argmax/argmin selection) is reused **unchanged** across both domains — only the pieces in the table above branch on `cfg.domain`. Phase 2 (`build-finetune`/`finetune`) stays `antenna_snapshot`-only, untouched by this mode.

`--train-channels`/`--val-channels`/`--early-stop-patience` are **not** domain-conditioned in code (still the antenna-domain's shared defaults, 12,000/2,000/`None`) — pass them explicitly for a time-freq run, e.g. `--train-channels 13500 --val-channels 2250 --early-stop-patience 8` (~1,500/250 channels per bucket).

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

# Time-freq domain (TDL-A/B/C x speeds, DMRS-comb fixed baseline, cluster-scale)
conda activate model_based_pilot_allocation
python new/error_estimators/cnn/train.py --phase sanity --domain time_freq
python new/error_estimators/cnn/train.py --phase build --domain time_freq --n-channels 18  # optional smoke (2/bucket)
python new/error_estimators/cnn/train.py --phase build --domain time_freq \
    --train-channels 13500 --val-channels 2250 --force-regen
python new/error_estimators/cnn/smoke_validation.py --suffix _tf_cf   # after smoke, before force-regen
python new/error_estimators/cnn/train.py --phase train --domain time_freq --early-stop-patience 8
python new/error_estimators/cnn/train.py --phase eval --domain time_freq --eval-n-mc 100
```

| `--phase` | `--label-mode current` output | `--label-mode counterfactual` output |
|-----------|-------------------------------|----------------------------------------|
| `sanity` | no save | no save (label mode not threaded into sanity path) |
| `build` | `data/train.pt`, `data/val.pt` | `data/train_cf.pt`, `data/val_cf.pt` |
| `train` | `checkpoints/phase1_best.pt`, `phase1_metrics.json` | `checkpoints/phase1_cf_best.pt`, `phase1_cf_metrics.json` |
| `eval` | `figures/phase1b_average_mse_vs_pilots.{png,json}` | `figures/phase1b_average_mse_vs_pilots_cf.{png,json}` |
| `build-finetune` | `data/phase2_{train,val}.pt` | n/a (Phase 2 not extended to counterfactual mode) |
| `finetune` | `checkpoints/phase2_best.pt`, `phase2_metrics.json` | n/a |

`domain="time_freq"` outputs: same table, with `label_suffix` prepending `_tf` (e.g. `data/train_tf.pt`/`data/train_tf_cf.pt`, `checkpoints/phase1_tf_cf_best.pt`, `figures/phase1b_average_mse_vs_pilots_tf_cf.{png,json}` — the eval JSON there is `{n_mc, checkpoint, pilots, buckets: {"<model>_<speed>": {curves, full_lmmse_mse}}}`, not a single `curves` dict).

After smoke build, always `--force-regen` before production train (same cache paths per mode).

## Artifacts (gitignored)

Under `new/error_estimators/cnn/`: `data/`, `checkpoints/`, `figures/`. Not committed. Current-mode (`antenna_snapshot`) artifacts live on the cluster only; counterfactual-mode artifacts from the original local session were produced on `gaussian_san_check`/MX330. `domain="time_freq"` was implemented and smoke-tested end-to-end (sanity/build/train/eval, both label modes) locally on MX330/`gaussian_san_check` — no cluster-scale run yet.

## Seeds (subset)

| Use | Formula |
|-----|---------|
| TDL | `seed + 100_000 + mc` |
| Empirical Σ̂ (antenna_snapshot) | `seed + 1_000_000 + k` |
| Empirical Σ̂ bucket (time_freq) | `seed + 1_000_000 + bucket_idx*100_000 + k` (`bucket_idx` = index into `[(m,s) for m in tdl_models for s in speeds]`) |
| First pilot SC | `Random(channel_seed + 777_777)` |
| Phase 1 train/val `mc` | `ch` / `500_000+ch` |
| Phase 1b eval `mc` | `2_000_000 + mc` (time_freq: `2_000_000 + (bucket_idx)*1_000_000 + mc`) |
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
| Re-`train` + re-`eval` after unused-SC loss (current mode) | **Done** (cluster) | Fresh run via `train_cnn.ssh` (smoke→build[`FORCE_REGEN=1`]→train→eval) on a checkout with both the Sigma-conjugation fix (`825d28a`) and unused-SC loss; defaults (`label_mode=current`, `n_cov_mc=512`, 12k/2k channels). Symptom (`cnn`/`fixed`/`active` nearly overlapping) **persisted** — ruled out both known bugs as the cause. Root-caused to a setup/comparability gap vs. `exp12_time_freq_pilots.py` (different "fixed" baseline — generic bisection vs. DMRS comb; different domain/Sigma-matching), not a defect. Motivated the `domain="time_freq"` mode below. |
| `counterfactual_labels` + `--label-mode` plumbing | **Done** | Vectorized batched-solve implementation; correctness spot-checked against generic `lmmse_incremental_update` |
| Counterfactual `build` (3k/600, local) | **Done** | Local MX330; 93,000 train / 18,600 val snapshots; ~0.55-0.6s/channel actual compute rate |
| Counterfactual `train` | **Done** | `checkpoints/phase1_cf_best.pt`; best val Huber 0.0263 @ epoch 14; early-stopped @ epoch 22 |
| Counterfactual `eval` (Phase 1b diagnostic) | **Done** | `figures/phase1b_average_mse_vs_pilots_cf.{png,json}`; cnn clearly beats active for pilots ~4-27 |
| `pilot_policy/cnn.py` (`CnnPolicy`, argmin) | **Done** | First-class `PilotPolicy`, sanctioned exception file |
| `exp11_cnn_allocator.py` (fixed/active/cnn figure) | **Done** | `new/figures/exp11_cnn_allocator.png`, 50 fresh held-out TDL-A channels; confirms Phase 1b finding — see conclusion delivered in chat |
| Phase 2 `build-finetune` / `finetune` | **Not started** | Current-mode-only scaffold; out of scope this session |
| Phase 3 HP sweep | **Deferred** | |
| `domain="time_freq"` plumbing (config, Sigma buckets, DMRS-comb rollout branch, per-cell CNN head, `build_features` grid/zscore/n_candidates split) | **Done** | See Domains table above; design plan at `.cursor/plans/cnn_time_freq_domain_redesign.plan.md` |
| `domain="time_freq"` local smoke validation | **Done** (MX330, `gaussian_san_check`) | `sanity`, tiny `build`→`train`→`eval` round trip, and `smoke_validation.py --suffix _tf_cf` all pass both domains; caught and fixed a real bug in `build_features`'s pilot-fraction denominator (was using grid width 32 instead of the flat 512-candidate count) during this testing |
| `domain="time_freq"` cluster-scale `build`/`train`/`eval` | **Not started** | Not yet run at the agreed scale (`--train-channels 13500 --val-channels 2250`, `--early-stop-patience 8`); pass `--domain time_freq` explicitly, these two flags are not domain-conditioned in code |

**Resume:** `domain="time_freq"` is implemented and locally smoke-tested (not yet run at cluster scale) — see the Domains table above for the antenna_snapshot vs. time_freq diff, and the CLI section for the exact commands. The counterfactual-mode antenna-domain CNN allocator from an earlier session is trained and evaluated; `new/pilot_policy/cnn.py` + `new/experiments/exp11_cnn_allocator.py` are the reusable entry points (antenna-domain only — no time-freq consumer policy/experiment yet, by design, until Phase 1/1b results are in). Follow-ups if picked back up: run the time-freq Phase 1/1b at cluster scale and check whether `active` now shows a clearer edge over `fixed` (the original motivating question), or extend counterfactual mode with Phase 2 on-policy fine-tuning / a cluster-scale (12k/2k) rebuild for tighter MC noise in the antenna domain.
