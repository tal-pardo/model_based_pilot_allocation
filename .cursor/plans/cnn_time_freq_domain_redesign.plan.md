# CNN training on a multi-model, multi-speed time-freq domain

## Context

The antenna-domain CNN Phase 1b eval (`new/error_estimators/cnn/train.py`, `domain="antenna_snapshot"`) showed `cnn`/`fixed`/`active` nearly overlapping in MSE, even after two previously-suspected bugs were fixed (a Sigma-conjugation bug, commit `825d28a`; and switching to unused-SC-only Huber loss) — the symptom persisted on a fresh post-fix cluster run. Root cause: not a bug, but a setup/comparability gap versus `new/experiments/exp12_time_freq_pilots.py`, which shows a much clearer `active > fixed` gap on a **time-frequency** grid with a **DMRS-comb** "fixed" baseline instead of the antenna-domain's generic 1D-bisection `FixedPolicy`.

This plan adds a `domain="time_freq"` mode to the same `train.py`/`CnnTrainConfig`, reproducing exp12's domain/baseline pairing inside the CNN pipeline: single-antenna, joint time-frequency OFDM grid (`n_symbols` OFDM symbols × `n_freq_subcarriers` subcarriers, flat index `s=k*n_symbols+t`), swept across **TDL-A/B/C × speeds** (9 buckets), with the goal of testing whether richer time-freq structure gives the CNN more to learn from, and whether `active` now shows a clearer edge over `fixed` inside this pipeline too.

**Decisions confirmed with the user:**
- Speed/model sampling: a **discrete grid** — `tdl_models=("A","B","C")` × `speeds=(3.0, 30.0, 120.0)` m/s = 9 buckets (reuses `exp13_unknown_sigma_time_freq.py`'s proven speed sweep), not continuous per-channel speed.
- Sigma per bucket: **Ledoit-Wolf shrinkage** (`new/channel_estimators/cov_update.py::ledoit_wolf_shrinkage`) from `n_cov_mc_per_bucket=256` draws per `(model,speed)` bucket, not a single shared Sigma (which would be statistically mismatched once channels vary in TDL model/speed).
- Training scale: keep total channel count at the antenna-domain's order of magnitude, split evenly across the 9 buckets (~1,500 train / ~250 val channels/bucket via `--train-channels 13500 --val-channels 2250`), not a 9x scale-up.
- The "fixed" baseline's DM-RS comb config (`comb_spacing`/`pilot_symbols`) stays **one fixed configuration shared across all 9 buckets**, not scaled per speed.
- Output artifacts (cache/checkpoint/figures) go in a **separate namespace** (`_tf`/`_tf_cf` suffix) from the existing antenna-domain artifacts.
- **Extend `train.py`/`CnnTrainConfig` in place via a `domain` field — not a new sibling module.** Most of `train.py` (training loop, CLI phase dispatch, checkpoint plumbing, label functions at block size 1, LMMSE primitives, `ActivePolicy`/`RandomPolicy`) is already domain-agnostic; duplicating the whole file would mean maintaining two copies of code that's identical in substance. The genuinely divergent pieces (channel/Sigma generation, "fixed" rollout policy, CNN head, feature grid/mask/z-score) each collapse into one well-scoped parameter/branch.

## Implementation (done — see `.claude/skills/cnn-error-estimator/SKILL.md` "Domains" table for the full antenna_snapshot-vs-time_freq diff)

1. **Config** — `CnnTrainConfig` gained `domain: Literal["antenna_snapshot","time_freq"] = "antenna_snapshot"` plus time-freq-only fields (`n_freq_subcarriers`, `n_symbols`, `tdl_models`, `speeds`, `n_cov_mc_per_bucket`, `dmrs_comb_spacing`, `dmrs_pilot_symbols`). `--domain time_freq` on the CLI auto-derives `n_antennas=1`/`n_subcarriers=n_symbols*n_freq_subcarriers` and defaults `label_mode` to `counterfactual` unless passed explicitly. `_check_domain_config` validates the derived shape.
2. **Channel generation** — `sample_tdl_channel` branches on `cfg.domain`; time-freq branch calls `sample_tdl_ofdm_channel` directly (no new per-model wrapper needed — `model="B"` already works via the generic function, same precedent as `exp10_genie_oracle_gap.py`). `bucket_for_channel(cfg, channel_mc)` round-robins the 9 `(model,speed)` buckets.
3. **Sigma** — `load_sigma_buckets(cfg, device)` returns `{(model,speed): Sigma}`; antenna mode: one entry via the existing `load_empirical_sigma`; time-freq mode: one Ledoit-Wolf-shrunk Sigma per bucket from `n_cov_mc_per_bucket` draws each.
4. **Policy-mix rollout** — `_make_policy` substitutes `DmrsCombPolicy` for `"fixed"` only in time-freq mode, mirroring `exp12_time_freq_pilots.py::_make_exp12_policy`; `comb_offset = channel_mc % cfg.dmrs_comb_spacing` gives "fixed" rollouts per-channel diversity (the policy itself has no seed).
5. **CNN architecture** — `CNNErrorEstimator` gained `pool_mode: Literal["mean","per_cell"]` and a `kernel_hw` param. `pool_mode="per_cell"` (time-freq) replaces the antenna-domain's mean-pool-the-nuisance-axis head with a per-cell `Conv2d` head, flattened via `permute(0,2,1).reshape(...)` to match `s=k*n_symbols+t`.
6. **Features** — `feature_grid_shape(cfg)` gives `(n_symbols, n_freq_subcarriers)` for time-freq vs `(n_antennas, n_subcarriers)` for antenna mode — distinct from the flat `(n_antennas=1, n_subcarriers)` candidate space used by labels/LMMSE/policies. `mask_to_grid` reshapes (time-freq, genuinely 2D) vs broadcasts (antenna, one pilot reveals all antennas) the flat pilot mask into that grid. `build_features` gained `zscore_mode` (`"global"` for time-freq, since both axes carry sample-varying model/speed-dependent structure that per-row z-scoring would flatten out) and a `n_candidates` param decoupling the pilot-fraction feature's denominator (always the true flat candidate count) from the grid width (only coincide in antenna mode) — **a real bug caught during local testing**: without this, time-freq's pilot-fraction feature used 32 (grid width) instead of 512 (flat count).
7. **Eval** — `eval_phase1b_time_freq` (new function) reports a `len(tdl_models) x len(speeds)` grid of per-bucket cnn/fixed/active/full-LMMSE curves, instead of one pooled curve — otherwise it's impossible to tell whether the CNN generalizes evenly across buckets or just wins on whichever it saw most.
8. **`smoke_validation.py`** — extended with domain-aware branches: reads `meta["domain"]`, uses `feature_grid_shape`-equivalent logic for expected tensor shapes, and skips the antenna-domain's mask-broadcast assertion in time-freq mode (the mask there is genuinely 2D by design, not a broadcast) — this is the one existing check that was actively wrong for the new mode, not just inapplicable. New `--suffix` CLI flag to point at a non-default cache (e.g. `--suffix _tf_cf`).

**Verified locally** (MX330, `gaussian_san_check`): `--phase sanity` for both domains; a full `build → train → eval` round trip at tiny scale for both domains; `smoke_validation.py` for both domains (all checks pass, including the "fixed" channel's bit-exact regeneration check, confirming the per-channel `comb_offset` is reproducible). Even with a near-untrained model on a 3-channel smoke cache, the time-freq eval already showed `active` clearly beating the DMRS-comb `fixed` baseline in every bucket (e.g. TDL-A @ 120 m/s: fixed=3.5e-1 vs active=1.9e-3) — structurally confirming this redesign reproduces exp12's kind of gap.

**Not yet done:** a cluster-scale run (`--train-channels 13500 --val-channels 2250 --early-stop-patience 8`, full `n_cov_mc_per_bucket=256`/`eval_n_mc=100`).

## Critical files

- `new/error_estimators/cnn/train.py` — `CnnTrainConfig` (domain + time-freq fields), `sample_tdl_channel`, `load_sigma_buckets`/`bucket_for_channel`, `_make_policy`, `build_features`/`zscore_across_nc`/`mask_to_grid`/`feature_grid_shape`, `CNNErrorEstimator`/`model_arch_kwargs`, `eval_phase1b_time_freq`, CLI argparse (`--domain`, `--tdl-models`, `--speeds`, `--n-cov-mc-per-bucket`, `--n-freq-subcarriers`, `--n-symbols`).
- `new/error_estimators/cnn/smoke_validation.py` — domain-aware checks, `--suffix` flag.
- `new/channel_generators/sionna.py`, `new/channel_estimators/cov_update.py`, `new/pilot_policy/dmrs_comb.py` — reused as-is, no edits.

## Verification (next steps, on the cluster)

1. `python new/error_estimators/cnn/train.py --phase sanity --domain time_freq` (and `--domain antenna_snapshot` regression check).
2. Small per-bucket smoke build (`--n-channels 18` ≈ 2/bucket) → `smoke_validation.py --suffix _tf_cf`.
3. Full `build --train-channels 13500 --val-channels 2250 --force-regen` → `train --early-stop-patience 8` → `eval --eval-n-mc 100`, all with `--domain time_freq`.
4. Inspect the 3×3 per-bucket Phase 1b figure: does `active` show a clearer edge over `fixed` than the antenna-domain pipeline did, and is that edge more pronounced at higher speed / larger-delay-spread TDL models (where channel structure is more exploitable) — the original motivating question.
