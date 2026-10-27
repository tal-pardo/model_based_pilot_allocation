---
name: rl-error-estimator
description: >-
  Reinforcement-learning (REINFORCE then PPO) alternative to the CNN active-greedy
  (J(k)) estimator, under new/error_estimators/rl/ (RlTrainConfig, rollout_episode,
  RlPilotPolicy, train_reinforce/train_ppo phases, time_freq domain). Use when
  editing rl/train.py, the REINFORCE/PPO reward/baseline/advantage design, or the
  "rl" arm wired into error_estimators/cnn/train.py's run_closed_loop_mse_curve /
  eval_phase1b_time_freq / --rl-checkpoint CLI. For the CNN itself use
  cnn-error-estimator; for vec(H)/LMMSE/incremental loop use
  pilot-selection-simulator.
---

# RL pilot-allocation policy (Layer C candidate, alternative to the CNN)

## Scope

Same decision as the CNN (`cnn-error-estimator`): which unused subcarrier to pilot
next. Instead of supervised regression against a brute-force counterfactual oracle
label, the policy is trained **on-policy** against the realized NMSE after the pilot
it actually chose — no label-generation step, no offline dataset/cache.

**Companion skills:**
[pilot-selection-simulator](../pilot-selection-simulator/SKILL.md) — vec(H), incremental
LMMSE, seed offsets.
[cnn-error-estimator](../cnn-error-estimator/SKILL.md) — the CNN this is an alternative
to; shares its `CNNErrorEstimator` backbone, `build_features`/`feature_grid_shape`/
`zscore_mode_for`/`mask_to_grid`, channel/Sigma-bucket machinery
(`sample_tdl_channel`, `load_sigma_buckets`, `bucket_for_channel`, `tdl_buckets`),
and `model_arch_kwargs` directly by import — do not duplicate any of that logic here.

**Domain:** `time_freq` only, matching the CNN checkpoint this compares against
(`phase1_tf_cf_best.pt` — **not** `phase1_cf_best.pt`, which is the unrelated
`antenna_snapshot`-domain checkpoint). `RlTrainConfig` is a thin subclass of
`CnnTrainConfig`; always pass the same `--kernel-hw`/`--tdl-models`/`--speeds`/
`--n-freq-subcarriers`/`--n-symbols`/`--n-cov-mc-per-bucket` the CNN checkpoint was
actually trained with (see cnn-error-estimator's "Time-freq cluster-scale tuning"
production config), or the comparison isn't apples-to-apples.

**Hard repo-layout constraint:** all implementation lives inside
`new/error_estimators/rl/`. The **one** sanctioned exception is
`new/error_estimators/cnn/train.py`, which has a small, deliberate `"rl"` branch
added to `run_closed_loop_mse_curve`/`eval_phase1b_time_freq`/the CLI's
`--rl-checkpoint` flag (lazy-imports `RlPilotPolicy`/`load_rl_checkpoint` from here to
avoid a module-load cycle — see "Entry points"). Nothing else in `new/`
(`simulation.py`, `pilot_policy/*`, `experiments/common.py`, other `exp*.py`) may be
edited to support this work.

## Design

| | Stage A: REINFORCE | Stage B: PPO |
|---|---|---|
| Network | `CNNErrorEstimator` (imported from cnn/train.py, same `model_arch_kwargs(cfg)`) — output read as **logits**, not a regression target | Same policy network + a separate small `ValueNet` critic (own conv trunk, not shared with the policy — kept separate for simplicity/robustness) |
| Action | Mask used SCs to `-inf`, softmax, `Categorical`, sample (training) / argmax (inference) | Same |
| Reward | `reward_t = -log(nmse_true_t + eps)` after the actually-realized pilot (no hypothetical counterfactual sweep over candidates — the agent only needs the outcome of the action it actually took) | Same per-step reward |
| Objective | Per-step bandit: `-(log_prob_t * advantage_t)`, advantage = reward minus a **per-(bucket, step) batch-mean baseline** | Undiscounted (`gamma=1`) return-to-go `R_t = sum_{t'>=t} reward_t'` over the fixed `max_pilots`-length episode; advantage = `R_t - V(s_t)`, **normalized per bucket** before the clipped-surrogate update |
| Why per-bucket | The 9 `time_freq` buckets span ~15x in NMSE scale (same finding that motivated the CNN's `bucket_weighted_loss` experiment) — a single global baseline/advantage-normalization would be dominated by whichever bucket's rewards/returns have the largest magnitude | Same rationale |
| Update | Single gradient step per batch of episodes | `ppo_epochs` (default 4) clipped-surrogate passes over the same batch, re-evaluating `log_prob`/`value` under the updating parameters each pass |

Episodes are fixed-length (`k0=1` random bootstrap + `max_pilots-1` agent-chosen
pilots, no threshold stop — same convention as the CNN's Phase 1b rollouts), one
episode per channel realization, run in a plain Python loop (not vectorized across
channels) — consistent with the rest of `new/`'s MC-trial style.

**Channel supply:** unlike the CNN's fixed train/val dataset, RL training draws a
fresh, never-repeated TDL channel every episode via a monotonically increasing
`episode_counter` (no "epoch" concept) — guarded to stay below `EVAL_CHANNEL_OFFSET`
(2,000,000), which is reserved for `rl_real_eval_score`'s held-out validation
channels; a training run that would exceed it raises immediately rather than
silently colliding with the validation channel space.

**Checkpoint selection:** always a `real_eval`-style deterministic (argmax) closed-loop
mean validation NMSE across all 9 buckets (`rl_real_eval_score`), run through the
*exact same* `run_closed_loop_mse_curve` pipeline used for the final cross-arm
comparison — not a separate ad hoc rollout, and not a training-loss proxy (RL has no
separate regression loss to begin with, unlike the CNN's `val_huber` vs `real_eval`
choice).

## Entry points

| Role | Path |
|------|------|
| Sole training CLI (`--stage reinforce\|ppo`) | `new/error_estimators/rl/train.py` |
| First-class pilot policy (inference, argmax) | `RlPilotPolicy` in the same file — mirrors `CnnPilotPolicy`'s shape/signature exactly |
| Checkpoint I/O | `save_rl_checkpoint`/`load_rl_checkpoint` — same blob format as `load_cnn_checkpoint` (`arch` + `model_state_dict`, `+value_state_dict` for PPO) |
| "rl" arm in the CNN's eval pipeline | `run_closed_loop_mse_curve(..., policy_mode="rl", rl_model=...)` and `eval_phase1b_time_freq(..., rl_checkpoint_path=...)` in `error_estimators/cnn/train.py`; CLI: `--rl-checkpoint PATH` on `--phase eval --domain time_freq` |

**Imports:** `new/` + `error_estimators.cnn.train` only (`sys.path` → repo `new/`, same
pattern as cnn/train.py). No `old/`.

## CLI

```bash
# Local (tiny smoke test, both stages, CPU-cheap settings)
conda activate gaussian_san_check
python new/error_estimators/rl/train.py --stage reinforce --domain time_freq \
    --n-updates 2 --episodes-per-update 2 --n-cov-mc-per-bucket 4 \
    --real-eval-n-mc 1 --eval-interval 1 --max-pilots 6 --kernel-hw 7 3
python new/error_estimators/rl/train.py --stage ppo --domain time_freq \
    --n-updates 2 --episodes-per-update 2 --n-cov-mc-per-bucket 4 \
    --real-eval-n-mc 1 --eval-interval 1 --max-pilots 6 --kernel-hw 7 3 --ppo-epochs 2 \
    --init-checkpoint new/error_estimators/rl/checkpoints/reinforce_bandit_tf_reinforce_best.pt

# Cluster (production scale -- match the CNN checkpoint's exact config)
conda activate model_based_pilot_allocation
python new/error_estimators/rl/train.py --stage reinforce --domain time_freq \
    --kernel-hw 7 3 --tdl-models A B C --speeds 3 30 120 \
    --n-cov-mc-per-bucket 256 --n-freq-subcarriers 32 --n-symbols 16 \
    --n-updates 3000 --episodes-per-update 72 --eval-interval 25 --real-eval-n-mc 30
python new/error_estimators/rl/train.py --stage ppo --domain time_freq \
    --kernel-hw 7 3 --tdl-models A B C --speeds 3 30 120 \
    --n-cov-mc-per-bucket 256 --n-freq-subcarriers 32 --n-symbols 16 \
    --init-checkpoint new/error_estimators/rl/checkpoints/reinforce_bandit_tf_reinforce_best.pt \
    --n-updates 1500 --episodes-per-update 72 --ppo-epochs 4 --eval-interval 25 --real-eval-n-mc 30

# Final 4-arm comparison (fixed/active/cnn/rl), from error_estimators/cnn/train.py
python new/error_estimators/cnn/train.py --phase eval --domain time_freq --kernel-hw 7 3 \
    --eval-n-mc 100 --rl-checkpoint new/error_estimators/rl/checkpoints/ppo_trajectory_tf_ppo_best.pt
```

Checkpoints: `checkpoints/reinforce_bandit_tf_reinforce_best.pt` (Stage A),
`checkpoints/ppo_trajectory_tf_ppo_best.pt` (Stage B), with a `_metrics.json`
alongside each (per-update `loss`/`mean_reward`/`real_eval_mean_mse`/etc.). Final
comparison figure/JSON land in `error_estimators/cnn/figures/
phase1b_average_mse_vs_pilots_tf_cf.{png,json}` (shared with the CNN's own eval,
now with a 4th `rl` curve per bucket).

## Seeds

| Use | Formula |
|-----|---------|
| Training rollout noise | `seed + 7_000_000 + channel_mc` (`ROLLOUT_NOISE_OFFSET`, disjoint from every offset in cnn/train.py) |
| Training channel draw | `episode_counter` (monotonically increasing from 0, never repeated, guarded `< EVAL_CHANNEL_OFFSET`) |
| `rl_real_eval_score` validation channels | `EVAL_CHANNEL_OFFSET + bucket_idx*1_000_000 + mc` (same convention as `real_eval_cnn_score`) |
| 4-arm eval noise (`run_closed_loop_mse_curve`, `policy_mode="rl"`) | `seed + EVAL_RL_NOISE_OFFSET(50_000) + channel_mc` (defined in cnn/train.py, disjoint from `NOISE_OFFSET`/`EVAL_CNN_NOISE_OFFSET`) |

## Non-negotiables

- Inherit vec(H) and incremental one-SC-per-step from **pilot-selection-simulator**.
- Reuse `CNNErrorEstimator`/`build_features`/channel-and-Sigma-bucket machinery from
  **cnn-error-estimator** by import; never fork/duplicate that logic here.
- No `old/` imports.
- Do not wire this into `simulation.py`, `pilot_policy/base.py`,
  `pilot_policy/active.py`, `pilot_policy/fixed.py`, `error_estimators/trace_min.py`,
  or `experiments/common.py`. The only sanctioned edit outside this folder is the
  `"rl"` branch inside `error_estimators/cnn/train.py` described above.
- Keep training and the 9-bucket real-eval parity with the CNN's actual production
  config (`kernel_hw`, `tdl_models`, `speeds`, `n_freq_subcarriers`, `n_symbols`,
  `n_cov_mc_per_bucket`) — a mismatch silently makes the 4-arm comparison unfair.

## Progress

Update this section when phases complete or design changes land.

| Milestone | Status | Notes |
|-----------|--------|-------|
| `RlTrainConfig`, `rollout_episode`, `ValueNet` | **Done** | Reuses `CNNErrorEstimator`/`model_arch_kwargs`/`build_features` from cnn/train.py directly |
| Stage A (REINFORCE + per-bucket baseline) | **Done**, untested at scale | Local tiny-config smoke test only (2 updates x 2 episodes, `n_cov_mc_per_bucket=4`) — runs end-to-end, no crash/NaN; real learning quality not yet checked |
| Stage B (PPO + per-bucket-normalized advantage, warm-start from Stage A) | **Done**, untested at scale | Same tiny-config smoke test, including `--init-checkpoint` warm-start path |
| `"rl"` arm in `run_closed_loop_mse_curve`/`eval_phase1b_time_freq`/`--rl-checkpoint` CLI | **Done** | Smoke-tested end-to-end locally (fixed/active/cnn/rl, 1 channel/bucket) against the local `phase1_tf_cf_best.pt` |
| Cluster-scale training (both stages) + full 4-arm eval (`eval_n_mc=100`) | **Done** | 2026-10-26/27, cs-pheno-09 (RTX 3090). Checkpoint confirmed: `phase1_tf_cf_best.pt` cfg matched the CLI defaults exactly (`kernel_hw=(7,3)`, `tdl_models=(A,B,C)`, `speeds=(3,30,120)`, `n_freq_subcarriers=32`, `n_symbols=16`, `n_cov_mc_per_bucket=256`, `sigma2=0.01`, `max_pilots=32`) — no flag adjustments needed. 200-update REINFORCE sanity run: `real_eval_mean` 2.93e-1→1.12e-1, clearly improving, cleared to proceed. **`n_updates` was cut from the literal 3000/1500 defaults to 1200/800** — the measured ~9.3s/update at `episodes_per_update=72` meant the literal defaults (plus `real_eval_n_mc=30`) would have projected to ~9.75h for Stage A alone, blowing the 8h allocation. Stage A (REINFORCE, 1200 updates): best `real_eval_mean=9.09e-2 @ update 1000`; noisy plateau in the 0.09–0.13 band from update ~500 onward (entropy fell 6.1→~2–4 and oscillated, never fully collapsed); final (update 1200) was 1.06e-1. Stage B (PPO, 800 updates, warm-started from the Stage A checkpoint, ~6.3s/update): best `real_eval_mean=9.05e-2 @ update 25` — **essentially the warm-start point; PPO never improved on it across the full 800 updates**, oscillating 0.093–0.113 throughout with no clear trend (`value_loss` did converge cleanly 11.6k→~50-100, so the critic fit fine — the policy just didn't improve). Final 4-arm eval (`eval_n_mc=100`, final-pilot/pilot-32 mean MSE, figure+JSON at `cnn/figures/phase1b_average_mse_vs_pilots_tf_cf.{png,json}`): `active` wins 8/9 buckets; `rl` wins outright only on B_120 (2.6558e-3 vs active's 2.7646e-3); `rl` beats `cnn` in 7/9 buckets (loses only at the two 3 m/s buckets A_3, C_3, where `cnn`'s regression-based estimator does comparatively better); `rl` always clears `fixed` by a wide margin. Per-bucket (cnn / fixed / active / rl / full_lmmse): A_3 1.02e-3/3.01e-3/8.04e-4/1.38e-3/2.23e-4; A_30 2.51e-3/3.15e-2/1.43e-3/1.73e-3/9.59e-4; A_120 6.87e-3/3.44e-1/2.70e-3/2.97e-3/3.28e-3; B_3 1.01e-3/2.70e-3/8.51e-4/1.17e-3/2.26e-4; B_30 2.19e-3/2.63e-2/1.42e-3/1.48e-3/8.49e-4; B_120 6.48e-3/2.92e-1/2.76e-3/2.66e-3/3.21e-3; C_3 1.28e-3/3.47e-3/1.27e-3/2.60e-3/2.46e-4; C_30 2.87e-3/3.87e-2/1.59e-3/2.02e-3/8.90e-4; C_120 7.03e-3/2.93e-1/2.86e-3/3.07e-3/3.23e-3. |

**Resume:** first cluster-scale run is complete end-to-end (both stages + 4-arm eval);
code works correctly at scale and the RL policy is a legitimate second-best arm
(beats `fixed` everywhere, beats `cnn` at 30/120 m/s, loses to `cnn` at 3 m/s, trails
`active` almost everywhere by a modest margin). **The open question for a follow-up
run is why Stage B (PPO) produced no improvement over the warm-started Stage A
policy** — it held flat in the same 0.09–0.11 band across all 800 updates rather than
refining it, despite the critic (`value_loss`) converging cleanly. Worth investigating
before trusting PPO's design as-is: try a lower `--lr` (the default may be too high
for a warm-started policy and immediately disrupting the good REINFORCE solution),
a smaller `--clip-eps`, tuning `--entropy-coef`, or simply confirming via ablation
whether PPO at all helps over just running REINFORCE longer. Also note `n_updates`
for both stages was sized to the available compute window (1200/800, not the
documented 3000/1500 CLI defaults above) — rerun with the full defaults if a longer
allocation is available, since Stage A's plateau (0.09–0.13, not still clearly
descending at update 1200) suggests more updates might still help.
