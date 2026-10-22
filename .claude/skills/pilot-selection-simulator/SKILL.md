---
name: pilot-selection-simulator
description: >-
  Conventions for the incremental pilot-selection simulator in new/.
  Use when editing simulation.run_until_threshold / run_until_threshold_genie,
  channel_generators, channel_estimators, error_estimators, pilot policies,
  or experiments exp1–exp9.
---

# pilot-selection-simulator

Legacy cumulative code lives in `old/`; see skill `model-based-gaussian`.

## Project core: `simulation.run_until_threshold`

**Entry:** `new/simulation.py` — `run_until_threshold(h_true, sigma, cfg, policy, device=..., generator=...) → RunTrace`.
Genie sibling: `run_until_threshold_genie(h_true, sigma_est, sigma_score, ...)` — estimator on `sigma_est`, active `J(k)` from a parallel shadow LMMSE on `sigma_score` (same `y_t`); logged NMSE from estimator only.

**Incremental rule:** each step measures **one** new subcarrier only (`X_t (Na,N)`, `y_t (Na,1)`, fresh AWGN). Never re-measure the full pilot set (unlike `old/` cumulative loop).

**Loop structure (fixed):**
1. `policy.reset()` → `initial_subcarriers()` (length `k0`; e.g. `k0=2`, `Nc=32` → `[0, 15]`)
2. First init SC: `measure_subcarrier` → `lmmse_initial_update(sigma, x0, y0, sigma2)` → log `nmse_true`, `nmse_hat`
3. Remaining `k0-1` init SCs: measure → `lmmse_incremental_update` → log
4. While not stopped: `k = policy.next_subcarrier(state, used)` → measure → `lmmse_incremental_update` → log
5. Stop when `target_pilots` reached, `nmse_hat ≤ nmse_threshold`, or `max_pilots`

**`RunTrace`:** `nmse_true` (oracle `(1/N)||h_hat-h||²`), `nmse_hat` (model-based), `subcarriers`, `n_pilots`, `stopped_reason` ∈ `{threshold, max_pilots, target_pilots}`.

**Experiments inject:** `h_true`, prior `Sigma`, `PilotPolicy`. Channel generators are **not** imported inside `simulation.py`.

## Swappable component layers

`new/` is a simulator framework: the loop is stable; experiments vary which implementations are composed.

### Layer A — `channel_generators/` (truth + prior construction)

Produce `h_true` and (usually) Kronecker `Sigma`; experiments pass both into `run_until_threshold`.

| Module | Role | Used in |
|--------|------|---------|
| `gaussian.py` | `build_sigma_kron`, `sample_gaussian_h` | exp1–4 prior; exp1–3 Gaussian truth; exp7 Kronecker |
| `sionna.py` | TDL-A/C, CDL-C, `vec_from_h` | exp2–5 TDL-A truth |
| `compound_gaussian.py` | `sample_compound_gaussian_h` (Gamma texture) | exp4 |
| `sigma_diag.py` | `build_sigma_diag_from_tdl`, `sigma_h_squared` | exp5 prior (`Σ = σ_H² I`) |
| `jakes.py` | `build_sigma_jakes_kron` (`R_freq ⊗ R_jakes`) | exp7 |
| `low_rank.py` | `build_sigma_low_rank`, `sample_low_rank_h` | exp7–8 |
| `block_freq.py` | `build_sigma_block_kron`, block AR(1) clusters | exp9 |

Empirical `Sigma_hat` from `utils.empirical_covariance` on Sionna draws (exp2/3) is a prior choice, not a separate generator.

**Extension:** add modules under `channel_generators/`; experiment selects sampler + prior.

### Layer B — `channel_estimators/` (belief update)

Maintain `EstimatorState(h_hat, P)` after each `(X_t, y_t)`.

| Module | API | Status |
|--------|-----|--------|
| `lmmse.py` | `lmmse_initial_update`, `lmmse_incremental_update` | **Only implementation today** |

Wired in `simulation.py` (hard-import). `experiments/common.py` reuses for full-subcarrier baselines.

**Extension:** new modules under `channel_estimators/`; may later inject into `run_until_threshold`.

### Layer C — `error_estimators/` (model-based error + pilot scores)

Estimate MSE without oracle `h_true` — stopping, logging, active scoring.

| Module | Functions | Used where |
|--------|-----------|------------|
| `trace_min.py` | `estimate_nmse` = `tr(P)/N` | `simulation.py` stop + trace; exp1 y-axis |
| `trace_min.py` | `active_subcarrier_score` → `J(k)` | `pilot_policy/active.py` |
| `cnn/` | supervised `CNNErrorEstimator` (train/eval CLI, current + counterfactual label modes) | **not wired** into `simulation.py`/`make_policy`; consumed only via the standalone `pilot_policy/cnn.py` (`CnnPolicy`) and `experiments/exp11_cnn_allocator.py` — see skill `cnn-error-estimator` |

**Extension:** new schemes in `error_estimators/<name>.py`; wire from simulation and/or policy. For the CNN sub-project under `error_estimators/cnn/`, use skill **`cnn-error-estimator`**.

### Pilot policy (related axis)

`pilot_policy/`: `fixed`, `random` (exp1, exp10), `repeat` (exp1b: re-measures last init SC), `active` (argmax `J(k)`), `cnn` (`CnnPolicy`, argmin predicted counterfactual NMSE — not registered in `make_policy`, only used directly by `exp11_cnn_allocator.py`; see skill `cnn-error-estimator`). Protocol in `base.py`; `make_policy` in `experiments/common.py` (fixed/random/active/repeat).

**Plot convention (all figures):** `common.POLICY_STYLES` — fixed `C0` blue `-` circles, random `C1` orange `--` squares, active `C2` green `-.` triangles; full-LMMSE baseline red dotted (`FULL_LMMSE_COLOR`/`FULL_LMMSE_LINESTYLE`). exp3-style true/est overlays: true = solid + marker, est = dotted same color, no marker. Oracle (exp10) purple `P`.

### Experiment layer combinations

| Exp | Truth | Prior `Sigma` | Estimator | Error est. | Policy |
|-----|-------|---------------|-----------|------------|--------|
| exp1 | Gaussian | Kronecker | LMMSE | trace_min | fixed / random / active |
| exp1b | Gaussian | Kronecker | LMMSE | trace_min | fixed / random / active / repeat |
| exp2 | Gaussian + TDL-A | Kronecker and/or empirical | LMMSE | trace_min | fixed / active |
| exp3 | Gaussian + TDL-A | Kronecker and/or empirical | LMMSE | trace_min | fixed / active |
| exp4 | Gaussian + compound-Gaussian | Kronecker (matched) | LMMSE | trace_min | fixed / active |
| exp5 | TDL-A | diagonal `σ_H² I` from TDL tap-power sum | LMMSE | trace_min | fixed / active |
| exp7 | 4 Gaussians (Toeplitz / Kronecker / Jakes+Rf / low-rank) | matched `Σ` vs identity `I` | LMMSE | trace_min | active |
| exp8 | low-rank only | est `I`; genie score uses matched `Σ` | LMMSE (+ shadow) | trace_min | active / fixed / genie-active |
| exp9 | block-cluster Gaussian | matched block `Σ` | LMMSE | trace_min | fixed / active |

When adding features, identify which layer(s) change; keep `run_until_threshold` unless deliberately extending the driver.

## Repo layout

| Path | Role |
|------|------|
| `new/simulation.py` | Canonical loop |
| `new/channel_generators/` | Layer A |
| `new/channel_estimators/` | Layer B |
| `new/error_estimators/` | Layer C |
| `new/pilot_policy/` | Subcarrier selection |
| `new/config.py` | `SimConfig`, `ExpRunConfig`, `Exp1Config`–`Exp9Config` |
| `new/utils.py` | `resolve_device`, `measure_subcarrier`, `empirical_covariance`, `empirical_nmse` |
| `new/experiments/` | exp1–exp9, `common.py`, `vis_channels.py` |
| `new/figures/` | Runtime PNG outputs |

**Imports:** no `__init__.py`; experiment scripts prepend `new/` to `sys.path` → `from simulation import run_until_threshold`.

**Environment:** 
on local PC: conda activate gaussian_san_check 
on cluster: conda activate model_based_pilot_allocation

## Math conventions

- `torch.complex64`, Hermitian `.mH`, prefer `torch.linalg.solve` over inverses
- **vec(H)** (CRITICAL): column-stack by subcarrier — `h = [H[:,0]; …; H[:,Nc-1]]`; SC `k` → indices `k*Na .. (k+1)*Na-1`
- Kronecker: `Sigma = kron(R_freq, R_space) + reg_kron·I`; `H = L_space @ Z @ L_freq.T`
- Empirical covariance (`utils.empirical_covariance`, rows = samples `h_k^T`): `Σ̂ = Hᵀ conj(H)/K = E[h hᴴ]`. **Not** `Hᴴ H` — that is `conj(Σ)`; identical for real Σ (Kronecker) but wrong for TDL/CDL (complex freq. correlation). Fixed 2026-10-20; CNN caches/checkpoints built before then used `conj(Σ)`.
- Sionna 2.x has its own RNG: seeding sets `sionna.phy.config.seed` (in `sionna._set_torch_seed`), otherwise TDL/CDL draws are not reproducible.
- `reg_kron` default `1e-9`; `reg_empirical` default `1e-3` (empirical cov when `n_cov_mc < N`)
- Active score for SC `k`, `s=k*Na`: `J(k)=tr(solve(σ²I+P_k, Q_k)).real` with `P_k=P[s:s+Na,s:s+Na]`, `Q_k=P[s:s+Na,:]@P[:,s:s+Na]`

## Shapes and defaults

| Symbol | Default | Meaning |
|--------|---------|---------|
| `Na`, `Nc`, `N` | 16, 32, 512 | antennas, subcarriers, `Na*Nc` |
| `k0` | 2 | `initial_pilot_subcarriers` |
| `sigma2` | `1e-2` | AWGN variance |
| `rho_space`, `rho_freq` | 0.8, 0.85 | Kronecker correlations |
| `device` | `"cuda"` | fails if CUDA unavailable |
| `nmse_threshold` | **0.1** (exp1) / **0.5** (exp5) | threshold stop on `tr(P)/N` |
| `target_pilots` | `None` (exp5) / **16** (exp2–4) / **32** (exp1, exp1b, exp7–10) | |
| `n_mc` | 50 (exp7 default **100**) | |
| `n_cov_mc` | 300 | exp2/3 empirical Σ |
| `texture_alpha` | 1.0 | exp4 compound-Gaussian |

Tensors: `H (Na,Nc)`, `h (N,1)`, `Sigma (N,N)`, `X_t (Na,N)`, `y_t (Na,1)`, `P (N,N)`. Docstrings: `Input:` / `Output:` with shapes.

## Time indexing (exp3/exp4)

- `RunTrace[0]` = after first init SC only; `RunTrace[k0-1]` = plot **t=0** (all `k0` bootstrap done)
- Policy time: `t=0..T`, `T = target_pilots - k0`; plot slice `slice(k0-1, target_pilots)`
- Differs from `old/` `T+1` prior-at-t=0 convention
- Full LMMSE baseline: all `Nc` SCs once; batch equivalent to incremental chain (`common.py`)

## Experiments

Run from repo root. Figures under `new/figures/`.

### exp1 — pilot policies

`python new/experiments/exp1.py` · `Exp1Config` · `exp1_pilot_policy.png`

Gaussian truth + Kronecker `Sigma`. Policies: fixed, random, active. Stop: `target_pilots=32` (`--target-pilots 0` → stop on `nmse_hat ≤ nmse_threshold`, 0.1). Y-axis: mean **estimated** `tr(P)/N` + full-LMMSE baseline (no threshold line). Noise seeds: fixed `+0`, random `+10_000`, active `+20_000`; channel `seed+mc`.

### exp1b — same subcarrier repeated

`python new/experiments/exp1b_repeat_subcarrier.py` · `Exp1bConfig` · `exp1b_repeat_subcarrier.png`

exp1 settings and curves plus `repeat`: init `[0, 15]`, then SC 15 re-measured every step (fresh noise, `+40_000`). Est. MSE decreases only slowly (≈0.704 → 0.702 from 2 to 32 pilots): SC 15 variance → σ²/m, rest only via correlation.

### exp2 — Gaussian vs TDL-A

`python new/experiments/exp2_gaussian_vs_tdl.py` · `Exp2Config`

Fixed/active; stop `target_pilots=16` (`--target-pilots 0` → threshold). Y-axis: mean **true** MSE. Three figures:
1. `exp2_gaussian_sigma.png` — Kronecker Σ for both families
2. `exp2_empirical_sigma.png` — Kronecker (Gaussian) + empirical `Sigma_hat` (TDL, `n_cov_mc=300`)
3. `exp2_tdl_sigma_compare.png` — TDL-A only; Kronecker vs empirical n=250 vs n=500

TDL channels: `seed+100_000+mc`. Cov: `seed+1_000_000+k` (fig2); `+2_000_000`, `+3_000_000` (fig3).

### exp3 — true vs estimated MSE

`python new/experiments/exp3_true_vs_estimated_err.py` · `Exp3Config` · requires `target_pilots` (default 16; deck figure uses `--target-pilots 32`)

1. `exp3_gaussian_err.png` — 4 curves (fixed/active × true/est) + full-LMMSE; Kronecker
2. `exp3_tdl_err.png` — 1×2: TDL+Kronecker | TDL+empirical `Sigma_hat`

Expected: Gaussian true≈est; TDL/Kronecker gap; TDL/empirical gap may shrink.

### exp4 — matched Sigma, non-Gaussian truth

`python new/experiments/exp4_known_sigma.py` · `Exp4Config` · `target_pilots=16`

`exp4_known_sigma.png` — 1×2: Gaussian control | compound-Gaussian (`texture_alpha`, default 1.0). Same Kronecker `Sigma` for LMMSE in both panels. Compound seeds: `seed+200_000+mc`.

### exp5 — TDL-A with diagonal Σ prior

`python new/experiments/exp5_diag_sigma.py` · `Exp5Config` · `exp5_diag_sigma.png`

TDL-A truth + **scalar diagonal prior** `Σ = σ_H² I + reg_kron·I`, with `σ_H² = Σ_l P_l` from the fixed 3GPP TDL tap-power profile (`sigma_diag.build_sigma_diag_from_tdl`; not per-realization, not empirical). Structural mismatch: correlated TDL truth vs i.i.d. LMMSE prior (simpler than exp3 Kronecker misspecification).

Fixed/active; stop `target_pilots=None`, `nmse_hat ≤ nmse_threshold` (default **0.5** on estimated MSE). Plot horizon `plot_len` from mean **estimated** curves crossing threshold. One shared MC run set for both panels. TDL channels: `seed+100_000+mc` (same as exp2/3). Noise: fixed `+0`, active `+10_000`.

`exp5_diag_sigma.png` — 1×2:
1. **Left** — mean **true** MSE for `tdl (fixed)` / `tdl (active)`; gray dotted hline at `nmse_threshold` (reference only; stopping uses est.)
2. **Right** — active policy only: true vs est (`tr(P)/N`) + full-LMMSE true baseline (red dotted)

Console: `sigma_H2`, `assert_sigma_diag` check, `mean_pilots_to_threshold` per policy, true/est gap at last plotted step. Expected: gap between true and est on right (prior mismatch); active may need fewer pilots than fixed.

### exp7 — matched LMMSE vs identity prior

`python new/experiments/exp7_matched_vs_identity.py` · `Exp7Config` · figures under `new/figures/exp7/`

Four Gaussian truths (matched prior = true `Σ`; suboptimal prior = exact `I`). Active only; `k0=1`, init SC **15**; stop `target_pilots=32`; `n_mc=100`. Y-axis: mean **estimated** `tr(P)/N` only (no true MSE). One figure per scenario:

1. `exp7_toeplitz.png` — Toeplitz `ρ=0.85` on full `N`
2. `exp7_kronecker.png` — Kronecker `ρ_space=ρ_freq=0.85`
3. `exp7_jakes.png` — Jakes `R_space` (`d/λ=0.5`) ⊗ AR(1) `R_freq` (`ρ=0.85`)
4. `exp7_low_rank.png` — factor `Σ=BB^H+εI`, `r=16`, `ε=1e-3`, random semi-orthogonal `B` scaled so `tr(Σ)/N=1`

Noise: matched `+0`, identity `+10_000`. Channel seeds: Toeplitz/Kronecker `seed+mc`; Jakes `+400_000+mc`; low-rank `+500_000+mc`; `B` factor `+3_000_000`. Expected: largest matched–vs–identity gap on low-rank.

### exp8 — low-rank genie active (matched J(k), identity estimator)

`python new/experiments/exp8_low_rank_genie.py` · `Exp8Config` · figures under `new/figures/exp8/`

Low-rank only (same `r`/`ε`/seeds as exp7). `k0=1`, init SC **15**; `target_pilots=32`; `n_mc=100`. Arms share `h_true`; matched + genie share noise offset `+0` (same pilots/`y_t`); identity-active `+10_000`; identity-fixed `+20_000`; matched-fixed `+30_000`. Fixed/active support `initial_indices=[15]`.

1. `exp8_low_rank_genie.png` — matched est, identity+active est, genie est, genie true
2. `exp8_genie_vs_fixed.png` — genie est/true + identity+fixed est/true
3. `exp8_matched_active_vs_fixed.png` — matched LMMSE: active est, fixed est, active true

### exp9 — block-cluster active vs fixed

`python new/experiments/exp9_block_cluster.py` · `Exp9Config` · `exp9_block_cluster.png`

Block-cluster Gaussian truth + matched `Σ = kron(R_freq, R_space)` (`block_freq.py`: ρ_intra=0.95, ρ_inter=0.05 default). Matched LMMSE; fixed vs active; `k0=2` uniform init; stop `target_pilots=32`; `n_mc=50`. Shared `h_true`; noise fixed `+0`, active `+10_000`; channel `seed+mc`. Figure 1×2: MSE curves (fixed/active × est/true) | gap vs pilot count at every m.

Expected: clear mid-horizon gap (active ≪ fixed est/true); curves meet at m=Nc.

### exp10 — generic Σ policy comparison (TDL+CDL sweep)

`python new/experiments/exp10_genie_oracle_gap.py` · `Exp10Config` · `figures/exp10/exp10_generic_sigma.png`

18 configs (TDL/CDL × A/B/C × 3 delay spreads, 120 kHz SCS), episodes round-robin. One pooled empirical Σ (`n_cov_mc=2000`, `+3_000_000`). `k0=1`, init SC 15 for all arms; `target_pilots=32`; `n_mc=36`. Arms: fixed, random, active, oracle (brute-force true-NMSE argmin, trial noise `+4_000_000`) + full-LMMSE true baseline. Y-axis: mean **true** NMSE vs number of pilots. (Genie per-config Σ arms were removed.)

### Seed offsets (summary)

| Purpose | Offset |
|---------|--------|
| Gaussian channel | `seed + mc` |
| Compound-Gaussian (exp4) | `seed + 200_000 + mc` |
| TDL test | `seed + 100_000 + mc` |
| Cov (exp2 fig2, exp3) | `seed + 1_000_000 + k` |
| Cov (exp2 fig3) | `seed + 2_000_000 + k`, `+ 3_000_000 + k` |
| Noise fixed | `seed + 0 + mc` |
| Noise active (exp2–5) | `seed + 10_000 + mc` |
| Noise random (exp1) | `seed + 10_000 + mc` |
| Noise active (exp1) | `seed + 20_000 + mc` |
| Noise repeat (exp1b) | `seed + 40_000 + mc` |
| Noise fixed / active / random / full-LMMSE (exp10) | `+10_000` / `+20_000` / `+40_000` / `+50_000` |
| Noise matched / identity (exp7) | `seed + 0 + mc` / `seed + 10_000 + mc` |
| Noise matched+genie / id-active / id-fixed / matched-fixed (exp8) | `+0` / `+10_000` / `+20_000` / `+30_000` |
| Jakes / low-rank channel (exp7–8) | `seed + 400_000 + mc` / `seed + 500_000 + mc` |
| Low-rank `B` (exp7–8) | `seed + 3_000_000` |

Same `h_true` per trial for fixed vs active; disjoint noise per policy.

**Utility:** `python new/experiments/vis_channels.py --channel gaussian|tdl_a|tdl_c|--all` → `{channel}_vis.png` (viz uses 300 ns delay spread; sim uses 100 ns).

## Device

`utils.resolve_device`: `cpu`, `cuda`, `cuda:N`, `gpu`; **no silent CPU fallback**. Per-trial `torch.Generator(device=device)`.

## Channel generator detail

- **Sionna** (`sionna.py`): 15 kHz, 3.5 GHz, 100 ns delay (sim); power-normalize `H`. Experiments use **TDL-A**.
- **Compound:** `h = sqrt(s)·g`, `s ~ Gamma(α, rate=α)`, `E[s]=1` ⇒ `E[hh^H]=Σ`.
- **Sigma_diag** (`sigma_diag.py`): `σ_H² = sum(tap_powers)` from Sionna `TDL(...).mean_powers` (same grid/delay/carrier as `sionna.py`); `Σ = σ_H² I + reg·I`. Normalized TDL-A ⇒ `σ_H² ≈ 1`.
- **Jakes** (`jakes.py`): `[R_space]_ij = J_0(2π |i-j| d/λ)`; Kronecker with AR(1) `R_freq` (exp7).
- **Low-rank** (`low_rank.py`): `h = Bz + √ε w`, `Σ = BB^H + εI`, semi-orthogonal `B` scaled to unit average power (exp7).
- **Block_freq** (`block_freq.py`): two contiguous SC clusters, AR(1) within (`ρ_intra`), constant cross-cluster `ρ_inter`; `Σ = kron(R_freq, R_space)` (exp9).

## Pilot policies

| Policy | Rule | Exps |
|--------|------|------|
| fixed | uniform init + largest-gap bisection | 1–5, 8 (`initial_indices` optional; exp8 uses `[15]`) |
| random | uniform random unused SC | 1, 10 (`initial_indices` optional; exp10 uses `[15]`) |
| repeat | uniform init, then last init SC every step | 1b |
| active | greedy argmax `J(k)` | 1–5, 7–8 (`initial_indices` optional; exp7/8 use `[15]`) |
