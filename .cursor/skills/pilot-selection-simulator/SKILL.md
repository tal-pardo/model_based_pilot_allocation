---
name: pilot-selection-simulator
description: >-
  Conventions for the incremental pilot-selection simulator in new/.
  Use when editing simulation.run_until_threshold, channel_generators,
  channel_estimators, error_estimators, pilot policies, or experiments exp1–exp4.
---

# pilot-selection-simulator

Legacy cumulative code lives in `old/`; see skill `model-based-gaussian`.

## Project core: `simulation.run_until_threshold`

**Entry:** `new/simulation.py` — `run_until_threshold(h_true, sigma, cfg, policy, device=..., generator=...) → RunTrace`.

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
| `gaussian.py` | `build_sigma_kron`, `sample_gaussian_h` | exp1–4 prior; exp1–3 Gaussian truth |
| `sionna.py` | TDL-A/C, CDL-C, `vec_from_h` | exp2–3 TDL-A truth |
| `compound_gaussian.py` | `sample_compound_gaussian_h` (Gamma texture) | exp4 |

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

**Extension:** new schemes in `error_estimators/<name>.py`; wire from simulation and/or policy.

### Pilot policy (related axis)

`pilot_policy/`: `fixed`, `random` (exp1 only), `active` (argmax `J(k)`). Protocol in `base.py`; `make_policy` in `experiments/common.py`.

### Experiment layer combinations

| Exp | Truth | Prior `Sigma` | Estimator | Error est. | Policy |
|-----|-------|---------------|-----------|------------|--------|
| exp1 | Gaussian | Kronecker | LMMSE | trace_min | fixed / random / active |
| exp2 | Gaussian + TDL-A | Kronecker and/or empirical | LMMSE | trace_min | fixed / active |
| exp3 | Gaussian + TDL-A | Kronecker and/or empirical | LMMSE | trace_min | fixed / active |
| exp4 | Gaussian + compound-Gaussian | Kronecker (matched) | LMMSE | trace_min | fixed / active |

When adding features, identify which layer(s) change; keep `run_until_threshold` unless deliberately extending the driver.

## Repo layout

| Path | Role |
|------|------|
| `new/simulation.py` | Canonical loop |
| `new/channel_generators/` | Layer A |
| `new/channel_estimators/` | Layer B |
| `new/error_estimators/` | Layer C |
| `new/pilot_policy/` | Subcarrier selection |
| `new/config.py` | `SimConfig`, `ExpRunConfig`, `Exp1Config`–`Exp4Config` |
| `new/utils.py` | `resolve_device`, `measure_subcarrier`, `empirical_covariance`, `empirical_nmse` |
| `new/experiments/` | exp1–exp4, `common.py`, `vis_channels.py` |
| `new/figures/` | Runtime PNG outputs |

**Imports:** no `__init__.py`; experiment scripts prepend `new/` to `sys.path` → `from simulation import run_until_threshold`.

**Environment:** 
on local PC: conda activate gaussian_san_check 
on cluster: conda activate model_based_pilot_allocation

## Math conventions

- `torch.complex64`, Hermitian `.mH`, prefer `torch.linalg.solve` over inverses
- **vec(H)** (CRITICAL): column-stack by subcarrier — `h = [H[:,0]; …; H[:,Nc-1]]`; SC `k` → indices `k*Na .. (k+1)*Na-1`
- Kronecker: `Sigma = kron(R_freq, R_space) + reg_kron·I`; `H = L_space @ Z @ L_freq.T`
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
| `nmse_threshold` | **0.1** | exp1 stop |
| `target_pilots` | `None` (exp1) / **16** (exp2–4) | |
| `n_mc` | 50 | |
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

Gaussian truth + Kronecker `Sigma`. Policies: fixed, random, active. Stop: `nmse_hat ≤ nmse_threshold` (default 0.1). Y-axis: mean **estimated** `tr(P)/N` + threshold + full-LMMSE baseline. Noise seeds: fixed `+0`, random `+10_000`, active `+20_000`; channel `seed+mc`.

### exp2 — Gaussian vs TDL-A

`python new/experiments/exp2_gaussian_vs_tdl.py` · `Exp2Config`

Fixed/active; stop `target_pilots=16` (`--target-pilots 0` → threshold). Y-axis: mean **true** MSE. Three figures:
1. `exp2_gaussian_sigma.png` — Kronecker Σ for both families
2. `exp2_empirical_sigma.png` — Kronecker (Gaussian) + empirical `Sigma_hat` (TDL, `n_cov_mc=300`)
3. `exp2_tdl_sigma_compare.png` — TDL-A only; Kronecker vs empirical n=250 vs n=500

TDL channels: `seed+100_000+mc`. Cov: `seed+1_000_000+k` (fig2); `+2_000_000`, `+3_000_000` (fig3).

### exp3 — true vs estimated MSE

`python new/experiments/exp3_true_vs_estimated_err.py` · `Exp3Config` · requires `target_pilots=16`

1. `exp3_gaussian_err.png` — 4 curves (fixed/active × true/est) + full-LMMSE; Kronecker
2. `exp3_tdl_err.png` — 1×2: TDL+Kronecker | TDL+empirical `Sigma_hat`

Expected: Gaussian true≈est; TDL/Kronecker gap; TDL/empirical gap may shrink.

### exp4 — matched Sigma, non-Gaussian truth

`python new/experiments/exp4_known_sigma.py` · `Exp4Config` · `target_pilots=16`

`exp4_known_sigma.png` — 1×2: Gaussian control | compound-Gaussian (`texture_alpha`, default 1.0). Same Kronecker `Sigma` for LMMSE in both panels. Compound seeds: `seed+200_000+mc`.

### Seed offsets (summary)

| Purpose | Offset |
|---------|--------|
| Gaussian channel | `seed + mc` |
| Compound-Gaussian (exp4) | `seed + 200_000 + mc` |
| TDL test | `seed + 100_000 + mc` |
| Cov (exp2 fig2, exp3) | `seed + 1_000_000 + k` |
| Cov (exp2 fig3) | `seed + 2_000_000 + k`, `+ 3_000_000 + k` |
| Noise fixed | `seed + 0 + mc` |
| Noise active (exp2–4) | `seed + 10_000 + mc` |
| Noise random (exp1) | `seed + 10_000 + mc` |
| Noise active (exp1) | `seed + 20_000 + mc` |

Same `h_true` per trial for fixed vs active; disjoint noise per policy.

**Utility:** `python new/experiments/vis_channels.py --channel gaussian|tdl_a|tdl_c|--all` → `{channel}_vis.png` (viz uses 300 ns delay spread; sim uses 100 ns).

## Device

`utils.resolve_device`: `cpu`, `cuda`, `cuda:N`, `gpu`; **no silent CPU fallback**. Per-trial `torch.Generator(device=device)`.

## Channel generator detail

- **Sionna** (`sionna.py`): 15 kHz, 3.5 GHz, 100 ns delay (sim); power-normalize `H`. Experiments use **TDL-A**.
- **Compound:** `h = sqrt(s)·g`, `s ~ Gamma(α, rate=α)`, `E[s]=1` ⇒ `E[hh^H]=Σ`.

## Pilot policies

| Policy | Rule | Exps |
|--------|------|------|
| fixed | uniform init + largest-gap bisection | 1–4 |
| random | uniform random unused SC | 1 only |
| active | greedy argmax `J(k)` | 1–4 |
