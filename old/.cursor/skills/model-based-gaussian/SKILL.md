---
name: model-based-gaussian
description: Conventions for the model_based_gaussian repo (complex torch, vec(H) ordering, Kronecker Sigma, pilot-as-subcarrier selection, fixed vs active allocation, time-step indexing, and experiment1 Sionna vs Gaussian MSE). Use when editing main.py/pilots.py/estimators.py/data_generator.py/sionna_channels.py or implementing new estimators or sampling strategies in this repo.
---

# model_based_gaussian conventions

## Repo scope
This repo’s source is limited to:
- `main.py`: `main()` (legacy Gaussian-only fixed vs active MSE + optional live heatmap) and `experiment1()` (Gaussian vs Sionna TDL/CDL comparison; no heatmap).
- `pilots.py`: fixed/active pilot samplers + shared sequential LMMSE driver.
- `estimators.py`: recursive + batch LMMSE for complex Gaussian prior.
- `data_generator.py`: complex Gaussian, exponential correlation, selection matrix helpers, `empirical_covariance`.
- `sionna_channels.py`: static Sionna TDL-C / CDL-C OFDM channels → `H (Na,Nc)` and `vec(H)`.

## Model & shapes
- Default **legacy** `main()`: `Na=32`, `Nc=64`, `N=Na*Nc=2048`.
- Default **`experiment1` `__main__`**: `Na=16`, `Nc=32`, `N=512` (faster).
- `H ∈ C^(Na×Nc)`, `h = vec(H) ∈ C^(N×1)`
- Noise `n ~ CN(0, sigma2 I)`
- Pilot subcarrier means observing the full `Na` antenna vector on that subcarrier.

## Non-negotiable conventions

### Complex linear algebra
- Use `torch.complex64` (unless explicitly requested otherwise).
- Hermitian transpose: `.mH` (not `.T`).
- Prefer `torch.linalg.solve(A, B)` over explicit matrix inverses.

### vec(H) ordering (CRITICAL)
This repo uses **column-stacking by subcarrier**:

`h = vec(H) = [H[:,0]; H[:,1]; ...; H[:,Nc-1]]`

So subcarrier `k` corresponds to indices `s = k*Na .. (k+1)*Na-1` in `h`.

If you change this ordering, you must update pilot indexing and Kronecker ordering.

### Prior covariance (Kronecker)
- `R_space[i,j] = rho_space^|i-j|`
- `R_freq[k,m] = rho_freq^|k-m|`
- `Sigma = kron(R_freq, R_space)`
- Add small regularization before Cholesky: `Sigma += 1e-9 * I`

### Pilot-as-subcarrier selection matrix
Use `pilot_matrix_from_indices(N, idx)` where `idx` contains the `vec(H)` indices for the chosen subcarrier(s).
For a single pilot subcarrier, `X_t` has shape `(Na, N)`.

### Active sampling score
For candidate subcarrier `k` with `s=k*Na`:
- `P_k = P[s:s+Na, s:s+Na]`
- `Q_k = P[s:s+Na, :] @ P[:, s:s+Na]`  (k-th diagonal block of `P^2`)
- Score: `J(k) = tr( solve(sigma2*I + P_k, Q_k) )` (compare `J(k).real`); default `ActivePilotSampler(..., score_fn=active_subcarrier_score_J)`. Alternatives such as `active_subcarrier_score_block_variance` plug in via `score_fn`.

### Time indexing / T meaning
- `T = ceil((kf-k0)/dk)` counts **new observation steps** (new pilots added).
- MSE curve is length `T+1` with:
  - `t=0`: prior-only estimate (`h_hat=0`)
  - `t=1..T`: after each measurement update

## experiment1 (Gaussian vs Sionna MSE)

**Entry:** `experiment1(cfg)` in `main.py`; current `__main__` calls it (not `main()`).

**Goal:** Compare mean sequential LMMSE MSE curves for three **static** channel truths — analytic Gaussian, Sionna **TDL-C**, Sionna **CDL-C** — under fixed vs active pilot growth. Two figures differ only in which `Sigma` the estimator/active scorer uses.

**Outputs (under `figures/`):**
- `experiment1_mse_kronecker.png` — **all six curves** use fixed Kronecker `Sigma_kron = kron(R_freq, R_space)` from `rho_space` / `rho_freq` (+ `1e-9 I`). Stresses prior mismatch on Sionna truths.
- `experiment1_mse_empirical.png` — **TDL** uses `Sigma_hat_TDL`, **CDL** uses `Sigma_hat_CDL` (from `n_cov_mc` Sionna-only draws via `empirical_covariance`); **Gaussian** still uses `Sigma_kron` (matched reference).
- `experiment1_mse_validation_gaussian.png` — per-step empirical vs `(1/N)tr(P_t)` for **fixed** and **active** pilots with Kronecker `Sigma` (two subplots).
- `experiment1_mse_validation_tdl.png` / `experiment1_mse_validation_cdl.png` — same overlay for **Kronecker** and **empirical `Sigma_hat`** priors, each with fixed and active pilots (four subplots).

**Posterior MSE:** after each sequential LMMSE update, `posterior_mse(P_t) = tr(P_t).real / N` from `recursive_lmmse_*` state `P_t`. Under a matched zero-mean Gaussian prior this is the Bayesian MSE; `P_t` depends only on pilot indices and `Sigma`, not on `h_true` or noise.

**Six curves per MSE figure:** channel family × pilot policy — Gaussian / TDL-C / CDL-C × fixed / active. Color = family; solid = fixed, dashed = active. Validation figures use **one PNG per channel family** with subplots for empirical vs dotted `tr(P)/N` (Gaussian: two panels; TDL/CDL: four panels for Kronecker vs empirical `Sigma` × fixed vs active). **No** `_LiveHeatmap` / `on_step` callbacks.

**Default `__main__` knobs:** `Na=16`, `Nc=32`, `initial_pilot_subcarriers=2`, `final_pilot_subcarriers=8`, `pilots_added_per_step=1`, cumulative → `T=6`, `n_mc=50`, `n_cov_mc=300`, `device="cuda"` (fails if CUDA unavailable; no silent CPU fallback).

**Monte Carlo budget:**
- `n_mc`: one `h_true` per family per trial; fixed and active reuse the same channel; Fig.1 and Fig.2 reuse the same test draws (only `Sigma` changes). Reseed noise per curve: fixed `seed+mc`, active `seed+10_000+mc` for both figures.
- `n_cov_mc`: separate TDL/CDL pools only for `Sigma_hat_*`; not averaged into plotted MSE.

**Channel generation:**
- **Gaussian:** `H = L_space @ Z @ L_freq.T`, `h = vec(H)` (column-stacked by subcarrier).
- **TDL-C / CDL-C:** `sionna_channels.py` — `fft_size=Nc`, `15` kHz spacing, `delay_spread=100` ns, static CIR (`num_time_steps=1`, `min_speed=max_speed=0`); per-realization power normalize `H` to unit mean `|H_{i,k}|^2`; `vec_from_H(H)`.
- **TDL:** `rx_corr_mat=exp_corr_mat(rho_space, Na)`; **CDL:** uplink, `1×Na` BS ULA, `polarization="single"`, `polarization_type="V"` on UT/BS `AntennaArray` (required by Sionna 2.x).

**Seeds (disjoint blocks):** cov TDL `seed+1_000_000+k`, cov CDL `seed+2_000_000+k`; test TDL `seed+100_000+mc`, CDL `seed+200_000+mc`.

**Sanity:** `mc==0` only — recursive vs batch LMMSE on Gaussian + fixed + `Sigma_kron`.

**Dependency:** `sionna` (PyTorch API). Estimator stack stays PyTorch; match `torch` CUDA build to `cfg.device` when using GPU.

