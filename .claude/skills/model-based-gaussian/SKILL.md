---
name: model-based-gaussian
description: >-
  Legacy cumulative Gaussian pilot allocation in old/ only (main.py experiment1,
  pilots.py, estimators.py, CNN training). For the incremental simulator in new/,
  use skill pilot-selection-simulator.
---

# model_based_gaussian (legacy `old/`)

Incremental simulator in `new/` → skill **`pilot-selection-simulator`**.

## Repo scope (`old/`)

- `old/main.py`: `main()` (Gaussian fixed vs active MSE + optional heatmap) and `experiment1()` (Gaussian vs Sionna TDL/CDL).
- `old/pilots.py`: fixed/active pilot samplers + **cumulative** sequential LMMSE driver.
- `old/estimators.py`: recursive + batch LMMSE for complex Gaussian prior.
- `old/data_generator.py`: complex Gaussian, exponential correlation, selection matrices, `empirical_covariance`.
- `old/sionna_channels.py`: static Sionna TDL-C / CDL-C OFDM → `H (Na,Nc)`, `vec(H)`.
- `old/train_cnn_pilot_allocator.py`, `old/inference_cnn_pilot_allocator.py`: cumulative CNN pilot allocator (reference only).

**Run:** `python old/main.py`

## Model & shapes

- Default `main()`: `Na=32`, `Nc=64`, `N=2048`.
- Default `experiment1` `__main__`: `Na=16`, `Nc=32`, `N=512`.
- `H ∈ C^(Na×Nc)`, `h = vec(H) ∈ C^(N×1)`; noise `n ~ CN(0, sigma2 I)`.
- Pilot subcarrier = observe full `Na` antenna vector on that subcarrier.

## Non-negotiable conventions

### Complex linear algebra
- `torch.complex64`; Hermitian `.mH`; prefer `torch.linalg.solve` over inverses.

### vec(H) ordering (CRITICAL)
Column-stack by subcarrier: `h = [H[:,0]; H[:,1]; …; H[:,Nc-1]]`. SC `k` → indices `s = k*Na .. (k+1)*Na-1`.

### Prior covariance (Kronecker)
- `R_space[i,j] = rho_space^|i-j|`, `R_freq[k,m] = rho_freq^|k-m|`
- `Sigma = kron(R_freq, R_space) + 1e-9·I`

### Cumulative measurements
Each step `X_t` includes **all** pilot subcarriers used so far (`pilot_matrix_from_indices`); fresh noise on full stack. Driver: `sequential_lmmse_mse_curve` in `pilots.py`. Init: `h=0`, `P=Sigma`, then `recursive_lmmse_*` updates.

### Pilot-as-subcarrier selection
`pilot_matrix_from_indices(N, idx)`; single SC → `X_t (Na, N)`.

### Active sampling score
For candidate SC `k`, `s=k*Na`: `P_k`, `Q_k` as k-th block of `P` and `P²`; `J(k)=tr(solve(σ²I+P_k, Q_k)).real`. `ActivePilotSampler(..., score_fn=active_subcarrier_score_J)`.

### Time indexing
- `T = ceil((kf-k0)/dk)` = new observation steps.
- MSE curve length `T+1`: `t=0` prior-only (`h_hat=0`); `t=1..T` after each update.

## experiment1 (Gaussian vs Sionna MSE)

**Entry:** `experiment1(cfg)` in `old/main.py`; `__main__` calls it.

**Goal:** Mean sequential LMMSE MSE for Gaussian, Sionna **TDL-C**, **CDL-C** under fixed vs active pilot growth.

**Outputs (`old/figures/`):**
- `experiment1_mse_kronecker.png` — six curves, Kronecker `Sigma` for all
- `experiment1_mse_empirical.png` — empirical `Sigma_hat` for TDL/CDL
- `experiment1_mse_validation_gaussian.png` — true vs `tr(P)/N` (fixed/active)
- `experiment1_mse_validation_tdl.png`, `experiment1_mse_validation_cdl.png` — Kronecker vs empirical panels

**Six curves per MSE figure:** Gaussian / TDL-C / CDL-C × fixed / active. Color = family; solid = fixed, dashed = active.

**Defaults:** `Na=16`, `Nc=32`, `initial_pilot_subcarriers=2`, `final_pilot_subcarriers=8`, `pilots_added_per_step=1`, `T=6`, `n_mc=50`, `n_cov_mc=300`, `device="cuda"` (fail if unavailable).

**MC:** one `h_true` per family per trial; fixed/active share channel. Noise: fixed `seed+mc`, active `seed+10_000+mc`.

**Channels:**
- Gaussian: `H = L_space @ Z @ L_freq.T`, `h = vec(H)`.
- TDL-C / CDL-C: `sionna_channels.py` — `fft_size=Nc`, 15 kHz, 100 ns delay, static CIR, unit mean `|H|²`; `vec_from_H(H)`.
- TDL: `rx_corr_mat=exp_corr_mat(rho_space, Na)`; CDL: uplink ULA, single-V polarization (Sionna 2.x).

**Seeds:** cov TDL `seed+1_000_000+k`, CDL `seed+2_000_000+k`; test TDL `seed+100_000+mc`, CDL `seed+200_000+mc`.

**Sanity:** `mc==0` — recursive vs batch LMMSE on Gaussian + fixed + `Sigma_kron`.

**Dependency:** `sionna` (PyTorch API).

## Environment

```bash
conda activate gaussian_san_check
python old/main.py
```

See `old/environment.yml`.
