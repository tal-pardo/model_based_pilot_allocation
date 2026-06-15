---
name: model-based-gaussian
description: Conventions for model_based_gaussian. Legacy code in old/; new incremental LMMSE loop in new/. Use when editing new/ simulation, pilots, or channel generators.
---

# model_based_gaussian

## Layout

- **`old/`** — previous project (cumulative measurements, experiment1, CNN training). Run: `python old/main.py`
- **`new/`** — reorganized incremental loop. Run: `python new/experiments/exp1.py`

## new/ structure

- `config.py` — SimConfig (`reg_kron`, `reg_empirical`), Exp1/2/3Config
- `utils.py` — device, noise, covariance, selection, `measure_subcarrier` (Na×1)
- `simulation.py` — `run_until_threshold` (canonical LMMSE loop)
- `channel_generators/` — gaussian.py, sionna.py (same conventions as old/data_generator + old/sionna_channels)
- `channel_estimators/lmmse.py` — information-form init + (I-KX) updates
- `error_estimators/trace_min.py` — tr(P)/N stopping + active J(k)
- `pilot_policy/` — fixed, random, active
- `experiments/` — exp1.py, exp2_gaussian_vs_tdl.py, exp3_true_vs_estimated_err.py

## Incremental measurements

Each step: `X_t` (Na, N), `y_t` (Na, 1), one subcarrier, fresh noise.

## vec(H)

Column-stack by subcarrier: `[H[:,0]; …; H[:,Nc-1]]`, `Sigma = kron(R_freq, R_space) + reg_kron·I`.

## Prior regularization

- **`reg_kron`** (default `1e-9`): ridge on analytic Kronecker prior (`build_sigma_kron`). Full-rank; numerical stability only.
- **`reg_empirical`** (default `1e-3`): ridge on sample covariance (`empirical_covariance`). Use larger value when `n_cov_mc < N` to avoid null-space overconfidence.

CLI: `--reg-kron`, `--reg-empirical` on exp1–exp3.

## Defaults (exp1)

Na=16, Nc=32, k0=2, max_pilots=32, nmse_threshold=0.02, n_mc=50, device=cuda (fail if unavailable).

## Environment

Use conda env `gaussian_san_check` (see `old/environment.yml`). Example:

```bash
conda activate gaussian_san_check
python new/experiments/exp1.py
```
