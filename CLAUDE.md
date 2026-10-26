# model_based_gaussian

Model-based pilot allocation for OFDM channel estimation (PyTorch, complex Gaussian priors, LMMSE, active vs fixed pilot selection).

- `new/`: current incremental simulator (one new subcarrier per step). Core loop: `new/simulation.py`; experiments `new/experiments/exp*.py`.
- `old/`: legacy cumulative code, kept for reference only. Do not import it from `new/`.

## Skills (read before editing code)

| Area | Skill |
|------|-------|
| `new/`: simulation loop, channel generators, LMMSE, error estimators, pilot policies, exp1–exp9, seeds, vec(H) convention | `pilot-selection-simulator` |
| `new/error_estimators/cnn/`: supervised CNN error estimator, `train.py` phases | `cnn-error-estimator` |
| `old/`: legacy cumulative Gaussian code | `model-based-gaussian` |

Skills live in `.claude/skills/` only (Cursor is no longer used; `.cursor/skills/` was removed). The CNN design plan is at `.cursor/plans/cnn_supervised_training_8ef4d544.plan.md`.

## Environment

- Local: `conda activate gaussian_san_check`
- Cluster: `conda activate model_based_pilot_allocation` (CNN build/train/eval phases run here)
- Run scripts from the repo root, e.g. `python new/experiments/exp1.py`. Figures go to `new/figures/`.
- Default device is `cuda`. It fails if CUDA is unavailable; there is no silent CPU fallback.
