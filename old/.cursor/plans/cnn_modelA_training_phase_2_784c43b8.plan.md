---
name: Phase 2 HP sweep
overview: "Completed staged hyperparameter sweep (A–E) and closed-loop TDL-A inference (fixed/active/CNN E0/CNN D2b, n_mc=50). E0 and D2b nearly identical on MSE curves; both beat fixed and approach active. Stage F skipped."
todos:
  - id: phase1-cache
    content: Phase 1 cache + baseline (val Huber 0.0434, top-1 0.248)
    status: completed
  - id: parametrize-model
    content: PilotScorerModelA(width, depth); model_arch in checkpoint
    status: completed
  - id: train-sweep-hooks
    content: train_loop, load_cached_datasets, max_train_label_sc, SweepTrainConfig, CLI
    status: completed
  - id: sweep-runner
    content: sweep + sweep-pick subcommands, stage_*.jsonl, sweep.ssh, results.csv
    status: completed
  - id: stage-a-e
    content: Stages A–E executed on cluster (retries for GPU + E0/E2 huber fix)
    status: completed
  - id: inference
    content: Closed-loop eval E0 vs D2/D2b vs fixed/active (inference_cnn_pilot_allocator.py)
    status: completed
  - id: snr-train-benchmark
    content: Train at multiple sigma2/SNR settings; closed-loop benchmark vs fixed/active per SNR
    status: pending
isProject: false
---

# Phase 2 — CNN pilot scorer hyperparameter sweep (completed)

## Status summary

| Item | State |
|------|--------|
| Phase 1 cache | Done — `data/cnn_pilot_scorer/{train,val}.pt` |
| Code (sweep infra) | Done — [`train_cnn_pilot_allocator.py`](train_cnn_pilot_allocator.py) |
| Stages A–E | Done — see [Full results](#full-results-resultscsv) |
| Stage F (lr refinement) | **Skipped** — `stage_f.jsonl` exists but not run; E0 already best val Huber |
| Closed-loop inference | **Done** — `figures/inference/models_comparison_after_sweep.png` + `.json` (`n_mc=50`) |
| Final model | **E0 or D2b** — closed-loop curves almost overlapping; promote either (see [Inference results](#closed-loop-inference-results-tdl-a-n_mc50)) |
| Phase 1 baseline | `best_val_huber=0.0434`, `best_val_top1≈0.248` @ epoch 14 |

**Selection rule used:** lowest **best val Huber** per stage for `winners.yaml`; tie-break **val top-1**. For deploy, also compare **closed-loop TDL-A MSE** (val top-1 can disagree with Huber — see Stage C/D).

---

## Prerequisite — Phase 1

- Job 17634546 on `dt-2080-14`: built cache (~70 min train + ~10 min val gen), trained default 64×3.
- Baseline: `checkpoints/model_a_phase1_best.pt`, metrics in `checkpoints/model_a_phase1_metrics.json`.
- CNN already beat fixed/active on TDL-A in informal eval before sweep.

**Caches (fixed for entire sweep):**

- `data/cnn_pilot_scorer/train.pt` — 72k snapshots, `(7, 32)` features  
- `data/cnn_pilot_scorer/val.pt` — 12k snapshots  

Sweep uses **cache only** — no `--force-regen`. Dataset-defining meta is checked via `CACHE_DATASET_META_KEYS` in code (excludes training-only fields like `huber_delta`).

---

## What was implemented (matches plan intent)

All items from the original “What to implement” section are in [`train_cnn_pilot_allocator.py`](train_cnn_pilot_allocator.py):

| Planned | Implemented |
|---------|-------------|
| `PilotScorerModelA(width, depth)` + `GroupNorm` groups | Yes — `group_norm_groups(width)`; depth ≥ 2, kernels `[5,5]` + `(depth-2)×3` |
| `train_loop` shared by Phase 1 and sweep | Yes |
| `load_cached_datasets()` | Yes — fails fast if cache missing / meta mismatch |
| `SweepTrainConfig` — max 35 epochs, min 5, patience 5 | Yes |
| `max_train_label_sc` — train-only mask subsampling | Yes — `subsample_train_label_mask()` |
| `sweep` subcommand — JSONL + `--index` | Yes — `run_sweep()` |
| `sweep-pick` — winner + optional `regenerate_stage_jsonl` | Yes — stages B, C, D, E presets |
| `results.csv` append per run | Yes — `checkpoints/sweep/results.csv` |
| Per-run `checkpoints/sweep/{run_id}/best.pt` + `metrics.json` | Yes |
| Checkpoint payload: `model_arch`, `cfg` | Yes — `load_checkpoint()` restores width/depth |
| CLI: `--weight-decay`, `--width`, `--depth`, etc. | Yes on `train`; sweep via JSONL |

**Repo layout (as run):**

```text
checkpoints/sweep/
  stage_a.jsonl … stage_e.jsonl   # stage_f.jsonl optional, not run
  winners.yaml
  results.csv
  {run_id}/best.pt, metrics.json
sweep.ssh                         # SLURM array driver
checkpoints/sweep/README.md       # submit cheat sheet
inference_cnn_pilot_allocator.py  # closed-loop eval (uses .pt only)
```

**Inference needs only `best.pt`** — not `metrics.json`. `load_checkpoint()` reads weights + `cfg` + `model_arch` from the `.pt` file.

---

## Deviations from original plan

### 1. Stage D expanded (4 runs, not 2)

**Plan:** Fix architecture from Stage C; sweep only `max_train_label_sc` ∈ {all, 16}.

**Actual:** After Stage C, **C8** (128×4) had best val **top-1 (0.256)** while **C4** (64×3) had best val **Huber (0.0430)**. Stage D was expanded to **2×2 grid**:

| run_id | width×depth | max_train_label_sc |
|--------|-------------|---------------------|
| D0 | 64×3 | null (all) |
| D1 | 64×3 | 16 |
| D2 / D2b | 128×4 | null |
| D3 | 128×4 | 16 |

`stage_d.jsonl` today: D0, D1, D2b, D3 (D2 folder from earlier run).

### 2. Duplicate `run_id` incident (important)

Early Stage D submit used **wrong `run_id`s** (reused D0/D1 for 128×4 lines). Effects:

- Best **top-1 ≈ 0.262** run (job 17658304 task 2) logged as **`run_id=D2` in CSV** but saved under **`checkpoints/sweep/D0/best.pt`** with log prefix `sweep D0`.
- That checkpoint was **overwritten** when the real **64×3 D0** run completed later.
- **`checkpoints/sweep/D2/best.pt`** is a **different** training (val top-1 **≈ 0.252**, epoch 6 per `D2/metrics.json`).
- **`results.csv` `checkpoint_path` for row D2** still says `D0/best.pt` — **do not trust CSV paths**; trust `{run_id}/metrics.json` next to the `.pt` you load.

**Recovery:** `D2b` rerun with clean `run_id` → `checkpoints/sweep/D2b/best.pt` (top-1 ≈ 0.252).

### 3. Stage E + `huber_delta` cache bug (fixed)

**Plan:** Sweep `huber_delta` without regen.

**Issue:** `meta_matches` initially included `huber_delta` → E0/E2 failed at cache load; E1 (`huber=1.0`) worked.

**Fix:** `CACHE_DATASET_META_KEYS` omits `huber_delta` (training-only). E0/E2 re-run after fix.

### 4. GPU / SLURM (Pascal vs Turing)

**Issue:** Array tasks on `cs-1080-*` / mixed `ise-pheno-*` got GTX 1080 / 1080 Ti (sm_61); PyTorch build requires sm_75+.

**Mitigations in [`sweep.ssh`](sweep.ssh):**

- `#SBATCH --constraint=rtx_2080|rtx_3090|rtx_4090|rtx_6000|rtx_pro_6000|l40s`
- `#SBATCH --gres=gpu:rtx_3090:1` (avoid generic `--gpus=1` on mixed nodes)
- `set -e` + preflight Python GPU CC check (exit before training if Pascal)

Stage A indices 0–5 failed once; retry with constraint succeeded.

### 5. Stage F not run

`stage_f.jsonl` still has old 64×3 / `batch=128` template. Skipped after E: **E0** best val Huber **0.0419** already beats Phase 1; narrow lr grid unlikely to beat closed-loop eval need.

### 6. `sweep-pick` on login node

Requires conda env (`torch`). Often updated `winners.yaml` / jsonl **manually** from `results.csv` instead of `python train_cnn_pilot_allocator.py sweep-pick`.

---

## Staged winners (by val Huber per stage)

Values from [`checkpoints/sweep/results.csv`](checkpoints/sweep/results.csv). **Bold** = stage winner (lowest Huber in that stage).

### Stage A — optimizer (9 runs) — `sbatch --array=0-8`

Fixed: `batch=128`, 64×3, all labels, `huber=1.0`.

| run_id | lr | wd | best val Huber | val top-1 | checkpoint |
|--------|-----|-----|--------------|-----------|------------|
| A6 | 3e-4 | 1e-3 | **0.04279** | 0.243 | `A6/best.pt` |
| A1 | 1e-3 | 0 | 0.04319 | **0.250** | `A1/best.pt` |
| A0 | 3e-4 | 0 | 0.04373 | 0.223 | `A0/best.pt` |
| A3 | 3e-4 | 1e-4 | 0.04352 | 0.232 | `A3/best.pt` |
| A5 | 3e-3 | 1e-4 | 0.04355 | 0.242 | `A5/best.pt` |
| A8 | 3e-3 | 1e-3 | 0.04363 | 0.240 | `A8/best.pt` |
| A2 | 3e-3 | 0 | 0.04407 | 0.245 | `A2/best.pt` |
| A7 | 1e-3 | 1e-3 | 0.04438 | 0.241 | `A7/best.pt` |
| A4 | 1e-3 | 1e-4 | 0.04419 | 0.182 | `A4/best.pt` |

**Winner:** **A6** → `lr*=3e-4`, `wd*=1e-3`.

### Stage B — batch size (3 runs) — `sbatch --array=0-2`

Fixed: A6 knobs, 64×3.

| run_id | batch | best val Huber | val top-1 | checkpoint |
|--------|-------|--------------|-----------|------------|
| B0 | 64 | **0.04304** | 0.241 | `B0/best.pt` |
| B1 | 128 | 0.04320 | 0.217 | `B1/best.pt` |
| B2 | 256 | 0.04527 | 0.210 | `B2/best.pt` |

**Winner:** **B0** → `batch*=64`.

### Stage C — architecture (9 runs) — `sbatch --array=0-8`

Fixed: `lr=3e-4`, `wd=1e-3`, `batch=64`.

| run_id | width | depth | best val Huber | val top-1 | checkpoint |
|--------|-------|-------|--------------|-----------|------------|
| C4 | 64 | 3 | **0.04300** | 0.243 | `C4/best.pt` |
| C5 | 64 | 4 | 0.04307 | 0.246 | `C5/best.pt` |
| C8 | 128 | 4 | 0.04319 | **0.256** | `C8/best.pt` |
| C7 | 128 | 3 | 0.04314 | 0.203 | `C7/best.pt` |
| C2 | 32 | 4 | 0.04334 | 0.243 | `C2/best.pt` |
| C1 | 32 | 3 | 0.04372 | 0.244 | `C1/best.pt` |
| C3 | 64 | 2 | 0.04373 | 0.245 | `C3/best.pt` |
| C0 | 32 | 2 | 0.04413 | 0.239 | `C0/best.pt` |
| C6 | 128 | 2 | 0.04398 | 0.235 | `C6/best.pt` |

**Winner (Huber):** **C4** → 64×3.  
**Best top-1 in stage:** **C8** → 128×4 (drove expanded Stage D).

Tasks C1, C8 failed once on Pascal; re-run with GPU constraint succeeded.

### Stage D — label subsampling + architecture (4 runs) — `sbatch --array=0-3`

Fixed: `lr=3e-4`, `wd=1e-3`, `batch=64`, `huber=1.0`.

| run_id | width×depth | max_train_label_sc | best val Huber | val top-1 | trustworthy checkpoint |
|--------|-------------|--------------------|----------------|-----------|-------------------------|
| D2 (CSV row) | 128×4 | null | 0.04281 | **0.262** | **Lost** (was mis-saved as `D0/`, overwritten) |
| D2 | 128×4 | null | 0.04360 | 0.252 | `D2/best.pt` + `D2/metrics.json` |
| D2b | 128×4 | null | 0.04315 | 0.252 | `D2b/best.pt` |
| D0 | 64×3 | null | 0.04305 | 0.245 | `D0/best.pt` |
| D3 | 128×4 | 16 | 0.04307 | 0.260 | CSV path `D1/best.pt` — **on disk `D1/` is 64×3** (top-1 0.238); D3 weights likely lost |
| D1 | 64×3 | 16 | 0.04370 (CSV) | 0.241 (CSV) | `D1/best.pt` — `metrics.json`: Huber 0.0431, top-1 0.238 |

**For inference (128×4, all labels, huber=1.0):** use **`D2b/best.pt`** or **`D2/best.pt`**, not the 0.262 CSV row.

### Stage E — Huber delta (3 runs) — `sbatch --array=0-2`

Fixed: 128×4, `batch=64`, all labels; sweep `huber_delta`.

| run_id | huber_delta | best val Huber | val top-1 | checkpoint |
|--------|-------------|--------------|-----------|------------|
| E0 | 0.5 | **0.04188** | 0.243 | `E0/best.pt` |
| E2 | 2.0 | 0.04293 | 0.248 | `E2/best.pt` |
| E1 | 1.0 | 0.04324 | 0.247 | `E1/best.pt` |

**Winner:** **E0** → `huber_delta*=0.5` (best val Huber in entire sweep).

---

## Full results (`results.csv`)

30 data rows (plus header; includes duplicate **D2b** line and mis-pathed **D2**/**D3** rows). File: [`checkpoints/sweep/results.csv`](checkpoints/sweep/results.csv).

**Global best val Huber:** **E0** @ `checkpoints/sweep/E0/best.pt` (0.0419, 128×4, huber 0.5).

**Global best val top-1 (among valid checkpoints):** **D3** row (0.260) or lost D2 CSV row (0.262); on disk **D2/D2b ≈ 0.252**, **C8 ≈ 0.256** on val during Stage C.

**Current [`winners.yaml`](checkpoints/sweep/winners.yaml)** (cumulative after stages):

```yaml
lr: 0.0003
weight_decay: 0.001
batch_size: 64
width: 128
depth: 4
huber_delta: 0.5
max_train_label_sc: null
```

---

## Cluster workflow (as executed)

Single [`sweep.ssh`](sweep.ssh): SLURM array + `STAGE_CONFIG` per stage.

```bash
cd /home/pardot/model_based_pilot_allocation
sbatch --array=0-8 --job-name=sweep_stage_X \
  --export=ALL,STAGE_CONFIG=checkpoints/sweep/stage_X.jsonl \
  sweep.ssh
```

| Stage | array | jsonl lines | Notes |
|-------|-------|-------------|--------|
| A | 0-8 | 9 | Retry 0-5 after Pascal failures |
| B | 0-2 | 3 | |
| C | 0-8 | 9 | Retry 1,8 if needed |
| D | 0-3 | 4 | Expanded grid; JSON typo once on index 2 |
| E | 0-2 | 3 | Retry 0,2 after huber cache fix |
| F | — | 6 | Not run |

Logs: `logs/sweep-<jobid>_<taskid>.out` (and `logs/stage_*` copies if kept).

**Between stages:** update next `stage_*.jsonl` with winners (manual or `sweep-pick` with conda).

---

## Hyperparameter inventory (reference)

### Fixed (in cache / problem definition)

`n_antennas=16`, `n_subcarriers=32`, `sigma2=1e-2`, pilots 2→8 cumulative, `rho_space=0.7`, `n_cov_mc=300`, `seed=0`, 50/50 random/active rollouts, 7 feature channels, AdamW, 12k/2k val channels.

### Swept (stages A–E)

| Parameter | Grid | Stage |
|-----------|------|-------|
| `lr` | 3e-4, 1e-3, 3e-3 | A |
| `weight_decay` | 0, 1e-4, 1e-3 | A |
| `batch_size` | 64, 128, 256 | B |
| `width` × `depth` | 32/64/128 × 2/3/4 | C |
| `max_train_label_sc` | null, 16 | D |
| `huber_delta` | 0.5, 1.0, 2.0 | E |

**Sweep training defaults:** `max_epochs=35`, `min_epochs=5`, `early_stop_patience=5`. Early stop on **val Huber**; log **val top-1** each epoch.

---

## Closed-loop inference results (TDL-A, n_mc=50)

**Script:** [`inference_cnn_pilot_allocator.py`](inference_cnn_pilot_allocator.py) — default sweep comparison mode.  
**Checkpoints:** `checkpoints/E0_best.pt` (huber δ=0.5, best val Huber in sweep) vs `checkpoints/D2b_best.pt` (128×4, all labels, huber δ=1.0).  
**Artifacts:** [`figures/inference/models_comparison_after_sweep.png`](figures/inference/models_comparison_after_sweep.png), [`figures/inference/models_comparison_after_sweep.json`](figures/inference/models_comparison_after_sweep.json).

**Setup:** Same `Sigma_hat` (TDL-A, `n_cov_mc=300`), same pilot schedule; per MC trial one `h_true`, then fixed / active / CNN E0 / CNN D2b with disjoint noise seeds (`EVAL_CHANNEL_SEED_OFFSET=300_000`).

### Offline training vs closed-loop (final time step t=6)

| Policy / model | best val Huber (train) | val top-1 (train) | mean final-step MSE (inference) |
|----------------|------------------------|-------------------|----------------------------------|
| — | — | — | fixed **0.00311** |
| — | — | — | active **0.00254** |
| **E0** | **0.0419** | 0.243 | **0.001602** |
| **D2b** | 0.0432 | **0.252** | **0.001586** |

Relative gap **E0 vs D2b** at final step: **~1.0%** (0.001602 vs 0.001586) — **negligible vs MC std** (±0.00048 / ±0.00047 on last step). Curves overlap for **t ≥ 1**; both CNNs sit **below fixed** and **between fixed and active** (active still best).

### Conclusion

- **E0 and D2b are effectively the same pilot policy in closed-loop** despite different offline metrics (E0 wins val Huber; D2b slightly higher val top-1). Architecture is the same 128×4; main training difference is `huber_delta` (0.5 vs 1.0).
- **Sweep validated 128×4 CNN** over Phase 1 64×3: both sweep models beat fixed clearly and approach active.
- **Promotion:** Either `E0_best.pt` (offline winner) or `D2b_best.pt` (marginally lower final MSE, within noise) → copy to `checkpoints/model_a_best.pt`. No need to re-run A–E or Stage F for this decision.

### MSE mean curve (all steps, from JSON)

| t | fixed | active | cnn E0 | cnn D2b |
|---|-------|--------|--------|---------|
| 0 | 0.0208 | 0.0211 | 0.0199 | 0.0202 |
| 1 | 0.0145 | 0.00564 | 0.00584 | 0.00594 |
| 2 | 0.0124 | 0.00423 | 0.00392 | 0.00396 |
| 3 | 0.0113 | 0.00346 | 0.00292 | 0.00294 |
| 4 | 0.00872 | 0.00305 | 0.00230 | 0.00231 |
| 5 | 0.00554 | 0.00279 | 0.00188 | 0.00190 |
| 6 | 0.00311 | 0.00254 | 0.00160 | 0.00159 |

---

## Next steps (post-inference)

1. **Promote** `E0_best.pt` or `D2b_best.pt` to `checkpoints/model_a_best.pt` (either justified; prefer **E0** if optimizing for offline val Huber consistency).

2. **Optional:** `n_mc=100` rerun for tighter CIs — unlikely to separate E0 vs D2b given overlap.

3. **Do not** re-run full A–E unless a new training change is made; Stage F remains optional/low value.

4. **Train at different SNR (`sigma2`) and benchmark**
   - Phase 1/2 cache and features assume **`sigma2=1e-2`** (SNR feature `log10(1/σ²)` in the 7-channel input). Each new noise level needs a **matching dataset** (`meta` includes `sigma2`) — regenerate `data/cnn_pilot_scorer/{train,val}.pt` per `sigma2` or add a dedicated cache subfolder (e.g. `data/cnn_pilot_scorer/sigma2_1e-3/`).
   - **Train** promoted architecture (128×4, E0-style knobs: `lr=3e-4`, `wd=1e-3`, `batch=64`, `huber_delta=0.5`) at a small grid of `sigma2` values (e.g. `1e-3`, `1e-2`, `1e-1` or equivalent SNR dB targets). Sweep only `sigma2` first — no full A–E repeat unless one SNR point underperforms.
   - **Benchmark** each checkpoint with [`inference_cnn_pilot_allocator.py`](inference_cnn_pilot_allocator.py): same TDL-A closed-loop setup (`fixed` / `active` / CNN), pass `--sigma2` to match training and eval noise; save figures under `figures/inference/` (e.g. `mse_snr_sigma2_{value}.png` + JSON).
   - **Goal:** see whether the CNN **generalizes across SNR** or needs per-SNR models; compare CNN vs active gap as SNR changes (active also uses `sigma2` in `J(k)` scoring).

---

## Original plan diagrams (workflow)

```mermaid
flowchart LR
  subgraph fixed [Fixed by cache]
    Sigma["Sigma_hat n_cov_mc=300"]
    Rollout["50/50 random/active"]
    Labels["Full counterfactual y_label"]
    Feat["7 channels"]
  end
  subgraph sweep [Swept per stage]
    Opt["lr, wd, batch"]
    Arch["width, depth"]
    TrainOnly["label K, huber_delta"]
  end
  fixed --> TrainLoop
  sweep --> TrainLoop
  TrainLoop --> Metric["val Huber + val top-1"]
```

```mermaid
flowchart TD
  doneA[Stage A done] --> doneB[Stage B done]
  doneB --> doneC[Stage C done]
  doneC --> doneD[Stage D expanded 4 runs]
  doneD --> doneE[Stage E done]
  doneE --> infer[Inference done E0≈D2b]
  infer --> promote[model_a_best.pt E0 or D2b]
  promote --> snr[Train + benchmark other SNR]
```

---

## Summary table (planned vs actual run count)

| Stage | Planned runs | Actual logged | Notes |
|-------|--------------|---------------|--------|
| A | 9 | 9 | |
| B | 3 | 3 | |
| C | 9 | 9 | |
| D | 2 | 4 + D2b rerun | Expanded arch; duplicate run_id issue |
| E | 3 | 3 | |
| F | 0–6 | 0 | Skipped |
| **Total** | **26–32** | **30 CSV rows, 28 run folders** | Duplicate D2b row; D2/D3 path collisions |

**Fixed throughout:** cached TDL-A data, AdamW, val Huber early stop (patience 5, max 35 epochs), 7 input channels. **Trust `{run_id}/best.pt` + sidecar `metrics.json`**, not `results.csv` checkpoint_path when run_ids were duplicated.
