---
name: DNN pilot scorer
overview: Model A in train_cnn_pilot_sampler.py — 1D CNN predicts counterfactual log-MSE per SC from H_hat/mask (+ optional innovation); TDL-A only; Sigma_hat from TDL-A pool; CNNPilotSampler for deploy. Phase 0 sanity_check done; Phase 1 full 12k/2k cached dataset + CUDA train; Phase 2 hyperparameter sweep later.
todos:
  - id: train-module
    content: "train_cnn_pilot_sampler.py: Model A, features, labels, dataset gen, CNNPilotSampler, sanity_check()"
    status: completed
  - id: sanity-check
    content: sanity_check() — few epochs on one minibatch; assert loss decreases and grads flow
    status: completed
  - id: phase1-dataset
    content: Phase 1 — generate/cache train+val tensors (12k/2k channels, 72k/12k snapshots) under data/cnn_pilot_scorer/
    status: pending
  - id: phase1-train
    content: Phase 1 — train() on CUDA, epoch logging, early stop, best checkpoint; val Huber must trend down
    status: pending
  - id: full-dataset-train
    content: Alias for phase1-dataset + phase1-train (single local CUDA session)
    status: pending
  - id: inference-stub
    content: inference_cnn_pilot_sampler.py — module docstring / placeholder only (no benchmarks yet)
    status: pending
  - id: experiment1-hook
    content: Later — wire CNNPilotSampler into experiment1 via inference_cnn_pilot_sampler.py
    status: pending
isProject: false
---

# CNN pilot selection — Model A

## Goal

**Model A** (`PilotScorerModelA` + **`CNNPilotSampler`**): predict counterfactual `e_{t+1}^{(k)}`, pick `k* = argmin` over unused SCs. Train on **Sionna TDL-A only**. LMMSE uses **`Sigma_hat`** from TDL-A `empirical_covariance`. **No `P_t` as CNN input.** **No BPTT.**

## Conventions

- `Na=16`, `Nc=32`, `N=512`; `h = vec(H)` column-stacked by subcarrier.
- `H_hat` shape `(16, 32)` via same reshape as [`_h_to_H`](c:\Users\talpa\Projects\model_based_gaussian\main.py).
- **Channels:** TDL-A static MIMO — `sample_tdl_ofdm_channel(model="A", n_ofdm_symbols=1, n_antennas=Na, rho_space=cfg.rho_space, ...)` ([`sionna_channels.py`](c:\Users\talpa\Projects\model_based_gaussian\sionna_channels.py)); power-normalized per repo.
- **`Sigma_hat`:** `n_cov_mc=300` TDL-A draws only.
- Pilots: `initial=2`, `final=8`, `+1`/step, cumulative → **T=6** decision rows per channel after init.

## Implementation files

| File | Status |
|------|--------|
| [`train_cnn_pilot_sampler.py`](c:\Users\talpa\Projects\model_based_gaussian\train_cnn_pilot_sampler.py) | **All training code** (single module): config, TDL-A sampling, features, counterfactual labels, dataset, `PilotScorerModelA`, `CNNPilotSampler`, `train()`, **`sanity_check()`** |
| [`inference_cnn_pilot_sampler.py`](c:\Users\talpa\Projects\model_based_gaussian\inference_cnn_pilot_sampler.py) | **Stub only** — docstring + `if __name__` placeholder; benchmarks / experiment1 hooks **not implemented** in v1 |
| [`pilots.py`](c:\Users\talpa\Projects\model_based_gaussian\pilots.py) | Unchanged until inference phase; optional later import of `CNNPilotSampler` |

**Do not add** `pilot_scorer.py` / `train_pilot_scorer.py`. Reuse repo: `estimators`, `pilots` (`FixedPilotSampler`, `active_subcarrier_score_J`, `recursive_lmmse_*`), `data_generator`, `sionna_channels`.

## `CNNPilotSampler` (Model A deploy)

- Class name: **`CNNPilotSampler`** (first DNN pilot policy for this project).
- Loads `PilotScorerModelA` checkpoint; builds features + norm as training; `argmin` over unused `k`; `ê[k]=+inf` on piloted SCs.
- Lives in **`train_cnn_pilot_sampler.py`**; `inference_cnn_pilot_sampler.py` will import it when benchmarks are added.

## Model A — `PilotScorerModelA`

| | Shape |
|--|--------|
| Input `X` | `(B, 7, 32)` float32 (v1 impl; innovation channel deferred) |
| Output `ê_log` | `(B, 32)` |

```text
Conv1d(7,64,k=5,pad=2,bias=False) → GroupNorm(8,64) → GELU
Conv1d(64,64,k=5,pad=2,bias=False) → GroupNorm(8,64) → GELU
Conv1d(64,64,k=3,pad=1,bias=False) → GroupNorm(8,64) → GELU
Conv1d(64,32,k=1) → GELU
Conv1d(32,1,k=1) → squeeze → (B,32)
```

### Features (7 × 32) — current code

| Ch | Feature |
|----|---------|
| 0–2 | `mean_a \|H_hat\|`, `Re`, `Im` → **per-snapshot z-score across 32 SCs** (each channel separately) |
| 3 | cumulative pilot `mask[k]` |
| 4–6 | `log10(1/σ²)`, `t/T`, `sum(mask)/Nc` broadcast |

**Deferred (not in v1):** innovation channel `log(1+ρ_k)` (`r = y - X @ h_prev`; `ρ_k = mean_a|r_{a,k}|` on piloted SCs). Sanity check and Phase 1 use **7 channels** only.

### Labels

Per decision snapshot, each **unused** `k`: one `recursive_lmmse_step` with pilot on `k` only →  
`e_{t+1}^{(k)} = (1/N)||ĥ_{t+1}^{(k)} - h_true||²`; store `y_label[k]=log(e+1e-8)`; `loss_mask[k]=1` iff unused.

**Rollout continuation:** per channel, **50% random** / **50% active** (`argmax J(k)`); no fixed training paths. Initial pilots from `FixedPilotSampler` step 0.

## Phase 0 — `sanity_check()` ✅ done

In [`train_cnn_pilot_sampler.py`](c:\Users\talpa\Projects\model_based_gaussian\train_cnn_pilot_sampler.py):

1. One minibatch (`B=12`) from on-the-fly TDL-A rollouts; `Sigma_hat` with reduced `n_cov_mc=64`.
2. **10 epochs** on the **same** minibatch; assert loss decreases, finite, non-zero grads.
3. **Passed** — proceed to Phase 1.

---

## Phase 1 — full dataset + local CUDA train (current)

**Goal:** One end-to-end session on your machine’s CUDA GPU: build the **full** offline set, train `PilotScorerModelA`, and confirm **validation Huber loss trends downward** (not hyperparameter tuning — that is **Phase 2**, later).

### Dataset size and split

| Split | Channels | Snapshots | Notes |
|-------|----------|-----------|--------|
| **Train** | **12,000** | **72,000** | 6 decision rows per channel (`T=6`) |
| **Val** | **2,000** | **12,000** | Same rollout rules as train |
| **Ratio** | **6 : 1** | **6 : 1** | Held-out channels only (no snapshot leakage) |

- **TDL-A only** — no CDL, no TDL-C, no analytic Gaussian channels in the training set.
- **Channel seeds (disjoint):** train channel `k` → `cfg.seed + k`; val channel `k` → `cfg.seed + 500_000 + k` (same convention as original plan).
- **Rollout policy:** per channel, **50% random** / **50% active** (`pick_rollout_subcarriers`); initial pilots from `FixedPilotSampler` step 0; 50/50 by `channel_seed % 2`.
- **Row tensors:** `X (7, 32)` float32, `y_label (32)` log-MSE, `loss_mask (32)` (1 on unused SCs only).
- **Labels:** **all unused** `k` on **both** train and val (no train subsampling in Phase 1 — simpler loss semantics; generation is the bottleneck but run once).
- **`Sigma_hat`:** single estimate at train start, `n_cov_mc=300` TDL-A draws (`estimate_sigma_hat_tdl_a`); **shared** by train and val rollouts (estimator prior only; val channels are unseen).

**Disk cache (required for Phase 1):** generate once, then reuse.

```text
data/cnn_pilot_scorer/
  train.pt   # dict: X (72000,7,32), y_label, loss_mask, meta
  val.pt     # dict: X (12000,7,32), y_label, loss_mask, meta
```

- `meta`: `TrainConfig` fields, `seed`, channel counts, timestamp, `n_cov_mc`.
- **Skip regen** if both files exist and `meta` matches (CLI `--force-regen` to rebuild).
- **Gen logging only:** print every **500** channels + final wall time and snapshot counts (no per-snapshot spam).

**Rough cost:** ~165 counterfactual `recursive_lmmse_step` calls per channel (sum over 6 steps of unused SCs). Dominates wall time vs CNN epochs — expect **hours** on a single GPU for 14k channels; acceptable for a one-time cache.

### Training loop (`train()`)

**Loss** (unchanged):

```text
L_b = mean_{k: loss_mask[k]} Huber(ê_log[k] - y_label[k]; δ=1.0)
loss = mean_b L_b
```

| Knob | Value | Rationale |
|------|--------|-----------|
| Optimizer | AdamW | Same as sanity |
| `lr` | `1e-3` | Default; fixed for Phase 1 |
| `weight_decay` | `1e-4` | Mild regularization |
| `batch_size` | **128** | Increase to 256 if VRAM allows; drop to 64 if OOM |
| `max_epochs` | **40** | Enough to see val trend on 72k rows |
| `min_epochs` | **10** | Don’t early-stop before signal |
| `early_stop_patience` | **8** | Stop if val Huber doesn’t improve for 8 epochs |
| `DataLoader` | `shuffle=True`, `num_workers=0` | Windows-safe; tensors already on disk |
| Device | `cfg.device="cuda"` | **Fail** if CUDA unavailable (match `experiment1`) |
| Checkpoint | `checkpoints/model_a_phase1_best.pt` | Best **val Huber**; also save last epoch optionally |

**Success criterion (Phase 1):**

- Best val Huber **<** epoch-1 val Huber at least once after epoch 5, **and**
- Val Huber at end **<** val Huber at epoch 1 (clear downward trend; need not be strictly monotonic every epoch).
- No NaN/Inf; training finishes or early-stops cleanly.

**Secondary metric (same val pass):** **top-1** — fraction of snapshots where `argmin_k ê_log[k]` over masked `k` equals `argmin_k y_label[k]` (true best SC in log-MSE). Log it each epoch; not used for early stopping in Phase 1.

### Terminal output (minimal)

**Once at start:**

```text
phase1: device=cuda:0  Sigma_hat n_cov_mc=300  train=72000 val=12000  batch=128
cache: data/cnn_pilot_scorer/train.pt (hit/miss)
```

**During dataset build (only if miss):**

```text
gen train: 500/12000 channels  (elapsed …)
gen train: 1000/12000 channels …
…
gen train done: 72000 snapshots in … min
gen val: …
```

**Each epoch — exactly one line:**

```text
epoch 012/040  train_huber=0.421  val_huber=0.398  val_top1=0.31  lr=1e-3  4.1s  *
```

- `*` marks new best val Huber (checkpoint written).
- No per-batch lines, no grad norms, unless `--verbose`.

**Once at end:**

```text
phase1 done: best_val_huber=0.381 @ epoch 23  checkpoint=checkpoints/model_a_phase1_best.pt
```

### CLI / `__main__`

```text
python train_cnn_pilot_sampler.py sanity          # Phase 0 (already passed)
python train_cnn_pilot_sampler.py train           # Phase 1: cache + train
python train_cnn_pilot_sampler.py train --force-regen
python train_cnn_pilot_sampler.py train --epochs 40 --batch-size 128
```

Wire `main()` to subcommands; default remains `sanity` until Phase 1 code lands, then document switching default to `train` after first successful Phase 1.

### Implementation checklist (code not written yet)

1. `build_dataset(split, n_channels, …)` → stacked tensors + `torch.save`.
2. `PilotSnapshotDataset` + `DataLoader`.
3. `train(cfg, …)` — epoch loop, `masked_huber_loss`, val top-1, early stop, checkpoint.
4. `load_checkpoint` helper for later `CNNPilotSampler` inference.

---

## Phase 2 — hyperparameter sweep (future, not now)

- Learning rate, weight decay, batch size, optional train label subsampling (16 unused `k`), architecture width/depth.
- Use **cached** `train.pt` / `val.pt` where possible so sweeps don’t repeat 14k-channel generation.
- Out of scope until Phase 1 val loss decrease is confirmed.

---

## Training (reference)

```text
L_b = mean_{k: loss_mask[k]} Huber(ê_log[k] - y_label[k]; δ=1.0)
loss = mean_b L_b
```

- Offline dataset; shuffled snapshots; **no BPTT**; AdamW `lr=1e-3`; early stop on **val Huber**; log top-1 for monitoring.

## Evaluation (later — `inference_cnn_pilot_sampler.py`)

- Closed-loop MSE vs fixed/active on TDL-A; empirical `Sigma_hat`.
- experiment1-style curves when implemented in inference module.

## Out of scope (v1)

- CDL / TDL-C / Gaussian training data; `P_t` CNN inputs; `DNNPilotSampler` name; separate `pilot_scorer.py`; inference benchmarks; BPTT / DAgger; fixed training rollouts; scalar head; ranking loss.
