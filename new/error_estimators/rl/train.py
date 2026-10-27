"""
RL (REINFORCE / PPO) alternative to the CNN active-greedy (J(k)) estimator -- time_freq domain.

Same decision as error_estimators/cnn/train.py's CNN: which unused subcarrier to pilot next.
Instead of supervised regression against a brute-force counterfactual oracle label, the policy
is trained on-policy against the realized NMSE after the pilot it actually chose. Reuses the
CNN module's CNNErrorEstimator backbone, feature/channel/Sigma-bucket machinery, and
model_arch_kwargs/feature_grid_shape/zscore_mode_for/mask_to_grid/build_features directly --
only the training objective and rollout differ.

Sections:
  1. sys.path -> repo new/
  2. RlTrainConfig
  3. ValueNet (PPO critic; separate small conv net, does not share the policy's trunk)
  4. rollout_episode (on-policy, autoregressive over the agent's own choices)
  5. REINFORCE update (Stage A: per-step bandit, per-bucket batch-mean baseline)
  6. PPO update (Stage B: full-episode return-to-go, per-bucket-normalized advantage)
  7. Checkpoint save/load, RlPilotPolicy (mirrors CnnPilotPolicy)
  8. rl_real_eval_score (checkpoint-selection proxy, mirrors real_eval_cnn_score)
  9. train_reinforce / train_ppo
  10. argparse / main
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. sys.path -> repo new/
# ---------------------------------------------------------------------------
_RL_DIR = Path(__file__).resolve().parent
_NEW_ROOT = _RL_DIR.parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from channel_estimators.lmmse import EstimatorState, lmmse_incremental_update, lmmse_initial_update
from error_estimators.cnn.train import (
    CnnTrainConfig,
    CNNErrorEstimator,
    EVAL_CHANNEL_OFFSET,
    _check_domain_config,
    bucket_for_channel,
    build_features,
    feature_grid_shape,
    load_sigma_buckets,
    make_grid,
    mask_to_grid,
    model_arch_kwargs,
    random_first_subcarrier,
    sample_tdl_channel,
    tdl_buckets,
    tdl_channel_seed,
    zscore_mode_for,
)
from utils import empirical_nmse, measure_subcarrier, resolve_device, set_seed

CHECKPOINT_DIR = _RL_DIR / "checkpoints"
FIGURES_DIR = _RL_DIR / "figures"
ROLLOUT_NOISE_OFFSET = 7_000_000  # disjoint from every offset used inside cnn/train.py


# ---------------------------------------------------------------------------
# 2. RlTrainConfig
# ---------------------------------------------------------------------------
@dataclass
class RlTrainConfig(CnnTrainConfig):
    stage: Literal["reinforce", "ppo"] = "reinforce"
    episodes_per_update: int = 36
    n_updates: int = 2_000
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    clip_eps: float = 0.2
    ppo_epochs: int = 4
    eval_interval: int = 20
    grad_clip_norm: float = 1.0
    init_checkpoint: Optional[str] = None  # PPO: warm-start from a Stage A (reinforce) checkpoint


def rl_label_suffix(cfg: RlTrainConfig) -> str:
    suffix = "_tf" if cfg.domain == "time_freq" else ""
    return f"{suffix}_{cfg.stage}"


# ---------------------------------------------------------------------------
# 3. ValueNet (PPO critic)
# ---------------------------------------------------------------------------
class ValueNet(nn.Module):
    """Small independent conv net -> pooled scalar V(s). Does not share weights with the
    policy's CNNErrorEstimator trunk -- kept separate for simplicity/robustness rather than
    reaching into CNNErrorEstimator's internals to share a trunk."""

    def __init__(self, kernel_hw: Tuple[int, int] = (5, 3)) -> None:
        super().__init__()
        kh, kw = kernel_hw
        ph, pw = kh // 2, kw // 2
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=(kh, kw), padding=(ph, pw), bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B,6,H,W). Output: (B,) scalar value estimates."""
        h = self.conv(x)
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 4. rollout_episode
# ---------------------------------------------------------------------------
def rollout_episode(
    cfg: RlTrainConfig,
    model: CNNErrorEstimator,
    *,
    h_true: torch.Tensor,
    sigma: torch.Tensor,
    device: torch.device,
    channel_mc: int,
    noise_offset: int = ROLLOUT_NOISE_OFFSET,
    deterministic: bool = False,
) -> Dict[str, Any]:
    """Run one full episode (k0=1 random bootstrap + max_pilots-1 agent-chosen pilots).

    deterministic=False (training): sample each action from the masked softmax, record
    log_prob/entropy/feat/mask/action for the gradient update.
    deterministic=True (eval): argmax each step, no grad, only final_nmse/rewards recorded.
    """
    na, nc = cfg.n_antennas, cfg.n_subcarriers
    n = cfg.n_total
    grid_na, grid_nc = feature_grid_shape(cfg)
    zscore_mode = zscore_mode_for(cfg)
    channel_seed = tdl_channel_seed(cfg, channel_mc)
    k_init = random_first_subcarrier(channel_seed, nc)
    noise_seed = cfg.seed + noise_offset + channel_mc
    gen = torch.Generator(device=device).manual_seed(noise_seed)

    x0, y0 = measure_subcarrier(
        k_init, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
    )
    state = lmmse_initial_update(sigma, x0, y0, cfg.sigma2)
    mask = torch.zeros(nc, device=device, dtype=torch.float32)
    mask[k_init] = 1.0
    n_pilots = 1

    feats: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    actions: List[int] = []
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    rewards: List[float] = []

    while n_pilots < cfg.max_pilots:
        mask_2d = mask_to_grid(mask, cfg)
        x_feat = build_features(
            state.h_hat, mask_2d, n_pilots, cfg.sigma2,
            na=grid_na, nc=grid_nc, n_candidates=nc, zscore_mode=zscore_mode,
        )
        no_grad_ctx = torch.no_grad() if deterministic else torch.enable_grad()
        with no_grad_ctx:
            logits = model(x_feat.unsqueeze(0)).squeeze(0)
            logits = logits.masked_fill(mask > 0.5, float("-inf"))
            if deterministic:
                k = int(torch.argmax(logits).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                k = int(action.item())
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())
                feats.append(x_feat)
                masks.append(mask.clone())
                actions.append(k)

        x, y = measure_subcarrier(
            k, h_true, cfg.sigma2, n_antennas=na, n_total=n, device=device, dtype=cfg.dtype, generator=gen
        )
        state = lmmse_incremental_update(state, x, y, cfg.sigma2)
        mask[k] = 1.0
        n_pilots += 1
        nmse_t = empirical_nmse(state.h_hat, h_true)
        rewards.append(-math.log(nmse_t + cfg.label_eps))

    return {
        "feats": feats,
        "masks": masks,
        "actions": actions,
        "log_probs": log_probs,
        "entropies": entropies,
        "rewards": rewards,
        "final_nmse": math.exp(-rewards[-1]) - cfg.label_eps,
    }


def _sample_bucket_channel(cfg: RlTrainConfig, channel_mc: int) -> Tuple[str, float]:
    return bucket_for_channel(cfg, channel_mc) if cfg.domain == "time_freq" else ("A", 0.0)


# ---------------------------------------------------------------------------
# 5. REINFORCE update (Stage A)
# ---------------------------------------------------------------------------
def reinforce_update(
    model: CNNErrorEstimator,
    optimizer: torch.optim.Optimizer,
    episodes: List[Dict[str, Any]],
    *,
    device: torch.device,
    entropy_coef: float,
    grad_clip_norm: float,
) -> Dict[str, float]:
    """Per-step bandit loss, reward_t = -log(nmse_true_t+eps), baseline = batch-mean reward at
    that same step index WITHIN this bucket (NMSE scale differs ~15x across the 9 time_freq
    buckets, so a single global baseline would be dominated by whichever bucket's rewards have
    the largest magnitude)."""
    by_bucket: Dict[Tuple[str, float], List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        by_bucket[ep["bucket"]].append(ep)

    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=device)
    total_entropy = 0.0
    total_reward = 0.0
    n_ep = 0

    for _bucket, eps in by_bucket.items():
        rewards = torch.tensor([ep["rewards"] for ep in eps], device=device, dtype=torch.float32)  # (B,T)
        baseline = rewards.mean(dim=0, keepdim=True)  # (1,T)
        advantage = (rewards - baseline).detach()  # (B,T)
        for i, ep in enumerate(eps):
            lp = torch.stack(ep["log_probs"])  # (T,)
            ent = torch.stack(ep["entropies"])  # (T,)
            loss_i = -(lp * advantage[i]).mean() - entropy_coef * ent.mean()
            total_loss = total_loss + loss_i
            total_entropy += float(ent.mean().item())
            total_reward += float(rewards[i].mean().item())
            n_ep += 1

    loss = total_loss / max(n_ep, 1)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "mean_entropy": total_entropy / max(n_ep, 1),
        "mean_reward": total_reward / max(n_ep, 1),
    }


# ---------------------------------------------------------------------------
# 6. PPO update (Stage B)
# ---------------------------------------------------------------------------
def _returns_to_go(rewards: List[float], device: torch.device) -> torch.Tensor:
    """Undiscounted (gamma=1) return-to-go over a fixed-length episode."""
    out = torch.zeros(len(rewards), device=device, dtype=torch.float32)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc += rewards[t]
        out[t] = acc
    return out


def ppo_update(
    model: CNNErrorEstimator,
    value_model: ValueNet,
    optimizer: torch.optim.Optimizer,
    episodes: List[Dict[str, Any]],
    *,
    device: torch.device,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    ppo_epochs: int,
    grad_clip_norm: float,
) -> Dict[str, float]:
    all_feats = torch.stack([f for ep in episodes for f in ep["feats"]]).to(device)  # (M,6,H,W)
    all_masks = torch.stack([m for ep in episodes for m in ep["masks"]]).to(device)  # (M,Nc)
    all_actions = torch.tensor(
        [a for ep in episodes for a in ep["actions"]], device=device, dtype=torch.long
    )
    old_log_probs = torch.stack([lp for ep in episodes for lp in ep["log_probs"]]).detach()

    returns_list: List[torch.Tensor] = []
    bucket_ids: List[int] = []
    bucket_to_id = {b: i for i, b in enumerate(sorted({ep["bucket"] for ep in episodes}))}
    for ep in episodes:
        r2g = _returns_to_go(ep["rewards"], device)
        returns_list.append(r2g)
        bucket_ids.extend([bucket_to_id[ep["bucket"]]] * len(ep["rewards"]))
    all_returns = torch.cat(returns_list).detach()
    all_bucket_ids = torch.tensor(bucket_ids, device=device, dtype=torch.long)

    metrics: Dict[str, float] = {}
    for _epoch in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(all_feats)
        logits = logits.masked_fill(all_masks > 0.5, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)
        new_log_probs = dist.log_prob(all_actions)
        entropy = dist.entropy()
        values = value_model(all_feats)

        advantage = (all_returns - values.detach()).clone()
        for b in all_bucket_ids.unique():
            sel = all_bucket_ids == b
            mu = advantage[sel].mean()
            sd = advantage[sel].std(unbiased=False).clamp(min=1e-6)
            advantage[sel] = (advantage[sel] - mu) / sd

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, all_returns)
        entropy_bonus = entropy.mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

        loss.backward()
        # Clipped separately: value_loss runs ~1e3-1e4x larger than policy_loss, so a joint
        # clip_grad_norm_ over both networks lets the value net's gradient dominate the combined
        # norm and the rescale crushes the policy's gradient to near-zero regardless of lr/value_coef.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mean_entropy": float(entropy_bonus.item()),
            "mean_return": float(all_returns.mean().item()),
        }
    return metrics


# ---------------------------------------------------------------------------
# 7. Checkpoint save/load, RlPilotPolicy
# ---------------------------------------------------------------------------
def save_rl_checkpoint(
    path: Path,
    *,
    model: CNNErrorEstimator,
    cfg: RlTrainConfig,
    arch_kwargs: Dict[str, Any],
    select_metric: float,
    update: int,
    value_model: Optional[ValueNet] = None,
) -> None:
    blob: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "cfg": {k: v for k, v in asdict(cfg).items() if not isinstance(v, torch.dtype)},
        "stage": cfg.stage,
        "domain": cfg.domain,
        "select_metric": select_metric,
        "update": update,
        "arch": {
            "class": "CNNErrorEstimator",
            "n_subcarriers": cfg.n_subcarriers,
            "pool_mode": arch_kwargs["pool_mode"],
            "kernel_hw": list(arch_kwargs["kernel_hw"]),
            "use_global_context": arch_kwargs["use_global_context"],
        },
    }
    if value_model is not None:
        blob["value_state_dict"] = value_model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)


def load_rl_checkpoint(path: Path, device: torch.device) -> CNNErrorEstimator:
    blob = torch.load(path, map_location=device, weights_only=False)
    arch = blob.get("arch", {})
    nc = int(arch.get("n_subcarriers", 512))
    pool_mode = arch.get("pool_mode", "per_cell")
    kernel_hw = tuple(arch.get("kernel_hw", (7, 3)))
    use_global_context = bool(arch.get("use_global_context", False))
    model = CNNErrorEstimator(
        n_subcarriers=nc, pool_mode=pool_mode, kernel_hw=kernel_hw, use_global_context=use_global_context
    ).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model


class RlPilotPolicy:
    """Greedy pilot selection from a trained RL policy network's logits (higher = better;
    argmax -- not the CNN's argmin-over-predicted-log-error convention). Mirrors
    error_estimators.cnn.train.CnnPilotPolicy's shape/signature exactly so it plugs into
    run_closed_loop_mse_curve the same way."""

    def __init__(self, model: CNNErrorEstimator, *, cfg: RlTrainConfig, device: torch.device) -> None:
        self._model = model
        self._cfg = cfg
        self._nc = cfg.n_subcarriers
        self._sigma2 = cfg.sigma2
        self._device = device
        self._grid_na, self._grid_nc = feature_grid_shape(cfg)
        self._zscore_mode = zscore_mode_for(cfg)

    def reset(self) -> None:
        pass

    def initial_subcarriers(self) -> List[int]:
        return []

    def next_subcarrier(self, state: EstimatorState, used: List[int]) -> int:
        mask = torch.zeros(self._nc, device=self._device, dtype=torch.float32)
        for k in used:
            mask[k] = 1.0
        return self.pick_next(state, used, mask, len(used))

    def pick_next(self, state: EstimatorState, used: List[int], mask: torch.Tensor, n_pilots: int) -> int:
        self._model.eval()
        mask_2d = mask_to_grid(mask, self._cfg)
        x = build_features(
            state.h_hat, mask_2d, n_pilots, self._sigma2,
            na=self._grid_na, nc=self._grid_nc, n_candidates=self._nc, zscore_mode=self._zscore_mode,
        )
        with torch.no_grad():
            logits = self._model(x.unsqueeze(0)).squeeze(0)
        scores = logits.clone()
        for k in used:
            scores[k] = float("-inf")
        return int(scores.argmax().item())


# ---------------------------------------------------------------------------
# 8. rl_real_eval_score (checkpoint-selection proxy, mirrors real_eval_cnn_score)
# ---------------------------------------------------------------------------
def rl_real_eval_score(
    cfg: RlTrainConfig,
    model: CNNErrorEstimator,
    *,
    sigma_by_bucket: Dict[Tuple[str, float], torch.Tensor],
    grid: Any,
    device: torch.device,
    n_mc: int,
) -> Dict[str, Any]:
    """Same closed-loop real-deployment proxy the CNN uses for select_checkpoint_by='real_eval'
    (error_estimators/cnn/train.py::real_eval_cnn_score), run through the exact same
    run_until_threshold/run_closed_loop_mse_curve pipeline used for the final cross-arm
    comparison -- so the checkpoint picked during training is scored under the identical
    procedure used to compare against fixed/active/cnn later, not a separate ad hoc rollout."""
    from error_estimators.cnn.train import run_closed_loop_mse_curve  # local: avoids a module-load cycle

    buckets = list(sigma_by_bucket.keys())
    per_bucket_mse: Dict[str, float] = {}
    for bi, (model_name, speed) in enumerate(buckets):
        sigma = sigma_by_bucket[(model_name, speed)]
        mses: List[float] = []
        for mc in range(n_mc):
            channel_mc = EVAL_CHANNEL_OFFSET + bi * 1_000_000 + mc
            h_true = sample_tdl_channel(cfg, channel_mc=channel_mc, device=device, grid=grid)
            curve = run_closed_loop_mse_curve(
                cfg, h_true=h_true, sigma=sigma, device=device, channel_mc=channel_mc,
                policy_mode="rl", rl_model=model,
            )
            mses.append(curve[-1])
        key = f"{model_name}_{speed:g}" if cfg.domain == "time_freq" else "antenna_snapshot"
        per_bucket_mse[key] = float(sum(mses) / len(mses))
    mean_mse = float(sum(per_bucket_mse.values()) / len(per_bucket_mse))
    worst_bucket = max(per_bucket_mse, key=per_bucket_mse.get)
    return {
        "mean_mse": mean_mse,
        "worst_bucket": worst_bucket,
        "worst_mse": per_bucket_mse[worst_bucket],
        "per_bucket_mse": per_bucket_mse,
    }


# ---------------------------------------------------------------------------
# 9. train_reinforce / train_ppo
# ---------------------------------------------------------------------------
def _init_model(cfg: RlTrainConfig, device: torch.device) -> Tuple[CNNErrorEstimator, Dict[str, Any]]:
    arch_kwargs = model_arch_kwargs(cfg)
    model = CNNErrorEstimator(n_subcarriers=cfg.n_subcarriers, **arch_kwargs).to(device)
    return model, arch_kwargs


def train_reinforce(cfg: RlTrainConfig) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    _check_domain_config(cfg)
    assert cfg.stage == "reinforce"

    model, arch_kwargs = _init_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    grid = make_grid(cfg)
    sigma_by_bucket = load_sigma_buckets(cfg, device)
    buckets = tdl_buckets(cfg) if cfg.domain == "time_freq" else [("A", 0.0)]

    suffix = rl_label_suffix(cfg)
    checkpoint_path = CHECKPOINT_DIR / f"reinforce_bandit{suffix}_best.pt"
    metrics_path = CHECKPOINT_DIR / f"reinforce_bandit{suffix}_metrics.json"

    print(
        f"train (reinforce): device={device}, domain={cfg.domain}, buckets={len(buckets)}, "
        f"episodes_per_update={cfg.episodes_per_update}, n_updates={cfg.n_updates}"
    )

    best_metric = float("inf")
    best_update = -1
    metrics: List[Dict[str, Any]] = []
    episode_counter = 0

    for update in range(1, cfg.n_updates + 1):
        t0 = time.perf_counter()
        episodes: List[Dict[str, Any]] = []
        for _ in range(cfg.episodes_per_update):
            channel_mc = episode_counter
            episode_counter += 1
            if channel_mc >= EVAL_CHANNEL_OFFSET:
                raise RuntimeError(
                    f"Training channel_mc ({channel_mc}) reached EVAL_CHANNEL_OFFSET "
                    f"({EVAL_CHANNEL_OFFSET}), which is reserved for rl_real_eval_score's "
                    "held-out validation channels -- reduce n_updates/episodes_per_update or "
                    "raise EVAL_CHANNEL_OFFSET."
                )
            bucket = _sample_bucket_channel(cfg, channel_mc)
            sigma = sigma_by_bucket[bucket]
            h_true = sample_tdl_channel(cfg, channel_mc=channel_mc, device=device, grid=grid)
            ep = rollout_episode(
                cfg, model, h_true=h_true, sigma=sigma, device=device, channel_mc=channel_mc,
                deterministic=False,
            )
            ep["bucket"] = bucket
            episodes.append(ep)

        stats = reinforce_update(
            model, optimizer, episodes, device=device,
            entropy_coef=cfg.entropy_coef, grad_clip_norm=cfg.grad_clip_norm,
        )
        elapsed = time.perf_counter() - t0

        star = ""
        record = {"update": update, **stats}
        if update % cfg.eval_interval == 0 or update == cfg.n_updates:
            real_eval = rl_real_eval_score(
                cfg, model, sigma_by_bucket=sigma_by_bucket, grid=grid, device=device,
                n_mc=cfg.real_eval_n_mc,
            )
            record["real_eval_mean_mse"] = real_eval["mean_mse"]
            record["real_eval_worst_bucket"] = real_eval["worst_bucket"]
            record["real_eval_worst_mse"] = real_eval["worst_mse"]
            if real_eval["mean_mse"] < best_metric:
                best_metric = real_eval["mean_mse"]
                best_update = update
                star = " *"
                save_rl_checkpoint(
                    checkpoint_path, model=model, cfg=cfg, arch_kwargs=arch_kwargs,
                    select_metric=best_metric, update=update,
                )
            print(
                f"update {update:05d}/{cfg.n_updates:05d}  loss={stats['loss']:.4f}  "
                f"mean_reward={stats['mean_reward']:.4f}  entropy={stats['mean_entropy']:.3f}  "
                f"real_eval_mean={real_eval['mean_mse']:.4e}  "
                f"real_eval_worst={real_eval['worst_bucket']}({real_eval['worst_mse']:.4e})  "
                f"{elapsed:.1f}s{star}"
            )
        metrics.append(record)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"train (reinforce) done: best_real_eval_mean={best_metric:.4e} @ update={best_update}  "
          f"checkpoint={checkpoint_path}")


def train_ppo(cfg: RlTrainConfig) -> None:
    device = resolve_device(cfg.device)
    set_seed(cfg.seed, device)
    _check_domain_config(cfg)
    assert cfg.stage == "ppo"

    model, arch_kwargs = _init_model(cfg, device)
    if cfg.init_checkpoint is not None:
        init_model = load_rl_checkpoint(Path(cfg.init_checkpoint), device)
        model.load_state_dict(init_model.state_dict())
        print(f"PPO: warm-started policy from {cfg.init_checkpoint}")
    value_model = ValueNet(kernel_hw=arch_kwargs["kernel_hw"]).to(device)
    optimizer = torch.optim.AdamW(
        chain(model.parameters(), value_model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    grid = make_grid(cfg)
    sigma_by_bucket = load_sigma_buckets(cfg, device)
    buckets = tdl_buckets(cfg) if cfg.domain == "time_freq" else [("A", 0.0)]

    suffix = rl_label_suffix(cfg)
    checkpoint_path = CHECKPOINT_DIR / f"ppo_trajectory{suffix}_best.pt"
    metrics_path = CHECKPOINT_DIR / f"ppo_trajectory{suffix}_metrics.json"

    print(
        f"train (ppo): device={device}, domain={cfg.domain}, buckets={len(buckets)}, "
        f"episodes_per_update={cfg.episodes_per_update}, n_updates={cfg.n_updates}, "
        f"ppo_epochs={cfg.ppo_epochs}"
    )

    best_metric = float("inf")
    best_update = -1
    metrics: List[Dict[str, Any]] = []
    episode_counter = 0

    for update in range(1, cfg.n_updates + 1):
        t0 = time.perf_counter()
        episodes: List[Dict[str, Any]] = []
        for _ in range(cfg.episodes_per_update):
            channel_mc = episode_counter
            episode_counter += 1
            if channel_mc >= EVAL_CHANNEL_OFFSET:
                raise RuntimeError(
                    f"Training channel_mc ({channel_mc}) reached EVAL_CHANNEL_OFFSET "
                    f"({EVAL_CHANNEL_OFFSET}), which is reserved for rl_real_eval_score's "
                    "held-out validation channels -- reduce n_updates/episodes_per_update or "
                    "raise EVAL_CHANNEL_OFFSET."
                )
            bucket = _sample_bucket_channel(cfg, channel_mc)
            sigma = sigma_by_bucket[bucket]
            h_true = sample_tdl_channel(cfg, channel_mc=channel_mc, device=device, grid=grid)
            ep = rollout_episode(
                cfg, model, h_true=h_true, sigma=sigma, device=device, channel_mc=channel_mc,
                deterministic=False,
            )
            ep["bucket"] = bucket
            episodes.append(ep)

        stats = ppo_update(
            model, value_model, optimizer, episodes, device=device,
            clip_eps=cfg.clip_eps, value_coef=cfg.value_coef, entropy_coef=cfg.entropy_coef,
            ppo_epochs=cfg.ppo_epochs, grad_clip_norm=cfg.grad_clip_norm,
        )
        elapsed = time.perf_counter() - t0

        star = ""
        record = {"update": update, **stats}
        if update % cfg.eval_interval == 0 or update == cfg.n_updates:
            real_eval = rl_real_eval_score(
                cfg, model, sigma_by_bucket=sigma_by_bucket, grid=grid, device=device,
                n_mc=cfg.real_eval_n_mc,
            )
            record["real_eval_mean_mse"] = real_eval["mean_mse"]
            record["real_eval_worst_bucket"] = real_eval["worst_bucket"]
            record["real_eval_worst_mse"] = real_eval["worst_mse"]
            if real_eval["mean_mse"] < best_metric:
                best_metric = real_eval["mean_mse"]
                best_update = update
                star = " *"
                save_rl_checkpoint(
                    checkpoint_path, model=model, cfg=cfg, arch_kwargs=arch_kwargs,
                    select_metric=best_metric, update=update, value_model=value_model,
                )
            print(
                f"update {update:05d}/{cfg.n_updates:05d}  loss={stats['loss']:.4f}  "
                f"policy_loss={stats['policy_loss']:.4f}  value_loss={stats['value_loss']:.4f}  "
                f"entropy={stats['mean_entropy']:.3f}  real_eval_mean={real_eval['mean_mse']:.4e}  "
                f"real_eval_worst={real_eval['worst_bucket']}({real_eval['worst_mse']:.4e})  "
                f"{elapsed:.1f}s{star}"
            )
        metrics.append(record)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"train (ppo) done: best_real_eval_mean={best_metric:.4e} @ update={best_update}  "
          f"checkpoint={checkpoint_path}")


# ---------------------------------------------------------------------------
# 10. argparse / main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL (REINFORCE/PPO) pilot-allocation policy training.")
    parser.add_argument("--stage", choices=("reinforce", "ppo"), default="reinforce")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--n-updates", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--real-eval-n-mc", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=str, default=None, help="ppo stage: warm-start from this reinforce checkpoint.")
    parser.add_argument(
        "--domain", choices=("antenna_snapshot", "time_freq"), default=None,
        help="Must match the CNN checkpoint's domain for a fair comparison.",
    )
    parser.add_argument("--tdl-models", nargs="+", default=None)
    parser.add_argument("--speeds", nargs="+", type=float, default=None)
    parser.add_argument("--n-cov-mc-per-bucket", type=int, default=None)
    parser.add_argument("--n-freq-subcarriers", type=int, default=None)
    parser.add_argument("--n-symbols", type=int, default=None)
    parser.add_argument(
        "--kernel-hw", type=int, nargs=2, default=None, metavar=("H", "W"),
        help="Must match the CNN production config for a fair comparison (e.g. --kernel-hw 7 3).",
    )
    parser.add_argument("--use-global-context", action="store_true", default=False)
    parser.add_argument("--max-pilots", type=int, default=None)
    parser.add_argument("--sigma2", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RlTrainConfig()
    cfg.stage = args.stage
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.episodes_per_update is not None:
        cfg.episodes_per_update = args.episodes_per_update
    if args.n_updates is not None:
        cfg.n_updates = args.n_updates
    if args.entropy_coef is not None:
        cfg.entropy_coef = args.entropy_coef
    if args.value_coef is not None:
        cfg.value_coef = args.value_coef
    if args.clip_eps is not None:
        cfg.clip_eps = args.clip_eps
    if args.ppo_epochs is not None:
        cfg.ppo_epochs = args.ppo_epochs
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.real_eval_n_mc is not None:
        cfg.real_eval_n_mc = args.real_eval_n_mc
    if args.init_checkpoint is not None:
        cfg.init_checkpoint = args.init_checkpoint
    if args.kernel_hw is not None:
        cfg.kernel_hw = tuple(args.kernel_hw)
    if args.use_global_context:
        cfg.use_global_context = True
    if args.tdl_models is not None:
        cfg.tdl_models = tuple(args.tdl_models)
    if args.speeds is not None:
        cfg.speeds = tuple(args.speeds)
    if args.n_cov_mc_per_bucket is not None:
        cfg.n_cov_mc_per_bucket = args.n_cov_mc_per_bucket
    if args.n_freq_subcarriers is not None:
        cfg.n_freq_subcarriers = args.n_freq_subcarriers
    if args.n_symbols is not None:
        cfg.n_symbols = args.n_symbols
    if args.max_pilots is not None:
        cfg.max_pilots = args.max_pilots
    if args.sigma2 is not None:
        cfg.sigma2 = args.sigma2
    if args.domain is not None:
        cfg.domain = args.domain
    if cfg.domain == "time_freq":
        cfg.n_antennas = 1
        cfg.n_subcarriers = cfg.n_symbols * cfg.n_freq_subcarriers
        cfg.label_mode = "counterfactual"  # irrelevant to RL's reward, kept for metadata/logging parity
    _check_domain_config(cfg)

    if cfg.stage == "reinforce":
        train_reinforce(cfg)
    else:
        train_ppo(cfg)


if __name__ == "__main__":
    main()
