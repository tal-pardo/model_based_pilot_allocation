from __future__ import annotations

import argparse
import sys
from pathlib import Path

_NEW_ROOT = Path(__file__).resolve().parents[1]
if str(_NEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEW_ROOT))

from config import Exp1bConfig
from experiments.exp1 import NOISE_SEED_OFFSET, compute_exp1_curves, plot_exp1

# Repeat arm: init [0, 15] like exp1, then SC 15 re-measured every step with fresh noise.
REPEAT_NOISE_SEED_OFFSET = 40_000
REPEAT_STYLE = ("C5", "-", "D")


def run_exp1b(cfg: Exp1bConfig) -> None:
    offsets = {**NOISE_SEED_OFFSET, "repeat": REPEAT_NOISE_SEED_OFFSET}
    curves, full_lmmse_nmse = compute_exp1_curves(cfg, offsets)
    title = (
        f"exp1b repeated subcarrier  Na={cfg.n_antennas}, Nc={cfg.n_subcarriers}, "
        f"n_mc={cfg.n_mc}, pilots={cfg.target_pilots}"
    )
    plot_exp1(
        {
            "fixed": curves["fixed"],
            "random": curves["random"],
            "active": curves["active"],
            "same SC repeated": curves["repeat"],
        },
        cfg,
        _NEW_ROOT / "figures" / "exp1b_repeat_subcarrier.png",
        full_lmmse_nmse=full_lmmse_nmse,
        title=title,
        extra_styles={"same SC repeated": REPEAT_STYLE},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="exp1b: exp1 policies + re-measuring the same subcarrier every step")
    p.add_argument("--n-mc", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    cfg = Exp1bConfig()
    args = parse_args()
    if args.n_mc is not None:
        cfg.n_mc = args.n_mc
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    run_exp1b(cfg)


if __name__ == "__main__":
    main()
