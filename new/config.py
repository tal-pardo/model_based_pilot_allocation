from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimConfig:
    n_antennas: int = 16
    n_subcarriers: int = 32
    rho_space: float = 0.8
    rho_freq: float = 0.85
    sigma2: float = 1e-2
    reg_kron: float = 1e-9
    reg_empirical: float = 1e-3
    device: str = "cuda"
    dtype: torch.dtype = torch.complex64
    seed: int = 1


@dataclass
class ExpRunConfig(SimConfig):
    initial_pilot_subcarriers: int = 2
    max_pilots: int = 32
    nmse_threshold: float = 0.1
    n_mc: int = 50
    target_pilots: Optional[int] = None


@dataclass
class Exp1Config(ExpRunConfig):
    pass


@dataclass
class Exp2Config(ExpRunConfig):
    target_pilots: Optional[int] = 16
    n_cov_mc: int = 300
    tdl_empirical_cov_sizes: tuple[int, ...] = (250, 500)


@dataclass
class Exp3Config(ExpRunConfig):
    target_pilots: Optional[int] = 16
    n_cov_mc: int = 300


@dataclass
class Exp4Config(ExpRunConfig):
    target_pilots: Optional[int] = 16
    texture_alpha: float = 1.0


@dataclass
class Exp5Config(ExpRunConfig):
    nmse_threshold: float = 0.5
    target_pilots: Optional[int] = None
    tdl_model: str = "A"
