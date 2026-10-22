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
    target_pilots: Optional[int] = 32


@dataclass
class Exp1bConfig(Exp1Config):
    """exp1 settings plus a 'repeat' arm that re-measures the last initial subcarrier every step."""


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


@dataclass
class Exp6Config(ExpRunConfig):
    target_pilots: Optional[int] = 32
    rho_toeplitz: float = 0.2
    reg_kron: float = 0.0
    tdl_model: str = "A"


@dataclass
class Exp7Config(ExpRunConfig):
    """Matched LMMSE vs identity prior across four Gaussian covariance models."""

    target_pilots: Optional[int] = 32
    initial_pilot_subcarriers: int = 1
    n_mc: int = 100
    rho_toeplitz: float = 0.85
    rho_space: float = 0.85
    rho_freq: float = 0.85
    d_over_lambda: float = 0.5
    low_rank_r: int = 16
    low_rank_eps: float = 1e-3
    initial_subcarrier: int = 15


@dataclass
class Exp8Config(ExpRunConfig):
    """Low-rank: matched vs identity vs genie-active (matched J(k), identity estimator)."""

    target_pilots: Optional[int] = 32
    initial_pilot_subcarriers: int = 1
    n_mc: int = 100
    low_rank_r: int = 16
    low_rank_eps: float = 1e-3
    initial_subcarrier: int = 15


@dataclass
class Exp9Config(ExpRunConfig):
    """Block-cluster Kronecker: matched LMMSE active vs fixed (mid-horizon gap)."""

    target_pilots: Optional[int] = 32
    initial_pilot_subcarriers: int = 2
    n_mc: int = 50
    rho_space: float = 0.8
    rho_intra: float = 0.95
    rho_inter: float = 0.05
    n_clusters: int = 2


@dataclass
class Exp10Config(ExpRunConfig):
    """Policy gap under a TDL+CDL model x delay-spread sweep, one generic (pooled) Sigma.

    18 configs = {TDL, CDL} x {A, B, C} x delay_spreads. Four arms, all sharing
    the same h_true per trial and the same pooled empirical Sigma: fixed, random,
    active, oracle (brute-force true-NMSE search).
    """

    target_pilots: Optional[int] = 32
    initial_pilot_subcarriers: int = 1
    initial_subcarrier: int = 15
    n_mc: int = 36
    rho_space: float = 0.85
    subcarrier_spacing: float = 120e3
    tdl_models: tuple[str, ...] = ("A", "B", "C")
    cdl_models: tuple[str, ...] = ("A", "B", "C")
    delay_spreads: tuple[float, ...] = (30e-9, 100e-9, 300e-9)
    n_cov_mc: int = 2000
