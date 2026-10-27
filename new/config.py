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


@dataclass
class Exp12Config(ExpRunConfig):
    """Time-frequency (OFDM symbol x subcarrier) single-antenna pilot allocation.

    A 'pilot' is one (subcarrier, OFDM-symbol) resource element, not a whole subcarrier
    column across antennas. Implemented via the existing generic machinery with
    n_antennas=1 and n_subcarriers=n_symbols*n_freq_subcarriers treated as a flat RE
    index space (see channel_generators/time_freq.py); n_subcarriers must equal
    n_symbols * n_freq_subcarriers (checked at runtime, not enforced in __post_init__
    to keep this a plain dataclass like its siblings).
    """

    n_antennas: int = 1
    n_freq_subcarriers: int = 32
    n_symbols: int = 16
    n_subcarriers: int = 512  # = n_symbols * n_freq_subcarriers
    rho_time: float = 0.85
    speed: float = 30.0
    target_pilots: Optional[int] = 64
    max_pilots: int = 64
    n_mc: int = 50
    # 'fixed' policy: NR PDSCH DM-RS Type-1-style comb (comb-2 in frequency, repeated at a
    # front-loaded + 3 additional OFDM symbols) instead of the 1D largest-gap bisection.
    # comb_spacing=2 x 16 freqs x 4 pilot_symbols = 64 = target_pilots, so the comb exactly
    # fills the pilot budget with no truncation.
    dmrs_comb_spacing: int = 2
    dmrs_pilot_symbols: tuple[int, ...] = (2, 6, 10, 14)
    dmrs_comb_offset: int = 0


@dataclass
class Exp13Config(ExpRunConfig):
    """TDL-A time-frequency deployment with an unknown true Sigma.

    Static Ledoit-Wolf shrinkage (floor) vs online covariance inflation vs online SBL
    delay-Doppler precision update, all starting from one small-sample empirical calibration
    Sigma_hat_raw, with ActivePolicy reacting to each arm's own evolving covariance belief.
    Swept across a few speeds (Doppler severity) to check whether method ranking holds across
    channel dynamics. Reuses exp12's single-antenna time-frequency RE-grid machinery (see
    channel_generators/time_freq.py); n_subcarriers must equal n_symbols*n_freq_subcarriers.
    """

    n_antennas: int = 1
    n_freq_subcarriers: int = 32
    n_symbols: int = 16
    n_subcarriers: int = 512  # = n_symbols * n_freq_subcarriers
    target_pilots: Optional[int] = 64
    max_pilots: int = 64
    initial_pilot_subcarriers: int = 2
    n_mc: int = 50
    speeds: tuple[float, ...] = (3.0, 30.0, 120.0)
    n_cov_mc_shrink: int = 64
    # Arm B: online covariance inflation.
    inflation_mode: str = "decaying"
    inflation_factor0: float = 0.05
    inflation_decay: float = 0.9
    inflation_eps: float = 1e-4
    # Arm C: online SBL precision update (delay-Doppler domain). burn-in is deliberately a large
    # fraction of the pilot budget (N=512 free coefficients is badly under-determined early) and
    # the rebuilt Sigma is damped toward the previous one rather than fully replacing it, so one
    # noisy early rebuild can't blow up the estimate (see exp13 plan risk #5).
    sbl_burn_in_pilots: int = 32
    sbl_update_every: int = 4
    # 0.5 let an under-informed rebuild candidate dominate and corrupt the posterior for the
    # rest of the run (NMSE 0.002->0.7 at the first rebuild); 0.99 keeps rebuilds stable.
    sbl_sigma_damping: float = 0.99
    # alpha_min bounds the max prior variance (1/alpha_min) any single delay-Doppler coefficient
    # may claim; channel power is unit-normalized, so letting one noisy early estimate claim
    # variance orders of magnitude above that (the original 1e-6 -> up to 1e6) makes the
    # posterior solve ill-conditioned in data-starved directions and blows up h_hat.
    sbl_alpha_min: float = 1e-2
    sbl_alpha_max: float = 1e8
    sbl_eps: float = 1e-10
