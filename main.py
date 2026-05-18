from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_generator import (
    complex_standard_normal,
    empirical_covariance,
    exponential_covariance,
    pilot_matrix_from_indices,
    stack_observations,
)
from estimators import batch_lmmse, recursive_lmmse
from pilots import (
    ActivePilotSampler,
    FixedPilotSampler,
    PilotScheduleConfig,
    num_timesteps_from_pilot_growth,
    ordered_subcarriers_from_vec,
    sequential_lmmse_mse_curve,
)
from sionna_channels import SionnaOFDMGrid, sample_cdl_c_channel, sample_tdl_c_channel, vec_from_H


def _h_to_H(h: torch.Tensor, n_antennas: int, n_subcarriers: int) -> torch.Tensor:
    # vec(H) ordering is column-stacking by subcarrier:
    # h = [H[:,0]; H[:,1]; ...; H[:,Nc-1]]
    N = n_antennas * n_subcarriers
    if h.shape != (N, 1):
        raise ValueError("Expected h with shape (N,1).")
    return h.view(n_subcarriers, n_antennas).T.contiguous()


class _LiveHeatmap:
    def __init__(self, *, Na: int, Nc: int, pause_s: float = 2.0) -> None:
        self.Na = Na
        self.Nc = Nc
        self.pause_s = float(pause_s)

        plt.ion()
        self.fig, (self.ax_true, self.ax_hat) = plt.subplots(
            2, 1, figsize=(10.0, 6.0), sharex=True, constrained_layout=True
        )
        self.im_true = self.ax_true.imshow(
            torch.zeros((Na, Nc), dtype=torch.float32).cpu().numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        self.im_hat = self.ax_hat.imshow(
            torch.zeros((Na, Nc), dtype=torch.float32).cpu().numpy(),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )

        self.ax_true.set_title(r"True $|H|$ (pilot subcarriers marked)")
        self.ax_hat.set_title(r"Estimate $|\hat H_t|$")
        self.ax_hat.set_xlabel("Subcarrier k")
        self.ax_true.set_ylabel("Antenna i")
        self.ax_hat.set_ylabel("Antenna i")

        self._pilot_lines = []
        self._text = self.ax_hat.text(
            0.01,
            0.99,
            "",
            transform=self.ax_hat.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )

        self.fig.colorbar(self.im_true, ax=self.ax_true, fraction=0.035, pad=0.02)
        self.fig.colorbar(self.im_hat, ax=self.ax_hat, fraction=0.035, pad=0.02)

    def _set_pilots(self, sc_list: list[int]) -> None:
        for ln in self._pilot_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._pilot_lines = []
        for k in sc_list:
            self._pilot_lines.append(self.ax_true.axvline(float(k), color="w", linewidth=1.2, alpha=0.9))

    def update(
        self,
        *,
        t: int,
        H_true_abs: torch.Tensor,
        H_hat_abs: torch.Tensor,
        pilot_subcarriers: list[int],
        mse: float,
        block: bool = False,
    ) -> None:
        self.im_true.set_data(H_true_abs.detach().to("cpu").to(torch.float32).numpy())
        self.im_hat.set_data(H_hat_abs.detach().to("cpu").to(torch.float32).numpy())

        vmin = float(min(H_true_abs.min().real.item(), H_hat_abs.min().real.item()))
        vmax = float(max(H_true_abs.max().real.item(), H_hat_abs.max().real.item()))
        self.im_true.set_clim(vmin=vmin, vmax=vmax)
        self.im_hat.set_clim(vmin=vmin, vmax=vmax)

        self._set_pilots(pilot_subcarriers)
        self._text.set_text(f"t={t} | pilots={len(pilot_subcarriers)} | MSE={mse:.3e}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if block:
            # Keep the final frame open until the user closes it.
            plt.ioff()
            plt.show(block=True)
        else:
            plt.pause(self.pause_s)


@dataclass
class ExperimentConfig:
    n_antennas: int = 32  # Na
    n_subcarriers: int = 64  # Nc
    rho_space: float = 0.7  # spatial correlation
    rho_freq: float = 0.8  # frequency correlation
    sigma2: float = 1e-2

    # Pilot schedule controls
    # Pilots are defined per-subcarrier: a pilot subcarrier observes all Na coefficients on that subcarrier.
    initial_pilot_subcarriers: int = 4  # start with 4 evenly-spaced pilot subcarriers
    final_pilot_subcarriers: int = 12  # end with 12 roughly-evenly-spaced pilot subcarriers
    pilots_added_per_step: int = 1  # add this many pilot subcarriers each step until final_pilot_subcarriers
    cumulative_pilots: bool = True  # if True, pilot set grows over time

    n_mc: int = 10  # Monte-Carlo trials
    n_cov_mc: int = 300  # Sionna-only empirical covariance samples (experiment1)

    seed: int = 0
    device: str = "cuda"  # "cpu" | "cuda" | "cuda:N" | "gpu" (alias for cuda). CUDA required if GPU requested.
    dtype: torch.dtype = torch.complex64

    # Numerical tolerance for recursive vs batch
    rtol: float = 1e-4
    atol: float = 5e-4

## Pilot scheduling and sequential estimation logic lives in pilots.py


def _resolve_device(device_str: str) -> torch.device:
    raw = device_str.lower().strip()
    if raw == "gpu":
        raw = "cuda"
    want_cuda = raw == "cuda" or raw.startswith("cuda:")
    if want_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (device=%r) but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and drivers, or set cfg.device='cpu'."
                % (device_str,)
            )
        device = torch.device(raw if raw.startswith("cuda:") else "cuda:0")
        cuda_index = device.index if device.index is not None else 0
        torch.cuda.set_device(cuda_index)
        print(
            f"Using CUDA device: {torch.cuda.get_device_name(cuda_index)} "
            f"(capability {torch.cuda.get_device_capability(cuda_index)})"
        )
        return device
    if raw != "cpu":
        raise ValueError("cfg.device must be 'cpu', 'cuda', 'cuda:N', or 'gpu'; got %r." % (device_str,))
    print("Using device: cpu")
    return torch.device("cpu")


def _build_sigma_kron(
    Na: int,
    Nc: int,
    rho_space: float,
    rho_freq: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R_space = exponential_covariance(Na, rho_space, device=device, dtype=dtype)
    R_freq = exponential_covariance(Nc, rho_freq, device=device, dtype=dtype)
    R_space = 0.5 * (R_space + R_space.mH)
    R_freq = 0.5 * (R_freq + R_freq.mH)
    Sigma = torch.kron(R_freq, R_space)
    Sigma = 0.5 * (Sigma + Sigma.mH)
    Sigma = Sigma + (1e-9 * torch.eye(Na * Nc, device=device, dtype=dtype))
    L_space = torch.linalg.cholesky(R_space)
    L_freq = torch.linalg.cholesky(R_freq)
    return Sigma, L_space, L_freq


def _sample_gaussian_h(
    Na: int,
    Nc: int,
    L_space: torch.Tensor,
    L_freq: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    Z = complex_standard_normal(Na, Nc, device=device, dtype=dtype, generator=generator)
    H_true = L_space @ Z @ L_freq.T
    return H_true.T.contiguous().view(Na * Nc, 1)


def _estimate_sionna_sigma_hat(
    *,
    n_cov_mc: int,
    seed: int,
    sample_fn,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    samples = torch.zeros((n_cov_mc, N), device=device, dtype=dtype)
    for k in range(n_cov_mc):
        H = sample_fn(seed=seed + k)
        samples[k] = vec_from_H(H).squeeze(-1)
    return empirical_covariance(samples, device=device, dtype=dtype)


def _plot_empirical_vs_theoretical_panel(
    ax,
    *,
    t: torch.Tensor,
    empirical: torch.Tensor,
    theoretical: torch.Tensor,
    title: str,
) -> None:
    t_np = t.numpy()
    ax.semilogy(
        t_np,
        empirical.numpy(),
        marker="o",
        linestyle="-",
        linewidth=1.6,
        color="C0",
        label="empirical",
    )
    ax.semilogy(
        t_np,
        theoretical.numpy(),
        linestyle=(0, (1, 2)),
        linewidth=1.0,
        color="C1",
        alpha=0.85,
        label="tr(P)/N",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Time step t")
    ax.set_ylabel("MSE  (1/N)||h_hat - h||^2 or tr(P_t)/N")
    ax.legend(fontsize=8)


def _plot_experiment1_family_validation(
    *,
    t: torch.Tensor,
    panels: list[tuple[str, torch.Tensor, torch.Tensor]],
    suptitle: str,
    out_path: Path,
    nrows: int,
    ncols: int,
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.8 * nrows), constrained_layout=True)
    axes_flat = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for ax, (title, empirical, theoretical) in zip(axes_flat, panels):
        _plot_empirical_vs_theoretical_panel(
            ax,
            t=t,
            empirical=empirical,
            theoretical=theoretical,
            title=title,
        )
    for ax in axes_flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()


def _print_family_validation_gaps(
    family: str,
    panels: list[tuple[str, torch.Tensor, torch.Tensor]],
) -> None:
    for title, empirical, theoretical in panels:
        emp_final = float(empirical[-1])
        theo_final = float(theoretical[-1])
        gap = emp_final - theo_final
        print(
            f"Validation {family} [{title}]: final empirical={emp_final:.6e}, "
            f"tr(P)/N={theo_final:.6e}, gap={gap:.6e}"
        )


def _print_kronecker_theo_consistency(theo_kron: dict[tuple[str, str], torch.Tensor]) -> None:
    for policy in ("fixed", "active"):
        ref = theo_kron[("gaussian", policy)]
        for family in ("tdl", "cdl"):
            diff = (theo_kron[(family, policy)] - ref).abs().max().item()
            print(f"Validation Kronecker {policy}: max |theo[{family}] - theo[gaussian]| = {diff:.3e}")


def _print_validation_final_gaps(
    curves_emp: dict[tuple[str, str], torch.Tensor],
    curves_theo: dict[tuple[str, str], torch.Tensor],
    *,
    label: str,
) -> None:
    for key, emp in curves_emp.items():
        theo = curves_theo[key]
        emp_final = float(emp[-1])
        theo_final = float(theo[-1])
        gap = emp_final - theo_final
        print(
            f"{label} {key[0]}/{key[1]}: final empirical={emp_final:.6e}, "
            f"tr(P)/N={theo_final:.6e}, gap={gap:.6e}"
        )


def _plot_experiment1_mse(
    *,
    t: torch.Tensor,
    curves: dict[tuple[str, str], torch.Tensor],
    title: str,
    out_path: Path,
) -> None:
    colors = {"gaussian": "C0", "tdl": "C1", "cdl": "C2"}
    labels = {"gaussian": "Gaussian", "tdl": "TDL-C", "cdl": "CDL-C"}
    plt.figure(figsize=(8.4, 4.8))
    for (family, policy), mse_mean in curves.items():
        linestyle = "-" if policy == "fixed" else "--"
        marker = "o" if policy == "fixed" else "s"
        label = f"{labels[family]} ({policy})"
        plt.semilogy(t.numpy(), mse_mean.numpy(), marker=marker, linestyle=linestyle, linewidth=1.6, color=colors[family], label=label)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Time step t")
    plt.ylabel("Mean MSE over MC  (1/N)||h_hat - h||^2")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.show()


def experiment1(cfg: ExperimentConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    dtype = cfg.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    Na, Nc = cfg.n_antennas, cfg.n_subcarriers
    N = Na * Nc
    Sigma_kron, L_space, L_freq = _build_sigma_kron(
        Na, Nc, cfg.rho_space, cfg.rho_freq, device=device, dtype=dtype
    )
    grid = SionnaOFDMGrid(fft_size=Nc, subcarrier_spacing=15e3, carrier_frequency=3.5e9, delay_spread=100e-9)

    T = num_timesteps_from_pilot_growth(
        cfg.initial_pilot_subcarriers, cfg.final_pilot_subcarriers, cfg.pilots_added_per_step
    )
    print(f"experiment1: T (new-observation steps) = {T}")

    sched_cfg = PilotScheduleConfig(
        n_subcarriers=Nc,
        n_antennas=Na,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        final_pilot_subcarriers=cfg.final_pilot_subcarriers,
        pilots_added_per_step=cfg.pilots_added_per_step,
        cumulative_pilots=cfg.cumulative_pilots,
    )
    T_total = T + 1
    fixed = FixedPilotSampler(sched_cfg, T=T_total, device=device)

    print(f"Estimating empirical Sigma from {cfg.n_cov_mc} Sionna draws per family...")
    Sigma_hat_tdl = _estimate_sionna_sigma_hat(
        n_cov_mc=cfg.n_cov_mc,
        seed=cfg.seed + 1_000_000,
        sample_fn=lambda seed: sample_tdl_c_channel(
            n_antennas=Na,
            n_subcarriers=Nc,
            rho_space=cfg.rho_space,
            grid=grid,
            device=device,
            dtype=dtype,
            seed=seed,
        ),
        N=N,
        device=device,
        dtype=dtype,
    )
    Sigma_hat_cdl = _estimate_sionna_sigma_hat(
        n_cov_mc=cfg.n_cov_mc,
        seed=cfg.seed + 2_000_000,
        sample_fn=lambda seed: sample_cdl_c_channel(
            n_antennas=Na,
            n_subcarriers=Nc,
            grid=grid,
            device=device,
            dtype=dtype,
            seed=seed,
        ),
        N=N,
        device=device,
        dtype=dtype,
    )

    families = ("gaussian", "tdl", "cdl")
    policies = ("fixed", "active")
    mse_kron = { (f, p): torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64) for f in families for p in policies }
    mse_emp = { (f, p): torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64) for f in families for p in policies }
    theo_kron = { (f, p): torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64) for f in families for p in policies }
    theo_emp = { (f, p): torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64) for f in families for p in policies }

    for mc in range(cfg.n_mc):
        h_gaussian = _sample_gaussian_h(
            Na,
            Nc,
            L_space,
            L_freq,
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(cfg.seed + mc),
        )
        h_tdl = vec_from_H(
            sample_tdl_c_channel(
                n_antennas=Na,
                n_subcarriers=Nc,
                rho_space=cfg.rho_space,
                grid=grid,
                device=device,
                dtype=dtype,
                seed=cfg.seed + 100_000 + mc,
            )
        )
        h_cdl = vec_from_H(
            sample_cdl_c_channel(
                n_antennas=Na,
                n_subcarriers=Nc,
                grid=grid,
                device=device,
                dtype=dtype,
                seed=cfg.seed + 200_000 + mc,
            )
        )
        h_by_family = {"gaussian": h_gaussian, "tdl": h_tdl, "cdl": h_cdl}
        sigma_emp_by_family = {"gaussian": Sigma_kron, "tdl": Sigma_hat_tdl, "cdl": Sigma_hat_cdl}

        for family in families:
            h_true = h_by_family[family]
            active_kron = ActivePilotSampler(sched_cfg, T=T_total, device=device, fixed=fixed, sigma2=cfg.sigma2)
            active_emp = ActivePilotSampler(sched_cfg, T=T_total, device=device, fixed=fixed, sigma2=cfg.sigma2)

            gen_fixed = torch.Generator(device=device).manual_seed(cfg.seed + mc)
            emp_fixed_kron, theo_fixed_kron = sequential_lmmse_mse_curve(
                Sigma_kron,
                h_true,
                cfg.sigma2,
                T,
                lambda t, _P: fixed.vec_indices_at_step(t),
                device=device,
                dtype=dtype,
                generator=gen_fixed,
            )
            mse_kron[(family, "fixed")][mc, :] = torch.tensor(emp_fixed_kron, dtype=torch.float64)
            theo_kron[(family, "fixed")][mc, :] = torch.tensor(theo_fixed_kron, dtype=torch.float64)
            gen_active_kron = torch.Generator(device=device).manual_seed(cfg.seed + 10_000 + mc)
            emp_active_kron, theo_active_kron = sequential_lmmse_mse_curve(
                Sigma_kron,
                h_true,
                cfg.sigma2,
                T,
                lambda t, P: active_kron.vec_indices_at_step(t, P),
                device=device,
                dtype=dtype,
                generator=gen_active_kron,
            )
            mse_kron[(family, "active")][mc, :] = torch.tensor(emp_active_kron, dtype=torch.float64)
            theo_kron[(family, "active")][mc, :] = torch.tensor(theo_active_kron, dtype=torch.float64)

            sigma_emp = sigma_emp_by_family[family]
            gen_fixed_emp = torch.Generator(device=device).manual_seed(cfg.seed + mc)
            emp_fixed_emp, theo_fixed_emp = sequential_lmmse_mse_curve(
                sigma_emp,
                h_true,
                cfg.sigma2,
                T,
                lambda t, _P: fixed.vec_indices_at_step(t),
                device=device,
                dtype=dtype,
                generator=gen_fixed_emp,
            )
            mse_emp[(family, "fixed")][mc, :] = torch.tensor(emp_fixed_emp, dtype=torch.float64)
            theo_emp[(family, "fixed")][mc, :] = torch.tensor(theo_fixed_emp, dtype=torch.float64)
            gen_active_emp = torch.Generator(device=device).manual_seed(cfg.seed + 10_000 + mc)
            emp_active_emp, theo_active_emp = sequential_lmmse_mse_curve(
                sigma_emp,
                h_true,
                cfg.sigma2,
                T,
                lambda t, P: active_emp.vec_indices_at_step(t, P),
                device=device,
                dtype=dtype,
                generator=gen_active_emp,
            )
            mse_emp[(family, "active")][mc, :] = torch.tensor(emp_active_emp, dtype=torch.float64)
            theo_emp[(family, "active")][mc, :] = torch.tensor(theo_active_emp, dtype=torch.float64)

        if mc == 0:
            gen_verify = torch.Generator(device=device).manual_seed(cfg.seed)
            X_list = [
                pilot_matrix_from_indices(N, fixed.vec_indices_at_step(t), device=device, dtype=dtype)
                for t in range(T + 1)
            ]
            y_list = []
            for t in range(T + 1):
                idx = fixed.vec_indices_at_step(t)
                n_t = (cfg.sigma2**0.5) * complex_standard_normal(
                    idx.numel(), 1, device=device, dtype=dtype, generator=gen_verify
                )
                y_list.append(X_list[t] @ h_gaussian + n_t)
            X_all, y_all = stack_observations(X_list, y_list)
            h_batch = batch_lmmse(Sigma_kron, X_all, y_all, cfg.sigma2)
            h_list, _ = recursive_lmmse(Sigma_kron, X_list, y_list, cfg.sigma2)
            max_abs_diff = (h_list[-1] - h_batch).abs().max().real.item()
            assert torch.allclose(h_list[-1], h_batch, rtol=cfg.rtol, atol=cfg.atol), (
                f"[mc=0, gaussian] Recursive final estimate and batch estimate differ. "
                f"max|diff|={max_abs_diff:.3e}, rtol={cfg.rtol}, atol={cfg.atol}"
            )

    curves_kron = {key: val.mean(dim=0) for key, val in mse_kron.items()}
    curves_emp = {key: val.mean(dim=0) for key, val in mse_emp.items()}
    curves_theo_kron = {key: val.mean(dim=0) for key, val in theo_kron.items()}
    curves_theo_emp = {key: val.mean(dim=0) for key, val in theo_emp.items()}
    t = torch.arange(0, T + 1, device="cpu")

    _print_kronecker_theo_consistency(theo_kron)

    for (family, policy), mse_mean in curves_kron.items():
        print(f"Fig1 Kronecker {family}/{policy}: final-step mean MSE = {float(mse_mean[-1]):.6e}")
    for (family, policy), mse_mean in curves_emp.items():
        print(f"Fig2 empirical {family}/{policy}: final-step mean MSE = {float(mse_mean[-1]):.6e}")

    fig_dir = Path(__file__).resolve().parent / "figures"
    common_title = (
        f"experiment1  Na={Na}, Nc={Nc}, n_mc={cfg.n_mc}, sigma2={cfg.sigma2}, "
        f"sc0={cfg.initial_pilot_subcarriers}, scf={cfg.final_pilot_subcarriers}"
    )
    _plot_experiment1_mse(
        t=t,
        curves=curves_kron,
        title=f"{common_title}  prior: Kronecker Sigma for all",
        out_path=fig_dir / "experiment1_mse_kronecker.png",
    )
    _plot_experiment1_mse(
        t=t,
        curves=curves_emp,
        title=f"{common_title}  prior: empirical Sigma (TDL/CDL), Kronecker (Gaussian)",
        out_path=fig_dir / "experiment1_mse_empirical.png",
    )

    _print_validation_final_gaps(curves_kron, curves_theo_kron, label="Validation Kronecker")
    _print_validation_final_gaps(curves_emp, curves_theo_emp, label="Validation empirical Sigma")

    gaussian_panels = [
        ("fixed pilots, Kronecker Sigma", curves_kron[("gaussian", "fixed")], curves_theo_kron[("gaussian", "fixed")]),
        ("active pilots, Kronecker Sigma", curves_kron[("gaussian", "active")], curves_theo_kron[("gaussian", "active")]),
    ]
    _print_family_validation_gaps("gaussian", gaussian_panels)
    _plot_experiment1_family_validation(
        t=t,
        panels=gaussian_panels,
        suptitle=f"{common_title}  Gaussian: empirical vs tr(P)/N",
        out_path=fig_dir / "experiment1_mse_validation_gaussian.png",
        nrows=1,
        ncols=2,
    )

    for family, label in (("tdl", "TDL-C"), ("cdl", "CDL-C")):
        family_panels = [
            (f"fixed pilots, Kronecker Sigma", curves_kron[(family, "fixed")], curves_theo_kron[(family, "fixed")]),
            (f"active pilots, Kronecker Sigma", curves_kron[(family, "active")], curves_theo_kron[(family, "active")]),
            (f"fixed pilots, empirical Sigma", curves_emp[(family, "fixed")], curves_theo_emp[(family, "fixed")]),
            (f"active pilots, empirical Sigma", curves_emp[(family, "active")], curves_theo_emp[(family, "active")]),
        ]
        _print_family_validation_gaps(family, family_panels)
        _plot_experiment1_family_validation(
            t=t,
            panels=family_panels,
            suptitle=f"{common_title}  {label}: empirical vs tr(P)/N",
            out_path=fig_dir / f"experiment1_mse_validation_{family}.png",
            nrows=2,
            ncols=2,
        )


def main(cfg: ExperimentConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    dtype = cfg.dtype
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Use a complex dtype (torch.complex64/128).")

    Na, Nc = cfg.n_antennas, cfg.n_subcarriers
    N = Na * Nc  # length of vec(H)

    # Structured Kronecker prior for h = vec(H) with column-stacking over subcarriers:
    # Sigma = R_freq \kron R_space
    R_space = exponential_covariance(Na, cfg.rho_space, device=device, dtype=dtype)
    R_freq = exponential_covariance(Nc, cfg.rho_freq, device=device, dtype=dtype)
    R_space = 0.5 * (R_space + R_space.mH)
    R_freq = 0.5 * (R_freq + R_freq.mH)
    Sigma = torch.kron(R_freq, R_space)  # (N,N)
    Sigma = 0.5 * (Sigma + Sigma.mH)
    Sigma = Sigma + (1e-9 * torch.eye(N, device=device, dtype=dtype))
    L_space = torch.linalg.cholesky(R_space)
    L_freq = torch.linalg.cholesky(R_freq)

    T = num_timesteps_from_pilot_growth(
        cfg.initial_pilot_subcarriers, cfg.final_pilot_subcarriers, cfg.pilots_added_per_step
    )
    # T is the number of new pilot-addition / new-observation steps.
    print(f"T (new-observation steps) = {T}")

    sched_cfg = PilotScheduleConfig(
        n_subcarriers=Nc,
        n_antennas=Na,
        initial_pilot_subcarriers=cfg.initial_pilot_subcarriers,
        final_pilot_subcarriers=cfg.final_pilot_subcarriers,
        pilots_added_per_step=cfg.pilots_added_per_step,
        cumulative_pilots=cfg.cumulative_pilots,
    )
    # Internally, samplers produce a schedule including the initial pilot layout, so use T_total = T+1.
    T_total = T + 1
    fixed = FixedPilotSampler(sched_cfg, T=T_total, device=device)
    active = ActivePilotSampler(sched_cfg, T=T_total, device=device, fixed=fixed, sigma2=cfg.sigma2)

    mse_fixed_mc = torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64)
    mse_active_mc = torch.zeros((cfg.n_mc, T + 1), device="cpu", dtype=torch.float64)
    maxdiff_mc = torch.zeros((cfg.n_mc,), device="cpu", dtype=torch.float64)
    fixed_pilots_used = None
    active_pilots_used = None

    for mc in range(cfg.n_mc):
        # torch.Generator must live on the same device as the tensors it seeds when using CUDA.
        gen = torch.Generator(device=device).manual_seed(cfg.seed + mc)

        # Sample H with vec(H) ~ CN(0, R_freq \kron R_space) via two-sided coloring:
        # H = L_space Z (L_freq^T), where Z is i.i.d. CN(0,1).
        Z = complex_standard_normal(Na, Nc, device=device, dtype=dtype, generator=gen)
        H_true = L_space @ Z @ L_freq.T  # (Na,Nc)
        # Column-stacked vec(H): [H[:,0]; H[:,1]; ...] so subcarrier k is a contiguous block.
        h_true = H_true.T.contiguous().view(N, 1)

        # Use only the first (T) observation steps from the (T_total) cumulative schedule:
        # step 0..T-1 correspond to pilot counts k0..kf-dk; step T corresponds to kf.
        emp_fixed, _ = sequential_lmmse_mse_curve(
            Sigma,
            h_true,
            cfg.sigma2,
            T,
            lambda t, _P: fixed.vec_indices_at_step(t),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        mse_fixed_mc[mc, :] = torch.tensor(emp_fixed, dtype=torch.float64)
        if fixed_pilots_used is None:
            fixed_pilots_used = ordered_subcarriers_from_vec(fixed.vec_indices_at_step(T_total - 1), Na)

        # Batch-verify the fixed path only (same pilot sequence).
        X_list = [
            pilot_matrix_from_indices(N, fixed.vec_indices_at_step(t), device=device, dtype=dtype)
            for t in range(T + 1)
        ]
        y_list = []
        for t in range(T + 1):
            idx = fixed.vec_indices_at_step(t)
            n_t = (cfg.sigma2**0.5) * complex_standard_normal(
                idx.numel(), 1, device=device, dtype=dtype, generator=gen
            )
            y_list.append(X_list[t] @ h_true + n_t)
        X_all, y_all = stack_observations(X_list, y_list)
        h_batch = batch_lmmse(Sigma, X_all, y_all, cfg.sigma2)
        h_list, _ = recursive_lmmse(Sigma, X_list, y_list, cfg.sigma2)

        h_final = h_list[-1]
        max_abs_diff = (h_final - h_batch).abs().max().real.item()
        maxdiff_mc[mc] = max_abs_diff
        assert torch.allclose(h_final, h_batch, rtol=cfg.rtol, atol=cfg.atol), (
            f"[mc={mc}] Recursive final estimate and batch estimate differ. "
            f"max|diff|={max_abs_diff:.3e}, rtol={cfg.rtol}, atol={cfg.atol}"
        )

        gen_active = torch.Generator(device=device).manual_seed(cfg.seed + 10_000 + mc)
        live = None
        if mc == 0:
            live = _LiveHeatmap(Na=Na, Nc=Nc, pause_s=2.0)

        def _on_step(t_vis, idx_vec, h_hat, _P, mse):
            if live is None:
                return
            if idx_vec is None:
                sc = []
            else:
                sc = sorted(set((idx_vec // Na).to("cpu").tolist()))
            H_hat = _h_to_H(h_hat, Na, Nc)
            live.update(
                t=int(t_vis),
                H_true_abs=H_true.abs(),
                H_hat_abs=H_hat.abs(),
                pilot_subcarriers=sc,
                mse=float(mse),
                block=bool(int(t_vis) == T),
            )

        emp_active, _ = sequential_lmmse_mse_curve(
            Sigma,
            h_true,
            cfg.sigma2,
            T,
            lambda t, P: active.vec_indices_at_step(t, P),
            device=device,
            dtype=dtype,
            generator=gen_active,
            on_step=_on_step if mc == 0 else None,
        )
        mse_active_mc[mc, :] = torch.tensor(emp_active, dtype=torch.float64)
        if active_pilots_used is None:
            active_pilots_used = list(active._used_sc)

    mse_fixed_mean = mse_fixed_mc.mean(dim=0).numpy()
    mse_active_mean = mse_active_mc.mean(dim=0).numpy()
    maxdiff_mean = maxdiff_mc.mean().item()
    maxdiff_max = maxdiff_mc.max().item()

    print("Monte-Carlo finished. Recursive final estimate matches batch LMMSE (all trials).")
    print(f"Mean max|h_rec(T)-h_batch| over MC = {maxdiff_mean:.3e}")
    print(f"Max  max|h_rec(T)-h_batch| over MC = {maxdiff_max:.3e}")
    print(f"Fixed:  mean final-step MSE over MC = {float(mse_fixed_mean[-1]):.6e}")
    if fixed_pilots_used is not None:
        print(f"Fixed pilots: {fixed_pilots_used}")
    print(f"Active: mean final-step MSE over MC = {float(mse_active_mean[-1]):.6e}")
    if active_pilots_used is not None:
        print(f"Active pilots (captured from mc=0): {active_pilots_used}")

    # Plot mean MSE over time (Monte-Carlo average)
    # Step indexing: t=0 corresponds to the estimate after the initial pilot set,
    # then each additional pilot-allocation/update increments t by 1.
    t = torch.arange(0, T + 1).cpu().numpy()
    plt.figure(figsize=(7.6, 4.4))
    plt.semilogy(t, mse_fixed_mean, marker="o", linewidth=1.6, label="Fixed (even)")
    plt.semilogy(t, mse_active_mean, marker="s", linewidth=1.6, label="Active (greedy)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("Time step t")
    plt.ylabel("Mean MSE over MC  (1/N)||h_hat - h||^2")
    title = (
        f"Recursive LMMSE (mean over {cfg.n_mc} MC)  Na={Na}, Nc={Nc}, "
        f"rho_space={cfg.rho_space}, rho_freq={cfg.rho_freq}, sigma2={cfg.sigma2}, "
        f"T={T}, sc0={cfg.initial_pilot_subcarriers}, scf={cfg.final_pilot_subcarriers}, "
        f"add/step={cfg.pilots_added_per_step}, cumulative={cfg.cumulative_pilots}"
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = ExperimentConfig(
        n_antennas=16,
        n_subcarriers=32,
        rho_space=0.8,
        rho_freq=0.85,
        sigma2=1e-2,
        initial_pilot_subcarriers=2,
        final_pilot_subcarriers=8,
        pilots_added_per_step=1,
        cumulative_pilots=True,
        n_mc=100,
        n_cov_mc=100,
        seed=1,
        device="cuda",
        dtype=torch.complex64,
    )
    experiment1(cfg)

