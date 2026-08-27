"""Deterministic periodic PDE simulator with intervention-separated actions.

The simulator is an outcome producer. Discovery code receives only ``x``,
``t``, and noisy ``u`` arrays; ``Family`` and coefficient metadata are private
evaluation material.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from .numerics import dealias_mask, integrate_etdrk4


class Family(str, Enum):
    ADVECTION_DIFFUSION = "advection_diffusion"
    VISCOUS_BURGERS = "viscous_burgers"
    FISHER_KPP = "fisher_kpp"
    ALLEN_CAHN = "allen_cahn"
    KDV = "kdv"
    HYPERDIFFUSION_OOG = "hyperdiffusion_oog"


TRUE_SUPPORTS: dict[Family, tuple[str, ...] | None] = {
    Family.ADVECTION_DIFFUSION: ("u_x", "u_xx"),
    Family.VISCOUS_BURGERS: ("u_xx", "u u_x"),
    Family.FISHER_KPP: ("u", "u^2", "u_xx"),
    Family.ALLEN_CAHN: ("u", "u^3", "u_xx"),
    Family.KDV: ("u_xxx", "u u_x"),
    Family.HYPERDIFFUSION_OOG: None,
}


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    description: str


BASELINE_ACTION = Action(
    "low_positive",
    0.35,
    "small positive low-modal initial condition",
)


def action_menu() -> tuple[Action, ...]:
    """Unique deployable second actions; no noise or cost duplicates."""

    return (
        Action("gentle_mode1", 0.45, "small zero-mean single-mode perturbation"),
        Action("sign_changing", 1.00, "zero-mean sign-changing modes 1 and 2"),
        Action("mean_shifted", 0.80, "positive mean with modes 1 and 2"),
        Action("gradient_rich", 1.15, "zero-mean modes 2, 3, and 5"),
        Action("mixed_scale", 0.95, "mean-shifted modes 1, 3, and 4"),
    )


@dataclass(frozen=True)
class SimulationConfig:
    nx: int = 64
    length: float = 2.0 * np.pi
    t_final: float = 0.24
    internal_dt: float = 0.00025
    save_dt: float = 0.01

    def validate(self) -> tuple[int, int]:
        if self.nx < 32 or self.nx % 2:
            raise ValueError("nx must be even and at least 32")
        if min(self.length, self.t_final, self.internal_dt, self.save_dt) <= 0:
            raise ValueError("grid/time quantities must be positive")
        steps = int(round(self.t_final / self.internal_dt))
        stride = int(round(self.save_dt / self.internal_dt))
        if not np.isclose(steps * self.internal_dt, self.t_final, atol=1e-13):
            raise ValueError("t_final must be an internal_dt multiple")
        if not np.isclose(stride * self.internal_dt, self.save_dt, atol=1e-13):
            raise ValueError("save_dt must be an internal_dt multiple")
        if steps % stride or (steps // stride + 1) % 2 == 0:
            raise ValueError("saved grid must include an odd number of points")
        return steps, stride


@dataclass(frozen=True)
class Trajectory:
    sample_id: str
    action: str
    x: np.ndarray
    t: np.ndarray
    u: np.ndarray
    noise_model: str
    relative_noise: float


@dataclass(frozen=True)
class HiddenTruth:
    family: Family
    support: tuple[str, ...] | None
    coefficients: tuple[tuple[str, float], ...]
    seed: int


def _initial_condition(action: Action, x: np.ndarray, public_seed: int) -> np.ndarray:
    rng = np.random.default_rng(public_seed)
    p1, p2, p3 = rng.uniform(0.0, 2.0 * np.pi, size=3)
    if action.name == "low_positive":
        return 0.18 + 0.035 * np.cos(x + p1) + 0.012 * np.sin(2.0 * x + p2)
    if action.name == "gentle_mode1":
        return 0.12 * np.cos(x + p1)
    if action.name == "sign_changing":
        return 0.58 * np.cos(x + p1) + 0.24 * np.sin(2.0 * x + p2)
    if action.name == "mean_shifted":
        return 0.52 + 0.21 * np.cos(x + p1) + 0.09 * np.sin(2.0 * x + p2)
    if action.name == "gradient_rich":
        return (
            0.34 * np.sin(2.0 * x + p1)
            + 0.20 * np.cos(3.0 * x + p2)
            + 0.08 * np.sin(5.0 * x + p3)
        )
    if action.name == "mixed_scale":
        return (
            0.30
            + 0.22 * np.sin(x + p1)
            + 0.14 * np.cos(3.0 * x + p2)
            + 0.06 * np.sin(4.0 * x + p3)
        )
    raise ValueError(f"unknown action {action.name}")


def _problem(
    family: Family,
    k: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], tuple[tuple[str, float], ...]]:
    mask = dealias_mask(k.size)

    def power_hat(v: np.ndarray, power: int) -> np.ndarray:
        u = np.fft.ifft(v).real
        result = np.fft.fft(u**power)
        result[~mask] = 0.0
        return result

    if family is Family.ADVECTION_DIFFUSION:
        c = float(rng.uniform(0.25, 0.55))
        d = float(rng.uniform(0.025, 0.050))
        return -d * k**2 - 1j * c * k, lambda v: np.zeros_like(v), (("u_x", -c), ("u_xx", d))
    if family is Family.VISCOUS_BURGERS:
        a = float(rng.uniform(0.75, 1.25))
        nu = float(rng.uniform(0.035, 0.070))
        return -nu * k**2, lambda v: -0.5j * a * k * power_hat(v, 2), (("u_xx", nu), ("u u_x", -a))
    if family is Family.FISHER_KPP:
        r = float(rng.uniform(0.8, 1.4))
        d = float(rng.uniform(0.020, 0.045))
        return -d * k**2, lambda v: r * (v - power_hat(v, 2)), (("u", r), ("u^2", -r), ("u_xx", d))
    if family is Family.ALLEN_CAHN:
        r = float(rng.uniform(0.7, 1.2))
        d = float(rng.uniform(0.020, 0.045))
        return -d * k**2, lambda v: r * (v - power_hat(v, 3)), (("u", r), ("u^3", -r), ("u_xx", d))
    if family is Family.KDV:
        a = float(rng.uniform(1.5, 3.0))
        delta = float(rng.uniform(0.18, 0.35))
        return 1j * delta * k**3, lambda v: -0.5j * a * k * power_hat(v, 2), (("u_xxx", -delta), ("u u_x", -a))
    if family is Family.HYPERDIFFUSION_OOG:
        gamma = float(rng.uniform(0.018, 0.030))
        r = float(rng.uniform(0.15, 0.30))
        return -gamma * k**4, lambda v: r * (v - power_hat(v, 3)), (("u", r), ("u^3", -r), ("u_xxxx", -gamma))
    raise ValueError(f"unsupported family {family}")


def _add_noise(
    clean: np.ndarray,
    relative_noise: float,
    noise_model: str,
    noise_seed: int,
) -> np.ndarray:
    if relative_noise == 0:
        return clean.copy()
    rng = np.random.default_rng(noise_seed)
    raw = rng.normal(size=clean.shape)
    if noise_model == "correlated":
        freq_t = np.fft.fftfreq(clean.shape[0])[:, None]
        freq_x = np.fft.fftfreq(clean.shape[1])[None, :]
        filt = np.exp(-0.5 * ((freq_t / 0.20) ** 2 + (freq_x / 0.15) ** 2))
        raw = np.fft.ifft2(np.fft.fft2(raw) * filt).real
    elif noise_model != "iid":
        raise ValueError("noise_model must be iid or correlated")
    raw /= max(float(np.std(raw)), 1e-15)
    scale = max(float(np.std(clean)), 1e-12)
    return clean + relative_noise * scale * raw


def simulate(
    family: Family,
    action: Action,
    episode_seed: int,
    public_action_seed: int,
    relative_noise: float = 0.0,
    noise_model: str = "iid",
    config: SimulationConfig | None = None,
) -> tuple[Trajectory, HiddenTruth]:
    cfg = config or SimulationConfig()
    steps, stride = cfg.validate()
    dx = cfg.length / cfg.nx
    x = np.arange(cfg.nx, dtype=np.float64) * dx
    k = 2.0 * np.pi * np.fft.fftfreq(cfg.nx, d=dx)
    initial = _initial_condition(action, x, public_action_seed)
    linear, nonlinear, coefficients = _problem(
        family, k, np.random.default_rng(episode_seed)
    )
    clean = integrate_etdrk4(
        initial, linear, nonlinear, cfg.internal_dt, steps, stride
    )
    noise_seed = int(np.random.SeedSequence([episode_seed, public_action_seed, 991]).generate_state(1)[0])
    observed = _add_noise(clean, relative_noise, noise_model, noise_seed)
    t = np.linspace(0.0, cfg.t_final, steps // stride + 1)
    trajectory = Trajectory(
        sample_id=f"e{episode_seed}-{action.name}",
        action=action.name,
        x=x,
        t=t,
        u=observed,
        noise_model=noise_model,
        relative_noise=float(relative_noise),
    )
    truth = HiddenTruth(
        family=family,
        support=TRUE_SUPPORTS[family],
        coefficients=coefficients,
        seed=episode_seed,
    )
    return trajectory, truth
