"""Finite-difference method-of-lines outcome producer.

This module intentionally does not import the spectral simulator.  It repeats
the public PDE specification with periodic finite differences and an implicit
SciPy time integrator so that producer-method sensitivity can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import solve_ivp


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
class ProducerConfig:
    nx: int
    t_final: float
    saved_nodes: int
    name: str


BASELINE = ProducerConfig(64, 0.24, 25, "baseline")
GRADIENT = ProducerConfig(96, 0.12, 49, "gradient_space")
NESTED_NODES = (7, 13, 25, 49)


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


def finite_difference_weights(derivative: int, radius: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Centered weights exact on polynomials through degree ``2*radius``."""

    if derivative < 1 or derivative > 4 or radius < derivative:
        raise ValueError("unsupported derivative/radius")
    offsets = np.arange(-radius, radius + 1, dtype=float)
    moments = np.vstack([offsets**power for power in range(offsets.size)])
    target = np.zeros(offsets.size)
    target[derivative] = math.factorial(derivative)
    return offsets.astype(int), np.linalg.solve(moments, target)


@lru_cache(maxsize=None)
def derivative_matrix(nx: int, derivative: int, length: float = 2.0 * np.pi) -> np.ndarray:
    if nx < 16:
        raise ValueError("nx must be at least 16")
    offsets, weights = finite_difference_weights(derivative)
    dx = length / nx
    matrix = np.zeros((nx, nx), dtype=float)
    rows = np.arange(nx)
    for offset, weight in zip(offsets, weights):
        matrix[rows, (rows + offset) % nx] += weight / dx**derivative
    return matrix


def initial_condition(config: ProducerConfig, x: np.ndarray, public_seed: int) -> np.ndarray:
    phases = np.random.default_rng(public_seed).uniform(0.0, 2.0 * np.pi, 3)
    if config.name == "baseline":
        return 0.18 + 0.035 * np.cos(x + phases[0]) + 0.012 * np.sin(2.0 * x + phases[1])
    if config.name == "gradient_space":
        return (
            0.42 * np.sin(2.0 * x + phases[0])
            + 0.24 * np.cos(4.0 * x + phases[1])
            + 0.10 * np.sin(6.0 * x + phases[2])
        )
    raise ValueError(config.name)


def draw_parameters(family: Family, episode_seed: int) -> tuple[dict[str, float], tuple[tuple[str, float], ...]]:
    rng = np.random.default_rng(episode_seed)
    if family is Family.ADVECTION_DIFFUSION:
        c, d = float(rng.uniform(0.25, 0.55)), float(rng.uniform(0.025, 0.050))
        return {"c": c, "d": d}, (("u_x", -c), ("u_xx", d))
    if family is Family.VISCOUS_BURGERS:
        a, nu = float(rng.uniform(0.75, 1.25)), float(rng.uniform(0.035, 0.070))
        return {"a": a, "nu": nu}, (("u_xx", nu), ("u u_x", -a))
    if family is Family.FISHER_KPP:
        r, d = float(rng.uniform(0.8, 1.4)), float(rng.uniform(0.020, 0.045))
        return {"r": r, "d": d}, (("u", r), ("u^2", -r), ("u_xx", d))
    if family is Family.ALLEN_CAHN:
        r, d = float(rng.uniform(0.7, 1.2)), float(rng.uniform(0.020, 0.045))
        return {"r": r, "d": d}, (("u", r), ("u^3", -r), ("u_xx", d))
    if family is Family.KDV:
        a, delta = float(rng.uniform(1.5, 3.0)), float(rng.uniform(0.18, 0.35))
        return {"a": a, "delta": delta}, (("u_xxx", -delta), ("u u_x", -a))
    if family is Family.HYPERDIFFUSION_OOG:
        gamma, r = float(rng.uniform(0.018, 0.030)), float(rng.uniform(0.15, 0.30))
        return {"gamma": gamma, "r": r}, (("u", r), ("u^3", -r), ("u_xxxx", -gamma))
    raise ValueError(family)


def _rhs_and_jacobian(family: Family, parameters: dict[str, float], nx: int, *, advection_sign: float = 1.0):
    d1 = derivative_matrix(nx, 1)
    d2 = derivative_matrix(nx, 2)
    d3 = derivative_matrix(nx, 3)
    d4 = derivative_matrix(nx, 4)
    if family is Family.ADVECTION_DIFFUSION:
        operator = -advection_sign * parameters["c"] * d1 + parameters["d"] * d2
        return lambda _t, u: operator @ u, lambda _t, _u: operator
    if family is Family.VISCOUS_BURGERS:
        a, nu = parameters["a"], parameters["nu"]
        def burgers_rhs(_t, u):
            return -a * u * (d1 @ u) + nu * (d2 @ u)
        def burgers_jac(_t, u):
            return -a * (np.diag(d1 @ u) + np.diag(u) @ d1) + nu * d2
        return burgers_rhs, burgers_jac
    if family is Family.FISHER_KPP:
        r, diffusion = parameters["r"], parameters["d"]
        return (
            lambda _t, u: diffusion * (d2 @ u) + r * (u - u**2),
            lambda _t, u: diffusion * d2 + np.diag(r * (1.0 - 2.0 * u)),
        )
    if family is Family.ALLEN_CAHN:
        r, diffusion = parameters["r"], parameters["d"]
        return (
            lambda _t, u: diffusion * (d2 @ u) + r * (u - u**3),
            lambda _t, u: diffusion * d2 + np.diag(r * (1.0 - 3.0 * u**2)),
        )
    if family is Family.KDV:
        a, delta = parameters["a"], parameters["delta"]
        def kdv_rhs(_t, u):
            return -delta * (d3 @ u) - a * u * (d1 @ u)
        def kdv_jac(_t, u):
            return -delta * d3 - a * (np.diag(d1 @ u) + np.diag(u) @ d1)
        return kdv_rhs, kdv_jac
    if family is Family.HYPERDIFFUSION_OOG:
        gamma, r = parameters["gamma"], parameters["r"]
        return (
            lambda _t, u: -gamma * (d4 @ u) + r * (u - u**3),
            lambda _t, u: -gamma * d4 + np.diag(r * (1.0 - 3.0 * u**2)),
        )
    raise ValueError(family)


def solve_clean(
    family: Family,
    config: ProducerConfig,
    episode_seed: int,
    public_seed: int,
    *,
    advection_sign: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HiddenTruth]:
    """Solve one clean PDE trajectory with BDF on a finite-difference MOL."""

    x = np.arange(config.nx, dtype=float) * (2.0 * np.pi / config.nx)
    t = np.linspace(0.0, config.t_final, config.saved_nodes)
    u0 = initial_condition(config, x, public_seed)
    parameters, coefficients = draw_parameters(family, episode_seed)
    rhs, jac = _rhs_and_jacobian(family, parameters, config.nx, advection_sign=advection_sign)
    solution = solve_ivp(
        rhs,
        (0.0, config.t_final),
        u0,
        t_eval=t,
        method="BDF",
        jac=jac,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.001,
    )
    if not solution.success or solution.y.shape != (config.nx, config.saved_nodes):
        raise RuntimeError(f"BDF solve failed for {family.value}/{config.name}: {solution.message}")
    clean = solution.y.T
    if not np.all(np.isfinite(clean)):
        raise FloatingPointError(f"nonfinite BDF trajectory for {family.value}/{config.name}")
    truth = HiddenTruth(family, TRUE_SUPPORTS[family], coefficients, episode_seed)
    return x, t, clean, truth


def add_noise(clean: np.ndarray, level: float, model: str, seed: int) -> np.ndarray:
    if level == 0.0:
        return clean.copy()
    raw = np.random.default_rng(seed).normal(size=clean.shape)
    if model == "correlated":
        ft = np.fft.fftfreq(clean.shape[0])[:, None]
        fx = np.fft.fftfreq(clean.shape[1])[None, :]
        kernel = np.exp(-0.5 * ((ft / 0.20) ** 2 + (fx / 0.15) ** 2))
        raw = np.fft.ifft2(np.fft.fft2(raw) * kernel).real
    elif model != "iid":
        raise ValueError(model)
    raw /= max(float(np.std(raw)), 1e-15)
    return clean + level * max(float(np.std(clean)), 1e-12) * raw


def observed_trajectory(
    config: ProducerConfig,
    episode_seed: int,
    public_seed: int,
    clean: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    noise: float,
    noise_model: str,
) -> Trajectory:
    noise_seed = int(
        np.random.SeedSequence(
            [episode_seed, public_seed, 991, sum(map(ord, config.name))]
        ).generate_state(1)[0]
    )
    observed = add_noise(clean, noise, noise_model, noise_seed)
    return Trajectory(
        f"fd-e{episode_seed}-{config.name}-{config.saved_nodes}",
        config.name,
        x.copy(),
        t.copy(),
        observed,
        noise_model,
        noise,
    )


def subsample(fine: Trajectory, nodes: int) -> Trajectory:
    if nodes not in NESTED_NODES or (fine.t.size - 1) % (nodes - 1):
        raise ValueError("requested grid is not nested")
    stride = (fine.t.size - 1) // (nodes - 1)
    index = np.arange(0, fine.t.size, stride)
    return Trajectory(
        f"{fine.sample_id}-n{nodes}",
        fine.action,
        fine.x.copy(),
        fine.t[index].copy(),
        fine.u[index].copy(),
        fine.noise_model,
        fine.relative_noise,
    )


def clean_pair(family: Family, episode_seed: int, public_seed: int):
    bx, bt, bu, truth = solve_clean(family, BASELINE, episode_seed, public_seed)
    gx, gt, gu, truth_gradient = solve_clean(family, GRADIENT, episode_seed, public_seed)
    if truth != truth_gradient:
        raise AssertionError("baseline/channel truth mismatch")
    return (bx, bt, bu), (gx, gt, gu), truth
