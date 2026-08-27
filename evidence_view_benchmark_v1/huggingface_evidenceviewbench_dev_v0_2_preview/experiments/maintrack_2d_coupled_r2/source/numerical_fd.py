"""Centered finite-difference trajectory producer with dense nonlocal operators.

This implementation never imports the spectral producer. Fractional and
Gaussian nonlocal terms use an eigendecomposition of the periodic centered
difference Laplacian, not FFT differentiation.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .contract import DOMAIN_LENGTH
from .owner_truth import (
    OWNER_CASE_BY_ID,
    analytic_state,
    opaque_episode_id,
    owner_forcing,
)
from .trajectory import Trajectory


ORIGIN_ID = "centered-fd-rk4-v1"


def _dx(u: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * h)


def _dy(u: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * h)


def _dxx(u: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / h**2


def _dyy(u: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / h**2


def _dxy(u: np.ndarray, h: float) -> np.ndarray:
    return (np.roll(np.roll(u, -1, 0), -1, 1)
            - np.roll(np.roll(u, -1, 0), 1, 1)
            - np.roll(np.roll(u, 1, 0), -1, 1)
            + np.roll(np.roll(u, 1, 0), 1, 1)) / (4.0 * h**2)


def _state_fields(u: np.ndarray, h: float, v: np.ndarray | None = None) -> dict[str, np.ndarray]:
    ux, uy = _dx(u, h), _dy(u, h)
    uxx, uyy = _dxx(u, h), _dyy(u, h)
    lap = uxx + uyy
    fields = {
        "u": u, "u2": u**2, "u3": u**3, "ux": ux, "uy": uy,
        "uxx": uxx, "uyy": uyy, "uxy": _dxy(u, h),
        "biharm": _dxx(lap, h) + _dyy(lap, h),
        "uux": u * ux, "uuy": u * uy,
    }
    if v is not None:
        fields.update({
            "v": v, "v2": v**2, "uv": u * v,
            "vx": _dx(v, h), "vy": _dy(v, h),
            "vxx": _dxx(v, h), "vyy": _dyy(v, h), "vxy": _dxy(v, h),
        })
    return fields


@lru_cache(maxsize=8)
def _laplacian_eigensystem(n: int) -> tuple[np.ndarray, np.ndarray]:
    h = DOMAIN_LENGTH / n
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i, i] = -2.0 / h**2
        matrix[i, (i - 1) % n] = 1.0 / h**2
        matrix[i, (i + 1) % n] = 1.0 / h**2
    values, vectors = np.linalg.eigh(matrix)
    return values, vectors


def _apply_laplacian_function(u: np.ndarray, function) -> np.ndarray:
    values, vectors = _laplacian_eigensystem(u.shape[0])
    coefficients = vectors.T @ u @ vectors
    lambda_2d = values[:, None] + values[None, :]
    return vectors @ (function(lambda_2d) * coefficients) @ vectors.T


def _linear_combination(fields: dict[str, np.ndarray], coefficients: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(fields["u"])
    for atom, coefficient in coefficients.items():
        out += coefficient * fields[atom]
    return out


def _omitted_rhs(case_id: str, seed: int, fields: dict[str, np.ndarray],
                 x2: np.ndarray, y2: np.ndarray, t: float) -> np.ndarray:
    case = OWNER_CASE_BY_ID[case_id]
    mechanism = case.omitted_mechanism
    if mechanism == "divergence_variable_coefficient":
        a = 1.0 + 0.25 * np.sin(x2) + 0.15 * np.cos(y2)
        ax = 0.25 * np.cos(x2)
        ay = -0.15 * np.sin(y2)
        return 0.055 * (a * (fields["uxx"] + fields["uyy"])
                        + ax * fields["ux"] + ay * fields["uy"])
    if mechanism == "fractional_laplacian_alpha_1p5":
        return 0.048 * _apply_laplacian_function(
            fields["u"], lambda lam: -np.maximum(-lam, 0.0) ** 0.75)
    if mechanism == "unobserved_v_coupling":
        tt = np.asarray(t)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        return -0.22 * fields["u"] + 0.37 * exact["v"] + 0.04 * (fields["uxx"] + fields["uyy"])
    if mechanism == "gaussian_periodic_convolution":
        filtered_minus_u = _apply_laplacian_function(
            fields["u"], lambda lam: np.exp(0.5 * 0.12 * lam) - 1.0)
        return 0.31 * filtered_minus_u - 0.12 * fields["u"]
    raise ValueError(f"unsupported omitted mechanism: {mechanism}")


def _simulate_fourth_order(case_id: str, seed: int, *, n: int,
                           snapshot_count: int, internal_steps: int) -> Trajectory:
    """Crank--Nicolson in the centered-difference eigenbasis."""
    if internal_steps % (snapshot_count - 1):
        raise ValueError("internal_steps must be divisible by snapshot_count-1")
    case = OWNER_CASE_BY_ID[case_id]
    x = DOMAIN_LENGTH * np.arange(n) / n
    y = DOMAIN_LENGTH * np.arange(n) / n
    x2, y2 = x[:, None], y[None, :]
    initial = analytic_state(seed, x2, y2, np.asarray(0.0)[None, None])
    values, vectors = _laplacian_eigensystem(n)
    lam = values[:, None] + values[None, :]
    linear = -0.22 - 0.081 * lam - 0.0045 * lam**2
    state_hat = vectors.T @ initial["u"] @ vectors
    dt = 1.0 / internal_steps
    numerator = 1.0 + 0.5 * dt * linear
    denominator = 1.0 - 0.5 * dt * linear
    stride = internal_steps // (snapshot_count - 1)
    times = [0.0]
    us = [vectors @ state_hat @ vectors.T]
    for step in range(internal_steps):
        midpoint = (step + 0.5) * dt
        tt = np.asarray(midpoint)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        forcing = owner_forcing(case, seed, exact, x2, y2, tt)
        forcing_hat = vectors.T @ forcing @ vectors
        state_hat = (numerator * state_hat + dt * forcing_hat) / denominator
        if (step + 1) % stride == 0:
            times.append((step + 1) * dt)
            us.append(vectors @ state_hat @ vectors.T)
    time_array = np.asarray(times)
    u_array = np.asarray(us)
    forcing_rows = []
    exact_error = 0.0
    for index, time in enumerate(time_array):
        tt = np.asarray(time)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        exact_error = max(exact_error, float(np.linalg.norm(u_array[index] - exact["u"])
                                             / max(np.linalg.norm(exact["u"]), 1e-15)))
        forcing_rows.append(owner_forcing(case, seed, exact, x2, y2, tt))
    trajectory = Trajectory(
        episode_id=opaque_episode_id(case_id, seed, ORIGIN_ID), origin=ORIGIN_ID,
        case_id=case_id, seed=seed, target="u", times=time_array, x=x, y=y,
        u=u_array, v=None, forcing=np.asarray(forcing_rows),
        convergence_error=exact_error,
    )
    trajectory.validate()
    return trajectory


def simulate(case_id: str, seed: int, *, n: int = 36, snapshot_count: int = 25,
             internal_steps: int = 720) -> Trajectory:
    if case_id == "S-fourth_order":
        return _simulate_fourth_order(
            case_id, seed, n=n, snapshot_count=snapshot_count,
            internal_steps=internal_steps)
    case = OWNER_CASE_BY_ID[case_id]
    if internal_steps % (snapshot_count - 1):
        raise ValueError("internal_steps must be divisible by snapshot_count-1")
    x = DOMAIN_LENGTH * np.arange(n) / n
    y = DOMAIN_LENGTH * np.arange(n) / n
    x2, y2 = x[:, None], y[None, :]
    h = DOMAIN_LENGTH / n
    t0 = np.asarray(0.0)[None, None]
    initial = analytic_state(seed, x2, y2, t0)
    u = np.array(initial["u"], copy=True)
    coupled = case.library_kind == "coupled"
    v = np.array(initial["v"], copy=True) if coupled else None
    if coupled:
        prefix = case_id.rsplit("-", 1)[0]
        u_case = OWNER_CASE_BY_ID[prefix + "-u"]
        v_case = OWNER_CASE_BY_ID[prefix + "-v"]

    def forcing_for(owner_case, time: float) -> np.ndarray:
        tt = np.asarray(time)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        return owner_forcing(owner_case, seed, exact, x2, y2, tt)

    def rhs(time: float, u_state: np.ndarray, v_state: np.ndarray | None):
        fields = _state_fields(u_state, h, v_state)
        if coupled:
            return (_linear_combination(fields, u_case.coefficient_map) + forcing_for(u_case, time),
                    _linear_combination(fields, v_case.coefficient_map) + forcing_for(v_case, time))
        dynamic = (_linear_combination(fields, case.coefficient_map)
                   if case.truth_kind == "in_library"
                   else _omitted_rhs(case_id, seed, fields, x2, y2, time))
        return dynamic + forcing_for(case, time), None

    dt = 1.0 / internal_steps
    stride = internal_steps // (snapshot_count - 1)
    times = [0.0]
    us = [u.copy()]
    vs = [v.copy()] if v is not None else None
    for step in range(internal_steps):
        time = step * dt
        k1u, k1v = rhs(time, u, v)
        k2u, k2v = rhs(time + 0.5 * dt, u + 0.5 * dt * k1u,
                        None if v is None else v + 0.5 * dt * k1v)
        k3u, k3v = rhs(time + 0.5 * dt, u + 0.5 * dt * k2u,
                        None if v is None else v + 0.5 * dt * k2v)
        k4u, k4v = rhs(time + dt, u + dt * k3u,
                        None if v is None else v + dt * k3v)
        u = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        if v is not None:
            v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        if (step + 1) % stride == 0:
            times.append((step + 1) * dt)
            us.append(u.copy())
            if vs is not None:
                vs.append(v.copy())

    time_array = np.asarray(times)
    u_array = np.asarray(us)
    v_array = None if vs is None else np.asarray(vs)
    exact_error = 0.0
    forcing_rows: list[np.ndarray] = []
    for index, time in enumerate(time_array):
        tt = np.asarray(time)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        exact_target = exact[case.target]
        numerical_target = u_array[index] if case.target == "u" else v_array[index]
        exact_error = max(exact_error, float(np.linalg.norm(numerical_target - exact_target)
                                             / max(np.linalg.norm(exact_target), 1e-15)))
        forcing_rows.append(owner_forcing(case, seed, exact, x2, y2, tt))
    trajectory = Trajectory(
        episode_id=opaque_episode_id(case_id, seed, ORIGIN_ID), origin=ORIGIN_ID,
        case_id=case_id, seed=seed, target=case.target, times=time_array,
        x=x, y=y, u=u_array, v=v_array, forcing=np.asarray(forcing_rows),
        convergence_error=exact_error,
    )
    trajectory.validate()
    return trajectory
