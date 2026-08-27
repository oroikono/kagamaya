"""Pseudo-spectral periodic trajectory producer.

This module never imports the finite-difference producer.  The manufactured
forcing is owner supplied; all runtime spatial operators are computed from the
current numerical state by FFT.
"""

from __future__ import annotations

import numpy as np

from .contract import DOMAIN_LENGTH
from .owner_truth import (
    OWNER_CASE_BY_ID,
    analytic_state,
    opaque_episode_id,
    owner_forcing,
)
from .trajectory import Trajectory


ORIGIN_ID = "spectral-rk4-v1"


def _wave_numbers(n: int) -> tuple[np.ndarray, np.ndarray]:
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=DOMAIN_LENGTH / n)
    return k[:, None], k[None, :]


def _state_fields(u: np.ndarray, v: np.ndarray | None = None) -> dict[str, np.ndarray]:
    kx, ky = _wave_numbers(u.shape[0])
    uh = np.fft.fft2(u)
    ux = np.fft.ifft2(1j * kx * uh).real
    uy = np.fft.ifft2(1j * ky * uh).real
    uxx = np.fft.ifft2(-(kx**2) * uh).real
    uyy = np.fft.ifft2(-(ky**2) * uh).real
    uxy = np.fft.ifft2(-(kx * ky) * uh).real
    biharm = np.fft.ifft2((kx**2 + ky**2) ** 2 * uh).real
    fields = {
        "u": u, "u2": u**2, "u3": u**3, "ux": ux, "uy": uy,
        "uxx": uxx, "uyy": uyy, "uxy": uxy, "biharm": biharm,
        "uux": u * ux, "uuy": u * uy,
    }
    if v is not None:
        vh = np.fft.fft2(v)
        vx = np.fft.ifft2(1j * kx * vh).real
        vy = np.fft.ifft2(1j * ky * vh).real
        fields.update({
            "v": v, "v2": v**2, "uv": u * v, "vx": vx, "vy": vy,
            "vxx": np.fft.ifft2(-(kx**2) * vh).real,
            "vyy": np.fft.ifft2(-(ky**2) * vh).real,
            "vxy": np.fft.ifft2(-(kx * ky) * vh).real,
        })
    return fields


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
        kx, ky = _wave_numbers(fields["u"].shape[0])
        multiplier = -(kx**2 + ky**2) ** 0.75
        multiplier[0, 0] = 0.0
        return 0.048 * np.fft.ifft2(multiplier * np.fft.fft2(fields["u"])).real
    if mechanism == "unobserved_v_coupling":
        tt = np.asarray(t)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        return -0.22 * fields["u"] + 0.37 * exact["v"] + 0.04 * (fields["uxx"] + fields["uyy"])
    if mechanism == "gaussian_periodic_convolution":
        kx, ky = _wave_numbers(fields["u"].shape[0])
        multiplier = np.exp(-0.5 * 0.12 * (kx**2 + ky**2)) - 1.0
        nonlocal_term = np.fft.ifft2(multiplier * np.fft.fft2(fields["u"])).real
        return 0.31 * nonlocal_term - 0.12 * fields["u"]
    raise ValueError(f"unsupported omitted mechanism: {mechanism}")


def _simulate_fourth_order(case_id: str, seed: int, *, n: int,
                           snapshot_count: int, internal_steps: int) -> Trajectory:
    """Crank--Nicolson in the spectral operator basis for the stiff linear case."""
    if internal_steps % (snapshot_count - 1):
        raise ValueError("internal_steps must be divisible by snapshot_count-1")
    case = OWNER_CASE_BY_ID[case_id]
    x = DOMAIN_LENGTH * np.arange(n) / n
    y = DOMAIN_LENGTH * np.arange(n) / n
    x2, y2 = x[:, None], y[None, :]
    initial = analytic_state(seed, x2, y2, np.asarray(0.0)[None, None])
    state_hat = np.fft.fft2(initial["u"])
    kx, ky = _wave_numbers(n)
    k2 = kx**2 + ky**2
    linear = -0.22 + 0.081 * k2 - 0.0045 * k2**2
    dt = 1.0 / internal_steps
    numerator = 1.0 + 0.5 * dt * linear
    denominator = 1.0 - 0.5 * dt * linear
    stride = internal_steps // (snapshot_count - 1)
    times = [0.0]
    us = [np.fft.ifft2(state_hat).real]
    for step in range(internal_steps):
        midpoint = (step + 0.5) * dt
        tt = np.asarray(midpoint)[None, None]
        exact = analytic_state(seed, x2, y2, tt)
        forcing = owner_forcing(case, seed, exact, x2, y2, tt)
        state_hat = (numerator * state_hat + dt * np.fft.fft2(forcing)) / denominator
        if (step + 1) % stride == 0:
            times.append((step + 1) * dt)
            us.append(np.fft.ifft2(state_hat).real)
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
             internal_steps: int = 480) -> Trajectory:
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
        fields = _state_fields(u_state, v_state)
        if coupled:
            ru = _linear_combination(fields, u_case.coefficient_map) + forcing_for(u_case, time)
            rv = _linear_combination(fields, v_case.coefficient_map) + forcing_for(v_case, time)
            return ru, rv
        if case.truth_kind == "in_library":
            dynamic = _linear_combination(fields, case.coefficient_map)
        else:
            dynamic = _omitted_rhs(case_id, seed, fields, x2, y2, time)
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
