"""Self-contained Fourier ETDRK4 utilities for the v4.1 deployment package."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _phi_functions(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=np.complex128)
    small = np.abs(z) < 1e-5
    phi1 = np.empty_like(z)
    phi2 = np.empty_like(z)
    phi3 = np.empty_like(z)
    zs = z[small]
    phi1[small] = 1.0 + zs / 2.0 + zs**2 / 6.0 + zs**3 / 24.0 + zs**4 / 120.0
    phi2[small] = 0.5 + zs / 6.0 + zs**2 / 24.0 + zs**3 / 120.0 + zs**4 / 720.0
    phi3[small] = (
        1.0 / 6.0
        + zs / 24.0
        + zs**2 / 120.0
        + zs**3 / 720.0
        + zs**4 / 5040.0
    )
    zl = z[~small]
    expm1 = np.expm1(zl)
    phi1[~small] = expm1 / zl
    phi2[~small] = (expm1 - zl) / zl**2
    phi3[~small] = (expm1 - zl - 0.5 * zl**2) / zl**3
    return phi1, phi2, phi3


def _etdrk4_coefficients(
    linear: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = dt * np.asarray(linear, dtype=np.complex128)
    phi1, phi2, phi3 = _phi_functions(z)
    half_phi1, _, _ = _phi_functions(0.5 * z)
    e = np.exp(z)
    e2 = np.exp(0.5 * z)
    q = 0.5 * dt * half_phi1
    f1 = dt * (phi1 - 3.0 * phi2 + 4.0 * phi3)
    f2 = dt * (phi2 - 2.0 * phi3)
    f3 = dt * (-phi2 + 4.0 * phi3)
    return e, e2, q, f1, f2, f3


def integrate_etdrk4(
    initial: np.ndarray,
    linear: np.ndarray,
    nonlinear: Callable[[np.ndarray], np.ndarray],
    dt: float,
    n_steps: int,
    save_stride: int,
) -> np.ndarray:
    e, e2, q, f1, f2, f3 = _etdrk4_coefficients(linear, dt)
    v = np.fft.fft(np.asarray(initial, dtype=np.float64))
    snapshots = [np.asarray(initial, dtype=np.float64).copy()]
    for step in range(1, n_steps + 1):
        nv = nonlinear(v)
        a = e2 * v + q * nv
        na = nonlinear(a)
        b = e2 * v + q * na
        nb = nonlinear(b)
        c = e2 * a + q * (2.0 * nb - nv)
        nc = nonlinear(c)
        v = e * v + f1 * nv + 2.0 * f2 * (na + nb) + f3 * nc
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"ETDRK4 produced a non-finite state at step {step}")
        if step % save_stride == 0:
            state = np.fft.ifft(v)
            imaginary_scale = float(np.max(np.abs(state.imag)))
            real_scale = max(float(np.max(np.abs(state.real))), 1.0)
            if imaginary_scale > 1e-9 * real_scale:
                raise FloatingPointError(
                    f"loss of conjugate symmetry at step {step}: {imaginary_scale}"
                )
            snapshots.append(state.real.copy())
    return np.asarray(snapshots, dtype=np.float64)


def dealias_mask(nx: int) -> np.ndarray:
    modes = np.fft.fftfreq(nx) * nx
    return np.abs(modes) <= nx // 3
