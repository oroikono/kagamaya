"""Closed-form 2-D weak-form development oracle.

This module never solves a PDE.  It evaluates a declared analytic field and
its derivatives, constructs the forcing exactly at quadrature nodes, and then
forms weak rows by integration by parts.  It is development infrastructure,
not a continuum certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ATOM_ORDER = (
    "u",
    "u2",
    "u3",
    "ux",
    "uy",
    "uxx",
    "uyy",
    "uxy",
    "biharm",
    "uux",
    "uuy",
)


@dataclass(frozen=True)
class Mode:
    amplitude: float
    time_coefficients: tuple[float, ...]
    kx: int
    ky: int
    phase: float
    trig: str


@dataclass(frozen=True)
class Family:
    name: str
    coefficients: dict[str, float]


MODES = (
    Mode(0.70, (1.0, 0.20, -0.05), 1, 2, 0.13, "sin"),
    Mode(-0.45, (0.35, -0.30, 0.11), 2, 1, -0.31, "cos"),
    Mode(0.32, (0.55, 0.10, 0.07), 3, 2, 0.47, "sin"),
    Mode(0.27, (0.25, -0.13, 0.03), 2, 3, -0.22, "cos"),
    Mode(-0.19, (0.40, 0.08, -0.02), 4, 1, 0.61, "sin"),
    Mode(0.16, (0.31, -0.07, 0.04), 1, 4, -0.52, "cos"),
    Mode(-0.14, (0.22, 0.16, -0.03), 3, 4, 0.38, "sin"),
    Mode(0.12, (0.47, -0.11, 0.02), 5, 2, -0.44, "cos"),
    Mode(0.10, (0.18, 0.09, 0.05), 2, 5, 0.72, "sin"),
    Mode(-0.09, (0.28, -0.04, 0.01), 5, 4, -0.67, "cos"),
)


FAMILIES = (
    Family("anisotropic_adr", {"u": -0.31, "ux": 0.27, "uy": -0.19, "uxx": 0.055, "uyy": 0.091}),
    Family("allen_cahn", {"u": 0.80, "u3": -0.62, "uxx": 0.047, "uyy": 0.047}),
    Family("two_direction_burgers", {"uux": -0.73, "uuy": 0.41, "uxx": 0.036, "uyy": 0.058}),
    Family("fourth_order", {"u": -0.22, "uxx": -0.081, "uyy": -0.081, "biharm": -0.0045}),
    Family("cross_diffusion", {"u": 0.18, "uxy": 0.067, "uxx": 0.029, "uyy": 0.044}),
)


def _poly_and_derivative(coefficients: tuple[float, ...], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.zeros_like(t, dtype=float)
    dp = np.zeros_like(t, dtype=float)
    for degree, coefficient in enumerate(coefficients):
        p += coefficient * t**degree
        if degree:
            dp += degree * coefficient * t ** (degree - 1)
    return p, dp


def analytic_fields(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate u and all candidate operators on broadcast-compatible arrays."""
    out = {key: np.zeros(np.broadcast_shapes(x.shape, y.shape, t.shape), dtype=float) for key in
           ("u", "ut", "ux", "uy", "uxx", "uyy", "uxy", "biharm")}
    for mode in MODES:
        p, dp = _poly_and_derivative(mode.time_coefficients, t)
        z = mode.kx * x + mode.ky * y + mode.phase
        if mode.trig == "sin":
            base, first = np.sin(z), np.cos(z)
        else:
            base, first = np.cos(z), -np.sin(z)
        k2 = mode.kx**2 + mode.ky**2
        a = mode.amplitude
        out["u"] += a * p * base
        out["ut"] += a * dp * base
        out["ux"] += a * p * mode.kx * first
        out["uy"] += a * p * mode.ky * first
        out["uxx"] += -a * p * mode.kx**2 * base
        out["uyy"] += -a * p * mode.ky**2 * base
        out["uxy"] += -a * p * mode.kx * mode.ky * base
        out["biharm"] += a * p * k2**2 * base
    u = out["u"]
    out["u2"] = u**2
    out["u3"] = u**3
    out["uux"] = u * out["ux"]
    out["uuy"] = u * out["uy"]
    return out


def forcing(fields: dict[str, np.ndarray], coefficients: dict[str, float]) -> np.ndarray:
    rhs = np.zeros_like(fields["u"])
    for atom, coefficient in coefficients.items():
        rhs += coefficient * fields[atom]
    return fields["ut"] - rhs


def _spatial_test(kind: str, kx: int, ky: int, phase: float,
                  x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    z = kx * x + ky * y + phase
    if kind == "sin":
        base, first = np.sin(z), np.cos(z)
    else:
        base, first = np.cos(z), -np.sin(z)
    k2 = kx**2 + ky**2
    return {
        "s": base,
        "sx": kx * first,
        "sy": ky * first,
        "sxx": -(kx**2) * base,
        "syy": -(ky**2) * base,
        "sxy": -(kx * ky) * base,
        "bilap_s": k2**2 * base,
    }


def _temporal_test(q: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # b and b' vanish as required only for b endpoints; phi itself vanishes,
    # which is sufficient for one integration by parts in time.
    b = t ** (q + 2) * (1.0 - t) ** 2
    db = (q + 2) * t ** (q + 1) * (1.0 - t) ** 2 - 2.0 * t ** (q + 2) * (1.0 - t)
    return b, db


def weak_system(family: Family, *, nx: int = 64, ny: int = 64, nt: int = 20,
                atoms: Iterable[str] = ATOM_ORDER) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Construct weak X,y using periodic trapezoidal and Gauss-Legendre rules."""
    atoms = tuple(atoms)
    xs = (2.0 * np.pi / nx) * np.arange(nx)
    ys = (2.0 * np.pi / ny) * np.arange(ny)
    nodes, weights = np.polynomial.legendre.leggauss(nt)
    ts = (nodes + 1.0) / 2.0
    tw = weights / 2.0
    t = ts[:, None, None]
    x = xs[None, :, None]
    y = ys[None, None, :]
    fields = analytic_fields(x, y, t)
    f = forcing(fields, family.coefficients)
    volume_weights = tw[:, None, None] * (2.0 * np.pi / nx) * (2.0 * np.pi / ny)

    rows: list[list[float]] = []
    targets: list[float] = []
    test_specs = []
    for q in range(4):
        for kx in range(0, 7):
            for ky in range(0, 7):
                if kx == 0 and ky == 0:
                    continue
                for kind, phase in (("sin", 0.17), ("cos", -0.29)):
                    test_specs.append((q, kx, ky, kind, phase))

    for q, kx, ky, kind, phase in test_specs:
        temporal, temporal_dt = _temporal_test(q, t)
        spatial = _spatial_test(kind, kx, ky, phase, x, y)
        phi = temporal * spatial["s"]
        target = np.sum((-fields["u"] * temporal_dt * spatial["s"] - f * phi) * volume_weights)
        row: list[float] = []
        for atom in atoms:
            if atom == "ux":
                integrand = -fields["u"] * temporal * spatial["sx"]
            elif atom == "uy":
                integrand = -fields["u"] * temporal * spatial["sy"]
            elif atom == "uxx":
                integrand = fields["u"] * temporal * spatial["sxx"]
            elif atom == "uyy":
                integrand = fields["u"] * temporal * spatial["syy"]
            elif atom == "uxy":
                integrand = fields["u"] * temporal * spatial["sxy"]
            elif atom == "biharm":
                integrand = fields["u"] * temporal * spatial["bilap_s"]
            elif atom == "uux":
                integrand = -0.5 * fields["u2"] * temporal * spatial["sx"]
            elif atom == "uuy":
                integrand = -0.5 * fields["u2"] * temporal * spatial["sy"]
            else:
                integrand = fields[atom] * phi
            row.append(float(np.sum(integrand * volume_weights)))
        rows.append(row)
        targets.append(float(target))
    return np.asarray(rows), np.asarray(targets), atoms


def fit_support(x: np.ndarray, y: np.ndarray, atoms: tuple[str, ...], *, coefficient_floor: float = 1e-8) -> dict:
    norms = np.linalg.norm(x, axis=0)
    available = norms > 1e-13
    normalized = x[:, available] / norms[available]
    beta_n, *_ = np.linalg.lstsq(normalized, y, rcond=None)
    beta = np.zeros(len(atoms), dtype=float)
    beta[available] = beta_n / norms[available]
    residual = np.linalg.norm(y - x @ beta) / max(np.linalg.norm(y), 1e-15)
    support = tuple(atom for atom, value in zip(atoms, beta) if abs(value) >= coefficient_floor)
    condition = float(np.linalg.cond(normalized))
    return {
        "coefficients": {atom: float(value) for atom, value in zip(atoms, beta)},
        "support": support,
        "relative_residual": float(residual),
        "condition": condition,
    }


def run_development() -> list[dict]:
    records = []
    for family in FAMILIES:
        x, y, atoms = weak_system(family)
        fit = fit_support(x, y, atoms)
        truth = tuple(atom for atom in atoms if atom in family.coefficients)
        records.append({
            "family": family.name,
            "truth_support": truth,
            "fit": fit,
            "support_correct": fit["support"] == truth,
        })
    return records


if __name__ == "__main__":
    import json
    print(json.dumps(run_development(), indent=2, sort_keys=True))
