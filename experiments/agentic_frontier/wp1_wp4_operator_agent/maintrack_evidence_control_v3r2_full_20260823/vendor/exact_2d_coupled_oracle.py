"""Closed-form coupled 2-D weak-form development oracle.

The owner truth is a pair of analytic Fourier-polynomial fields.  No PDE
solver is used to create labels.  Each target equation is evaluated against
the same complete, declared two-field operator dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_2d_oracle import Mode, _poly_and_derivative, _spatial_test, _temporal_test


COUPLED_ATOMS = (
    "u", "v", "u2", "v2", "uv",
    "ux", "uy", "vx", "vy",
    "uxx", "uyy", "vxx", "vyy", "uxy", "vxy",
)


V_MODES = (
    Mode(0.53, (0.41, -0.18, 0.06), 1, 3, -0.21, "cos"),
    Mode(-0.37, (0.62, 0.09, -0.04), 3, 1, 0.35, "sin"),
    Mode(0.29, (0.28, 0.21, 0.02), 2, 4, -0.49, "cos"),
    Mode(-0.24, (0.49, -0.12, 0.05), 4, 2, 0.58, "sin"),
    Mode(0.20, (0.19, 0.17, -0.03), 5, 1, -0.73, "cos"),
    Mode(0.17, (0.36, -0.05, 0.01), 1, 5, 0.66, "sin"),
    Mode(-0.15, (0.24, 0.13, 0.04), 4, 5, -0.11, "cos"),
    Mode(0.13, (0.44, -0.09, 0.03), 5, 3, 0.27, "sin"),
    Mode(-0.11, (0.33, 0.06, -0.02), 3, 5, -0.62, "cos"),
    Mode(0.09, (0.16, -0.14, 0.06), 6, 1, 0.42, "sin"),
)


@dataclass(frozen=True)
class CoupledFamily:
    name: str
    u_equation: dict[str, float]
    v_equation: dict[str, float]


COUPLED_FAMILIES = (
    CoupledFamily(
        "coupled_reaction_diffusion",
        {"u": -0.31, "v": 0.24, "uxx": 0.047, "uyy": 0.081},
        {"u": -0.19, "v": 0.28, "vxx": 0.063, "vyy": 0.039},
    ),
    CoupledFamily(
        "cross_advection_diffusion",
        {"v": -0.22, "ux": 0.17, "vy": -0.13, "uxx": 0.026, "uyy": 0.041},
        {"u": 0.21, "vx": -0.16, "uy": 0.12, "vxx": 0.034, "vyy": 0.052},
    ),
    CoupledFamily(
        "nonlinear_coupling",
        {"u": -0.18, "uv": 0.33, "uxx": 0.031, "uyy": 0.022},
        {"v": -0.14, "u2": -0.27, "vxx": 0.043, "vyy": 0.057},
    ),
)


def _field_from_modes(modes: tuple[Mode, ...], x: np.ndarray, y: np.ndarray,
                      t: np.ndarray) -> dict[str, np.ndarray]:
    shape = np.broadcast_shapes(x.shape, y.shape, t.shape)
    out = {key: np.zeros(shape, dtype=float) for key in ("value", "dt", "x", "y", "xx", "yy", "xy")}
    for mode in modes:
        p, dp = _poly_and_derivative(mode.time_coefficients, t)
        z = mode.kx * x + mode.ky * y + mode.phase
        if mode.trig == "sin":
            base, first = np.sin(z), np.cos(z)
        else:
            base, first = np.cos(z), -np.sin(z)
        a = mode.amplitude
        out["value"] += a * p * base
        out["dt"] += a * dp * base
        out["x"] += a * p * mode.kx * first
        out["y"] += a * p * mode.ky * first
        out["xx"] += -a * p * mode.kx**2 * base
        out["yy"] += -a * p * mode.ky**2 * base
        out["xy"] += -a * p * mode.kx * mode.ky * base
    return out


def analytic_coupled_fields(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> dict[str, np.ndarray]:
    from .exact_2d_oracle import MODES

    uf = _field_from_modes(MODES, x, y, t)
    vf = _field_from_modes(V_MODES, x, y, t)
    u, v = uf["value"], vf["value"]
    return {
        "u": u, "v": v, "ut": uf["dt"], "vt": vf["dt"],
        "u2": u**2, "v2": v**2, "uv": u * v,
        "ux": uf["x"], "uy": uf["y"], "vx": vf["x"], "vy": vf["y"],
        "uxx": uf["xx"], "uyy": uf["yy"], "vxx": vf["xx"], "vyy": vf["yy"],
        "uxy": uf["xy"], "vxy": vf["xy"],
    }


def _forcing(fields: dict[str, np.ndarray], target: str, coefficients: dict[str, float]) -> np.ndarray:
    rhs = np.zeros_like(fields[target])
    for atom, coefficient in coefficients.items():
        rhs += coefficient * fields[atom]
    return fields[f"{target}t"] - rhs


def weak_system(family: CoupledFamily, target: str, *, nx: int = 64, ny: int = 64,
                nt: int = 20, atoms: tuple[str, ...] = COUPLED_ATOMS
                ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if target not in {"u", "v"}:
        raise ValueError("target must be u or v")
    coefficients = family.u_equation if target == "u" else family.v_equation
    xs = (2.0 * np.pi / nx) * np.arange(nx)
    ys = (2.0 * np.pi / ny) * np.arange(ny)
    nodes, weights = np.polynomial.legendre.leggauss(nt)
    ts = (nodes + 1.0) / 2.0
    tw = weights / 2.0
    t = ts[:, None, None]
    x = xs[None, :, None]
    y = ys[None, None, :]
    fields = analytic_coupled_fields(x, y, t)
    f = _forcing(fields, target, coefficients)
    volume_weights = tw[:, None, None] * (2.0 * np.pi / nx) * (2.0 * np.pi / ny)

    rows: list[list[float]] = []
    targets: list[float] = []
    for q in range(4):
        for kx in range(0, 8):
            for ky in range(0, 8):
                if kx == 0 and ky == 0:
                    continue
                for kind, phase in (("sin", 0.17), ("cos", -0.29)):
                    temporal, temporal_dt = _temporal_test(q, t)
                    spatial = _spatial_test(kind, kx, ky, phase, x, y)
                    phi = temporal * spatial["s"]
                    target_row = np.sum(
                        (-fields[target] * temporal_dt * spatial["s"] - f * phi) * volume_weights
                    )
                    row: list[float] = []
                    for atom in atoms:
                        base = "u" if atom.startswith("u") else "v"
                        if atom in {"ux", "vx"}:
                            integrand = -fields[base] * temporal * spatial["sx"]
                        elif atom in {"uy", "vy"}:
                            integrand = -fields[base] * temporal * spatial["sy"]
                        elif atom in {"uxx", "vxx"}:
                            integrand = fields[base] * temporal * spatial["sxx"]
                        elif atom in {"uyy", "vyy"}:
                            integrand = fields[base] * temporal * spatial["syy"]
                        elif atom in {"uxy", "vxy"}:
                            integrand = fields[base] * temporal * spatial["sxy"]
                        else:
                            integrand = fields[atom] * phi
                        row.append(float(np.sum(integrand * volume_weights)))
                    rows.append(row)
                    targets.append(float(target_row))
    return np.asarray(rows), np.asarray(targets), atoms


def run_development() -> list[dict]:
    from .exact_2d_oracle import fit_support

    records: list[dict] = []
    for family in COUPLED_FAMILIES:
        for target in ("u", "v"):
            coefficients = family.u_equation if target == "u" else family.v_equation
            x, y, atoms = weak_system(family, target)
            fit = fit_support(x, y, atoms)
            truth = tuple(atom for atom in atoms if atom in coefficients)
            records.append({
                "family": family.name,
                "target": target,
                "truth_support": truth,
                "fit": fit,
                "support_correct": fit["support"] == truth,
            })
    return records

