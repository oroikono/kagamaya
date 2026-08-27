"""Weak-form evidence construction without differentiating observed fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .manufactured import ObservedField
from .serialization import array_digest, content_digest


TERM_NAMES = ("u", "ux", "uxx", "uux", "uxxx")


@dataclass(frozen=True)
class WeakEvidence:
    seed: int
    row_count: int
    y: np.ndarray
    columns: dict[str, np.ndarray]
    bank_sha256: str
    metadata: dict[str, Any]


def _spatial_factor(x: np.ndarray, kind: str, mode: int, derivative: int) -> np.ndarray:
    if kind == "constant":
        return np.ones_like(x) if derivative == 0 else np.zeros_like(x)
    phase = mode * x
    coefficient = float(mode**derivative)
    shifted = phase + derivative * np.pi / 2.0
    if kind == "cos":
        return coefficient * np.cos(shifted)
    if kind == "sin":
        return coefficient * np.sin(shifted)
    raise AssertionError(f"unsupported spatial factor {kind}")


def build_weak_evidence(
    observed: ObservedField,
    *,
    seed: int,
    row_count: int,
    max_spatial_mode: int,
    max_temporal_mode: int,
) -> WeakEvidence:
    """Project an observed field onto a deterministic seeded weak-test bank."""
    x = np.asarray(observed.x, dtype=float)
    t = np.asarray(observed.t, dtype=float)
    u = np.asarray(observed.u, dtype=float)
    f = np.asarray(observed.f, dtype=float)
    if u.shape != (t.size, x.size) or f.shape != u.shape:
        raise ValueError("u and f must have shape (len(t), len(x))")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(f)):
        raise ValueError("u and f must contain only finite observations")
    if not np.allclose(np.diff(x), np.diff(x)[0]):
        raise ValueError("x must be a uniform periodic grid")
    if not np.allclose(np.diff(t), np.diff(t)[0]):
        raise ValueError("t must be a uniform endpoint-inclusive grid")
    dx = float(x[1] - x[0])
    if not (
        np.isclose(x[0], 0.0)
        and np.isclose(x[-1] + dx, 2.0 * np.pi)
    ):
        raise ValueError("x must contain the unique periodic grid on [0, 2*pi)")
    if not (np.isclose(t[0], 0.0) and np.isclose(t[-1], 1.0)):
        raise ValueError("t must include both endpoints of [0, 1]")
    if row_count < 1 or max_spatial_mode < 0 or max_temporal_mode < 1:
        raise ValueError("weak-bank dimensions and temporal modes must be positive")

    spatial_factors = [("constant", 0)]
    for mode in range(1, max_spatial_mode + 1):
        spatial_factors.extend((("cos", mode), ("sin", mode)))
    basis_index = [
        (kind, mode, temporal_mode)
        for temporal_mode in range(1, max_temporal_mode + 1)
        for kind, mode in spatial_factors
    ]

    basis: dict[str, list[np.ndarray]] = {
        "psi": [],
        "psi_t": [],
        "psi_x": [],
        "psi_xx": [],
        "psi_xxx": [],
    }
    for kind, spatial_mode, temporal_mode in basis_index:
        omega = temporal_mode * np.pi
        temporal = np.sin(omega * t)[:, None]
        temporal_t = omega * np.cos(omega * t)[:, None]
        basis["psi"].append(
            temporal * _spatial_factor(x, kind, spatial_mode, 0)[None, :]
        )
        basis["psi_t"].append(
            temporal_t * _spatial_factor(x, kind, spatial_mode, 0)[None, :]
        )
        basis["psi_x"].append(
            temporal * _spatial_factor(x, kind, spatial_mode, 1)[None, :]
        )
        basis["psi_xx"].append(
            temporal * _spatial_factor(x, kind, spatial_mode, 2)[None, :]
        )
        basis["psi_xxx"].append(
            temporal * _spatial_factor(x, kind, spatial_mode, 3)[None, :]
        )
    stacked = {name: np.stack(values, axis=0) for name, values in basis.items()}

    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((row_count, len(basis_index)))
    tests = {
        name: np.tensordot(coefficients, values, axes=(1, 0))
        for name, values in stacked.items()
    }

    dt = float(t[1] - t[0])
    time_weights = np.ones(t.size, dtype=float)
    time_weights[[0, -1]] = 0.5
    quadrature = time_weights[:, None] * (dt * dx)
    norms = np.sqrt(np.einsum("rtx,tx->r", tests["psi"] ** 2, quadrature))
    if np.any(norms <= 0.0):
        raise RuntimeError("weak-test normalization encountered a zero function")
    for name in tests:
        tests[name] = tests[name] / norms[:, None, None]

    def project(test: np.ndarray, field: np.ndarray) -> np.ndarray:
        return np.einsum("rtx,tx,tx->r", test, field, quadrature)

    y = -project(tests["psi_t"], u) - project(tests["psi"], f)
    columns = {
        "u": project(tests["psi"], u),
        "ux": -project(tests["psi_x"], u),
        "uxx": project(tests["psi_xx"], u),
        "uux": -0.5 * project(tests["psi_x"], u**2),
        "uxxx": -project(tests["psi_xxx"], u),
    }
    digest_payload = {
        "seed": int(seed),
        "row_count": int(row_count),
        "basis_index": [list(item) for item in basis_index],
        "coefficient_sha256": array_digest(coefficients),
        "y_sha256": array_digest(y),
        "column_sha256": {
            name: array_digest(columns[name]) for name in sorted(columns)
        },
        "observed_field_sha256": observed.digest(),
    }
    bank_sha256 = content_digest(digest_payload)
    return WeakEvidence(
        seed=seed,
        row_count=row_count,
        y=y,
        columns=columns,
        bank_sha256=bank_sha256,
        metadata={
            **digest_payload,
            "bank_sha256": bank_sha256,
            "normalization": "unit_discrete_L2",
            "space_quadrature": "periodic_rectangle",
            "time_quadrature": "trapezoidal",
            "observed_derivatives_used": False,
        },
    )


def candidate_matrix(evidence: WeakEvidence, terms: tuple[str, ...]) -> np.ndarray:
    unknown = set(terms) - set(evidence.columns)
    if unknown:
        raise ValueError(f"unsupported weak terms: {sorted(unknown)}")
    return np.column_stack([evidence.columns[name] for name in terms])
