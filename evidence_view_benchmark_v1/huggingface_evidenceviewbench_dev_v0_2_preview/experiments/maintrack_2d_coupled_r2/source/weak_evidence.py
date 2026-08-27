"""Public weak-form evidence construction from a registered trajectory view."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .contract import COUPLED_ATOMS, SCALAR_ATOMS, ViewSpec
from .trajectory import Trajectory


@dataclass(frozen=True)
class WeakEvidence:
    episode_id: str
    view_id: str
    origin: str
    atoms: tuple[str, ...]
    design: np.ndarray
    target: np.ndarray
    row_roles: tuple[str, ...]
    registered_cost: float
    evidence_digest: str


def _time_indices(times: np.ndarray, view: ViewSpec) -> np.ndarray:
    eligible = np.flatnonzero(
        (times >= view.window_start - 1e-12) & (times <= view.window_end + 1e-12)
    )
    if len(eligible) < view.temporal_samples:
        raise ValueError("base trajectory has too few registered time samples")
    positions = np.linspace(0, len(eligible) - 1, view.temporal_samples)
    indices = eligible[np.rint(positions).astype(int)]
    if len(np.unique(indices)) != view.temporal_samples:
        raise ValueError("temporal selection is not unique")
    return indices


def _spatial_indices(base_nodes: int, desired_nodes: int) -> np.ndarray:
    if base_nodes % desired_nodes:
        raise ValueError("view is not an exact nested spatial subsample")
    return np.arange(0, base_nodes, base_nodes // desired_nodes, dtype=int)


def observed_arrays(trajectory: Trajectory, *, noise_level: float,
                    noise_model: str, noise_seed: int
                    ) -> tuple[np.ndarray, np.ndarray | None]:
    """Create one owner-blind noisy high-resolution record per episode.

    Every view is a strict subsample of this record, so comparisons do not
    receive separately redrawn measurement noise.
    """
    if noise_level < 0:
        raise ValueError("noise level must be nonnegative")
    rng = np.random.default_rng(noise_seed)

    def perturb(values: np.ndarray) -> np.ndarray:
        if noise_level == 0.0:
            return np.array(values, copy=True)
        scale = noise_level * max(float(np.std(values)), 1e-12)
        eps = rng.standard_normal(values.shape)
        if noise_model == "iid":
            pass
        elif noise_model == "correlated":
            eps = (eps + np.roll(eps, 1, axis=1) + np.roll(eps, -1, axis=1)
                   + np.roll(eps, 1, axis=2) + np.roll(eps, -1, axis=2)) / np.sqrt(5.0)
            eps = 0.75 * eps + 0.25 * np.roll(eps, 1, axis=0)
        else:
            raise ValueError("noise model must be iid or correlated")
        return values + scale * eps

    return perturb(trajectory.u), None if trajectory.v is None else perturb(trajectory.v)


def _spatial_test(kind: str, kx: int, ky: int,
                  x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    phase = 0.173 if kind == "sin" else -0.291
    z = kx * x + ky * y + phase
    if kind == "sin":
        base, first = np.sin(z), np.cos(z)
    else:
        base, first = np.cos(z), -np.sin(z)
    k2 = kx**2 + ky**2
    return {
        "s": base, "sx": kx * first, "sy": ky * first,
        "sxx": -(kx**2) * base, "syy": -(ky**2) * base,
        "sxy": -(kx * ky) * base, "bilap": k2**2 * base,
    }


def _temporal_test(q: int, times: np.ndarray, start: float,
                   end: float) -> tuple[np.ndarray, np.ndarray]:
    s = (times - start) / (end - start)
    b = s ** (q + 2) * (1.0 - s) ** 2
    db_ds = ((q + 2) * s ** (q + 1) * (1.0 - s) ** 2
             - 2.0 * s ** (q + 2) * (1.0 - s))
    return b, db_ds / (end - start)


def build_weak_evidence(trajectory: Trajectory, view: ViewSpec, *,
                        observed_u: np.ndarray, observed_v: np.ndarray | None,
                        episode_id: str | None = None,
                        ) -> WeakEvidence:
    trajectory.validate()
    if observed_u.shape != trajectory.u.shape:
        raise ValueError("observed u shape mismatch")
    if (observed_v is None) != (trajectory.v is None):
        raise ValueError("observed v availability mismatch")
    tidx = _time_indices(trajectory.times, view)
    sidx = _spatial_indices(len(trajectory.x), view.spatial_nodes)
    times = trajectory.times[tidx]
    xs = trajectory.x[sidx]
    ys = trajectory.y[sidx]
    u = observed_u[np.ix_(tidx, sidx, sidx)]
    v = None if observed_v is None else observed_v[np.ix_(tidx, sidx, sidx)]
    forcing = trajectory.forcing[np.ix_(tidx, sidx, sidx)]
    target_field = u if trajectory.target == "u" else v
    atoms = SCALAR_ATOMS if v is None else COUPLED_ATOMS
    dx = 2.0 * np.pi / view.spatial_nodes
    x2, y2 = xs[:, None], ys[None, :]

    rows: list[list[float]] = []
    targets: list[float] = []
    roles: list[str] = []
    for q in range(view.test_mode_count):
        temporal, temporal_dt = _temporal_test(q, times, view.window_start, view.window_end)
        for kx in range(view.test_mode_count + 1):
            for ky in range(view.test_mode_count + 1):
                if kx == 0 and ky == 0:
                    continue
                for kind_index, kind in enumerate(("sin", "cos")):
                    spatial = _spatial_test(kind, kx, ky, x2, y2)
                    phi = temporal[:, None, None] * spatial["s"][None, :, :]
                    target_integrand = (-target_field * temporal_dt[:, None, None]
                                        * spatial["s"][None, :, :] - forcing * phi)
                    target_spatial = np.sum(target_integrand, axis=(1, 2)) * dx**2
                    target_row = float(np.trapezoid(target_spatial, times))
                    row: list[float] = []
                    for atom in atoms:
                        base = u if atom.startswith("u") else v
                        if atom in {"u", "v"}:
                            integrand = base * phi
                        elif atom == "u2":
                            integrand = u**2 * phi
                        elif atom == "v2":
                            integrand = v**2 * phi
                        elif atom == "uv":
                            integrand = u * v * phi
                        elif atom == "u3":
                            integrand = u**3 * phi
                        elif atom in {"ux", "vx"}:
                            integrand = -base * temporal[:, None, None] * spatial["sx"][None, :, :]
                        elif atom in {"uy", "vy"}:
                            integrand = -base * temporal[:, None, None] * spatial["sy"][None, :, :]
                        elif atom in {"uxx", "vxx"}:
                            integrand = base * temporal[:, None, None] * spatial["sxx"][None, :, :]
                        elif atom in {"uyy", "vyy"}:
                            integrand = base * temporal[:, None, None] * spatial["syy"][None, :, :]
                        elif atom in {"uxy", "vxy"}:
                            integrand = base * temporal[:, None, None] * spatial["sxy"][None, :, :]
                        elif atom == "biharm":
                            integrand = u * temporal[:, None, None] * spatial["bilap"][None, :, :]
                        elif atom == "uux":
                            integrand = -0.5 * u**2 * temporal[:, None, None] * spatial["sx"][None, :, :]
                        elif atom == "uuy":
                            integrand = -0.5 * u**2 * temporal[:, None, None] * spatial["sy"][None, :, :]
                        else:
                            raise ValueError(f"unhandled atom {atom}")
                        spatial_row = np.sum(integrand, axis=(1, 2)) * dx**2
                        row.append(float(np.trapezoid(spatial_row, times)))
                    rows.append(row)
                    targets.append(target_row)
                    roles.append("train" if (q + kx + 2 * ky + kind_index) % 2 == 0 else "validate")

    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    if set(roles) != {"train", "validate"}:
        raise ValueError("weak rows do not contain both roles")
    h = hashlib.sha256()
    h.update(trajectory.public_digest.encode())
    h.update(view.view_id.encode())
    h.update(np.ascontiguousarray(design).tobytes())
    h.update(np.ascontiguousarray(target).tobytes())
    evidence = WeakEvidence(
        episode_id=trajectory.episode_id if episode_id is None else episode_id,
        view_id=view.view_id,
        origin=trajectory.origin, atoms=atoms, design=design, target=target,
        row_roles=tuple(roles), registered_cost=view.registered_cost,
        evidence_digest=h.hexdigest(),
    )
    return evidence
