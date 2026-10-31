"""Neumann-compatible weak evidence for PDEBench diffusion-reaction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from .contract import ATOMS, DU, DV, K, RAW_TIME_POINTS

@dataclass(frozen=True)
class Evidence:
    episode_id: str
    view_id: str
    target_name: str
    atoms: tuple[str, ...]
    design: np.ndarray
    target: np.ndarray
    roles: tuple[str, ...]
    cost: int
    digest: str

@dataclass(frozen=True)
class ProjectionBundle:
    times: np.ndarray
    coefficients: dict[str, np.ndarray]

def nested_time_indices(count: int) -> np.ndarray:
    if count not in {7, 13, 25, 49, 101}:
        raise ValueError("unregistered temporal sample count")
    return np.rint(np.linspace(0, RAW_TIME_POINTS - 1, count)).astype(int)

def validate_nesting() -> None:
    previous: set[int] = set()
    for count in (7, 13, 25, 49, 101):
        current = set(nested_time_indices(count).tolist())
        if previous and not previous <= current:
            raise AssertionError("temporal grids are not nested")
        previous = current

def add_noise(data: np.ndarray, level: float, seed: int) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if level == 0.0:
        return values.copy()
    rng = np.random.default_rng(seed)
    scale = level * max(float(np.std(values)), 1.0e-12)
    return values + scale * rng.standard_normal(values.shape)

def _temporal_test(q: int, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    start, end = float(times[0]), float(times[-1])
    s = (times - start) / (end - start)
    value = s ** (q + 2) * (1.0 - s) ** 2
    derivative = ((q + 2) * s ** (q + 1) * (1.0 - s) ** 2
                  - 2.0 * s ** (q + 2) * (1.0 - s)) / (end - start)
    return value, derivative

def _spatial_cosine(kx: int, ky: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = x[:, None], y[None, :]
    cx = np.cos(kx * math.pi * (xx + 1.0) / 2.0)
    cy = np.cos(ky * math.pi * (yy + 1.0) / 2.0)
    phi = cx * cy
    eigenvalue = -((kx * math.pi / 2.0) ** 2 + (ky * math.pi / 2.0) ** 2)
    return phi, eigenvalue * phi

def project(*, source_id: int, data: np.ndarray, x: np.ndarray, y: np.ndarray,
            times: np.ndarray, noise_level: float) -> ProjectionBundle:
    if data.shape != (101, 128, 128, 2):
        raise ValueError(f"unexpected publisher data shape {data.shape}")
    seed_material = f"pdebench-v2|{source_id}|{noise_level:.9g}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    u = add_noise(np.asarray(data)[..., 0], noise_level, seed)
    v = add_noise(np.asarray(data)[..., 1], noise_level, seed ^ 0x9E3779B97F4A7C15)
    dx, dy = 2.0 / len(x), 2.0 / len(y)
    modes = np.arange(6)
    bx = np.cos(modes[:, None] * math.pi * (x[None, :] + 1.0) / 2.0)
    by = np.cos(modes[:, None] * math.pi * (y[None, :] + 1.0) / 2.0)

    def transform(values: np.ndarray) -> np.ndarray:
        # Two matrix contractions are substantially cheaper than rebuilding a
        # 128x128 spatial test for every weak row and every library contract.
        first = np.einsum("txy,ly->txl", values, by, optimize=True)
        return np.einsum("kx,txl->tkl", bx, first, optimize=True) * dx * dy

    coefficients = {
        "u": transform(u), "v": transform(v), "u3": transform(u**3),
        "one": transform(np.ones_like(u)),
    }
    return ProjectionBundle(np.asarray(times, dtype=float), coefficients)

def build_from_projection(*, source_id: int, bundle: ProjectionBundle,
                          target_name: str, allowed_atoms: tuple[str, ...],
                          temporal_samples: int, test_modes: int,
                          noise_level: float) -> Evidence:
    if target_name not in {"u", "v"}:
        raise ValueError("target must be u or v")
    if not set(allowed_atoms) <= set(ATOMS[target_name]):
        raise ValueError("contract atoms do not match target")
    tidx = nested_time_indices(temporal_samples)
    selected_times = bundle.times[tidx]
    coefficients = {key: value[tidx] for key, value in bundle.coefficients.items()}
    field = coefficients[target_name]
    rows, targets, roles = [], [], []
    for q in range(test_modes):
        temporal, temporal_dt = _temporal_test(q, selected_times)
        for kx in range(test_modes + 1):
            for ky in range(test_modes + 1):
                eigenvalue = -((kx * math.pi / 2.0) ** 2 + (ky * math.pi / 2.0) ** 2)
                target_series = -field[:, kx, ky] * temporal_dt
                full = {
                    "Du_lap_u": DU * eigenvalue * coefficients["u"][:, kx, ky] * temporal,
                    "Dv_lap_v": DV * eigenvalue * coefficients["v"][:, kx, ky] * temporal,
                    "u": coefficients["u"][:, kx, ky] * temporal,
                    "neg_u3": -coefficients["u3"][:, kx, ky] * temporal,
                    "neg_k": -K * coefficients["one"][:, kx, ky] * temporal,
                    "neg_v": -coefficients["v"][:, kx, ky] * temporal,
                }
                rows.append([float(np.trapezoid(full[a], selected_times)) for a in allowed_atoms])
                targets.append(float(np.trapezoid(target_series, selected_times)))
                roles.append("train" if (q + kx + 2 * ky) % 2 == 0 else "validate")
    design = np.asarray(rows, dtype=float)
    target = np.asarray(targets, dtype=float)
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(design).tobytes())
    digest.update(np.ascontiguousarray(target).tobytes())
    view_id = f"T{temporal_samples:03d}_M{test_modes}"
    return Evidence(
        episode_id=f"PDB-{source_id:04d}-{target_name}-{noise_level:.3g}",
        view_id=view_id, target_name=target_name, atoms=allowed_atoms,
        design=design, target=target, roles=tuple(roles),
        cost=128 * 128 * temporal_samples + test_modes * 128 * 128 // 10,
        digest=digest.hexdigest(),
    )
