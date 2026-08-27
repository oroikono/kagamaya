"""Observed-data weak-form support enumeration and fail-closed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .simulator import Trajectory


OPERATORS = ("1", "u", "u^2", "u^3", "u_x", "u_xx", "u_xxx", "u u_x")
SUPPORTS = tuple(
    support
    for size in range(1, 4)
    for support in combinations(OPERATORS, size)
)


@dataclass(frozen=True)
class DiscoveryConfig:
    temporal_modes: int = 5
    spatial_modes: int = 7
    fit_residual_max: float = 0.10
    score_gap_min: float = 0.002
    complexity_penalty: float = 0.0015
    coefficient_floor: float = 1e-4
    condition_max: float = 20.0
    rank_rtol: float = 1e-9
    # Scientific admission is evidence-driven.  A trajectory count is not a
    # surrogate for rank, residual, coefficient, or separation evidence.
    min_trajectories_for_identification: int = 1


@dataclass(frozen=True)
class SupportFit:
    support: tuple[str, ...]
    coefficients: tuple[float, ...]
    train_residual: float
    validation_residual: float
    score: float
    rank: int
    condition: float


@dataclass(frozen=True)
class DiscoveryResult:
    status: str
    support: tuple[str, ...] | None
    best_residual: float
    score_gap: float
    ambiguity_set: tuple[tuple[str, ...], ...]
    fits: tuple[SupportFit, ...]


def _space_tests(x: np.ndarray, max_mode: int) -> tuple[tuple[np.ndarray, ...], ...]:
    length = float((x[1] - x[0]) * x.size)
    tests: list[tuple[np.ndarray, ...]] = [
        (np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x))
    ]
    for mode in range(1, max_mode + 1):
        omega = 2.0 * np.pi * mode / length
        phase = omega * (x - x[0])
        for trig in (np.cos, np.sin):
            tests.append(
                tuple(
                    omega**order * trig(phase + order * np.pi / 2.0)
                    for order in range(4)
                )
            )
    return tuple(tests)


def trajectory_evidence(
    trajectory: Trajectory,
    temporal_modes: int,
    spatial_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return design, target, and deterministic train/validation mask."""

    x, t, u = trajectory.x, trajectory.t, trajectory.u
    if u.shape != (t.size, x.size) or t.size % 2 == 0:
        raise ValueError("trajectory requires matching shapes and odd saved-time count")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("uniform time grid required")
    dx = float((x[1] - x[0]))
    duration = float(t[-1] - t[0])
    scaled = (t - t[0]) / duration
    q = np.ones(t.size)
    q[1:-1:2] = 4.0
    q[2:-1:2] = 2.0
    q *= dt / 3.0
    fields = (np.ones_like(u), u, u**2, u**3)
    rows: list[list[float]] = []
    targets: list[float] = []
    train: list[bool] = []
    for tm in range(1, temporal_modes + 1):
        eta = np.sin(tm * np.pi * scaled)
        deta = tm * np.pi / duration * np.cos(tm * np.pi * scaled)
        wet = q * eta
        wdt = q * deta
        for sm, (psi, psi1, psi2, psi3) in enumerate(_space_tests(x, spatial_modes)):
            def integ(wt: np.ndarray, field: np.ndarray, wx: np.ndarray) -> float:
                return float(dx * np.einsum("t,tx,x->", wt, field, wx))

            targets.append(-integ(wdt, u, psi))
            rows.append(
                [
                    integ(wet, fields[0], psi),
                    integ(wet, fields[1], psi),
                    integ(wet, fields[2], psi),
                    integ(wet, fields[3], psi),
                    -integ(wet, u, psi1),
                    integ(wet, u, psi2),
                    -integ(wet, u, psi3),
                    -0.5 * integ(wet, fields[2], psi1),
                ]
            )
            train.append((tm + sm) % 2 == 0)
    return np.asarray(rows), np.asarray(targets), np.asarray(train, dtype=bool)


def build_evidence(
    trajectories: tuple[Trajectory, ...], config: DiscoveryConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    pieces = [
        trajectory_evidence(item, config.temporal_modes, config.spatial_modes)
        for item in trajectories
    ]
    return (
        np.vstack([piece[0] for piece in pieces]),
        np.concatenate([piece[1] for piece in pieces]),
        np.concatenate([piece[2] for piece in pieces]),
    )


def enumerate_fits(
    design: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    config: DiscoveryConfig,
) -> tuple[SupportFit, ...]:
    index = {name: idx for idx, name in enumerate(OPERATORS)}
    y_train, y_val = target[train_mask], target[~train_mask]
    train_norm = max(float(np.linalg.norm(y_train)), 1e-15)
    val_norm = max(float(np.linalg.norm(y_val)), 1e-15)
    result: list[SupportFit] = []
    for support in SUPPORTS:
        columns = [index[name] for name in support]
        a_train = design[train_mask][:, columns]
        a_val = design[~train_mask][:, columns]
        norms = np.linalg.norm(a_train, axis=0)
        if np.any(norms <= 1e-14):
            continue
        a_normal = a_train / norms
        singular = np.linalg.svd(a_normal, compute_uv=False)
        rank = int(np.count_nonzero(singular > config.rank_rtol * singular[0]))
        if rank < len(support):
            continue
        beta_normal = np.linalg.lstsq(a_normal, y_train / train_norm, rcond=config.rank_rtol)[0]
        beta = beta_normal * train_norm / norms
        train_resid = float(np.linalg.norm(y_train - a_train @ beta) / train_norm)
        val_resid = float(np.linalg.norm(y_val - a_val @ beta) / val_norm)
        # Penalize support size explicitly; validation residual remains reported
        # independently and is the observable out-of-grammar diagnostic.
        score = val_resid + config.complexity_penalty * len(support)
        condition = float(singular[0] / singular[-1])
        result.append(
            SupportFit(
                support=support,
                coefficients=tuple(float(v) for v in beta),
                train_residual=train_resid,
                validation_residual=val_resid,
                score=score,
                rank=rank,
                condition=condition,
            )
        )
    return tuple(sorted(result, key=lambda item: (item.score, len(item.support), item.support)))


def discover(
    trajectories: tuple[Trajectory, ...], config: DiscoveryConfig | None = None
) -> DiscoveryResult:
    cfg = config or DiscoveryConfig()
    design, target, train = build_evidence(trajectories, cfg)
    fits = enumerate_fits(design, target, train, cfg)
    if len(fits) < 2:
        return DiscoveryResult("FAILED", None, float("inf"), 0.0, (), fits)
    best = fits[0]
    # Supports with coefficients indistinguishable from zero are not promoted.
    coefficient_ok = all(abs(value) >= cfg.coefficient_floor for value in best.coefficients)
    # Strict supersets of the best support do not constitute a distinct
    # minimal explanation.  They are still retained in ``fits`` and can win
    # if their validation improvement pays the complexity penalty.  Once a
    # smaller support wins, however, only non-superset alternatives challenge
    # its minimal-equivalence class.
    best_terms = set(best.support)
    competitors = tuple(
        item for item in fits[1:] if not best_terms < set(item.support)
    )
    if not competitors:
        return DiscoveryResult("FAILED", None, best.validation_residual, 0.0, (), fits)
    gap = float(competitors[0].score - best.score)
    ambiguity = tuple(
        item.support
        for item in (best, *competitors)
        if item.score <= best.score + cfg.score_gap_min
    )
    if best.validation_residual > cfg.fit_residual_max:
        status, support = "OUT_OF_GRAMMAR", None
    elif len(trajectories) < cfg.min_trajectories_for_identification:
        # A single experiment may rank hypotheses but cannot promote a
        # support.  This is a protocol requirement, not a truth-dependent
        # exception, and is enforced identically for every proposal source.
        status, support = "UNIDENTIFIED", None
    elif (
        not coefficient_ok
        or gap < cfg.score_gap_min
        or best.condition > cfg.condition_max
    ):
        status, support = "UNIDENTIFIED", None
    else:
        status, support = "IDENTIFIED", best.support
    return DiscoveryResult(
        status=status,
        support=support,
        best_residual=best.validation_residual,
        score_gap=gap,
        ambiguity_set=ambiguity,
        fits=fits,
    )
