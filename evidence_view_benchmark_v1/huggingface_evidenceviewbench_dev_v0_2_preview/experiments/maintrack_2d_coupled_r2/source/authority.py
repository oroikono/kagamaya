"""Locked finite-support weak-form authority.

Policies may observe serialized records returned here but may not change its
library, thresholds, candidate ordering, or terminal semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .contract import DEFAULT_DEVELOPMENT_THRESHOLDS, TerminalThresholds
from .weak_evidence import WeakEvidence


@dataclass(frozen=True)
class AuthorityRecord:
    episode_id: str
    view_id: str
    origin: str
    status: str
    support: tuple[str, ...]
    coefficients: tuple[tuple[str, float], ...]
    residual: float
    score_gap: float
    condition: float
    minimum_coefficient: float
    ambiguity_size: int
    registered_cost: float
    evidence_digest: str

    def public_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "view_id": self.view_id,
            "origin": self.origin,
            "status": self.status,
            "support": list(self.support),
            "residual": self.residual,
            "score_gap": self.score_gap,
            "condition": self.condition,
            "minimum_coefficient": self.minimum_coefficient,
            "ambiguity_size": self.ambiguity_size,
            "registered_cost": self.registered_cost,
            "evidence_digest": self.evidence_digest,
        }


def _candidate_fit(x_train: np.ndarray, y_train: np.ndarray,
                   x_validate: np.ndarray, y_validate: np.ndarray,
                   columns: tuple[int, ...]) -> tuple[float, np.ndarray]:
    selected = x_train[:, columns]
    norms = np.linalg.norm(selected, axis=0)
    if np.any(norms <= 1e-13):
        return math.inf, np.zeros(len(columns))
    normalized = selected / norms
    beta_n, _, rank, _ = np.linalg.lstsq(normalized, y_train, rcond=None)
    if rank != len(columns):
        return math.inf, np.zeros(len(columns))
    beta = beta_n / norms
    residual = np.linalg.norm(y_validate - x_validate[:, columns] @ beta)
    denominator = max(np.linalg.norm(y_validate), 1e-15)
    # Fixed small complexity preference. It only resolves near-equal empirical
    # fits and is part of the locked authority, not selected by a policy.
    score = float(residual / denominator + 2.0e-5 * len(columns))
    return score, beta


def evaluate(evidence: WeakEvidence, *, thresholds: TerminalThresholds = DEFAULT_DEVELOPMENT_THRESHOLDS,
             max_support_size: int = 5) -> AuthorityRecord:
    roles = np.asarray(evidence.row_roles)
    train = roles == "train"
    validate = roles == "validate"
    x_train, y_train = evidence.design[train], evidence.target[train]
    x_validate, y_validate = evidence.design[validate], evidence.target[validate]
    candidates: list[tuple[float, int, tuple[int, ...], np.ndarray]] = []
    for size in range(1, min(max_support_size, len(evidence.atoms)) + 1):
        for columns in combinations(range(len(evidence.atoms)), size):
            score, beta = _candidate_fit(x_train, y_train, x_validate, y_validate, columns)
            if np.isfinite(score):
                candidates.append((score, size, columns, beta))
    if not candidates:
        return AuthorityRecord(
            evidence.episode_id, evidence.view_id, evidence.origin,
            "INCONCLUSIVE", (), (), math.inf, 0.0, math.inf, 0.0, 0,
            evidence.registered_cost, evidence.evidence_digest,
        )
    candidates.sort(key=lambda item: (item[0], item[1], tuple(evidence.atoms[i] for i in item[2])))
    best = candidates[0]
    columns = best[2]
    winner_set = set(columns)
    # Adding a numerically inert atom to the same explanation is not treated
    # as a scientifically distinct rival. This strict-superset exclusion is
    # fixed in the authority and is applied identically to every policy.
    competitors = [item for item in candidates[1:]
                   if not winner_set < set(item[2])]
    second_score = competitors[0][0] if competitors else math.inf
    full_x = evidence.design[:, columns]
    norms = np.linalg.norm(full_x, axis=0)
    beta_n, _, rank, _ = np.linalg.lstsq(full_x / norms, evidence.target, rcond=None)
    beta = beta_n / norms
    residual = float(np.linalg.norm(evidence.target - full_x @ beta)
                     / max(np.linalg.norm(evidence.target), 1e-15))
    condition = float(np.linalg.cond(full_x / norms)) if rank == len(columns) else math.inf
    support = tuple(evidence.atoms[index] for index in columns)
    minimum = float(np.min(np.abs(beta))) if len(beta) else 0.0
    gap = float(second_score - best[0]) if np.isfinite(second_score) else math.inf
    ambiguity = 1 + int(sum(item[0] <= best[0] + thresholds.support_gap_min
                            for item in competitors))

    if residual >= thresholds.outside_library_residual_min:
        status = "OUTSIDE_LIBRARY"
        emitted_support: tuple[str, ...] = ()
    elif (residual <= thresholds.identified_residual_max
          and condition <= thresholds.condition_max
          and minimum >= thresholds.coefficient_floor
          and gap >= thresholds.support_gap_min):
        status = "IDENTIFIED"
        emitted_support = support
    else:
        status = "INCONCLUSIVE"
        emitted_support = ()
    coefficients = tuple((atom, float(value)) for atom, value in zip(support, beta))
    return AuthorityRecord(
        evidence.episode_id, evidence.view_id, evidence.origin, status,
        emitted_support, coefficients, residual, gap, condition, minimum,
        ambiguity, evidence.registered_cost, evidence.evidence_digest,
    )
