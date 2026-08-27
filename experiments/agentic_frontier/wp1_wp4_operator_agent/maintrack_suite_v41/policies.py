"""Truth-blind, ambiguity-aware second-intervention policies for v4.1."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .simulator import Action, Family, SimulationConfig, action_menu, simulate
from .weakform import DiscoveryConfig, DiscoveryResult, OPERATORS, build_evidence


DEPLOYABLE_POLICIES = (
    "no_second_action",
    "random",
    "cheapest",
    "weak_gram",
    "disagreement",
    "joint",
)


def _plausible_supports(round1: DiscoveryResult) -> tuple[tuple[str, ...], ...]:
    """Extract a deterministic public ambiguity set from the round-one fit."""

    supports: list[tuple[str, ...]] = []
    for support in round1.ambiguity_set:
        if support not in supports:
            supports.append(support)
    # A narrow ambiguity threshold can leave only one public member.  Retain
    # the next best distinct hypotheses so action design still targets the
    # observed local competition rather than falling back to a global lookup.
    for fit in round1.fits:
        if fit.support not in supports:
            supports.append(fit.support)
        if len(supports) >= 4:
            break
    return tuple(supports)


@lru_cache(maxsize=32)
def _public_design_bank(public_action_seed: int) -> dict[str, np.ndarray]:
    """Frozen public weak designs; no episode outcome or hidden truth is used."""

    cfg = SimulationConfig()
    dcfg = DiscoveryConfig()
    reference_families = tuple(Family)[:5]
    bank: dict[str, np.ndarray] = {}
    for action in action_menu():
        trajectories = tuple(
            simulate(
                family,
                action,
                episode_seed=424242 + idx,
                public_action_seed=public_action_seed,
                relative_noise=0.0,
                config=cfg,
            )[0]
            for idx, family in enumerate(reference_families)
        )
        bank[action.name] = build_evidence(trajectories, dcfg)[0]
    return bank


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    keep = diagonal > 1e-10 * max(float(diagonal.max()), 1e-15)
    return q[:, keep]


def _subspace_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Normalized chordal distance between two column subspaces."""

    ql = _orthonormal_basis(left)
    qr = _orthonormal_basis(right)
    if ql.shape[1] == 0 or qr.shape[1] == 0:
        return 0.0
    overlap = float(np.linalg.norm(ql.T @ qr, ord="fro") ** 2)
    distance2 = max(0.0, ql.shape[1] + qr.shape[1] - 2.0 * overlap)
    return float(np.sqrt(distance2 / max(ql.shape[1], qr.shape[1])))


def ambiguity_action_scores(
    round1: DiscoveryResult, public_action_seed: int
) -> dict[str, dict[str, float]]:
    """Score actions against the observed round-one ambiguity geometry."""

    supports = _plausible_supports(round1)
    operator_index = {name: idx for idx, name in enumerate(OPERATORS)}
    bank = _public_design_bank(public_action_seed)
    # Use one public scale per operator across the whole action menu.  Scaling
    # each action independently would erase the amplitude/polynomial-order
    # differences an intervention is supposed to create and produced a fixed
    # ranking under distinct ambiguity sets in the adversarial test.
    operator_scales = np.sqrt(
        sum(np.square(design).sum(axis=0) for design in bank.values())
    )
    raw_gram: list[float] = []
    raw_disagreement: list[float] = []
    for action in action_menu():
        design = bank[action.name] / np.maximum(operator_scales, 1e-15)
        union = tuple(dict.fromkeys(term for support in supports for term in support))
        columns = [operator_index[term] for term in union]
        restricted = design[:, columns]
        singular = np.linalg.svd(restricted, compute_uv=False)
        raw_gram.append(float(singular[-1]) if singular.size else 0.0)

        pair_scores: list[float] = []
        for idx, left in enumerate(supports):
            left_columns = [operator_index[term] for term in left]
            for right in supports[idx + 1 :]:
                right_columns = [operator_index[term] for term in right]
                pair_scores.append(
                    _subspace_distance(design[:, left_columns], design[:, right_columns])
                )
        raw_disagreement.append(min(pair_scores) if pair_scores else 0.0)

    def scaled(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        span = float(array.max() - array.min())
        return np.ones_like(array) if span <= 1e-15 else (array - array.min()) / span

    scaled_gram = scaled(raw_gram)
    scaled_disagreement = scaled(raw_disagreement)
    table: dict[str, dict[str, float]] = {}
    for idx, action in enumerate(action_menu()):
        table[action.name] = {
            "weak_gram": raw_gram[idx],
            "disagreement": raw_disagreement[idx],
            "joint": float(
                (scaled_gram[idx] + scaled_disagreement[idx]) / (2.0 * action.cost)
            ),
            "cost": action.cost,
        }
    return table


def independent_policy_seed(base_seed: int, episode_ordinal: int, policy_index: int) -> int:
    """Derive policy randomness without using simulator/generation seeds."""

    return int(
        np.random.SeedSequence([base_seed, episode_ordinal, policy_index])
        .generate_state(1)[0]
    )


def choose_action(
    policy: str,
    round1: DiscoveryResult,
    public_action_seed: int,
    policy_seed: int,
) -> Action | None:
    """Choose from public state only; identified episodes stop without cost."""

    if round1.status in {"IDENTIFIED", "OUT_OF_GRAMMAR"}:
        return None
    if policy == "no_second_action":
        return None
    menu = action_menu()
    if policy == "random":
        return menu[int(np.random.default_rng(policy_seed).integers(len(menu)))]
    if policy == "cheapest":
        return min(menu, key=lambda item: (item.cost, item.name))
    if policy not in {"weak_gram", "disagreement", "joint"}:
        raise ValueError(f"unknown policy {policy}")
    scores = ambiguity_action_scores(round1, public_action_seed)
    return max(menu, key=lambda item: (scores[item.name][policy], -item.cost, item.name))
