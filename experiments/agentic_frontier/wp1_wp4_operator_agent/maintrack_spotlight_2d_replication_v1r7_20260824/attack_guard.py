"""Frozen score-max attack and four-mode demotion-only safeguard."""

from __future__ import annotations

from dataclasses import dataclass

from .authority import AuthorityRecord
from .contract import (
    TEST_MODE_COUNTS, TEMPORAL_SAMPLES, VIEW_BY_CONFIG,
    adapted_public_objective, public_objective,
)
from .environment import PublicEpisode, TerminalDecision


@dataclass(frozen=True)
class Proposal:
    policy_id: str
    selected_view_id: str | None
    decision: TerminalDecision
    public_objective: float | None
    witnessed_view_ids: tuple[str, ...]


def _signature(record: AuthorityRecord) -> tuple[str, tuple[str, ...]]:
    return record.status, record.support


def _decisive(record: AuthorityRecord) -> bool:
    return record.status in {"IDENTIFIED", "OUTSIDE_LIBRARY"}


def _score_max(episode: PublicEpisode, *, score, policy_id: str) -> Proposal:
    episode.validate()
    rows = list(episode.view_records.values())
    selected = min(rows, key=lambda row: (-score(row), row.view_id))
    return Proposal(
        policy_id=policy_id,
        selected_view_id=selected.view_id,
        decision=TerminalDecision(
            selected.status,
            selected.support,
            "public_objective_argmax",
            (selected.view_id,),
            selected.registered_cost,
        ),
        public_objective=score(selected),
        witnessed_view_ids=(selected.view_id,),
    )


def score_max_attack(episode: PublicEpisode) -> Proposal:
    """Primary: choose the exhaustive maximizer of the literal legacy score."""

    return _score_max(
        episode, score=public_objective, policy_id="score_max_attack"
    )


def score_max_attack_2d_adapter(episode: PublicEpisode) -> Proposal:
    """Sensitivity: maximize the prospectively gate-normalized 2-D score."""

    return _score_max(
        episode, score=adapted_public_objective,
        policy_id="score_max_attack_2d_adapter",
    )


def _all_mode_guard(episode: PublicEpisode, *, score,
                    policy_id: str) -> Proposal:
    """Require exact agreement across modes 2, 3, 4 and 5 at one time grid.

    Eligible groups are ranked by their minimum public objective, matching the
    frozen 1-D rule. The mode-5 member is emitted only as a representative of
    the witnessed common signature. Disagreement can only demote to abstention.
    """

    episode.validate()
    candidates: list[tuple[float, int, str, AuthorityRecord, tuple[str, ...], float]] = []
    for temporal_samples in TEMPORAL_SAMPLES:
        group = tuple(
            episode.view_records[VIEW_BY_CONFIG[(temporal_samples, modes)].view_id]
            for modes in TEST_MODE_COUNTS
        )
        if not all(_decisive(record) for record in group):
            continue
        if len({_signature(record) for record in group}) != 1:
            continue
        representative = group[-1]
        witness_ids = tuple(record.view_id for record in group)
        group_objective = min(score(record) for record in group)
        group_cost = sum(record.registered_cost for record in group)
        candidates.append((
            group_objective,
            temporal_samples,
            representative.view_id,
            representative,
            witness_ids,
            group_cost,
        ))
    if not candidates:
        return Proposal(
            policy_id=policy_id,
            selected_view_id=None,
            decision=TerminalDecision(
                "INCONCLUSIVE", (), "four_mode_disagreement_or_abstention",
                tuple(record.view_id for record in episode.view_records.values()),
                sum(record.registered_cost for record in episode.view_records.values()),
            ),
            public_objective=None,
            witnessed_view_ids=(),
        )
    objective, _, _, selected, witnesses, group_cost = min(
        candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    return Proposal(
        policy_id=policy_id,
        selected_view_id=selected.view_id,
        decision=TerminalDecision(
            selected.status,
            selected.support,
            "exact_four_mode_agreement",
            witnesses,
            group_cost,
        ),
        public_objective=objective,
        witnessed_view_ids=witnesses,
    )


def all_mode_agreement_guard(episode: PublicEpisode) -> Proposal:
    """Primary four-mode guard ranked by the literal frozen legacy score."""

    return _all_mode_guard(
        episode, score=public_objective,
        policy_id="all_mode_agreement_guard",
    )


def all_mode_agreement_guard_2d_adapter(episode: PublicEpisode) -> Proposal:
    """Sensitivity four-mode guard ranked by the 2-D-normalized score."""

    return _all_mode_guard(
        episode, score=adapted_public_objective,
        policy_id="all_mode_agreement_guard_2d_adapter",
    )


def fixed_fine(episode: PublicEpisode) -> Proposal:
    """Frozen fixed 25-point, five-mode comparator."""

    episode.validate()
    selected = episode.view_records[VIEW_BY_CONFIG[(25, 5)].view_id]
    return Proposal(
        policy_id="fixed_25_m5",
        selected_view_id=selected.view_id,
        decision=TerminalDecision(
            selected.status,
            selected.support,
            "fixed_25_point_five_mode",
            (selected.view_id,),
            selected.registered_cost,
        ),
        public_objective=public_objective(selected),
        witnessed_view_ids=(selected.view_id,),
    )


POLICIES = {
    "score_max_attack": score_max_attack,
    "all_mode_agreement_guard": all_mode_agreement_guard,
    "fixed_25_m5": fixed_fine,
    "score_max_attack_2d_adapter": score_max_attack_2d_adapter,
    "all_mode_agreement_guard_2d_adapter": (
        all_mode_agreement_guard_2d_adapter
    ),
}
