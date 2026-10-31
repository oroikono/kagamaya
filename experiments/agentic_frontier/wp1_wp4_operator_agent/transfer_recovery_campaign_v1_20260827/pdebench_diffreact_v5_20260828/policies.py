"""Frozen public score and fixed controls for prospective confirmation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import contract
from .pipeline import Diagnostics


TERMINAL_STATUSES = frozenset({"IDENTIFIED", "OUTSIDE_LIBRARY"})


def view_cost(temporal_samples: int, test_modes: int) -> int:
    return (
        contract.SPATIAL_NODES**2 * temporal_samples
        + test_modes * contract.SPATIAL_NODES**2 // 10
    )


MAX_SELECTABLE_VIEW_COST = max(
    view_cost(n, m)
    for n in contract.TEMPORAL_SAMPLES
    for m in contract.TEST_MODE_COUNTS
)


def public_gate_margin(
    diagnostics: Diagnostics,
    status: str,
    thresholds: contract.Thresholds,
) -> float:
    """Threshold-normalized analogue of the frozen one-dimensional margin."""

    identified = min(
        (thresholds.identified_residual_max - diagnostics.residual)
        / thresholds.identified_residual_max,
        (thresholds.condition_max - diagnostics.condition)
        / thresholds.condition_max,
    )
    omitted = (
        diagnostics.residual - thresholds.outside_library_residual_min
    ) / thresholds.outside_library_residual_min
    if status == "IDENTIFIED":
        margin = identified
    elif status == "OUTSIDE_LIBRARY":
        margin = omitted
    else:
        margin = max(identified, omitted)
    if not math.isfinite(margin):
        margin = -contract.MARGIN_CLIP
    return float(max(-contract.MARGIN_CLIP, min(contract.MARGIN_CLIP, margin)))


def public_objective(
    diagnostics: Diagnostics,
    status: str,
    cost: int,
    thresholds: contract.Thresholds,
) -> float:
    decisive = status in TERMINAL_STATUSES
    return float(
        contract.DECISIVE_REWARD * int(decisive)
        + public_gate_margin(diagnostics, status, thresholds)
        - contract.COST_WEIGHT * cost / MAX_SELECTABLE_VIEW_COST
    )


def signature(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["status"]), tuple(row.get("support") or ())


def _proposal(
    *,
    policy: str,
    status: str,
    support: Sequence[str],
    selected_view_id: str | None,
    witnessed_view_ids: Sequence[str],
    realized_cost: int,
    reason: str,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "status": status,
        "support": list(support),
        "selected_view_id": selected_view_id,
        "witnessed_view_ids": list(witnessed_view_ids),
        "realized_cost": int(realized_cost),
        "reason": reason,
    }


def score_max(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise ValueError("score_max requires the exact twelve-view bank")
    chosen = min(
        rows,
        key=lambda row: (-float(row["public_objective"]), str(row["view_id"])),
    )
    return _proposal(
        policy="score_max",
        status=str(chosen["status"]),
        support=tuple(chosen.get("support") or ()),
        selected_view_id=str(chosen["view_id"]),
        witnessed_view_ids=(str(chosen["view_id"]),),
        realized_cost=int(chosen["cost"]),
        reason="canonical_public_objective_argmax",
    )


def fixed_fine(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if str(row["view_id"]) == "T025_M5"]
    if len(matches) != 1:
        raise ValueError("fixed_fine requires exactly one T025_M5 row")
    chosen = matches[0]
    return _proposal(
        policy="fixed_fine",
        status=str(chosen["status"]),
        support=tuple(chosen.get("support") or ()),
        selected_view_id=str(chosen["view_id"]),
        witnessed_view_ids=(str(chosen["view_id"]),),
        realized_cost=int(chosen["cost"]),
        reason="fixed_T025_M5",
    )


def all_mode_agreement(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise ValueError("all_mode_agreement requires the exact twelve-view bank")
    by = {
        (int(row["temporal_samples"]), int(row["test_modes"])): row
        for row in rows
    }
    expected = {
        (n, m)
        for n in contract.TEMPORAL_SAMPLES
        for m in contract.TEST_MODE_COUNTS
    }
    if set(by) != expected:
        raise ValueError("view bank is not the frozen temporal/test-mode Cartesian product")
    candidates: list[tuple[float, int, int, Mapping[str, Any], tuple[str, ...]]] = []
    for n in contract.TEMPORAL_SAMPLES:
        group = [by[(n, m)] for m in contract.TEST_MODE_COUNTS]
        signatures = {signature(row) for row in group}
        if len(signatures) != 1 or str(group[0]["status"]) not in TERMINAL_STATUSES:
            continue
        witness = by[(n, max(contract.TEST_MODE_COUNTS))]
        candidates.append(
            (
                min(float(row["public_objective"]) for row in group),
                sum(int(row["cost"]) for row in group),
                n,
                witness,
                tuple(str(row["view_id"]) for row in group),
            )
        )
    if not candidates:
        return _proposal(
            policy="all_mode_agreement",
            status="INCONCLUSIVE",
            support=(),
            selected_view_id=None,
            witnessed_view_ids=(),
            realized_cost=sum(int(row["cost"]) for row in rows),
            reason="no_exact_definitive_four_mode_group",
        )
    worst_score, cost, _, witness, view_ids = min(
        candidates,
        key=lambda item: (-item[0], item[1], item[2], str(item[3]["view_id"])),
    )
    del worst_score
    return _proposal(
        policy="all_mode_agreement",
        status=str(witness["status"]),
        support=tuple(witness.get("support") or ()),
        selected_view_id=str(witness["view_id"]),
        witnessed_view_ids=view_ids,
        realized_cost=cost,
        reason="exact_four_mode_signature_agreement",
    )


def evaluate_all(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [score_max(rows), fixed_fine(rows), all_mode_agreement(rows)]
