"""Truth-blind policies over serialized evidence-view records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contract import (
    MODES,
    NODES,
    POLICIES,
    TERMINAL_STATUSES,
    VIEW_BY_CONFIG,
    VIEW_BY_TOKEN,
    VIEWS,
    digest,
    is_terminal,
    terminal_signature,
)


FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "family",
        "family_index",
        "truth_support",
        "correct",
        "false_promotion",
        "reference_final",
        "reference_support",
        "reference_status",
    }
)


def validate_public_episode(episode: Mapping[str, Any]) -> None:
    if FORBIDDEN_PUBLIC_KEYS & set(episode):
        raise ValueError("public episode contains an owner-only key")
    rows = episode.get("views")
    if not isinstance(rows, list) or len(rows) != len(VIEWS):
        raise ValueError("public episode must contain the complete sixteen-view table")
    tokens = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("view row must be a mapping")
        if FORBIDDEN_PUBLIC_KEYS & set(row):
            raise ValueError("public view contains an owner-only key")
        token = str(row.get("view_token"))
        if token not in VIEW_BY_TOKEN:
            raise ValueError("unknown view token")
        view = VIEW_BY_TOKEN[token]
        if (row.get("nodes"), row.get("temporal_modes")) != (
            view.nodes,
            view.temporal_modes,
        ):
            raise ValueError("view token/configuration mismatch")
        if row.get("terminal_status") not in {
            "IDENTIFIED",
            "OUT_OF_GRAMMAR",
            "UNIDENTIFIED",
            "FAILED",
        }:
            raise ValueError("unknown terminal status")
        tokens.append(token)
    if set(tokens) != set(VIEW_BY_TOKEN) or len(tokens) != len(set(tokens)):
        raise ValueError("view coverage is incomplete or duplicated")
    body = dict(episode)
    recorded = body.pop("episode_sha256", None)
    if recorded != digest(body):
        raise ValueError("public episode digest mismatch")


def _by_config(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], Mapping[str, Any]]:
    return {
        (int(row["nodes"]), int(row["temporal_modes"])): row
        for row in rows
    }


def _selected(
    episode: Mapping[str, Any],
    policy: str,
    row: Mapping[str, Any] | None,
    *,
    acquired: Sequence[str],
    stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    terminal = row is not None and is_terminal(row)
    body: dict[str, Any] = {
        "schema_version": "verisigs.sequential_consensus_controller.proposal.v1",
        "episode_id": str(episode["episode_id"]),
        "public_episode_sha256": str(episode["episode_sha256"]),
        "policy": policy,
        "proposal_valid": bool(terminal),
        "selected_view_token": str(row["view_token"]) if terminal else None,
        "selected_status": str(row["terminal_status"]) if terminal else "INCONCLUSIVE",
        "selected_support": list(row.get("terminal_support") or ()) if terminal else None,
        "acquired_view_tokens": list(acquired),
        "acquisition_stages": [dict(stage) for stage in stages],
        "views_evaluated": len(acquired),
        "max_nodes_acquired": max(
            (VIEW_BY_TOKEN[token].nodes for token in acquired), default=0
        ),
        "stage_count": len(stages),
        "scientific_authority": False,
        "hidden_truth_accessed": False,
        "hidden_reference_accessed": False,
    }
    body["proposal_sha256"] = digest(body)
    return body


def _sequential(episode: Mapping[str, Any]) -> dict[str, Any]:
    by = _by_config(episode["views"])
    acquired: list[str] = []
    stages: list[dict[str, Any]] = []
    for nodes in NODES:
        group = [by[(nodes, mode)] for mode in MODES]
        tokens = [str(row["view_token"]) for row in group]
        acquired.extend(tokens)
        signatures = [terminal_signature(row) for row in group]
        consensus = all(is_terminal(row) for row in group) and len(set(signatures)) == 1
        stages.append(
            {
                "action": "ACQUIRE_AND_CHECK",
                "nodes": nodes,
                "view_tokens": tokens,
                "terminal_consensus": consensus,
                "consensus_signature": (
                    [signatures[0][0], list(signatures[0][1])] if consensus else None
                ),
                "next": "STOP" if consensus else ("ABSTAIN" if nodes == 49 else "ESCALATE"),
            }
        )
        if consensus:
            return _selected(
                episode,
                "sequential_all_mode",
                by[(nodes, 5)],
                acquired=acquired,
                stages=stages,
            )
    return _selected(
        episode,
        "sequential_all_mode",
        None,
        acquired=acquired,
        stages=stages,
    )


def _fullbank_consensus(episode: Mapping[str, Any]) -> dict[str, Any]:
    by = _by_config(episode["views"])
    candidates: list[tuple[float, int, str, Mapping[str, Any]]] = []
    for nodes in NODES:
        group = [by[(nodes, mode)] for mode in MODES]
        if all(is_terminal(row) for row in group) and len(
            {terminal_signature(row) for row in group}
        ) == 1:
            selected = by[(nodes, 5)]
            candidates.append(
                (
                    min(float(row["public_objective"]) for row in group),
                    nodes,
                    str(selected["view_token"]),
                    selected,
                )
            )
    chosen = min(candidates, key=lambda item: (-item[0], item[1], item[2]))[3] if candidates else None
    tokens = [view.token for view in VIEWS]
    return _selected(
        episode,
        "fullbank_all_mode",
        chosen,
        acquired=tokens,
        stages=[{"action": "READ_COMPLETE_BANK", "nodes": 49, "view_tokens": tokens, "next": "STOP"}],
    )


def _one_shot(episode: Mapping[str, Any], policy: str) -> dict[str, Any]:
    rows = list(episode["views"])
    by = _by_config(rows)
    if policy == "fixed_25_m5":
        chosen = by[(25, 5)]
        acquired = [str(chosen["view_token"])]
    elif policy == "fixed_49_m5":
        chosen = by[(49, 5)]
        acquired = [str(chosen["view_token"])]
    elif policy == "confidence_max":
        chosen = min(rows, key=lambda row: (-float(row["public_objective"]), str(row["view_token"])))
        acquired = [view.token for view in VIEWS]
    elif policy == "d_optimal":
        chosen = min(
            rows,
            key=lambda row: (
                -float(row["gram_logdet"]),
                int(row["nodes"]),
                int(row["temporal_modes"]),
                str(row["view_token"]),
            ),
        )
        acquired = [view.token for view in VIEWS]
    elif policy == "e_optimal":
        chosen = min(
            rows,
            key=lambda row: (
                -float(row["gram_lambda_min"]),
                int(row["nodes"]),
                int(row["temporal_modes"]),
                str(row["view_token"]),
            ),
        )
        acquired = [view.token for view in VIEWS]
    else:
        raise ValueError(f"unsupported one-shot policy {policy}")
    return _selected(
        episode,
        policy,
        chosen,
        acquired=acquired,
        stages=[
            {
                "action": "SELECT_DECLARED_VIEW",
                "nodes": int(chosen["nodes"]),
                "view_tokens": list(acquired),
                "next": "STOP" if is_terminal(chosen) else "ABSTAIN",
            }
        ],
    )


def propose(episode: Mapping[str, Any], policy: str) -> dict[str, Any]:
    validate_public_episode(episode)
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy == "sequential_all_mode":
        return _sequential(episode)
    if policy == "fullbank_all_mode":
        return _fullbank_consensus(episode)
    return _one_shot(episode, policy)

