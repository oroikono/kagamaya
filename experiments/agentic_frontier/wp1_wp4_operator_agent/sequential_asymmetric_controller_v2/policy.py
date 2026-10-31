"""Asymmetric terminal-evidence policy frozen for v2."""

from __future__ import annotations

from typing import Any, Mapping

from ..sequential_consensus_controller_v1.policy import (
    _by_config,
    _selected,
    propose as propose_v1,
    validate_public_episode,
)
from .contract import MODES, NODES, POLICIES, is_terminal, terminal_signature


def _sequential_asymmetric(episode: Mapping[str, Any]) -> dict[str, Any]:
    by = _by_config(episode["views"])
    acquired: list[str] = []
    stages: list[dict[str, Any]] = []
    previous_oog: tuple[str, tuple[str, ...]] | None = None
    for nodes in NODES:
        group = [by[(nodes, mode)] for mode in MODES]
        tokens = [str(row["view_token"]) for row in group]
        acquired.extend(tokens)
        signatures = [terminal_signature(row) for row in group]
        consensus = all(is_terminal(row) for row in group) and len(set(signatures)) == 1
        signature = signatures[0] if consensus else None
        identified_ready = signature is not None and signature[0] == "IDENTIFIED"
        oog_ready = (
            signature is not None
            and signature[0] == "OUT_OF_GRAMMAR"
            and previous_oog == signature
        )
        ready = identified_ready or oog_ready
        stages.append(
            {
                "action": "ACQUIRE_AND_CHECK_ASYMMETRIC",
                "nodes": nodes,
                "view_tokens": tokens,
                "terminal_consensus": consensus,
                "consensus_signature": (
                    [signature[0], list(signature[1])] if signature is not None else None
                ),
                "previous_oog_signature": (
                    [previous_oog[0], list(previous_oog[1])]
                    if previous_oog is not None
                    else None
                ),
                "identified_ready": identified_ready,
                "oog_two_resolution_ready": oog_ready,
                "next": "STOP" if ready else ("ABSTAIN" if nodes == 49 else "ESCALATE"),
            }
        )
        if ready:
            return _selected(
                episode,
                "sequential_all_mode",
                by[(nodes, 5)],
                acquired=acquired,
                stages=stages,
            )
        if signature is not None and signature[0] == "OUT_OF_GRAMMAR":
            previous_oog = signature
        else:
            previous_oog = None
    return _selected(
        episode,
        "sequential_all_mode",
        None,
        acquired=acquired,
        stages=stages,
    )


def propose(episode: Mapping[str, Any], policy: str) -> dict[str, Any]:
    validate_public_episode(episode)
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    if policy == "sequential_all_mode":
        return _sequential_asymmetric(episode)
    return propose_v1(episode, policy)

