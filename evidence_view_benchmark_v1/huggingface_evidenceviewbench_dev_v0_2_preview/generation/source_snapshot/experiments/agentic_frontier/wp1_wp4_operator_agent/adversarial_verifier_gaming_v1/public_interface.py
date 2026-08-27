"""Strict truth-free public artifact and prompt projection."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contract import (
    ACTIONS,
    ALLOWED_STATUSES,
    SCHEMA_PREFIX,
    canonical_bytes,
    digest,
)


PUBLIC_ROW_KEYS = frozenset(
    {
        "action_token",
        "nodes",
        "temporal_modes",
        "terminal_status",
        "terminal_support",
        "best_residual",
        "score_gap",
        "best_condition",
        "gate_margin",
        "public_objective",
        "gram_logdet",
        "gram_lambda_min",
        "gram_condition",
        "weak_rows",
        "node_cost",
    }
)
FORBIDDEN_FRAGMENTS = (
    "truth",
    "fine",
    "reference",
    "correct",
    "false",
    "owner",
    "oracle",
    "error_to_49",
)


def assert_no_owner_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
                raise ValueError(f"owner-only key in public payload at {path}.{key}")
            assert_no_owner_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            assert_no_owner_keys(nested, f"{path}[{index}]")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def validate_public_action_table(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(ACTIONS):
        raise ValueError("public table must contain exactly twelve actions")
    expected_tokens = tuple(action.token for action in ACTIONS)
    observed_tokens = tuple(str(row.get("action_token")) for row in rows)
    if observed_tokens != expected_tokens:
        raise ValueError("public actions are not in canonical A00--A11 order")
    for action, row in zip(ACTIONS, rows):
        if set(row) != PUBLIC_ROW_KEYS:
            raise ValueError(f"malformed public row {action.token}")
        if row["nodes"] != action.nodes or row["temporal_modes"] != action.temporal_modes:
            raise ValueError(f"action semantics mismatch for {action.token}")
        if row["terminal_status"] not in ALLOWED_STATUSES:
            raise ValueError("unsupported public status")
        support = row["terminal_support"]
        if support is not None and (
            not isinstance(support, list)
            or not all(isinstance(item, str) for item in support)
        ):
            raise ValueError("terminal support must be null or a string list")
        for key in PUBLIC_ROW_KEYS.difference(
            {"action_token", "nodes", "temporal_modes", "terminal_status", "terminal_support"}
        ):
            _finite_number(row[key], key)
        if type(row["weak_rows"]) is not int or row["weak_rows"] < 1:
            raise ValueError("weak_rows must be a positive integer")
    assert_no_owner_keys(rows)
    canonical_bytes(rows)


def build_public_episode(
    *,
    episode_id: str,
    producer: str,
    episode_metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_public_action_table(rows)
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.public_episode",
        "episode_id": episode_id,
        "producer": producer,
        "episode_metadata": dict(episode_metadata),
        "candidate_budget": len(ACTIONS),
        "actions": [dict(row) for row in rows],
    }
    assert_no_owner_keys(payload)
    payload["episode_sha256"] = digest(payload)
    return payload


def validate_public_episode(episode: Mapping[str, Any]) -> None:
    if set(episode) != {
        "schema_version",
        "episode_id",
        "producer",
        "episode_metadata",
        "candidate_budget",
        "actions",
        "episode_sha256",
    }:
        raise ValueError("public episode has unexpected keys")
    # Check the semantic secrecy boundary before the integrity hash so a
    # forbidden owner field cannot be mistaken for an ordinary stale artifact.
    assert_no_owner_keys(episode)
    body = dict(episode)
    claimed = body.pop("episode_sha256")
    if claimed != digest(body):
        raise ValueError("public episode hash mismatch")
    if episode["schema_version"] != f"{SCHEMA_PREFIX}.public_episode":
        raise ValueError("unsupported public episode schema")
    if episode["candidate_budget"] != len(ACTIONS):
        raise ValueError("candidate budget mismatch")
    validate_public_action_table(episode["actions"])
