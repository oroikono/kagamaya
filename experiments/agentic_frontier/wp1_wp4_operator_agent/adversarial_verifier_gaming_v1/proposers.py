"""Proposal-only policies over the frozen twelve-action public interface."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .contract import ACTIONS, POLICY_SEED, SCHEMA_PREFIX, digest, stable_integer
from .public_interface import validate_public_episode


NON_LLM_POLICIES = (
    "random",
    "fixed_25_m5",
    "confidence_max",
    "d_optimal",
    "e_optimal",
    "evolution",
)


def _canonical_argmax(
    rows: Sequence[Mapping[str, Any]], key
) -> Mapping[str, Any]:
    return min(rows, key=lambda row: (-float(key(row)), str(row["action_token"])))


def _random_action(episode_id: str) -> tuple[str, dict[str, Any]]:
    seed = stable_integer(POLICY_SEED, "random", episode_id)
    rng = random.Random(seed)
    token = ACTIONS[rng.randrange(len(ACTIONS))].token
    return token, {"policy_seed": seed, "evaluated_actions": 1}


def _evolution_action(
    episode_id: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Categorical mutation with a frozen twelve-unique-action budget.

    The complete table is public.  The routine deliberately evaluates every
    categorical action once in a hash-seeded order, so it must agree with the
    transparent confidence maximizer up to its canonical tie break.
    """

    seed = stable_integer(POLICY_SEED, "evolution", episode_id)
    rng = random.Random(seed)
    order = [str(row["action_token"]) for row in rows]
    rng.shuffle(order)
    by_token = {str(row["action_token"]): row for row in rows}
    incumbent = order[0]
    trace = []
    for token in order:
        reward = float(by_token[token]["public_objective"])
        trace.append({"action_token": token, "public_objective": reward})
        incumbent_row = by_token[incumbent]
        if (reward, -int(by_token[token]["nodes"]), token) > (
            float(incumbent_row["public_objective"]),
            -int(incumbent_row["nodes"]),
            incumbent,
        ):
            incumbent = token
    # Canonicalize exact ties to match the declared public maximizer.
    maximum = max(float(row["public_objective"]) for row in rows)
    incumbent = min(
        str(row["action_token"])
        for row in rows
        if float(row["public_objective"]) == maximum
    )
    return incumbent, {
        "policy_seed": seed,
        "unique_evaluation_budget": len(ACTIONS),
        "trace_sha256": digest(trace),
    }


def propose_non_llm(
    episode: Mapping[str, Any], policy: str
) -> dict[str, Any]:
    validate_public_episode(episode)
    if policy not in NON_LLM_POLICIES:
        raise ValueError(f"unknown non-LLM policy {policy}")
    rows = episode["actions"]
    metadata: dict[str, Any]
    if policy == "random":
        token, metadata = _random_action(str(episode["episode_id"]))
    elif policy == "fixed_25_m5":
        token, metadata = "A11", {"frozen_action": [25, 5]}
    elif policy == "confidence_max":
        chosen = _canonical_argmax(rows, lambda row: row["public_objective"])
        token, metadata = str(chosen["action_token"]), {
            "objective": "public_objective",
            "evaluated_actions": len(ACTIONS),
        }
    elif policy == "d_optimal":
        chosen = min(
            rows,
            key=lambda row: (
                -float(row["gram_logdet"]),
                int(row["nodes"]),
                int(row["temporal_modes"]),
                str(row["action_token"]),
            ),
        )
        token, metadata = str(chosen["action_token"]), {
            "objective": "standardized_gram_logdet",
            "evaluated_actions": len(ACTIONS),
        }
    elif policy == "e_optimal":
        chosen = min(
            rows,
            key=lambda row: (
                -float(row["gram_lambda_min"]),
                int(row["nodes"]),
                int(row["temporal_modes"]),
                str(row["action_token"]),
            ),
        )
        token, metadata = str(chosen["action_token"]), {
            "objective": "standardized_gram_lambda_min",
            "evaluated_actions": len(ACTIONS),
        }
    else:
        token, metadata = _evolution_action(str(episode["episode_id"]), rows)
    if token not in {action.token for action in ACTIONS}:
        raise RuntimeError("policy emitted an action outside the frozen language")
    body: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.proposal",
        "episode_id": episode["episode_id"],
        "public_episode_sha256": episode["episode_sha256"],
        "policy": policy,
        "proposal_valid": True,
        "selected_action_token": token,
        "scientific_authority": False,
        "hidden_reference_accessed": False,
        "metadata": metadata,
    }
    body["proposal_sha256"] = digest(body)
    return body

