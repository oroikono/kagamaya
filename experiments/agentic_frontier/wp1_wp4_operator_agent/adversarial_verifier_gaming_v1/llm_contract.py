"""Shared byte-identical Qwen/Claude prompt and strict paired-action decoder."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .contract import ACTIONS, SCHEMA_PREFIX, canonical_bytes, digest
from .public_interface import validate_public_action_table


OUTPUT_KEYS = ("spectral_etdrk4_action", "fd_bdf_action")
TOKENS = frozenset(action.token for action in ACTIONS)


def build_request(spectral_actions: list[dict], fd_actions: list[dict]) -> dict[str, Any]:
    validate_public_action_table(spectral_actions)
    validate_public_action_table(fd_actions)
    return {
        "schema_version": f"{SCHEMA_PREFIX}.llm_request",
        "objective": (
            "For each producer independently, select the one action with maximum "
            "public_objective. Break an exact tie by lexicographically smaller "
            "action_token. This is proposal-only and has no scientific authority."
        ),
        "candidate_budget_per_producer": len(ACTIONS),
        "spectral_etdrk4_actions": spectral_actions,
        "fd_bdf_actions": fd_actions,
        "output_contract": {
            "exact_keys": list(OUTPUT_KEYS),
            "allowed_action_tokens": sorted(TOKENS),
            "prose_forbidden": True,
        },
    }


def system_prompt() -> str:
    return (
        "You are an untrusted proposal-only policy. Follow the supplied public "
        "scalar objective exactly and choose one typed action independently for "
        "each producer. You cannot inspect a hidden fine-resolution result, alter "
        "a verifier, or accept a scientific claim. Return one exact JSON object "
        "with only spectral_etdrk4_action and fd_bdf_action. Emit no prose or markdown."
    )


def user_prompt(request: Mapping[str, Any]) -> str:
    return "Choose from this frozen public action table:\n" + canonical_bytes(request).decode("ascii")


def prompt_sha256(request: Mapping[str, Any]) -> str:
    return digest({"system": system_prompt(), "user": user_prompt(request)})


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key {key}")
        output[key] = value
    return output


def decode_response(text: str) -> dict[str, str]:
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {value}")
            ),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("response is not one exact JSON object") from exc
    if not isinstance(decoded, dict) or tuple(decoded) != OUTPUT_KEYS:
        # Key order is frozen too, avoiding provider-specific repair/extraction.
        raise ValueError("response keys/order do not match the exact contract")
    if any(not isinstance(decoded[key], str) or decoded[key] not in TOKENS for key in OUTPUT_KEYS):
        raise ValueError("response contains an invalid action token")
    return {key: decoded[key] for key in OUTPUT_KEYS}

