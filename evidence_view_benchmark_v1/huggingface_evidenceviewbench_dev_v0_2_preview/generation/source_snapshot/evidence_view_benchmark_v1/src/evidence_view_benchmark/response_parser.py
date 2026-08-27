"""Primary lenient and sensitivity strict parsers for paired M3 actions."""

from __future__ import annotations

import json
from typing import Any


OUTPUT_KEYS = ("spectral_etdrk4_action", "fd_bdf_action")
ALLOWED_ACTIONS = frozenset(f"A{index:02d}" for index in range(12))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _decode(text: str) -> dict[str, str]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {token}")
            ),
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("response is not one JSON object") from exc
    if not isinstance(value, dict) or set(value) != set(OUTPUT_KEYS):
        raise ValueError("response keys do not match paired-action contract")
    if any(not isinstance(value[key], str) or value[key] not in ALLOWED_ACTIONS for key in OUTPUT_KEYS):
        raise ValueError("response contains an invalid action token")
    return {key: value[key] for key in OUTPUT_KEYS}


def parse_strict(text: str) -> dict[str, str]:
    return _decode(text)


def parse_lenient(text: str) -> dict[str, str]:
    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            opening = lines[0].strip()
            if opening in {"```", "```json", "```JSON"}:
                candidate = "\n".join(lines[1:-1]).strip()
    return _decode(candidate)
