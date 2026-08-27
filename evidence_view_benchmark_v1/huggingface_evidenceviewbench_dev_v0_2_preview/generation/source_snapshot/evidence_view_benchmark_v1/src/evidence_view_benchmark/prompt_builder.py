"""Byte-bound prompt construction for the Gate-4 M3/M5 smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = LANE_ROOT / "prompts"
ALLOWED_ACTIONS = tuple(f"A{index:02d}" for index in range(12))
PROMPT_FILES = {
    "shared_preamble": "shared_preamble.txt",
    "A1": "goal_A1_obedient.txt",
    "A2": "goal_A2_neutral.txt",
    "A3": "goal_A3_incentivized.txt",
    "A4": "goal_A4_proxy_incentive.txt",
    "A5": "goal_A5_novelty_seeking.txt",
    "oneshot_tail": "oneshot_tail.txt",
    "capability_probe": "gate4/capability_probe.txt",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def read_fragment(name: str, *, prompt_root: Path = PROMPT_ROOT) -> tuple[str, str]:
    try:
        filename = PROMPT_FILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown frozen prompt fragment: {name}") from exc
    payload = (prompt_root / filename).read_bytes()
    return payload.decode("utf-8"), sha256_bytes(payload)


def _validate_public_rows(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = case.get("public_rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("one-shot cases require twelve public rows")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError("public row must be an object")
        row = json.loads(canonical_json(dict(source)))
        expected = ALLOWED_ACTIONS[index]
        if row.get("action_token") != expected:
            raise ValueError("public rows are not in canonical A00-A11 order")
        lowered = canonical_json(row).lower()
        for forbidden in (
            "truth",
            "held_out",
            "owner",
            "oracle",
            "reference",
            "error_to_49",
        ):
            if forbidden in lowered:
                raise ValueError("owner-side material reached prompt construction")
        output.append(row)
    return output


def build_oneshot_prompt(
    case: Mapping[str, Any],
    *,
    arm: str,
    visibility: str,
    prompt_root: Path = PROMPT_ROOT,
) -> dict[str, Any]:
    """Build the only Gate-4-authorized M3 prompt: A1/all_verdicts."""

    if arm != "A1":
        raise ValueError("Gate 4 authorizes only A1")
    if visibility != "all_verdicts":
        raise ValueError("visibility serialization is unresolved; fail closed")
    preamble, preamble_sha = read_fragment("shared_preamble", prompt_root=prompt_root)
    goal, goal_sha = read_fragment(arm, prompt_root=prompt_root)
    tail, tail_sha = read_fragment("oneshot_tail", prompt_root=prompt_root)
    table = canonical_json(_validate_public_rows(case))
    prompt = preamble + goal + table + "\n" + tail
    return {
        "prompt_text": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "fragment_sha256": {
            "shared_preamble": preamble_sha,
            "goal": goal_sha,
            "oneshot_tail": tail_sha,
        },
        "arm": arm,
        "visibility": visibility,
    }


def build_probe_prompt(
    case: Mapping[str, Any],
    *,
    visibility: str,
    prompt_root: Path = PROMPT_ROOT,
) -> dict[str, Any]:
    """Build M5 using the byte-exact question copied from RUN-SPEC."""

    if visibility != "all_verdicts":
        raise ValueError("visibility serialization is unresolved; fail closed")
    preamble, preamble_sha = read_fragment("shared_preamble", prompt_root=prompt_root)
    question, question_sha = read_fragment("capability_probe", prompt_root=prompt_root)
    table = canonical_json(_validate_public_rows(case))
    prompt = preamble + table + "\n" + question
    return {
        "prompt_text": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "fragment_sha256": {
            "shared_preamble": preamble_sha,
            "capability_probe": question_sha,
        },
        "visibility": visibility,
    }
