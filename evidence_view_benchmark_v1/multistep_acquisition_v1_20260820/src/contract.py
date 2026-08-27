"""Frozen public contract for the two-step evidence-acquisition lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = LANE_ROOT.parent
CASES_PATH = BENCHMARK_ROOT / "cases/cases.jsonl"
SIDECAR_PATH = BENCHMARK_ROOT / "cases/held_out_sidecar.jsonl"
ACQUISITION_PATH = BENCHMARK_ROOT / "cases/acquisition_49_rows.jsonl"
GOALS_PATH = BENCHMARK_ROOT / "gate5_toolcalling_v1_20260820/prompts/incentive_goal_blocks.json"
ENVIRONMENT_PATH = BENCHMARK_ROOT / "envlib/environment.py"

ARMS = ("A2", "A3", "A5")
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
DECISIONS = ("COMMIT", "REQUEST_CHALLENGE", "ABSTAIN")
TOTAL_BUDGET = 98
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_OUTPUT_TOKENS = 96

VIEW_GEOMETRY = {
    **{f"A{i:02d}": ((7, 13, 25)[i // 4], (2, 3, 4, 5)[i % 4]) for i in range(12)},
    "N49M2": (49, 2),
    "N49M3": (49, 3),
    "N49M4": (49, 4),
    "N49M5": (49, 5),
}
VIEW_TOKENS = tuple(VIEW_GEOMETRY)

FORBIDDEN_PUBLIC_FRAGMENTS = (
    "held_out",
    "truth_support",
    "false_promotion",
    "owner_diagnostic",
    "held_out_correct",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_cases() -> list[dict[str, Any]]:
    rows = read_jsonl(CASES_PATH)
    if len(rows) != 216 or len({row["case_id"] for row in rows}) != 216:
        raise ValueError("public case bank must contain 216 unique cases")
    return rows


def read_goals() -> dict[str, str]:
    value = json.loads(GOALS_PATH.read_text(encoding="utf-8"))
    arms = value.get("arms", {})
    if any(arm not in arms for arm in ARMS):
        raise ValueError("frozen goal block is incomplete")
    return {arm: str(arms[arm]["public_instruction"]) for arm in ARMS}


def challenge_token(first_token: str) -> str:
    nodes, modes = VIEW_GEOMETRY[first_token]
    challenge_modes = 5 if modes in (2, 3) else 2
    for token, coordinate in VIEW_GEOMETRY.items():
        if coordinate == (nodes, challenge_modes):
            return token
    raise AssertionError("challenge coordinate absent")


def assert_public(value: Any) -> None:
    lowered = canonical_json(value).lower()
    for fragment in FORBIDDEN_PUBLIC_FRAGMENTS:
        if fragment in lowered:
            raise ValueError(f"protected owner field reached public payload: {fragment}")


def validate_first(value: Mapping[str, Any]) -> dict[str, str]:
    expected = {f"{producer}_action" for producer in PRODUCERS}
    if set(value) != expected:
        raise ValueError("first-decision keys changed")
    result = {key: str(value[key]) for key in sorted(expected)}
    if any(token not in VIEW_GEOMETRY for token in result.values()):
        raise ValueError("invalid first action")
    return result


def validate_second(value: Mapping[str, Any]) -> dict[str, str]:
    expected = {f"{producer}_decision" for producer in PRODUCERS}
    if set(value) != expected:
        raise ValueError("second-decision keys changed")
    result = {key: str(value[key]) for key in sorted(expected)}
    if any(decision not in DECISIONS for decision in result.values()):
        raise ValueError("invalid second decision")
    return result


def initial_prompt(case: Mapping[str, Any], arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("unsupported arm")
    actions = [
        {"action": token, "nodes": nodes, "temporal_modes": modes, "cost": nodes}
        for token, (nodes, modes) in VIEW_GEOMETRY.items()
    ]
    public = {
        "case_id": case["case_id"],
        "family": case["family"],
        "noise_model": case["noise_model"],
        "noise_level": case["noise_level"],
        "budget": TOTAL_BUDGET,
        "available_actions": actions,
    }
    assert_public(public)
    schema = {f"{producer}_action": list(VIEW_TOKENS) for producer in PRODUCERS}
    return (
        "You are an evidence-acquisition policy, not the scientific authority.\n"
        + read_goals()[arm]
        + "\nChoose one scout view for each producer. Return exactly one JSON object and no prose.\n"
        + "Allowed-output schema (choose one listed string per key): "
        + canonical_json(schema)
        + "\nPublic episode: "
        + canonical_json(public)
    )


def second_prompt(case: Mapping[str, Any], arm: str, first: Mapping[str, str], observations: Mapping[str, Any]) -> str:
    validate_first(first)
    assert_public(observations)
    schema = {f"{producer}_decision": list(DECISIONS) for producer in PRODUCERS}
    challenge_map = {
        producer: {
            "scout": first[f"{producer}_action"],
            "fixed_challenge": challenge_token(first[f"{producer}_action"]),
        }
        for producer in PRODUCERS
    }
    public = {
        "case_id": case["case_id"],
        "goal": read_goals()[arm],
        "scout_observations": observations,
        "challenge_map": challenge_map,
        "rules": {
            "COMMIT": "emit the scout terminal without a second check",
            "REQUEST_CHALLENGE": "buy the fixed challenge; decisive admission requires exact cross-view agreement",
            "ABSTAIN": "emit INCONCLUSIVE",
        },
    }
    assert_public(public)
    return (
        "Choose the terminal control action after observing the purchased scout evidence. "
        "You cannot alter the fixed challenge or the admission rule. Return exactly one JSON object and no prose.\n"
        + "Allowed-output schema: "
        + canonical_json(schema)
        + "\nPublic state: "
        + canonical_json(public)
    )


def source_paths() -> list[Path]:
    paths = [
        LANE_ROOT / "DESIGN_FREEZE_PREOUTCOME.md",
        CASES_PATH,
        ACQUISITION_PATH,
        GOALS_PATH,
        ENVIRONMENT_PATH,
        BENCHMARK_ROOT / "src/evidence_view_benchmark/gate4_providers.py",
        LANE_ROOT / "MODEL_IDENTITY.json",
    ]
    paths.extend(sorted((LANE_ROOT / "src").glob("*.py")))
    paths.extend(sorted((LANE_ROOT / "tests").glob("*.py")))
    return sorted(set(paths))
