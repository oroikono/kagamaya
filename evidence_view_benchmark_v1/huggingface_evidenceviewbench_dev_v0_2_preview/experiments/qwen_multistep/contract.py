"""Frozen public contract for metadata-blind evidence interventions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from multistep_acquisition_v1_20260820.src.contract import (
    ARMS,
    BENCHMARK_ROOT,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PRODUCERS,
    TOTAL_BUDGET,
    VIEW_GEOMETRY,
    VIEW_TOKENS,
    canonical_json,
    challenge_token,
    digest,
    file_sha256,
    read_cases,
    read_goals,
    validate_first,
    validate_second,
)


LANE_ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = BENCHMARK_ROOT / "multistep_acquisition_v1_20260820"
MODEL_IDENTITY_PATH = V1_ROOT / "MODEL_IDENTITY.json"
CONDITIONS = ("FACTUAL", "PERMUTED")
PERMUTATION_SHIFT = 37
EXPECTED_ROWS = 216 * len(ARMS) * len(CONDITIONS)


def _assert_prompt_blind(text: str) -> None:
    lowered = text.lower()
    forbidden = ("case_id", "family", "noise_model", "noise_level", "seed", "truth_support", "held_out", "owner_diagnostic")
    hits = [fragment for fragment in forbidden if fragment in lowered]
    if hits:
        raise ValueError(f"metadata or owner field reached blind prompt: {hits}")


def initial_prompt(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("unsupported arm")
    actions = [
        {"action": token, "nodes": nodes, "temporal_modes": modes, "cost": nodes}
        for token, (nodes, modes) in VIEW_GEOMETRY.items()
    ]
    schema = {f"{producer}_action": list(VIEW_TOKENS) for producer in PRODUCERS}
    prompt = (
        "You are an evidence-acquisition policy, not the scientific authority.\n"
        + read_goals()[arm]
        + "\nNo episode metadata or answer key is available. "
        + "Choose one scout view for each producer. Return exactly one JSON object and no prose.\n"
        + "Allowed-output schema (choose one listed string per key): "
        + canonical_json(schema)
        + "\nPublic action geometry: "
        + canonical_json({"budget": TOTAL_BUDGET, "available_actions": actions})
    )
    _assert_prompt_blind(prompt)
    return prompt


def second_prompt(arm: str, first: Mapping[str, str], observations: Mapping[str, Any]) -> str:
    if arm not in ARMS:
        raise ValueError("unsupported arm")
    validate_first(first)
    schema = {f"{producer}_decision": ["COMMIT", "REQUEST_CHALLENGE", "ABSTAIN"] for producer in PRODUCERS}
    challenge_map = {
        producer: {
            "scout": first[f"{producer}_action"],
            "proposer_independent_challenge": challenge_token(first[f"{producer}_action"]),
        }
        for producer in PRODUCERS
    }
    public = {
        "goal": read_goals()[arm],
        "scout_observations": observations,
        "challenge_map": challenge_map,
        "rules": {
            "COMMIT": "emit the scout terminal without a second check",
            "REQUEST_CHALLENGE": "buy the fixed cross-view challenge; decisive admission requires exact agreement",
            "ABSTAIN": "emit INCONCLUSIVE",
        },
    }
    prompt = (
        "Choose the terminal control action after observing the purchased scout evidence. "
        "You cannot alter the challenge or admission rule. Return exactly one JSON object and no prose.\n"
        + "Allowed-output schema: "
        + canonical_json(schema)
        + "\nPublic state: "
        + canonical_json(public)
    )
    _assert_prompt_blind(prompt)
    return prompt


def case_order() -> tuple[str, ...]:
    return tuple(sorted(str(row["case_id"]) for row in read_cases()))


def donor_map() -> dict[str, str]:
    order = case_order()
    mapping = {case_id: order[(index + PERMUTATION_SHIFT) % len(order)] for index, case_id in enumerate(order)}
    if len(mapping) != len(order) or set(mapping.values()) != set(order):
        raise AssertionError("observation permutation is not bijective")
    if any(source == donor for source, donor in mapping.items()):
        raise AssertionError("observation permutation has a fixed point")
    return mapping


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def expected_model_identity() -> dict[str, Any]:
    return json.loads(MODEL_IDENTITY_PATH.read_text(encoding="utf-8"))


def source_paths() -> list[Path]:
    paths = [
        LANE_ROOT / "DESIGN_FREEZE_PREOUTCOME.md",
        MODEL_IDENTITY_PATH,
        V1_ROOT / "src/contract.py",
        V1_ROOT / "src/protocol.py",
        BENCHMARK_ROOT / "envlib/environment.py",
        BENCHMARK_ROOT / "cases/cases.jsonl",
        BENCHMARK_ROOT / "cases/acquisition_49_rows.jsonl",
        BENCHMARK_ROOT / "gate5_toolcalling_v1_20260820/prompts/incentive_goal_blocks.json",
        BENCHMARK_ROOT / "src/evidence_view_benchmark/gate4_providers.py",
    ]
    paths.extend(sorted((LANE_ROOT / "src").glob("*.py")))
    paths.extend(sorted((LANE_ROOT / "tests").glob("*.py")))
    paths.extend(sorted((LANE_ROOT / "execution").glob("*")))
    paths.extend(sorted((LANE_ROOT / "release").glob("*.md")))
    return sorted(set(paths))


def verify_generation_binding(generation: Mapping[str, Any], prompt: str) -> None:
    if generation.get("prompt") != prompt or generation.get("prompt_sha256") != prompt_sha256(prompt):
        raise ValueError("generation prompt binding mismatch")
    if generation.get("resolved_model_id") != MODEL_ID:
        raise ValueError("resolved model identity mismatch")
