"""Frozen contracts for the full Claude evidence-choice campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = LANE_ROOT.parent
PARENT_TOOL_LANE = BENCHMARK_ROOT / "gate5_toolcalling_v1_20260820"
CASES_PATH = BENCHMARK_ROOT / "cases/cases.jsonl"
SIDECAR_PATH = BENCHMARK_ROOT / "cases/held_out_sidecar.jsonl"
GOAL_BLOCKS_PATH = PARENT_TOOL_LANE / "prompts/incentive_goal_blocks.json"
M3_SCHEMA_PATH = PARENT_TOOL_LANE / "schema/choose_evidence_view.schema.json"
M5_SCHEMA_PATH = LANE_ROOT / "schema/report_evidence_trust.schema.json"
AUTHORIZATION_PATH = LANE_ROOT / "AUTHORIZATION.json"

ARMS = ("A1", "A2", "A3", "A5")
MODELS = ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5")
SMOKE_CASES = ("case_00", "case_01")
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
M3_TOOL_NAME = "choose_evidence_view"
M5_TOOL_NAME = "report_evidence_trust"
M3_MAX_TOKENS = 128
M5_MAX_TOKENS = 256
HARD_CAP_USD = 100.0
CONCURRENCY = 6


@dataclass(frozen=True)
class ModelSpec:
    request_id: str
    accepted_resolved_ids: tuple[str, ...]
    input_usd_per_million: float
    output_usd_per_million: float
    temperature: float | None


MODEL_SPECS = {
    "claude-opus-4-8": ModelSpec(
        "claude-opus-4-8", ("claude-opus-4-8",), 5.0, 25.0, None
    ),
    "claude-sonnet-4-6": ModelSpec(
        "claude-sonnet-4-6", ("claude-sonnet-4-6",), 3.0, 15.0, 0.0
    ),
    "claude-haiku-4-5": ModelSpec(
        "claude-haiku-4-5",
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        1.0,
        5.0,
        0.0,
    ),
}


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
        raise ValueError("public case bank is not the frozen 216-case bank")
    return rows


def read_authorization() -> dict[str, Any]:
    value = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    expected = {
        "models": list(MODELS),
        "cases": 216,
        "arms": list(ARMS),
        "m5_cases_per_model": 216,
        "repeats": 1,
        "max_paid_api_usd": HARD_CAP_USD,
        "qwen_is_a_gate": False,
        "automatic_continuation_after_valid_smoke": True,
    }
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f"authorization mismatch: {key}")
    return value


def _parent_contract():
    import sys

    source_root = BENCHMARK_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from evidence_view_benchmark import gate5_contract

    return gate5_contract


def _goal_instruction(arm: str) -> str:
    if arm == "A1":
        return (BENCHMARK_ROOT / "prompts/goal_A1_obedient.txt").read_text(encoding="utf-8")
    blocks = json.loads(GOAL_BLOCKS_PATH.read_text(encoding="utf-8"))
    return str(blocks["arms"][arm]["public_instruction"]).strip() + "\n\n"


def build_m3_prompt(case: Mapping[str, Any], arm: str) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("arm outside frozen campaign")
    parent = _parent_contract()
    preamble = (BENCHMARK_ROOT / "prompts/shared_preamble.txt").read_text(encoding="utf-8")
    tail = (BENCHMARK_ROOT / "prompts/oneshot_tail.txt").read_text(encoding="utf-8")
    rows = parent.serialize_rows(case, arm=arm, visibility="all_verdicts")
    prompt = preamble + _goal_instruction(arm) + canonical_json(rows) + "\n" + tail
    return {
        "module": "M3",
        "arm": arm,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "tool_name": M3_TOOL_NAME,
        "max_output_tokens": M3_MAX_TOKENS,
    }


def build_m5_prompt(case: Mapping[str, Any]) -> dict[str, Any]:
    parent = _parent_contract()
    row = parent.build_m5_prompt(case)
    return {
        **row,
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "tool_name": M5_TOOL_NAME,
        "max_output_tokens": M5_MAX_TOKENS,
    }


def prompt_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in read_cases():
        for arm in ARMS:
            rows.append(build_m3_prompt(case, arm))
        rows.append(build_m5_prompt(case))
    if len(rows) != 1080:
        raise AssertionError("prompt inventory cardinality changed")
    return rows


def tool_definition(module: str) -> dict[str, Any]:
    if module == "M3":
        name, description, path = (
            M3_TOOL_NAME,
            "Select one public evidence-view action for each locked numerical producer.",
            M3_SCHEMA_PATH,
        )
    elif module == "M5":
        name, description, path = (
            M5_TOOL_NAME,
            "Report the least trustworthy public evidence view for each producer and the public warning signals used.",
            M5_SCHEMA_PATH,
        )
    else:
        raise ValueError("unknown module")
    return {"name": name, "description": description, "input_schema": json.loads(path.read_text())}


def validate_tool_arguments(module: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tool input is not an object")
    parent = _parent_contract()
    if module == "M3":
        expected = {"spectral_etdrk4_action", "fd_bdf_action"}
        if set(value) != expected:
            raise ValueError("M3 key set changed")
        for key in expected:
            if value[key] not in parent.ALLOWED_ACTIONS:
                raise ValueError("invalid M3 action")
    else:
        parent.validate_m5_response(dict(value))
    return json.loads(canonical_json(dict(value)))


def action_geometry(action: str) -> tuple[int, int]:
    index = int(action[1:])
    return (7, 13, 25)[index // 4], (2, 3, 4, 5)[index % 4]


def source_paths() -> list[Path]:
    paths = [
        LANE_ROOT / "DESIGN_FREEZE_PREOUTCOME.md",
        LANE_ROOT / "BUDGET_R2_AMENDMENT.md",
        AUTHORIZATION_PATH,
        M5_SCHEMA_PATH,
        M3_SCHEMA_PATH,
        GOAL_BLOCKS_PATH,
        CASES_PATH,
        BENCHMARK_ROOT / "prompts/goal_A1_obedient.txt",
        BENCHMARK_ROOT / "prompts/shared_preamble.txt",
        BENCHMARK_ROOT / "prompts/oneshot_tail.txt",
        BENCHMARK_ROOT / "prompts/gate5/capability_probe_v1_2.txt",
        BENCHMARK_ROOT / "src/evidence_view_benchmark/gate5_contract.py",
    ]
    paths.extend(sorted((LANE_ROOT / "src").glob("*.py")))
    paths.extend(sorted((LANE_ROOT / "tests").glob("*.py")))
    return sorted(set(paths))
