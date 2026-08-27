"""Frozen contracts for the M5 transport repair."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = LANE_ROOT.parent
PARENT_FULL = BENCHMARK_ROOT / "gate5a4_claude_full_v1_20260820"
CASES_PATH = BENCHMARK_ROOT / "cases/cases.jsonl"
OLD_NORMALIZED_PATH = PARENT_FULL / "results/full_public_normalized.jsonl"
OLD_INVENTORY_PATH = PARENT_FULL / "PROMPT_INVENTORY.jsonl"
SCHEMA_PATH = LANE_ROOT / "schema/report_evidence_trust_v2.schema.json"
TAIL_PATH = LANE_ROOT / "prompts_capability_probe_v2.txt"
GATES_PATH = LANE_ROOT / "NULL_AND_CLAIM_GATES.json"
AUTHORIZATION_PATH = LANE_ROOT / "AUTHORIZATION.json"
SOURCE_MANIFEST_PATH = LANE_ROOT / "SOURCE_MANIFEST_R2.json"

REPAIR_MODELS = ("claude-opus-4-8", "claude-haiku-4-5")
BRIDGE_MODEL = "claude-sonnet-4-6"
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
ARMS = ("A1", "A2", "A3", "A5")
ALLOWED_ACTIONS = tuple(f"A{i:02d}" for i in range(12))
SIGNALS = (
    "terminal_disagreement",
    "cross_producer_disagreement",
    "residual",
    "score_gap",
    "conditioning",
    "resolution_test_space",
    "cost",
    "other",
)
TOOL_NAME = "report_evidence_trust_v2"
MAX_OUTPUT_TOKENS = 1024
CONCURRENCY = 6
PRIOR_SPEND_USD = 61.808521
HARD_CAP_USD = 100.0

PARENT_BINDINGS = {
    "cases/cases.jsonl": "cda68d3da3b74edc0f998e651f4d23c0c5c9add2d18ff55294db38db86451f6d",
    "prompts/shared_preamble.txt": "82f168322ef147677f010d4fdeb5db60fe8eb76349d169ab94d6329498904065",
    "prompts/gate5/capability_probe_v1_2.txt": "2824766fc56ceb3e8d9cfd8874ef8063d89a70e618c13c7bad992835d75340f5",
    "src/evidence_view_benchmark/gate5_contract.py": "5834ed14bfd4bee89d331a82f20f7998a5bf6a70f708c991e43b719e40a4ed51",
    "gate5a4_claude_full_v1_20260820/results/full_public_normalized.jsonl": "97831d88f2375a267f06e48233b3ff59478c28a7b8a060b6ed28c6834b861d06",
    "gate5a4_claude_full_v1_20260820/PROMPT_INVENTORY.jsonl": "6303e300556130ec92ca4c6e17669cc3c1309fd93bfd0ab4b1b5765ce770bc5e",
    "gate5a4_claude_full_v1_20260820/results/FULL_CAMPAIGN_SUMMARY.json": "7095b0c6a767321a1b101fea17299d71023d90514ec7ab1c1ada548d3a535b47",
}


@dataclass(frozen=True)
class ModelSpec:
    request_id: str
    accepted_resolved_ids: tuple[str, ...]
    input_usd_per_million: float
    output_usd_per_million: float
    temperature: float | None


MODEL_SPECS = {
    "claude-opus-4-8": ModelSpec("claude-opus-4-8", ("claude-opus-4-8",), 5.0, 25.0, None),
    "claude-sonnet-4-6": ModelSpec("claude-sonnet-4-6", ("claude-sonnet-4-6",), 3.0, 15.0, 0.0),
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


def verify_parent_bindings() -> None:
    for relative, expected in PARENT_BINDINGS.items():
        path = BENCHMARK_ROOT / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"parent binding mismatch: {relative}")


def parent_contract():
    path = BENCHMARK_ROOT / "src"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from evidence_view_benchmark import gate5_contract

    return gate5_contract


def read_cases() -> list[dict[str, Any]]:
    verify_parent_bindings()
    rows = read_jsonl(CASES_PATH)
    if len(rows) != 216 or len({row["case_id"] for row in rows}) != 216:
        raise ValueError("case bank cardinality mismatch")
    return rows


def bridge_case_ids() -> tuple[str, ...]:
    ids = [row["case_id"] for row in read_cases()]
    return tuple(sorted(ids, key=lambda x: hashlib.sha256(x.encode()).hexdigest())[:24])


def build_prompt(case: Mapping[str, Any]) -> dict[str, Any]:
    parent = parent_contract()
    preamble = (BENCHMARK_ROOT / "prompts/shared_preamble.txt").read_text(encoding="utf-8")
    rows = parent.serialize_rows(case, arm="A2", visibility="all_verdicts")
    prompt = preamble + canonical_json(rows) + "\n" + TAIL_PATH.read_text(encoding="utf-8")
    return {
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "prompt_text": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "tool_name": TOOL_NAME,
    }


def prompt_inventory() -> list[dict[str, Any]]:
    return [build_prompt(case) for case in read_cases()]


def schedule() -> list[dict[str, Any]]:
    inventory = prompt_inventory()
    bridge = set(bridge_case_ids())
    rows = [
        {"model": model, "request": request}
        for model in REPAIR_MODELS
        for request in inventory
    ]
    rows.extend(
        {"model": BRIDGE_MODEL, "request": request}
        for request in inventory
        if request["case_id"] in bridge
    )
    if len(rows) != 456:
        raise AssertionError("schedule cardinality mismatch")
    return rows


def tool_definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Report the least trustworthy public evidence view for each producer, one primary warning signal, and a concise explanation.",
        "input_schema": json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
    }


def validate_tool_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tool input is not an object")
    expected = {"spectral_etdrk4_action", "fd_bdf_action", "primary_signal", "explanation"}
    if set(value) != expected:
        raise ValueError("tool input key set mismatch")
    for key in ("spectral_etdrk4_action", "fd_bdf_action"):
        if value[key] not in ALLOWED_ACTIONS:
            raise ValueError("invalid action")
    if value["primary_signal"] not in SIGNALS:
        raise ValueError("invalid primary signal")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or not explanation.strip() or len(explanation) > 600:
        raise ValueError("invalid explanation")
    if len(explanation.split()) > 80:
        raise ValueError("explanation exceeds 80 words")
    return json.loads(canonical_json(dict(value)))


def read_authorization() -> dict[str, Any]:
    value = json.loads(AUTHORIZATION_PATH.read_text(encoding="utf-8"))
    expected = {
        "generation_calls": 456,
        "prior_spend_usd": PRIOR_SPEND_USD,
        "hard_cumulative_cap_usd": HARD_CAP_USD,
        "no_retry": True,
    }
    for key, target in expected.items():
        if value.get(key) != target:
            raise ValueError(f"authorization mismatch: {key}")
    return value


def source_paths() -> list[Path]:
    paths = [
        LANE_ROOT / "DESIGN_FREEZE_PREOUTCOME.md",
        LANE_ROOT / "PREOUTCOME_AMENDMENT_R2.md",
        LANE_ROOT / "NULL_AND_CLAIM_GATES.json",
        AUTHORIZATION_PATH,
        TAIL_PATH,
        SCHEMA_PATH,
    ]
    paths.extend(sorted((LANE_ROOT / "src").glob("*.py")))
    paths.extend(sorted((LANE_ROOT / "tests").glob("*.py")))
    return sorted(paths)
