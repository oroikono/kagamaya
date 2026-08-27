"""Outcome-blind contract for the Gemini-deferred Gate-5A ladder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .gate5_contract import (
    build_m3_prompt,
    build_m5_prompt,
    canonical_json,
    read_cases,
    sha256_bytes,
)


LANE_ROOT = Path(__file__).resolve().parents[2]
ROSTER_PATH = LANE_ROOT / "model_roster_v1_2a.json"
PREFLIGHT_CASE_ID = "case_00"
M3_MAX_OUTPUT = 512
M5_MAX_OUTPUT = 384
PROPOSED_API_CAP_USD = 0.25
EXPECTED_MODELS = (
    ("qwen_local", "Qwen/Qwen2.5-7B-Instruct"),
    ("anthropic", "claude-opus-4-8"),
    ("anthropic", "claude-sonnet-4-6"),
    ("anthropic", "claude-haiku-4-5-20251001"),
)


def read_gate5a_roster(path: Path = ROSTER_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_gate5a_roster(roster: Mapping[str, Any]) -> None:
    models = roster.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("Gate 5A requires exactly four primary models")
    observed = tuple((row.get("provider"), row.get("model_id")) for row in models)
    if observed != EXPECTED_MODELS:
        raise ValueError("Gate 5A roster identity or order changed")
    if len({row["vendor"] for row in models}) != 2:
        raise ValueError("Gate 5A must honestly record two primary vendors")
    deferred = {(row.get("provider"), row.get("reason")) for row in roster.get("deferred", [])}
    if ("google", "DEFERRED_BY_OWNER") not in deferred:
        raise ValueError("Google deferral is absent")
    if ("openai_api", "DEFERRED_NO_API_CREDENTIAL") not in deferred:
        raise ValueError("OpenAI API deferral is absent")
    secondary = roster.get("secondary_hosted_stratum", {})
    if secondary.get("pooled_with_primary") is not False:
        raise ValueError("hosted OpenAI stratum must not be pooled")


def preflight_case(cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = cases if cases is not None else read_cases()
    matches = [row for row in rows if row.get("case_id") == PREFLIGHT_CASE_ID]
    if len(matches) != 1:
        raise ValueError("frozen Gate-5A preflight case is missing or duplicated")
    return matches[0]


def preflight_requests(case: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    selected = dict(case) if case is not None else preflight_case()
    m3 = build_m3_prompt(selected, arm="A1", visibility="all_verdicts")
    m5 = build_m5_prompt(selected)
    return [
        {
            **m3,
            "case_id": selected["case_id"],
            "case_sha256": selected["case_sha256"],
            "max_output_tokens": M3_MAX_OUTPUT,
        },
        {
            **m5,
            "case_id": selected["case_id"],
            "case_sha256": selected["case_sha256"],
            "max_output_tokens": M5_MAX_OUTPUT,
        },
    ]


def byte_upper_cost() -> dict[str, Any]:
    roster = read_gate5a_roster()
    validate_gate5a_roster(roster)
    requests = preflight_requests()
    input_bytes = sum(len(row["prompt_text"].encode("utf-8")) for row in requests)
    output_cap = sum(row["max_output_tokens"] for row in requests)
    rows = []
    for model in roster["models"]:
        value = None
        if model["input_usd_per_mtoken"] is not None:
            value = (
                input_bytes * model["input_usd_per_mtoken"]
                + output_cap * model["output_usd_per_mtoken"]
            ) / 1_000_000
        rows.append(
            {
                "provider": model["provider"],
                "model_id": model["model_id"],
                "calls": 2,
                "input_bytes_token_upper_bound": input_bytes,
                "output_token_ceiling": output_cap,
                "api_cost_byte_upper_usd": None if value is None else round(value, 9),
            }
        )
    total = sum(row["api_cost_byte_upper_usd"] or 0.0 for row in rows)
    return {
        "model_rows": rows,
        "paid_api_cost_byte_upper_total_usd": round(total, 9),
        "proposed_owner_cap_usd": PROPOSED_API_CAP_USD,
        "within_proposed_cap": total <= PROPOSED_API_CAP_USD,
        "note": (
            "ASCII bytes are treated as input tokens and all outputs reach their cap. "
            "This is a conservative list-price ceiling, not observed spend."
        ),
    }


def gate5a_source_bindings() -> dict[str, Any]:
    paths = {
        LANE_ROOT / "PROTOCOL-v1_2.md",
        LANE_ROOT / "PROTOCOL-v1_2a_AMENDMENT.md",
        LANE_ROOT / "GATE5A_LADDER_DESIGN.md",
        LANE_ROOT / "GATE5A1_EXECUTION_INCIDENT_20260820.md",
        LANE_ROOT / "model_roster_v1_2a.json",
        LANE_ROOT / "cases/cases.jsonl",
    }
    paths.update((LANE_ROOT / "prompts").rglob("*.txt"))
    paths.update((LANE_ROOT / "src/evidence_view_benchmark").glob("*.py"))
    paths.update((LANE_ROOT / "tools").glob("*gate5a*.py"))
    paths.update((LANE_ROOT / "tests").glob("test_gate5a*.py"))
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT)),
                "bytes": len(path.read_bytes()),
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    return {
        "files": rows,
        "source_closure_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }
