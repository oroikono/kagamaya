"""Validate the raw schedule and normalize recorded incidents to invalid proposals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..src.contract import (
    LANE_ROOT,
    MODEL_SPECS,
    MODELS,
    canonical_json,
    digest,
    file_sha256,
    prompt_inventory,
    validate_tool_arguments,
)


def expected_schedule() -> dict[tuple[str, str, str, str | None], dict[str, Any]]:
    return {
        (model, prompt["case_id"], prompt["module"], prompt["arm"]): prompt
        for model in MODELS
        for prompt in prompt_inventory()
    }


def validate_and_normalize(raw_path: Path, normalized_path: Path) -> dict[str, Any]:
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    expected = expected_schedule()
    if len(raw) != 3240 or len(expected) != 3240:
        raise ValueError("raw or expected schedule cardinality changed")
    observed = set()
    normalized = []
    incidents = []
    r2_manifest = json.loads((LANE_ROOT / "SOURCE_MANIFEST_FULL_R2.json").read_text())
    r2_closure = r2_manifest["source_closure_sha256"]
    for row in raw:
        claimed = row.get("record_sha256")
        unsigned = dict(row)
        unsigned.pop("record_sha256", None)
        if claimed != digest(unsigned):
            raise ValueError("raw record hash mismatch")
        key = (row.get("model_requested"), row.get("case_id"), row.get("module"), row.get("arm"))
        if key not in expected or key in observed:
            raise ValueError(f"unknown or duplicate schedule key: {key}")
        observed.add(key)
        prompt = expected[key]
        if row.get("prompt_sha256") != prompt["prompt_sha256"]:
            raise ValueError("prompt digest mismatch")
        if row["schema_version"] == "evidence_view_benchmark.claude_full_raw.v1":
            if row.get("source_closure_sha256") != r2_closure:
                raise ValueError("raw response has wrong R2 source closure")
            if row.get("resolved_model_id") not in MODEL_SPECS[row["model_requested"]].accepted_resolved_ids:
                raise ValueError("resolved model identity mismatch")
            if row.get("tool_call_valid"):
                validate_tool_arguments(row["module"], row["parsed_tool_arguments"])
            normalized.append(row)
            continue
        if row["schema_version"] != "evidence_view_benchmark.claude_full_incident.v1":
            raise ValueError("unexpected record schema")
        incidents.append(row)
        replacement = {
            "schema_version": "evidence_view_benchmark.claude_full_raw.v1",
            "stage": "full",
            "scored": False,
            "owner_side_opened": False,
            "ordinal": row["ordinal"],
            "model_requested": row["model_requested"],
            "resolved_model_id": None,
            "provider_response_id": None,
            "case_id": row["case_id"],
            "case_sha256": prompt["case_sha256"],
            "module": row["module"],
            "arm": row["arm"],
            "prompt_sha256": row["prompt_sha256"],
            "tool_name": prompt["tool_name"],
            "max_output_tokens": prompt["max_output_tokens"],
            "token_count_plan": None,
            "list_price_upper_usd": row["cost_upper_retained_usd"],
            "observed_list_price_usd": None,
            "source_closure_sha256": r2_closure,
            "credential_recorded_or_hashed": False,
            "stop_reason": "API_TIMEOUT_NO_RETRY",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "latency_seconds": None,
            "raw_content_blocks": [],
            "tool_call_valid": False,
            "tool_parse_error": row["error"],
            "parsed_tool_arguments": None,
            "recorded_at_utc": row["recorded_at_utc"],
            "transport_incident_record_sha256": row["record_sha256"],
        }
        replacement["record_sha256"] = digest(replacement)
        normalized.append(replacement)
    if observed != set(expected):
        raise ValueError("raw schedule is incomplete")
    if normalized_path.exists():
        raise FileExistsError(normalized_path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_path.open("x", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    result = {
        "schema_version": "evidence_view_benchmark.claude_full_raw_validation.r2",
        "raw_rows": len(raw),
        "schedule_keys": len(observed),
        "raw_incidents": len(incidents),
        "normalized_invalid_incidents": len(incidents),
        "raw_sha256": file_sha256(raw_path),
        "normalized_sha256": file_sha256(normalized_path),
        "source_closure_sha256": r2_closure,
        "valid": True,
    }
    result["content_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = validate_and_normalize(args.raw, args.normalized)
    with args.report.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
