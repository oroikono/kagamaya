"""Independent structural, budget, and artifact validator for the M5 repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .contract import (
    HARD_CAP_USD,
    LANE_ROOT,
    MODEL_SPECS,
    PRIOR_SPEND_USD,
    SOURCE_MANIFEST_PATH,
    bridge_case_ids,
    canonical_json,
    digest,
    file_sha256,
    prompt_inventory,
    schedule,
    validate_tool_arguments,
    verify_parent_bindings,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _content_hash(value: dict[str, Any], field: str = "content_sha256") -> bool:
    body = dict(value)
    claimed = body.pop(field, None)
    return claimed == digest(body)


def validate_source() -> dict[str, Any]:
    verify_parent_bindings()
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    for row in manifest["files"]:
        path = LANE_ROOT / row["path"]
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"source hash mismatch: {row['path']}")
    body = dict(manifest)
    claimed = body.pop("source_closure_sha256")
    if digest(body) != claimed:
        raise ValueError("source closure mismatch")
    inventory_path = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    if file_sha256(inventory_path) != manifest["prompt_inventory_sha256"]:
        raise ValueError("prompt inventory file hash mismatch")
    if canonical_json(_rows(inventory_path)) != canonical_json(prompt_inventory()):
        raise ValueError("prompt inventory replay mismatch")
    selection = json.loads((LANE_ROOT / "CASE_SELECTION.json").read_text())
    if not _content_hash(selection) or tuple(selection["case_ids"]) != bridge_case_ids():
        raise ValueError("bridge selection mismatch")
    return manifest


def validate_plan() -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    errors: list[str] = []
    plan_path = LANE_ROOT / "results/TOKEN_PLAN.jsonl"
    summary_path = LANE_ROOT / "results/TOKEN_PLAN_SUMMARY.json"
    plan = _rows(plan_path)
    summary = json.loads(summary_path.read_text())
    expected = schedule()
    if len(plan) != 456:
        errors.append("token-plan row count")
    if [row.get("ordinal") for row in plan] != list(range(len(plan))):
        errors.append("token-plan ordinals")
    for ordinal, (row, item) in enumerate(zip(plan, expected)):
        request = item["request"]
        spec = MODEL_SPECS[item["model"]]
        prefix = f"plan {ordinal}"
        if row.get("model") != item["model"] or row.get("case_id") != request["case_id"]:
            errors.append(f"{prefix} schedule binding")
        if row.get("prompt_sha256") != request["prompt_sha256"]:
            errors.append(f"{prefix} prompt hash")
        if not isinstance(row.get("input_tokens"), int) or row["input_tokens"] <= 0:
            errors.append(f"{prefix} input tokens")
            continue
        if row.get("full_output_ceiling_tokens") != request["max_output_tokens"]:
            errors.append(f"{prefix} output ceiling")
        upper = (
            row["input_tokens"] * spec.input_usd_per_million
            + request["max_output_tokens"] * spec.output_usd_per_million
        ) / 1_000_000
        if not _close(row.get("list_price_upper_usd", -1), upper):
            errors.append(f"{prefix} upper price")
    total = sum(float(row.get("list_price_upper_usd", 0)) for row in plan)
    if not _content_hash(summary):
        errors.append("token-plan summary content hash")
    expected_summary = {
        "call_count": len(plan),
        "prior_spend_usd": PRIOR_SPEND_USD,
        "incremental_upper_usd": total,
        "cumulative_upper_usd": PRIOR_SPEND_USD + total,
        "hard_cap_usd": HARD_CAP_USD,
        "plan_sha256": file_sha256(plan_path),
    }
    for key, expected_value in expected_summary.items():
        observed = summary.get(key)
        if isinstance(expected_value, float):
            if not _close(observed, expected_value):
                errors.append(f"token-plan summary {key}")
        elif observed != expected_value:
            errors.append(f"token-plan summary {key}")
    if PRIOR_SPEND_USD + total > HARD_CAP_USD + 1e-12:
        errors.append("token plan exceeds hard cap")
    return plan, summary, errors


def validate_raw(path: Path) -> dict[str, Any]:
    manifest = validate_source()
    plan, _, errors = validate_plan()
    rows = _rows(path)
    expected = schedule()
    if len(rows) != 456:
        errors.append("raw record count mismatch")
    by_key = {(row.get("model_requested"), row.get("case_id")): row for row in rows}
    if len(by_key) != 456:
        errors.append("duplicate model/case record")
    response_ids: list[str] = []
    valid = 0
    incidents = 0
    realized = 0.0
    for item, plan_row in zip(expected, plan):
        model = item["model"]
        request = item["request"]
        row = by_key.get((model, request["case_id"]))
        prefix = f"{model}/{request['case_id']}"
        if row is None:
            errors.append(f"missing {prefix}")
            continue
        body = dict(row)
        claimed = body.pop("record_sha256", None)
        if digest(body) != claimed:
            errors.append(f"record hash {prefix}")
        for key, target in (("ordinal", plan_row["ordinal"]), ("prompt_sha256", request["prompt_sha256"])):
            if row.get(key) != target:
                errors.append(f"{key} {prefix}")
        upper = float(plan_row["list_price_upper_usd"])
        if row.get("schema_version", "").endswith("incident.v1"):
            incidents += 1
            if not _close(row.get("cost_upper_retained_usd", -1), upper):
                errors.append(f"incident upper price {prefix}")
            realized += upper
            continue
        spec = MODEL_SPECS[model]
        if row.get("case_sha256") != request["case_sha256"]:
            errors.append(f"case hash {prefix}")
        if row.get("max_output_tokens") != request["max_output_tokens"]:
            errors.append(f"output ceiling {prefix}")
        if row.get("token_count_plan") != plan_row["input_tokens"]:
            errors.append(f"token-plan binding {prefix}")
        if not _close(row.get("list_price_upper_usd", -1), upper):
            errors.append(f"upper price {prefix}")
        if row.get("source_closure_sha256") != manifest["source_closure_sha256"]:
            errors.append(f"source closure {prefix}")
        if row.get("resolved_model_id") not in spec.accepted_resolved_ids:
            errors.append(f"model identity {prefix}")
        if row.get("credential_recorded_or_hashed") is not False or row.get("owner_side_opened") is not False:
            errors.append(f"boundary flag {prefix}")
        usage = row.get("usage", {})
        if usage.get("input_tokens") != plan_row["input_tokens"]:
            errors.append(f"usage input {prefix}")
        if not isinstance(usage.get("output_tokens"), int) or not 0 <= usage["output_tokens"] <= request["max_output_tokens"]:
            errors.append(f"usage output {prefix}")
            output_tokens = 0
        else:
            output_tokens = usage["output_tokens"]
        actual = (
            int(usage.get("input_tokens", 0)) * spec.input_usd_per_million
            + output_tokens * spec.output_usd_per_million
        ) / 1_000_000
        if not _close(row.get("observed_list_price_usd", -1), actual):
            errors.append(f"observed price {prefix}")
        realized += actual
        response_id = row.get("provider_response_id")
        if not isinstance(response_id, str) or not response_id:
            errors.append(f"response id {prefix}")
        else:
            response_ids.append(response_id)
        if row.get("tool_call_valid") is True:
            try:
                validate_tool_arguments(row.get("parsed_tool_arguments"))
                valid += 1
            except ValueError as exc:
                errors.append(f"parsed arguments {prefix}: {exc}")
    if len(response_ids) != len(set(response_ids)):
        errors.append("duplicate provider response id")

    run_summary_path = LANE_ROOT / "results/RUN_SUMMARY.json"
    run_summary = json.loads(run_summary_path.read_text())
    if not _content_hash(run_summary):
        errors.append("run summary content hash")
    expected_run = {
        "expected_records": 456,
        "observed_records": len(rows),
        "valid_tool_calls": valid,
        "incident_records": incidents,
        "prior_spend_usd": PRIOR_SPEND_USD,
        "incremental_observed_or_reserved_usd": realized,
        "cumulative_observed_or_reserved_usd": PRIOR_SPEND_USD + realized,
        "hard_cap_usd": HARD_CAP_USD,
        "raw_sha256": file_sha256(path),
    }
    for key, expected_value in expected_run.items():
        observed = run_summary.get(key)
        if isinstance(expected_value, float):
            if not _close(observed, expected_value):
                errors.append(f"run summary {key}")
        elif observed != expected_value:
            errors.append(f"run summary {key}")
    if PRIOR_SPEND_USD + realized > HARD_CAP_USD + 1e-12:
        errors.append("realized cumulative spend exceeds hard cap")
    report = {
        "schema_version": "evidence_view_benchmark.m5_repair_validation.v2",
        "valid": not errors,
        "row_count": len(rows),
        "valid_tool_calls": valid,
        "incident_records": incidents,
        "incremental_observed_or_reserved_usd": realized,
        "cumulative_observed_or_reserved_usd": PRIOR_SPEND_USD + realized,
        "errors": errors,
        "raw_sha256": file_sha256(path),
        "source_closure_sha256": manifest["source_closure_sha256"],
    }
    report["content_sha256"] = digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=LANE_ROOT / "results/public_raw.jsonl")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    report = (
        {"source_valid": True, "source_closure_sha256": validate_source()["source_closure_sha256"]}
        if args.source_only
        else validate_raw(args.raw)
    )
    print(json.dumps(report, indent=2))
    if report.get("valid") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
