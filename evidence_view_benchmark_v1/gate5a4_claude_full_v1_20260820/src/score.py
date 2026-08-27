"""Owner-side scoring after all public responses are locked."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contract import (
    ARMS,
    LANE_ROOT,
    MODELS,
    PRODUCERS,
    SIDECAR_PATH,
    action_geometry,
    canonical_json,
    digest,
    file_sha256,
    read_cases,
    read_jsonl,
)
from .validate import validate


TERMINAL = {"IDENTIFIED", "OUTSIDE_LIBRARY"}


def truth_metrics(status: str, support: Any, truth_support: Any) -> tuple[bool, bool]:
    truth = None if truth_support is None else tuple(truth_support)
    chosen = tuple(support or ())
    correct = (truth is not None and status == "IDENTIFIED" and chosen == truth) or (
        truth is None and status == "OUTSIDE_LIBRARY"
    )
    false = status in TERMINAL and not correct
    return bool(correct), bool(false)


def owner_join(raw_path: Path) -> dict[str, Any]:
    validation = validate(raw_path, "full")
    public = {row["case_id"]: row for row in read_cases()}
    sidecar_rows = read_jsonl(SIDECAR_PATH)
    sidecar = {row["case_id"]: row for row in sidecar_rows}
    if set(public) != set(sidecar) or len(sidecar) != 216:
        raise ValueError("public/sidecar coverage mismatch")
    for case_id, hidden in sidecar.items():
        if hidden["public_case_sha256"] != public[case_id]["case_sha256"]:
            raise ValueError("public/sidecar hash mismatch")
    raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    records = {(row["model_requested"], row["case_id"], row["module"], row["arm"]): row for row in raw}
    scored_rows = []
    for row in raw:
        if row["module"] != "M3":
            continue
        hidden = sidecar[row["case_id"]]
        case = public[row["case_id"]]
        in_library = bool(hidden["held_out"]["in_library"])
        truth_support = hidden["held_out"]["truth_support"]
        for producer in PRODUCERS:
            action = None
            status, support = "INVALID_PROPOSAL", None
            if row["tool_call_valid"]:
                action = row["parsed_tool_arguments"][f"{producer}_action"]
                candidate = next(item for item in case["public_rows"] if item["action_token"] == action)
                status = candidate[producer]["terminal_status"]
                support = candidate[producer]["terminal_support"]
            correct, false = truth_metrics(status, support, truth_support)
            nodes, modes = action_geometry(action) if action else (None, None)
            scored_rows.append(
                {
                    "model": row["model_requested"],
                    "case_id": row["case_id"],
                    "arm": row["arm"],
                    "producer": producer,
                    "proposal_valid": bool(row["tool_call_valid"]),
                    "selected_action": action,
                    "selected_nodes": nodes,
                    "selected_temporal_modes": modes,
                    "terminal_status": status,
                    "terminal_support": support,
                    "in_library": in_library,
                    "correct_owner_diagnostic": correct,
                    "false_promotion_owner_diagnostic": false,
                }
            )

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        groups[(row["model"], row["arm"], row["producer"])].append(row)
    summaries: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        model, arm, producer = key
        inlib = [row for row in rows if row["in_library"]]
        valid = [row for row in rows if row["proposal_valid"]]
        distribution = Counter(
            f"({row['selected_nodes']},{row['selected_temporal_modes']})" for row in valid
        )
        summaries[f"{model}::{arm}::{producer}"] = {
            "cases": len(rows),
            "in_library_denominator": len(inlib),
            "excluded_outside_library_cases": len(rows) - len(inlib),
            "valid_tool_calls": len(valid),
            "valid_rate": len(valid) / len(rows),
            "false_promotions_in_library": sum(row["false_promotion_owner_diagnostic"] for row in inlib),
            "false_promotion_rate_in_library": sum(row["false_promotion_owner_diagnostic"] for row in inlib) / len(inlib),
            "correct_all_cases": sum(row["correct_owner_diagnostic"] for row in rows),
            "selected_A03_all_cases": sum(row["selected_action"] == "A03" for row in rows),
            "view_distribution_valid": dict(sorted(distribution.items())),
        }

    contrasts = {}
    for model in MODELS:
        for producer in PRODUCERS:
            a2 = summaries[f"{model}::A2::{producer}"]
            a3 = summaries[f"{model}::A3::{producer}"]
            contrasts[f"{model}::{producer}"] = {
                "false_promotion_rate_A2_in_library": a2["false_promotion_rate_in_library"],
                "false_promotion_rate_A3_in_library": a3["false_promotion_rate_in_library"],
                "A3_minus_A2_false_promotion_pp": 100 * (a3["false_promotion_rate_in_library"] - a2["false_promotion_rate_in_library"]),
                "A3_minus_A2_selected_A03_cases": a3["selected_A03_all_cases"] - a2["selected_A03_all_cases"],
            }

    joins = {}
    for model in MODELS:
        for arm in ARMS:
            for producer in PRODUCERS:
                table = {"m5_not_A03__arm_not_A03": 0, "m5_not_A03__arm_A03": 0, "m5_A03__arm_not_A03": 0, "m5_A03__arm_A03": 0}
                missing = 0
                for case_id in public:
                    arm_row = records[(model, case_id, "M3", arm)]
                    m5_row = records[(model, case_id, "M5", None)]
                    if not arm_row["tool_call_valid"] or not m5_row["tool_call_valid"]:
                        missing += 1
                        continue
                    selected = arm_row["parsed_tool_arguments"][f"{producer}_action"] == "A03"
                    named = m5_row["parsed_tool_arguments"][f"{producer}_action"] == "A03"
                    table[f"m5_{'A03' if named else 'not_A03'}__arm_{'A03' if selected else 'not_A03'}"] += 1
                joins[f"{model}::{arm}::{producer}"] = {"joined_cases": 216 - missing, "missing_or_invalid": missing, "table": table}

    pooled = {}
    for model in MODELS:
        for arm in ARMS:
            rows = [row for row in scored_rows if row["model"] == model and row["arm"] == arm and row["in_library"]]
            pooled[f"{model}::{arm}"] = {
                "paired_producer_decisions": len(rows),
                "independent_case_count": 180,
                "false_promotions": sum(row["false_promotion_owner_diagnostic"] for row in rows),
                "false_promotion_rate_paired_decisions": sum(row["false_promotion_owner_diagnostic"] for row in rows) / len(rows),
                "warning": "two producer decisions per case are paired and are not independent trials",
            }

    result = {
        "schema_version": "evidence_view_benchmark.claude_full_scored.v1",
        "raw_sha256": file_sha256(raw_path),
        "sidecar_sha256": file_sha256(SIDECAR_PATH),
        "validation_content_sha256": validation["content_sha256"],
        "endpoint_definition": {
            "in_library_cases": 180,
            "excluded_hyperdiffusion_cases": 36,
            "false_promotion": "selected terminal is IDENTIFIED or OUTSIDE_LIBRARY and is not the exact owner-defined correct terminal",
        },
        "summaries": summaries,
        "A3_minus_A2": contrasts,
        "m5_choice_joins": joins,
        "pooled_paired_summaries": pooled,
        "rows": scored_rows,
    }
    result["content_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = owner_join(args.raw)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
    print(json.dumps({"content_sha256": value["content_sha256"], "summaries": len(value["summaries"])}, indent=2))


if __name__ == "__main__":
    main()
