"""Owner-side scoring and frozen gate evaluation for the Qwen loop."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contract import ARMS, PRODUCERS, SIDECAR_PATH, canonical_json, digest, read_jsonl
from .score import truth_metrics
from .validate_qwen import validate


VARIANTS = ("actual", "forced_commit", "forced_challenge")


def _endpoint(row: dict[str, Any], variant: str, producer: str, hidden: dict[str, Any]) -> dict[str, Any]:
    public = row.get(variant)
    if public is None:
        status, support, node_cost, decision = "INCONCLUSIVE", None, 0, "INVALID_FIRST"
    else:
        record = public["producers"][producer]
        status = record["admission"]["status"]
        support = record["admission"]["support"]
        node_cost = record["node_cost"]
        decision = record["second_decision"]
    in_library = bool(hidden["held_out"]["in_library"])
    correct, false = truth_metrics(status, support, in_library, hidden["held_out"]["truth_support"])
    return {
        "status": status,
        "support": support,
        "node_cost": node_cost,
        "decision": decision,
        "correct": correct,
        "false_promotion": false,
        "in_library": in_library,
    }


def score(path: Path) -> dict[str, Any]:
    validation = validate(path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    hidden_by_case = {row["case_id"]: row for row in read_jsonl(SIDECAR_PATH)}
    endpoint_rows = []
    for row in artifact["rows"]:
        hidden = hidden_by_case[row["case_id"]]
        if hidden["public_case_sha256"] != row["case_sha256"]:
            raise ValueError("sidecar/public case binding mismatch")
        for variant in VARIANTS:
            for producer in PRODUCERS:
                endpoint_rows.append({
                    "case_id": row["case_id"],
                    "arm": row["arm"],
                    "variant": variant,
                    "producer": producer,
                    "first_valid": row["first_valid"],
                    "second_valid": row["second_valid"],
                    **_endpoint(row, variant, producer, hidden),
                })
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        groups[(row["arm"], row["variant"], row["producer"])].append(row)
    summaries = {}
    for key, rows in sorted(groups.items()):
        summaries["::".join(key)] = {
            "cases": len(rows),
            "first_valid": sum(row["first_valid"] for row in rows),
            "second_valid": sum(row["second_valid"] for row in rows),
            "correct": sum(row["correct"] for row in rows),
            "false_promotions": sum(row["false_promotion"] for row in rows),
            "inconclusive": sum(row["status"] == "INCONCLUSIVE" for row in rows),
            "challenge_requests": sum(row["decision"] == "REQUEST_CHALLENGE" for row in rows),
            "mean_node_cost": sum(row["node_cost"] for row in rows) / len(rows),
            "correct_minus_false": sum(row["correct"] for row in rows) - sum(row["false_promotion"] for row in rows),
        }

    paired = {}
    safe_to_false = 0
    forced_commit_false = 0
    forced_challenge_false = 0
    forced_commit_correct = 0
    forced_challenge_correct = 0
    for arm in ARMS:
        for producer in PRODUCERS:
            keyed = {
                (row["case_id"], row["variant"]): row
                for row in endpoint_rows
                if row["arm"] == arm and row["producer"] == producer
            }
            transitions: dict[str, int] = defaultdict(int)
            for case_id in hidden_by_case:
                commit = keyed[(case_id, "forced_commit")]
                challenge = keyed[(case_id, "forced_challenge")]
                c_label = "C" if commit["correct"] else "F" if commit["false_promotion"] else "U"
                h_label = "C" if challenge["correct"] else "F" if challenge["false_promotion"] else "U"
                transitions[f"{c_label}->{h_label}"] += 1
                safe_to_false += int(not commit["false_promotion"] and challenge["false_promotion"])
                forced_commit_false += int(commit["false_promotion"])
                forced_challenge_false += int(challenge["false_promotion"])
                forced_commit_correct += int(commit["correct"])
                forced_challenge_correct += int(challenge["correct"])
            paired[f"{arm}::{producer}"] = dict(sorted(transitions.items()))

    actual_challenge_rates = [
        summaries[f"{arm}::actual::{producer}"]["challenge_requests"] / 216
        for arm in ARMS
        for producer in PRODUCERS
    ]
    actual_utility = sum(
        summaries[f"{arm}::actual::{producer}"]["correct_minus_false"]
        for arm in ARMS
        for producer in PRODUCERS
    )
    commit_utility = sum(
        summaries[f"{arm}::forced_commit::{producer}"]["correct_minus_false"]
        for arm in ARMS
        for producer in PRODUCERS
    )
    gates = {
        "G0_integrity": validation["valid"],
        "G1_sequentiality": max(actual_challenge_rates) >= 0.10,
        "G2_safety": forced_challenge_false < forced_commit_false and safe_to_false == 0,
        "G3_utility_retention": forced_challenge_correct >= 0.50 * forced_commit_correct,
        "G4_policy_value": actual_utility > commit_utility,
        "details": {
            "max_actual_challenge_rate": max(actual_challenge_rates),
            "forced_commit_false": forced_commit_false,
            "forced_challenge_false": forced_challenge_false,
            "safe_commit_to_false_challenge": safe_to_false,
            "forced_commit_correct": forced_commit_correct,
            "forced_challenge_correct": forced_challenge_correct,
            "actual_correct_minus_false": actual_utility,
            "forced_commit_correct_minus_false": commit_utility,
        },
    }
    gates["all_passed"] = all(gates[key] for key in ("G0_integrity", "G1_sequentiality", "G2_safety", "G3_utility_retention", "G4_policy_value"))
    result = {
        "schema_version": "evidence_view_benchmark.multistep_qwen_scored.v1",
        "validation": validation,
        "summaries": summaries,
        "paired_commit_to_challenge": paired,
        "gates": gates,
        "rows": endpoint_rows,
    }
    result["content_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.raw)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(result) + "\n")
    print(json.dumps({"gates": result["gates"], "content_sha256": result["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

