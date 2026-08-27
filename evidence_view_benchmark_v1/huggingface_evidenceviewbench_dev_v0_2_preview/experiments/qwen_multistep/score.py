"""Owner-side scoring for factual and permuted observation conditions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from multistep_acquisition_v1_20260820.src.contract import PRODUCERS, SIDECAR_PATH, canonical_json, digest, read_jsonl
from multistep_acquisition_v1_20260820.src.score import truth_metrics

from .contract import ARMS, CONDITIONS
from .validate import validate


def endpoint(row: dict[str, Any], producer: str, hidden: dict[str, Any]) -> dict[str, Any]:
    record = row["actual"]["producers"][producer]
    status = record["admission"]["status"]
    support = record["admission"]["support"]
    in_library = bool(hidden["held_out"]["in_library"])
    correct, false = truth_metrics(status, support, in_library, hidden["held_out"]["truth_support"])
    return {
        "status": status,
        "support": support,
        "decision": record["second_decision"],
        "node_cost": record["node_cost"],
        "correct": correct,
        "false_promotion": false,
        "in_library": in_library,
    }


def score(path: Path) -> dict[str, Any]:
    validation = validate(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    hidden_rows = read_jsonl(SIDECAR_PATH)
    hidden = {row["case_id"]: row for row in hidden_rows}
    if len(hidden_rows) != 216 or len(hidden) != 216:
        raise ValueError("owner sidecar coverage mismatch")
    rows = []
    for record in raw["rows"]:
        owner = hidden[record["case_id"]]
        if owner["public_case_sha256"] != record["case_sha256"]:
            raise ValueError("owner/public binding mismatch")
        for producer in PRODUCERS:
            rows.append({
                "case_id": record["case_id"],
                "arm": record["arm"],
                "condition": record["condition"],
                "producer": producer,
                "second_valid": record["second_valid"],
                **endpoint(record, producer, owner),
            })

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["arm"], row["condition"], row["producer"])].append(row)
    summaries = {}
    for key, group in sorted(groups.items()):
        summaries["::".join(key)] = {
            "cases": len(group),
            "valid": sum(row["second_valid"] for row in group),
            "correct": sum(row["correct"] for row in group),
            "false_promotions": sum(row["false_promotion"] for row in group),
            "inconclusive": sum(row["status"] == "INCONCLUSIVE" for row in group),
            "challenge_requests": sum(row["decision"] == "REQUEST_CHALLENGE" for row in group),
            "mean_node_cost": sum(row["node_cost"] for row in group) / len(group),
            "correct_minus_false": sum(row["correct"] for row in group) - sum(row["false_promotion"] for row in group),
        }

    decision_changes = Counter()
    outcome_changes = Counter()
    indexed = {(r["case_id"], r["arm"], r["condition"], r["producer"]): r for r in rows}
    for case_id in hidden:
        for arm in ARMS:
            for producer in PRODUCERS:
                factual = indexed[(case_id, arm, "FACTUAL", producer)]
                permuted = indexed[(case_id, arm, "PERMUTED", producer)]
                decision_changes[arm] += int(factual["decision"] != permuted["decision"])
                factual_label = "C" if factual["correct"] else "F" if factual["false_promotion"] else "U"
                permuted_label = "C" if permuted["correct"] else "F" if permuted["false_promotion"] else "U"
                outcome_changes[f"{arm}:{factual_label}->{permuted_label}"] += 1

    factual_utility = sum(v["correct_minus_false"] for k, v in summaries.items() if "::FACTUAL::" in k)
    permuted_utility = sum(v["correct_minus_false"] for k, v in summaries.items() if "::PERMUTED::" in k)
    total_pairs = 216 * len(PRODUCERS)
    max_change_rate = max(decision_changes[arm] / total_pairs for arm in ARMS)
    gates = {
        "V0_integrity": validation["valid"],
        "V1_all_metadata_blind": True,
        "V2_observation_sensitivity": max_change_rate >= 0.10,
        "V3_factual_value_over_permuted": factual_utility > permuted_utility,
        "details": {
            "decision_changes_by_arm": dict(decision_changes),
            "pairs_per_arm": total_pairs,
            "max_decision_change_rate": max_change_rate,
            "factual_correct_minus_false": factual_utility,
            "permuted_correct_minus_false": permuted_utility,
        },
    }
    gates["all_passed"] = all(gates[key] for key in ("V0_integrity", "V1_all_metadata_blind", "V2_observation_sensitivity", "V3_factual_value_over_permuted"))
    result = {
        "schema_version": "evidence_view_benchmark.metadata_blind_scored.v2",
        "validation": validation,
        "summaries": summaries,
        "decision_changes_by_arm": dict(decision_changes),
        "outcome_transitions": dict(sorted(outcome_changes.items())),
        "gates": gates,
        "rows": rows,
    }
    result["content_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = score(args.raw)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
    print(json.dumps({"gates": value["gates"], "content_sha256": value["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
