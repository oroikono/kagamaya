"""Owner-side scoring after a complete public artifact is locked."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contract import PRODUCERS, SIDECAR_PATH, canonical_json, digest, file_sha256, read_jsonl


DECISIVE = {"IDENTIFIED", "OUTSIDE_LIBRARY"}


def truth_metrics(status: str, support: Any, in_library: bool, truth_support: Any) -> tuple[bool, bool]:
    truth = tuple(truth_support or ())
    chosen = tuple(support or ())
    correct = (in_library and status == "IDENTIFIED" and chosen == truth) or (
        not in_library and status == "OUTSIDE_LIBRARY"
    )
    return bool(correct), bool(status in DECISIVE and not correct)


def score(public_path: Path) -> dict[str, Any]:
    artifact = json.loads(public_path.read_text(encoding="utf-8"))
    claimed = artifact.pop("content_sha256")
    if claimed != digest(artifact):
        raise ValueError("public artifact content hash mismatch")
    artifact["content_sha256"] = claimed
    sidecar = {row["case_id"]: row for row in read_jsonl(SIDECAR_PATH)}
    if len(sidecar) != 216:
        raise ValueError("held-out sidecar coverage changed")
    rows = []
    for entry in artifact["rows"]:
        public = entry["public"]
        hidden = sidecar[public["case_id"]]
        if hidden["public_case_sha256"] != public["case_sha256"]:
            raise ValueError("public/owner binding mismatch")
        in_library = bool(hidden["held_out"]["in_library"])
        truth_support = hidden["held_out"]["truth_support"]
        for producer in PRODUCERS:
            record = public["producers"][producer]
            admission = record["admission"]
            correct, false = truth_metrics(admission["status"], admission["support"], in_library, truth_support)
            rows.append({
                "kind": entry["kind"],
                "policy": entry["policy"],
                "arm": entry["arm"],
                "case_id": public["case_id"],
                "producer": producer,
                "in_library": in_library,
                "status": admission["status"],
                "support": admission["support"],
                "correct": correct,
                "false_promotion": false,
                "node_cost": record["node_cost"],
                "second_decision": record["second_decision"],
                "first_action": record["first_action"],
            })
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["policy"], row["arm"], row["producer"])].append(row)
    summaries = {}
    for key, group in sorted(groups.items()):
        summaries["::".join(key)] = {
            "cases": len(group),
            "correct": sum(row["correct"] for row in group),
            "false_promotions": sum(row["false_promotion"] for row in group),
            "inconclusive": sum(row["status"] == "INCONCLUSIVE" for row in group),
            "challenge_requests": sum(row["second_decision"] == "REQUEST_CHALLENGE" for row in group),
            "mean_node_cost": sum(row["node_cost"] for row in group) / len(group),
        }
    result = {
        "schema_version": "evidence_view_benchmark.multistep_scored.v1",
        "public_content_sha256": claimed,
        "sidecar_file_sha256": file_sha256(SIDECAR_PATH),
        "rows": rows,
        "summaries": summaries,
    }
    result["content_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = score(args.public)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
    print(json.dumps({"rows": len(value["rows"]), "summaries": len(value["summaries"]), "content_sha256": value["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
