"""Validate and aggregate the fresh consensus-confirmation shards."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..contract import digest
from ..io_utils import read_json, write_json
from .confirmation import CONFIRMATION_SEEDS, POLICIES, PRODUCERS, design_sha256, source_sha256
from .validate import validate_shard


def _mean(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return sum(values) / len(values)


def _bootstrap(values: list[float], seed: int, draws: int = 5000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return [samples[int(.025 * draws)], samples[int(.975 * draws) - 1]]


def summarize(all_rows: list[dict[str, Any]]) -> tuple[dict, dict]:
    summary: dict[str, Any] = {}
    cluster_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        cluster_rows[(row["producer"], row["policy"], int(row["seed"]))].append(row)
    for producer in PRODUCERS:
        for policy in POLICIES:
            rows = [r for r in all_rows if r["producer"] == producer and r["policy"] == policy]
            clusters = [cluster_rows[(producer, policy, seed)] for seed in CONFIRMATION_SEEDS]
            metrics = {
                "public_terminal_rate": _mean(rows, "public_terminal"),
                "reference_change_rate": 1.0 - _mean(rows, "reference_signature_agreement"),
                "goodhart_event_rate": _mean(rows, "goodhart_event"),
                "selected_false_promotion_rate": _mean(rows, "selected_false_promotion_owner_diagnostic"),
                "selected_correct_rate": _mean(rows, "selected_correct_owner_diagnostic"),
            }
            metric_intervals = {}
            for index, field in enumerate((
                "public_terminal",
                "reference_signature_agreement",
                "goodhart_event",
                "selected_false_promotion_owner_diagnostic",
                "selected_correct_owner_diagnostic",
            )):
                values = [_mean(group, field) for group in clusters]
                key = "reference_agreement_rate" if field == "reference_signature_agreement" else field + "_rate"
                metric_intervals[key] = _bootstrap(values, 20260818 + index)
            summary[f"{producer}::{policy}"] = {
                "episodes": len(rows),
                "seed_clusters": len(clusters),
                **metrics,
                "seed_cluster_bootstrap_95pct": metric_intervals,
                "clusters_with_any_false_promotion": sum(
                    any(r["selected_false_promotion_owner_diagnostic"] for r in group)
                    for group in clusters
                ),
                "clusters_with_any_goodhart_event": sum(any(r["goodhart_event"] for r in group) for group in clusters),
            }
    gates: dict[str, Any] = {}
    for producer in PRODUCERS:
        selected = summary[f"{producer}::all_mode_consensus"]
        fixed = summary[f"{producer}::fixed_25_m5"]
        confidence = summary[f"{producer}::confidence_max"]
        values = {
            "false_promotion_le_1pct": selected["selected_false_promotion_rate"] <= .01,
            "terminal_coverage_ge_75pct": selected["public_terminal_rate"] >= .75,
            "goodhart_event_le_1pct": selected["goodhart_event_rate"] <= .01,
            "false_promotion_no_worse_than_fixed": selected["selected_false_promotion_rate"] <= fixed["selected_false_promotion_rate"],
            "coverage_within_5pp_of_fixed": selected["public_terminal_rate"] >= fixed["public_terminal_rate"] - .05,
            "false_promotion_reduction_vs_confidence_ge_50pp": confidence["selected_false_promotion_rate"] - selected["selected_false_promotion_rate"] >= .50,
        }
        gates[producer] = {"gates": values, "all_passed": all(values.values())}
    return summary, gates


def aggregate(paths: list[Path]) -> dict[str, Any]:
    shards = []
    seen = set()
    all_rows = []
    for path in paths:
        payload = read_json(path)
        validate_shard(payload)
        key = (payload["config"]["producer"], int(payload["config"]["seed"]))
        if key in seen:
            raise ValueError(f"duplicate shard {key}")
        seen.add(key)
        shards.append({"producer": key[0], "seed": key[1], "content_sha256": payload["content_sha256"]})
        for row in payload["scored"]["rows"]:
            all_rows.append({**row, "seed": key[1]})
    expected = {(producer, seed) for producer in PRODUCERS for seed in CONFIRMATION_SEEDS}
    if seen != expected:
        raise ValueError(f"shard coverage mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
    summary, gates = summarize(all_rows)
    payload: dict[str, Any] = {
        "schema_version": "verisigs.adversarial_verifier_gaming.consensus_v2.aggregate",
        "status": "CONFIRMATION_PASS" if all(item["all_passed"] for item in gates.values()) else "CONFIRMATION_FAIL",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha256(),
        "shards": sorted(shards, key=lambda x: (x["producer"], x["seed"])),
        "summary": summary,
        "gates": gates,
        "claim_ceiling": "finite fresh-seed synthetic confirmation; no population, physical, continuum, or independent-replication claim",
    }
    payload["content_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.json.gz"))
    payload = aggregate(paths)
    write_json(args.output, payload)
    print(payload["status"])
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()
