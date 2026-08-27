#!/usr/bin/env python3
"""Build the compact sequential-confirmation release artifact.

The 65 MB upstream aggregate contains episode-level evidence and policy rows.
This release keeps the frozen summary, dependence-cluster statistics, shard
receipts, final gates, and an independently recomputed paired transition table.
It deliberately omits the episode arrays; their hashes remain recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def outcome(row: dict) -> str:
    if row["false_promotion_owner_diagnostic"]:
        return "false_promotion"
    if row["correct_owner_diagnostic"]:
        return "correct"
    return "inconclusive"


def build(source_path: Path, verdict_path: Path) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))

    by_key: dict[tuple[str, str, str], dict] = {}
    for row in source["scored_rows"]:
        key = (row["producer"], row["episode_id"], row["policy"])
        if key in by_key:
            raise ValueError(f"duplicate scored row: {key}")
        by_key[key] = row

    transitions: dict[str, dict[str, int]] = {}
    for producer in ("fd_bdf", "spectral_etdrk4"):
        table = {
            "correct_to_correct": 0,
            "correct_to_inconclusive": 0,
            "correct_to_false_promotion": 0,
            "inconclusive_to_correct": 0,
            "inconclusive_to_inconclusive": 0,
            "inconclusive_to_false_promotion": 0,
            "false_promotion_to_correct": 0,
            "false_promotion_to_inconclusive": 0,
            "false_promotion_to_false_promotion": 0,
        }
        episode_ids = sorted(
            episode_id
            for (prod, episode_id, policy) in by_key
            if prod == producer and policy == "fixed_49_m5"
        )
        if len(episode_ids) != 1800:
            raise ValueError(f"{producer}: expected 1800 fixed-fine episodes")
        for episode_id in episode_ids:
            fixed = by_key[(producer, episode_id, "fixed_49_m5")]
            sequential = by_key[(producer, episode_id, "sequential_all_mode")]
            table[f"{outcome(fixed)}_to_{outcome(sequential)}"] += 1
        if sum(table.values()) != 1800:
            raise ValueError(f"{producer}: transition table does not total 1800")
        transitions[producer] = table

    fullbank = source["summary"]["fd_bdf::fullbank_all_mode"]
    sequential = source["summary"]["fd_bdf::sequential_all_mode"]
    savings = {
        "views_fraction_reduced": 1.0
        - sequential["mean_views_evaluated"] / fullbank["mean_views_evaluated"],
        "max_nodes_fraction_reduced": 1.0
        - sequential["mean_max_nodes_acquired"] / fullbank["mean_max_nodes_acquired"],
    }

    compact = {
        "schema_version": "evidenceviewbench.sequential_confirmation_compact.v1",
        "release_scope": "compact aggregate; episode arrays and 100 raw shards intentionally omitted",
        "upstream": {
            "aggregate_file_sha256": file_sha256(source_path),
            "aggregate_content_sha256": source["content_sha256"],
            "final_verdict_file_sha256": file_sha256(verdict_path),
            "source_closure_sha256": source["source_manifest"]["source_closure_sha256"],
            "freeze_content_sha256": source["freeze_sha256"],
        },
        "split": source["split"],
        "dependence": {
            "seed_clusters": 50,
            "producer_rows_are_paired": True,
            "independent_trial_count_not_claimed": True,
        },
        "counts": {
            "unique_hidden_episodes_per_producer": 1800,
            "paired_producer_episode_cells": source["episode_count"],
            "policy_decisions_per_producer": source["proposal_count"] // 2,
            "paired_policy_decisions": source["proposal_count"],
            "validated_shards": source["shard_count"],
        },
        "summary_by_producer_policy": source["summary"],
        "seed_cluster_statistics": source["seed_cluster_statistics"],
        "shard_validation_receipts": source["shard_reports"],
        "final_gate_verdict": verdict,
        "fixed49_to_sequential_transitions": transitions,
        "efficiency_vs_fullbank": savings,
        "claim_ceiling": source["claim_ceiling"],
    }
    compact["content_sha256"] = canonical_sha256(compact)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verdict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = build(args.source, args.verdict)
    args.output.write_text(
        json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
