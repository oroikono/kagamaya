"""Validate, aggregate, and evaluate frozen confirmation gates."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adversarial_verifier_gaming_v1.io_utils import write_json
from ..sequential_consensus_controller_v1.campaign import summarize
from .contract import (
    CONFIRMATION_SEEDS,
    FAMILIES,
    POLICIES,
    PRODUCERS,
    digest,
    validate_freeze,
)
from .validate import validate_shard


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows)


def confirmation_gates(
    summary: Mapping[str, Any],
    scores: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for producer in PRODUCERS:
        seq_summary = summary[f"{producer}::sequential_all_mode"]
        full = summary[f"{producer}::fullbank_all_mode"]
        fixed25 = summary[f"{producer}::fixed_25_m5"]
        fixed49 = summary[f"{producer}::fixed_49_m5"]
        confidence = summary[f"{producer}::confidence_max"]
        seq_rows = [
            row
            for row in scores
            if row["producer"] == producer and row["policy"] == "sequential_all_mode"
        ]
        family_rates: dict[str, float] = {}
        for family in FAMILIES:
            group = [
                row
                for row in seq_rows
                if owners[str(row["episode_id"])]["owner_episode_key"][0] == family
            ]
            family_rates[family] = _rate(group, "correct_owner_diagnostic")
        nonzero_noise_terminal: dict[str, float] = {}
        for level in (0.001, 0.005):
            group = [
                row
                for row in seq_rows
                if float(owners[str(row["episode_id"])]["owner_episode_key"][3]) == level
            ]
            nonzero_noise_terminal[str(level)] = _rate(group, "terminal")
        checks = {
            "zero_observed_false_promotions": seq_summary["false_promotion_count"] == 0,
            "terminal_rate_ge_75pct": seq_summary["terminal_rate"] >= 0.75,
            "correct_rate_ge_75pct": seq_summary["correct_rate"] >= 0.75,
            "correct_within_5pp_fullbank": seq_summary["correct_rate"] >= full["correct_rate"] - 0.05,
            "false_count_no_higher_fixed25": seq_summary["false_promotion_count"] <= fixed25["false_promotion_count"],
            "false_count_no_higher_fixed49": seq_summary["false_promotion_count"] <= fixed49["false_promotion_count"],
            "false_reduction_vs_confidence_ge_50pp": confidence["false_promotion_rate"] - seq_summary["false_promotion_rate"] >= 0.50,
            "mean_views_le_12": seq_summary["mean_views_evaluated"] <= 12.0,
            "mean_max_nodes_le_30": seq_summary["mean_max_nodes_acquired"] <= 30.0,
            "every_family_correct_ge_50pct": all(value >= 0.50 for value in family_rates.values()),
            "each_nonzero_noise_terminal_ge_50pct": all(value >= 0.50 for value in nonzero_noise_terminal.values()),
            "exact_confirmation_split_used": True,
        }
        output[producer] = {
            "checks_before_numerical_origin_replay": checks,
            "all_scientific_checks_passed": all(checks.values()),
            "family_correct_rates": family_rates,
            "nonzero_noise_terminal_rates": nonzero_noise_terminal,
        }
    return output


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_statistics(
    scores: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Mapping[str, Any]],
    producer: str,
) -> dict[str, Any]:
    by_key = {(str(row["episode_id"]), str(row["policy"])): row for row in scores}
    per_seed: list[dict[str, float]] = []
    for seed in CONFIRMATION_SEEDS:
        episode_ids = [
            episode_id
            for episode_id, owner in owners.items()
            if int(owner["owner_episode_key"][1]) == seed
            and by_key[(episode_id, "sequential_all_mode")]["producer"] == producer
        ]
        sequential = [by_key[(episode_id, "sequential_all_mode")] for episode_id in episode_ids]
        fullbank = [by_key[(episode_id, "fullbank_all_mode")] for episode_id in episode_ids]
        per_seed.append(
            {
                "seed": float(seed),
                "correct_rate": _rate(sequential, "correct_owner_diagnostic"),
                "false_rate": _rate(sequential, "false_promotion_owner_diagnostic"),
                "terminal_rate": _rate(sequential, "terminal"),
                "correct_difference_vs_fullbank": _rate(
                    sequential, "correct_owner_diagnostic"
                )
                - _rate(fullbank, "correct_owner_diagnostic"),
                "mean_views": sum(float(row["views_evaluated"]) for row in sequential)
                / len(sequential),
            }
        )
    metrics = (
        "correct_rate",
        "false_rate",
        "terminal_rate",
        "correct_difference_vs_fullbank",
        "mean_views",
    )
    rng = random.Random(445566)
    boot: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(10_000):
        sample = [per_seed[rng.randrange(len(per_seed))] for _ in per_seed]
        for metric in metrics:
            boot[metric].append(sum(row[metric] for row in sample) / len(sample))
    intervals = {
        metric: {
            "estimate": sum(row[metric] for row in per_seed) / len(per_seed),
            "cluster_bootstrap_95pct": [
                _quantile(boot[metric], 0.025),
                _quantile(boot[metric], 0.975),
            ],
        }
        for metric in metrics
    }
    clusters_with_any_false = sum(row["false_rate"] > 0.0 for row in per_seed)
    iid_working_model_upper = (
        1.0 - 0.05 ** (1.0 / len(per_seed)) if clusters_with_any_false == 0 else None
    )
    return {
        "seed_cluster_count": len(per_seed),
        "clusters_with_any_false": clusters_with_any_false,
        "metrics": intervals,
        "zero_event_one_sided_95pct_upper_under_unverified_iid_cluster_model": iid_working_model_upper,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 445566,
    }


def build_aggregate(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    freeze = validate_freeze()
    reports = [validate_shard(shard) for shard in shards]
    if any(not report["valid"] for report in reports):
        raise ValueError(f"invalid shard(s): {[r for r in reports if not r['valid']]}")
    expected = {(producer, seed) for producer in PRODUCERS for seed in CONFIRMATION_SEEDS}
    observed = {(str(row["producer"]), int(row["seed"])) for row in shards}
    if observed != expected or len(shards) != len(expected):
        raise ValueError("shard coverage is not the exact frozen producer x seed cross")
    publics: list[dict[str, Any]] = []
    owners_list: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for shard in sorted(shards, key=lambda row: (str(row["producer"]), int(row["seed"]))):
        publics.extend(shard["public_episodes"])
        owners_list.extend(shard["owner_episodes"])
        proposals.extend(shard["proposals"])
        scores.extend(shard["scored_rows"])
    owners = {str(row["episode_id"]): row for row in owners_list}
    if len(owners) != len(owners_list):
        raise ValueError("duplicate episode IDs across shards")
    summary = summarize(scores)
    gates = confirmation_gates(summary, scores, owners)
    cluster_stats = {
        producer: cluster_statistics(scores, owners, producer) for producer in PRODUCERS
    }
    body = {
        "schema_version": "verisigs.sequential_asymmetric.confirmation_aggregate.v1",
        "status": "NUMERICAL_ORIGIN_REPLAY_PENDING",
        "freeze_sha256": freeze["freeze_sha256"],
        "source_manifest": freeze["source_manifest"],
        "split": {
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "confirmation_seed_cluster_access_count": len(CONFIRMATION_SEEDS),
        },
        "shard_count": len(shards),
        "episode_count": len(publics),
        "proposal_count": len(proposals),
        "shard_reports": reports,
        "public_episodes": publics,
        "owner_episodes": owners_list,
        "proposals": proposals,
        "scored_rows": scores,
        "summary": summary,
        "confirmation_gates_before_numerical_replay": gates,
        "seed_cluster_statistics": cluster_stats,
        "claim_ceiling": (
            "50-seed finite synthetic confirmation; zero observed error is not a population guarantee; "
            "producer cells are paired and repeated within seed cluster"
        ),
    }
    return {**body, "content_sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    paths = sorted(args.shard_dir.glob("*.json"))
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    payload = build_aggregate(shards)
    write_json(args.output, payload)
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()
