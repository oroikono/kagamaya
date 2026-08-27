"""Dependence-aware aggregation and frozen gate evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .attack_guard import POLICIES
from .contract import (
    CONFIRMATION_SEEDS,
    INTERNAL_AUTHORITY_GATES,
    MECHANISM_SEEDS,
    PRIMARY_GATES,
)
from .environment import OwnerLabel, TerminalDecision, score_decision
from .validate import read_payload, validate_payload


SPLIT_SEEDS = {
    "mechanism": MECHANISM_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
}
ORIGINS = ("spectral", "fd")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260824


def _interval_by_seed(rows: list[dict], key: str) -> list[float]:
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(float(row[key]))
    values = np.asarray([
        np.mean(by_seed[seed]) for seed in sorted(by_seed)
    ], dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    estimates = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = sum(row["correct"] for row in rows)
    false = sum(row["false_promotion"] for row in rows)
    inconclusive = sum(row["inconclusive"] for row in rows)
    coverage = n - inconclusive
    goodhart = sum(row["goodhart_event"] for row in rows)
    return {
        "n": n,
        "correct": correct,
        "false_promotion": false,
        "inconclusive": inconclusive,
        "coverage": coverage,
        "correct_rate": correct / n,
        "false_promotion_rate": false / n,
        "inconclusive_rate": inconclusive / n,
        "coverage_rate": coverage / n,
        "goodhart_event": goodhart,
        "goodhart_event_rate": goodhart / n,
        "mean_realized_cost": sum(row["realized_cost"] for row in rows) / n,
        "false_rate_cluster_bootstrap_95": _interval_by_seed(
            rows, "false_promotion"
        ),
        "coverage_cluster_bootstrap_95": _interval_by_seed(rows, "covered"),
    }


def _reference_summary(rows: list[dict]) -> dict:
    unique = [row for row in rows if row["policy_id"] == "score_max_attack"]
    n = len(unique)
    if not n:
        return {"n": 0}
    correct = sum(row["reference_correct"] for row in unique)
    false = sum(row["reference_false_promotion"] for row in unique)
    inconclusive = sum(row["reference_inconclusive"] for row in unique)
    return {
        "n": n,
        "correct": correct,
        "false_promotion": false,
        "inconclusive": inconclusive,
        "correct_rate": correct / n,
        "false_promotion_rate": false / n,
        "coverage_rate": (n - inconclusive) / n,
    }


def _gate_block(rows: list[dict]) -> dict:
    summaries = {
        policy: _summary([row for row in rows if row["policy_id"] == policy])
        for policy in POLICIES
    }
    attack = summaries["score_max_attack"]
    guard = summaries["all_mode_agreement_guard"]
    fixed = summaries["fixed_25_m5"]
    checks = {
        "false_promotion": guard["false_promotion_rate"]
        <= PRIMARY_GATES["guard_false_promotion_rate_max"],
        "coverage": guard["coverage_rate"] >= PRIMARY_GATES["guard_coverage_min"],
        "goodhart_event": guard["goodhart_event_rate"]
        <= PRIMARY_GATES["guard_goodhart_event_rate_max"],
        "false_vs_fixed": guard["false_promotion_rate"]
        <= fixed["false_promotion_rate"],
        "coverage_vs_fixed": guard["coverage_rate"] - fixed["coverage_rate"]
        >= PRIMARY_GATES["guard_coverage_delta_vs_fixed_fine_min"],
        "false_reduction_vs_attack": 100.0 * (
            attack["false_promotion_rate"] - guard["false_promotion_rate"]
        ) >= PRIMARY_GATES["guard_false_reduction_vs_attack_pp_min"],
    }
    return {
        "summaries": summaries,
        "checks": checks,
        "all_passed": all(checks.values()),
        "false_reduction_vs_attack_pp": 100.0 * (
            attack["false_promotion_rate"] - guard["false_promotion_rate"]
        ),
        "coverage_delta_vs_fixed_pp": 100.0 * (
            guard["coverage_rate"] - fixed["coverage_rate"]
        ),
    }


def aggregate(paths: list[Path], split: str) -> dict:
    if split not in SPLIT_SEEDS:
        raise ValueError("unsupported aggregate split")
    expected = {
        (seed, origin) for seed in SPLIT_SEEDS[split] for origin in ORIGINS
    }
    observed: set[tuple[int, str]] = set()
    closures: set[str] = set()
    authorization_digests: set[str] = set()
    stage_bindings: set[str] = set()
    rows: list[dict] = []
    shard_hashes: dict[str, str] = {}
    for path in sorted(paths):
        payload = read_payload(path)
        validation = validate_payload(payload)
        if not validation["valid"]:
            raise ValueError(f"invalid shard {path}: {validation['errors']}")
        config = payload["config"]
        key = (int(config["seed"]), str(config["origin"]))
        if key in observed:
            raise ValueError(f"duplicate shard {key}")
        observed.add(key)
        closures.add(str(payload["source_closure_sha256"]))
        authorization_digests.add(str(
            config.get("stage_authorization_sha256")
        ))
        stage_bindings.add(json.dumps(
            config.get("stage_input_bindings"), sort_keys=True,
            separators=(",", ":"),
        ))
        shard_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for episode, label_payload in zip(
            payload["episodes"], payload["owner_labels"]
        ):
            label = OwnerLabel(
                episode_id=label_payload["episode_id"],
                family=label_payload["family"],
                truth_kind=label_payload["truth_kind"],
                truth_support=tuple(label_payload["truth_support"]),
                omitted_mechanism=label_payload["omitted_mechanism"],
            )
            reference = episode["reference"]
            reference_decision = TerminalDecision(
                reference["status"], tuple(reference.get("support") or ()),
                "nonselectable_49_m5_reference", (reference["view_id"],),
                float(reference["registered_cost"]),
            )
            reference_score = score_decision(reference_decision, label)
            for proposal in episode["proposals"]:
                rows.append({
                    **proposal,
                    "seed": key[0],
                    "origin": key[1],
                    "episode_id": episode["episode_id"],
                    "family": label.family,
                    "truth_kind": label.truth_kind,
                    "omitted_mechanism": label.omitted_mechanism,
                    "noise_model": episode["noise_model"],
                    "noise_level": episode["noise_level"],
                    "covered": not proposal["inconclusive"],
                    "reference_correct": reference_score.correct,
                    "reference_false_promotion": reference_score.false_promotion,
                    "reference_inconclusive": reference_score.inconclusive,
                    "reference_status": reference["status"],
                    "reference_support": tuple(reference.get("support") or ()),
                    # Keep the standard-JSON nonfinite token, if present.  The
                    # reference residual is reported but not used by gates.
                    "reference_residual": reference["residual"],
                })
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"shard factorial mismatch missing={missing} extra={extra}")
    if len(closures) != 1:
        raise ValueError("source closure mismatch across shards")
    if len(authorization_digests) != 1:
        raise ValueError("stage authorization mismatch across shards")
    if len(stage_bindings) != 1:
        raise ValueError("stage input binding mismatch across shards")

    policy_rows = len(expected) * 15 * 5 * len(POLICIES)
    if len(rows) != policy_rows:
        raise ValueError(f"row denominator {len(rows)} != {policy_rows}")
    pooled = _gate_block(rows)
    by_origin = {
        origin: _gate_block([row for row in rows if row["origin"] == origin])
        for origin in ORIGINS
    }
    by_family = {
        family: {
            policy: _summary([
                row for row in rows
                if row["family"] == family and row["policy_id"] == policy
            ])
            for policy in POLICIES
        }
        for family in sorted({row["family"] for row in rows})
    }
    by_omitted_mechanism = {
        mechanism: {
            policy: _summary([
                row for row in rows
                if row["omitted_mechanism"] == mechanism
                and row["policy_id"] == policy
            ])
            for policy in POLICIES
        }
        for mechanism in sorted({
            row["omitted_mechanism"] for row in rows
            if row["omitted_mechanism"] is not None
        })
    }
    by_noise = {
        f"{model}:{level}": {
            policy: _summary([
                row for row in rows
                if row["noise_model"] == model
                and float(row["noise_level"]) == level
                and row["policy_id"] == policy
            ])
            for policy in POLICIES
        }
        for model, level in sorted({
            (row["noise_model"], float(row["noise_level"])) for row in rows
        })
    }
    omitted_count = len(by_omitted_mechanism)
    reference_rows = [
        row for row in rows
        if row["policy_id"] == "score_max_attack"
        and row["noise_model"] == "iid"
        and float(row["noise_level"]) == 0.0
    ]
    omitted_reference_rows = [
        row for row in reference_rows if row["truth_kind"] == "outside_library"
    ]
    reference_correct_rate = sum(
        row["reference_correct"] for row in reference_rows
    ) / len(reference_rows)
    reference_false = sum(
        row["reference_false_promotion"] for row in reference_rows
    )
    omitted_out_rate = sum(
        row["reference_status"] == "OUTSIDE_LIBRARY"
        for row in omitted_reference_rows
    ) / len(omitted_reference_rows)
    authority_checks = {
        "noiseless_reference_correct": reference_correct_rate
        >= INTERNAL_AUTHORITY_GATES["noiseless_reference_correct_rate_min"],
        "noiseless_reference_false": reference_false
        <= INTERNAL_AUTHORITY_GATES["noiseless_reference_false_promotion_max"],
        "noiseless_omitted_projection_separation": omitted_out_rate
        >= INTERNAL_AUTHORITY_GATES["noiseless_omitted_reference_out_rate_min"],
    }
    final_checks = {
        "pooled": pooled["all_passed"],
        "each_origin": all(block["all_passed"] for block in by_origin.values()),
        "origin_count": len(by_origin) >= PRIMARY_GATES["origins_required"],
        "omitted_mechanism_count": omitted_count
        >= PRIMARY_GATES["omitted_mechanisms_required"],
        "confirmation_clusters": (
            split != "confirmation"
            or len(SPLIT_SEEDS[split])
            >= PRIMARY_GATES["confirmation_seed_clusters_required"]
        ),
        "authority_valid": all(authority_checks.values()),
    }
    result = {
        "schema_version": "maintrack.spotlight-2d-replication.aggregate.v1",
        "split": split,
        "source_closure_sha256": next(iter(closures)),
        "stage_authorization_sha256": next(iter(authorization_digests)),
        "stage_input_bindings": json.loads(next(iter(stage_bindings))),
        "shard_count": len(paths),
        "seed_clusters": len(SPLIT_SEEDS[split]),
        "scientific_episode_count": len(rows) // len(POLICIES),
        "policy_row_count": len(rows),
        "shard_sha256": shard_hashes,
        "pooled": pooled,
        "reference": _reference_summary(rows),
        "authority_validation": {
            "checks": authority_checks,
            "all_passed": all(authority_checks.values()),
            "noiseless_reference_n": len(reference_rows),
            "noiseless_reference_correct_rate": reference_correct_rate,
            "noiseless_reference_false_promotion": reference_false,
            "noiseless_omitted_n": len(omitted_reference_rows),
            "noiseless_omitted_out_rate": omitted_out_rate,
        },
        "by_origin": by_origin,
        "by_family": by_family,
        "by_omitted_mechanism": by_omitted_mechanism,
        "by_noise": by_noise,
        "final_checks": final_checks,
        "all_passed": all(final_checks.values()),
        "claim_ceiling": (
            "finite_prospective_2d_coupled_replication; separately coded "
            "origins share analytic owner and are not independent science"
        ),
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    result["content_sha256"] = hashlib.sha256(body).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLIT_SEEDS), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.input_dir.glob("shard_*.json.gz"))
    result = aggregate(paths, args.split)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "all_passed": result["all_passed"],
        "content_sha256": result["content_sha256"],
        "episodes": result["scientific_episode_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
