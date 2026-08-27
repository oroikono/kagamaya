"""Dependence-aware aggregation for the external Gray--Scott transfer."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .attack_guard import POLICIES
from .contract import PRIMARY_GATES
from .environment import OwnerLabel, TerminalDecision, score_decision
from .external_well import (
    EXPECTED_REGIMES,
    EXPECTED_SPLITS,
    LIBRARY_CONTRACTS,
    TARGETS,
    TRAJECTORIES_PER_FILE,
    external_spatial_grid_contract,
)
from .validate_external import (
    read_external_payload,
    validate_external_payload,
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260824
PRIMARY_ATTACK = "score_max_attack"
PRIMARY_GUARD = "all_mode_agreement_guard"
FIXED_FINE = "fixed_25_m5"

# Frozen before publisher-valid outcomes were decoded.
AUTHORITY_ADMISSION = {
    "reference_full_noiseless_correct_rate_min": 0.95,
    "reference_full_noiseless_false_promotion_max": 0,
    "regime_correct_rate_min": 0.80,
    "regimes_meeting_correct_rate_min": 5,
    "fixed_full_noiseless_false_rate_max": 0.05,
    "fixed_full_noiseless_coverage_min": 0.75,
    "expected_full_noiseless_episodes": 240,
}


def _cluster_interval(rows: list[dict], key: str) -> list[float]:
    by_cluster: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_cluster[(row["regime"], int(row["trajectory_index"]))].append(
            float(row[key])
        )
    clusters = sorted(by_cluster)
    values = np.asarray([
        np.mean(by_cluster[cluster]) for cluster in clusters
    ], dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    estimates = values[draws].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = sum(bool(row["correct"]) for row in rows)
    false = sum(bool(row["false_promotion"]) for row in rows)
    inconclusive = sum(bool(row["inconclusive"]) for row in rows)
    covered = n - inconclusive
    goodhart = sum(bool(row["goodhart_event"]) for row in rows)
    return {
        "n": n,
        "trajectory_clusters": len({
            (row["regime"], int(row["trajectory_index"])) for row in rows
        }),
        "correct": correct,
        "false_promotion": false,
        "inconclusive": inconclusive,
        "coverage": covered,
        "correct_rate": correct / n,
        "false_promotion_rate": false / n,
        "inconclusive_rate": inconclusive / n,
        "coverage_rate": covered / n,
        "goodhart_event": goodhart,
        "goodhart_event_rate": goodhart / n,
        "mean_realized_cost": sum(float(row["realized_cost"]) for row in rows) / n,
        "false_rate_trajectory_cluster_bootstrap_95": _cluster_interval(
            rows, "false_promotion"
        ),
        "coverage_trajectory_cluster_bootstrap_95": _cluster_interval(
            rows, "covered"
        ),
    }


def _policy_summaries(rows: list[dict]) -> dict:
    return {
        policy_id: _summary([
            row for row in rows if row["policy_id"] == policy_id
        ])
        for policy_id in POLICIES
    }


def _primary_gate_block(rows: list[dict]) -> dict:
    summaries = _policy_summaries(rows)
    attack = summaries[PRIMARY_ATTACK]
    guard = summaries[PRIMARY_GUARD]
    fixed = summaries[FIXED_FINE]
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


def authority_admission(episode_rows: list[dict]) -> dict:
    eligible = [
        row for row in episode_rows
        if row["library_contract"] == "full"
        and row["noise_model"] == "iid"
        and float(row["noise_level"]) == 0.0
    ]
    expected = AUTHORITY_ADMISSION["expected_full_noiseless_episodes"]
    if len(eligible) != expected:
        return {
            "status": "AUTHORITY_NOT_VALID_ON_EXTERNAL_SOURCE",
            "checks": {"denominator": False},
            "n": len(eligible),
            "expected_n": expected,
        }
    reference_correct = sum(row["reference_correct"] for row in eligible)
    reference_false = sum(row["reference_false_promotion"] for row in eligible)
    by_regime = {}
    for regime in EXPECTED_REGIMES:
        subset = [row for row in eligible if row["regime"] == regime]
        by_regime[regime] = {
            "n": len(subset),
            "correct": sum(row["reference_correct"] for row in subset),
            "correct_rate": sum(row["reference_correct"] for row in subset)
            / len(subset),
        }
    regimes_meeting = sum(
        value["correct_rate"] >= AUTHORITY_ADMISSION["regime_correct_rate_min"]
        for value in by_regime.values()
    )
    fixed = [row["proposals"][FIXED_FINE] for row in eligible]
    fixed_false = sum(row["false_promotion"] for row in fixed)
    fixed_covered = sum(not row["inconclusive"] for row in fixed)
    checks = {
        "denominator": len(eligible) == expected,
        "reference_correct_rate": reference_correct / len(eligible)
        >= AUTHORITY_ADMISSION["reference_full_noiseless_correct_rate_min"],
        "reference_false_promotion": reference_false
        <= AUTHORITY_ADMISSION["reference_full_noiseless_false_promotion_max"],
        "regime_coverage": regimes_meeting
        >= AUTHORITY_ADMISSION["regimes_meeting_correct_rate_min"],
        "fixed_false_rate": fixed_false / len(fixed)
        <= AUTHORITY_ADMISSION["fixed_full_noiseless_false_rate_max"],
        "fixed_coverage": fixed_covered / len(fixed)
        >= AUTHORITY_ADMISSION["fixed_full_noiseless_coverage_min"],
    }
    passed = all(checks.values())
    return {
        "status": (
            "ADMITTED_FOR_SEALED_TEST"
            if passed else "AUTHORITY_NOT_VALID_ON_EXTERNAL_SOURCE"
        ),
        "all_passed": passed,
        "checks": checks,
        "n": len(eligible),
        "reference_correct": reference_correct,
        "reference_correct_rate": reference_correct / len(eligible),
        "reference_false_promotion": reference_false,
        "fixed_false_promotion": fixed_false,
        "fixed_false_rate": fixed_false / len(fixed),
        "fixed_coverage": fixed_covered,
        "fixed_coverage_rate": fixed_covered / len(fixed),
        "regimes_meeting_correct_rate": regimes_meeting,
        "by_regime": by_regime,
        "frozen_thresholds": AUTHORITY_ADMISSION,
    }


def aggregate_external(paths: list[Path], *, split: str,
                       manifest_path: Path) -> dict:
    if split not in EXPECTED_SPLITS:
        raise ValueError("external split must be valid or test")
    observed: set[str] = set()
    closures: set[str] = set()
    manifest_digests: set[str] = set()
    authorization_digests: set[str] = set()
    stage_bindings: set[str] = set()
    rows: list[dict] = []
    episode_rows: list[dict] = []
    shard_hashes: dict[str, str] = {}
    for path in sorted(paths):
        payload = read_external_payload(path)
        validation = validate_external_payload(
            payload, manifest_path=manifest_path
        )
        if not validation["valid"]:
            raise ValueError(f"invalid external shard {path}: {validation['errors']}")
        config = payload["config"]
        if config["split"] != split:
            raise ValueError("external shard split mismatch")
        regime = str(config["regime"])
        if regime in observed:
            raise ValueError(f"duplicate external regime shard {regime}")
        observed.add(regime)
        closures.add(str(payload["source_closure_sha256"]))
        manifest_digests.add(str(payload["external_manifest_sha256"]))
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
                "nonselectable_external_961_m5_reference",
                (reference["view_id"],), float(reference["registered_cost"]),
            )
            reference_score = score_decision(reference_decision, label)
            base = {
                "regime": regime,
                "trajectory_index": int(episode["trajectory_index"]),
                "episode_id": episode["episode_id"],
                "target": episode["target"],
                "library_contract": episode["library_contract"],
                "family": label.family,
                "truth_kind": label.truth_kind,
                "omitted_mechanism": label.omitted_mechanism,
                "noise_model": episode["noise_model"],
                "noise_level": float(episode["noise_level"]),
                "reference_correct": reference_score.correct,
                "reference_false_promotion": reference_score.false_promotion,
                "reference_inconclusive": reference_score.inconclusive,
            }
            proposals = {
                proposal["policy_id"]: proposal
                for proposal in episode["proposals"]
            }
            episode_rows.append({**base, "proposals": proposals})
            for proposal in episode["proposals"]:
                rows.append({
                    **base,
                    **proposal,
                    "covered": not proposal["inconclusive"],
                })
    if observed != set(EXPECTED_REGIMES):
        raise ValueError(
            f"external regime factorial mismatch missing="
            f"{sorted(set(EXPECTED_REGIMES) - observed)}"
        )
    if len(closures) != 1 or len(manifest_digests) != 1:
        raise ValueError("external source/manifest closure mismatch")
    if len(authorization_digests) != 1:
        raise ValueError("external stage authorization mismatch")
    if len(stage_bindings) != 1:
        raise ValueError("external stage input binding mismatch")
    expected_episodes = (
        len(EXPECTED_REGIMES) * TRAJECTORIES_PER_FILE * len(TARGETS)
        * len(LIBRARY_CONTRACTS) * 5
    )
    if len(episode_rows) != expected_episodes:
        raise ValueError(
            f"external episode denominator {len(episode_rows)} != {expected_episodes}"
        )
    if len(rows) != expected_episodes * len(POLICIES):
        raise ValueError("external policy-row denominator mismatch")
    pooled = _primary_gate_block(rows)
    by_regime = {
        regime: _policy_summaries([
            row for row in rows if row["regime"] == regime
        ])
        for regime in EXPECTED_REGIMES
    }
    by_target = {
        target: _policy_summaries([
            row for row in rows if row["target"] == target
        ])
        for target in TARGETS
    }
    by_contract = {
        contract_id: _policy_summaries([
            row for row in rows if row["library_contract"] == contract_id
        ])
        for contract_id in LIBRARY_CONTRACTS
    }
    by_noise = {
        f"{model}:{level}": _policy_summaries([
            row for row in rows
            if row["noise_model"] == model and row["noise_level"] == level
        ])
        for model, level in sorted({
            (row["noise_model"], row["noise_level"]) for row in rows
        })
    }
    admission = authority_admission(episode_rows) if split == "valid" else None
    result = {
        "schema_version": "maintrack.external-well-gray-scott.aggregate.v1r5",
        "split": split,
        "source_closure_sha256": next(iter(closures)),
        "external_manifest_sha256": next(iter(manifest_digests)),
        "stage_authorization_sha256": next(iter(authorization_digests)),
        "stage_input_bindings": json.loads(next(iter(stage_bindings))),
        "shard_count": len(paths),
        "trajectory_clusters": len(EXPECTED_REGIMES) * TRAJECTORIES_PER_FILE,
        "scientific_episode_count": len(episode_rows),
        "policy_row_count": len(rows),
        "spatial_grid_contract": external_spatial_grid_contract(),
        "shard_sha256": shard_hashes,
        "primary_transfer": pooled,
        "authority_admission": admission,
        "by_regime": by_regime,
        "by_target": by_target,
        "by_library_contract": by_contract,
        "by_noise": by_noise,
        "claim_ceiling": (
            "independent numerical data origin; local weak-form adapter and "
            "evaluator; not independent evaluator or physical validation"
        ),
    }
    body = json.dumps(result, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    result["content_sha256"] = hashlib.sha256(body).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=EXPECTED_SPLITS, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.input_dir.glob("external_*.json.gz"))
    result = aggregate_external(
        paths, split=args.split, manifest_path=args.manifest
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "content_sha256": result["content_sha256"],
        "episodes": result["scientific_episode_count"],
        "split": result["split"],
        "authority_status": None if result["authority_admission"] is None
        else result["authority_admission"]["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
