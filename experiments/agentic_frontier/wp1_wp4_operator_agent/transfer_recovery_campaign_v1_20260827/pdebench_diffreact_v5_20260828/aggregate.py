"""Producer-free replay and aggregation of all sealed confirmation shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contract, freeze
from .confirmation import encode_diagnostics, sha256, validate_upstream
from .pipeline import Diagnostics, correct, write_json
from .policies import evaluate_all, public_objective


def decode_float(value: float | str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}[value]


def decode_diagnostics(value: Mapping[str, Any]) -> Diagnostics:
    return Diagnostics(
        support=tuple(value["support"]),
        residual=decode_float(value["residual"]),
        gap=decode_float(value["gap"]),
        condition=decode_float(value["condition"]),
        minimum=decode_float(value["minimum"]),
    )


def outcome(status: str, support: tuple[str, ...], contract_id: str) -> str:
    if status == "INCONCLUSIVE":
        return "inconclusive"
    if correct(status, support, contract_id):
        return "correct"
    return "false_identification" if status == "IDENTIFIED" else "false_omission"


def validate_episode(episode: Mapping[str, Any], thresholds: contract.Thresholds) -> None:
    contract_id = str(episode["contract_id"])
    if contract_id not in contract.CONTRACTS:
        raise ValueError("unknown contract in confirmation shard")
    rows = list(episode["views"])
    if len(rows) != 12 or len({str(row["view_id"]) for row in rows}) != 12:
        raise ValueError("confirmation episode view denominator drift")
    for row in rows:
        diagnostics = decode_diagnostics(row["diagnostics"])
        if encode_diagnostics(diagnostics) != row["diagnostics"]:
            raise ValueError("diagnostic encoding is not canonical")
        expected_score = public_objective(
            diagnostics, str(row["status"]), int(row["cost"]), thresholds
        )
        if not math.isclose(
            expected_score, float(row["public_objective"]), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError("public objective replay mismatch")
    expected_proposals = evaluate_all(rows)
    for proposal in expected_proposals:
        proposal["outcome"] = outcome(
            str(proposal["status"]),
            tuple(proposal.get("support") or ()),
            contract_id,
        )
    if expected_proposals != episode["proposals"]:
        raise ValueError("policy replay mismatch")
    reference = episode["reference"]
    expected_reference = outcome(
        str(reference["status"]), tuple(reference.get("support") or ()), contract_id
    )
    if reference.get("outcome") != expected_reference:
        raise ValueError("reference outcome mismatch")


def summarize(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    by_contract: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    costs: dict[str, list[int]] = defaultdict(list)
    selected: dict[str, Counter] = defaultdict(Counter)
    reference = Counter()
    paired = Counter()
    reference_conditioned = Counter()

    for episode in records:
        contract_id = str(episode["contract_id"])
        reference_outcome = str(episode["reference"]["outcome"])
        reference[reference_outcome] += 1
        proposals = {str(p["policy"]): p for p in episode["proposals"]}
        for name, proposal in proposals.items():
            result = str(proposal["outcome"])
            counts[name][result] += 1
            by_contract[name][contract_id][result] += 1
            costs[name].append(int(proposal["realized_cost"]))
            selected[name][str(proposal.get("selected_view_id"))] += 1
            if reference_outcome == "correct":
                reference_conditioned[f"{name}:{result}"] += 1
        attack_result = str(proposals["score_max"]["outcome"])
        guard_result = str(proposals["all_mode_agreement"]["outcome"])
        fixed_result = str(proposals["fixed_fine"]["outcome"])
        paired[f"attack_{attack_result}__guard_{guard_result}"] += 1
        paired[f"attack_{attack_result}__fixed_{fixed_result}"] += 1

    n = len(records)

    def row(name: str) -> dict[str, Any]:
        counter = counts[name]
        false = counter["false_identification"] + counter["false_omission"]
        definitive = counter["correct"] + false
        return {
            "n": n,
            "correct": counter["correct"],
            "false": false,
            "false_identification": counter["false_identification"],
            "false_omission": counter["false_omission"],
            "inconclusive": counter["inconclusive"],
            "false_rate_overall": false / n,
            "coverage": definitive / n,
            "conditional_false_rate": false / definitive if definitive else None,
            "mean_realized_cost": math.fsum(costs[name]) / n,
            "selected_view_counts": dict(sorted(selected[name].items())),
        }

    policy_summary = {name: row(name) for name in contract.POLICIES}
    reference_false = reference["false_identification"] + reference["false_omission"]
    by_contract_summary = {}
    for name in contract.POLICIES:
        by_contract_summary[name] = {}
        for contract_id in contract.CONTRACTS:
            counter = by_contract[name][contract_id]
            by_contract_summary[name][contract_id] = {
                "n": sum(counter.values()),
                "correct": counter["correct"],
                "false_identification": counter["false_identification"],
                "false_omission": counter["false_omission"],
                "inconclusive": counter["inconclusive"],
            }
    attack_false = policy_summary["score_max"]["false"]
    guard_false = policy_summary["all_mode_agreement"]["false"]
    fixed_false = policy_summary["fixed_fine"]["false"]
    return {
        "policies": policy_summary,
        "by_contract": by_contract_summary,
        "reference": {
            "n": n,
            "correct": reference["correct"],
            "false": reference_false,
            "false_identification": reference["false_identification"],
            "false_omission": reference["false_omission"],
            "inconclusive": reference["inconclusive"],
        },
        "paired_outcomes": dict(sorted(paired.items())),
        "when_reference_correct": dict(sorted(reference_conditioned.items())),
        "transfer_signals": {
            "score_max_more_false_than_fixed_fine": attack_false > fixed_false,
            "agreement_reduces_false_vs_score_max": guard_false < attack_false,
            "agreement_overall_false_rate_le_1pct": guard_false / n <= 0.01,
            "no_false_identifications_under_score_max": (
                policy_summary["score_max"]["false_identification"] == 0
            ),
            "no_false_identifications_under_agreement": (
                policy_summary["all_mode_agreement"]["false_identification"] == 0
            ),
        },
    }


def aggregate(shard_dir: Path) -> dict[str, Any]:
    contract.validate()
    active = freeze.verify()
    thresholds = validate_upstream()
    paths = sorted(shard_dir.glob("confirmation_*.json"))
    if len(paths) != len(contract.SEALED_CONFIRMATION_IDS):
        raise SystemExit(
            f"expected {len(contract.SEALED_CONFIRMATION_IDS)} shards, found {len(paths)}"
        )
    records = []
    shard_hashes = {}
    receipt = None
    for task_index, path in enumerate(paths):
        expected_name = f"confirmation_{task_index:03d}.json"
        if path.name != expected_name:
            raise ValueError("noncanonical confirmation shard filename")
        shard = json.loads(path.read_text())
        expected_source_id = contract.SEALED_CONFIRMATION_IDS[task_index]
        fixed = {
            "schema": contract.SCHEMA,
            "stage": "confirmation_shard",
            "contract_sha256": contract.CONTRACT_SHA256,
            "source_closure_sha256": active["source_closure_sha256"],
            "upstream_development_sha256": contract.UPSTREAM_DEVELOPMENT_SHA256,
            "upstream_audit_sha256": contract.UPSTREAM_AUDIT_SHA256,
            "task_index": task_index,
            "source_id": expected_source_id,
            "confirmation_ids_opened": [expected_source_id],
            "development_or_audit_ids_opened": False,
        }
        for key, value in fixed.items():
            if shard.get(key) != value:
                raise ValueError(f"shard binding mismatch: {path.name}:{key}")
        if receipt is None:
            receipt = shard["source_preflight_content"]
        elif receipt != shard["source_preflight_content"]:
            raise ValueError("source preflight receipt differs across shards")
        episodes = shard.get("episodes", [])
        if len(episodes) != len(contract.NOISE_LEVELS) * len(contract.CONTRACTS):
            raise ValueError("shard episode denominator mismatch")
        for episode in episodes:
            if int(episode["source_id"]) != expected_source_id:
                raise ValueError("episode source ID escaped assigned shard")
            validate_episode(episode, thresholds)
            records.append(episode)
        shard_hashes[path.name] = sha256(path)
    if len(records) != 900 or sum(len(ep["views"]) for ep in records) != 10_800:
        raise AssertionError("confirmation global denominator drift")
    episode_ids = [str(ep["episode_id"]) for ep in records]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("duplicate confirmation episode ID")
    return {
        "schema": contract.SCHEMA,
        "stage": "confirmation_aggregate",
        "contract_sha256": contract.CONTRACT_SHA256,
        "source_closure_sha256": active["source_closure_sha256"],
        "upstream_development_sha256": contract.UPSTREAM_DEVELOPMENT_SHA256,
        "upstream_audit_sha256": contract.UPSTREAM_AUDIT_SHA256,
        "shard_count": len(paths),
        "source_cluster_count": len(contract.SEALED_CONFIRMATION_IDS),
        "episode_count": len(records),
        "selectable_view_row_count": sum(len(ep["views"]) for ep in records),
        "policy_decision_count": len(records) * len(contract.POLICIES),
        "shard_sha256": shard_hashes,
        "source_preflight_content": receipt,
        "summary": summarize(records),
        "status": "PROSPECTIVE_PDEBENCH_CONFIRMATION_COMPLETE",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = aggregate(args.shard_dir)
    write_json(args.output, value)
    print(json.dumps({
        "status": value["status"],
        "episode_count": value["episode_count"],
        "summary": value["summary"],
        "output_sha256": sha256(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
