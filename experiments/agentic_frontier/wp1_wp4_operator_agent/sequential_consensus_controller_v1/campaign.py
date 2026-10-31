"""Generate, execute, and owner-score the sequential consensus pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adversarial_verifier_gaming_v1.campaign import (
    _cell,
    _producer_episode,
    _public_row,
)
from ..maintrack_suite_v41.simulator import Family
from .contract import (
    DEVELOPMENT_SEEDS,
    NOISE_MODELS,
    POLICIES,
    PRODUCERS,
    PUBLIC_SEED,
    RELATIVE_NOISES,
    RESERVED_CALIBRATION_SEEDS,
    RESERVED_CONFIRMATION_SEEDS,
    TERMINAL_STATUSES,
    VIEW_BY_CONFIG,
    VIEWS,
    design_sha256,
    digest,
    source_manifest,
)
from .policy import propose, validate_public_episode


def _truth_record(value: Any) -> list[str] | None:
    return None if value is None else [str(item) for item in value]


def _public_view(view, cell: Mapping[str, Any], baseline: Any, channel: Any) -> dict[str, Any]:
    row = _public_row(view, dict(cell), baseline, channel)
    row["view_token"] = row.pop("action_token")
    row.pop("node_cost", None)
    return row


def _build_episode(
    *, producer: str, seed: int, family: Family, noise_model: str, relative_noise: float, ordinal: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    channels, baseline, truth = _producer_episode(
        producer,
        family,
        seed,
        noise_model,
        relative_noise,
        PUBLIC_SEED,
    )
    rows = []
    for view in VIEWS:
        cell = _cell(
            producer,
            baseline,
            channels[view.nodes],
            channels[49],
            truth.support,
            view.temporal_modes,
        )
        rows.append(_public_view(view, cell, baseline, channels[view.nodes]))
    reference = _cell(
        producer,
        baseline,
        channels[49],
        channels[49],
        truth.support,
        5,
    )
    public_body: dict[str, Any] = {
        "schema_version": "verisigs.sequential_consensus_controller.public_episode.v1",
        "episode_id": f"seq_dev_{producer}_{ordinal:04d}",
        "producer": producer,
        "public_metadata": {
            "noise_model": noise_model,
            "relative_noise": relative_noise,
        },
        "views": rows,
    }
    public = {**public_body, "episode_sha256": digest(public_body)}
    validate_public_episode(public)
    owner_body: dict[str, Any] = {
        "schema_version": "verisigs.sequential_consensus_controller.owner_episode.v1",
        "episode_id": public["episode_id"],
        "public_episode_sha256": public["episode_sha256"],
        "owner_episode_key": [family.value, seed, noise_model, relative_noise],
        "truth_support": _truth_record(truth.support),
        "reference_status": str(reference["final"]["status"]),
        "reference_support": _truth_record(reference["final"].get("support")),
    }
    owner = {**owner_body, "owner_sha256": digest(owner_body)}
    return public, owner


def _signature(status: str, support: Any) -> tuple[str, tuple[str, ...]]:
    return str(status), tuple(str(item) for item in (support or ()))


def _truth_metrics(status: str, support: Any, truth_support: Any) -> tuple[bool, bool]:
    truth = None if truth_support is None else tuple(str(item) for item in truth_support)
    selected = tuple(str(item) for item in (support or ()))
    correct = (truth is not None and status == "IDENTIFIED" and selected == truth) or (
        truth is None and status == "OUT_OF_GRAMMAR"
    )
    false = status in TERMINAL_STATUSES and not correct
    return bool(correct), bool(false)


def score_proposal(
    public: Mapping[str, Any], owner: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    if proposal["public_episode_sha256"] != public["episode_sha256"]:
        raise ValueError("proposal/public binding mismatch")
    if owner["public_episode_sha256"] != public["episode_sha256"]:
        raise ValueError("owner/public binding mismatch")
    token = proposal.get("selected_view_token")
    selected = None
    if proposal.get("proposal_valid") is True:
        selected = {row["view_token"]: row for row in public["views"]}.get(token)
        if selected is None:
            raise ValueError("proposal selected an unknown public view")
        status = str(selected["terminal_status"])
        support = selected.get("terminal_support")
    else:
        status, support = "INCONCLUSIVE", None
    correct, false = _truth_metrics(status, support, owner["truth_support"])
    reference_agreement = _signature(status, support) == _signature(
        owner["reference_status"], owner["reference_support"]
    )
    body = {
        "episode_id": public["episode_id"],
        "producer": public["producer"],
        "policy": proposal["policy"],
        "proposal_sha256": proposal["proposal_sha256"],
        "proposal_valid": bool(proposal["proposal_valid"]),
        "selected_view_token": token if selected is not None else None,
        "selected_status": status,
        "selected_support": list(support or ()) if selected is not None else None,
        "terminal": status in TERMINAL_STATUSES,
        "correct_owner_diagnostic": correct,
        "false_promotion_owner_diagnostic": false,
        "reference_signature_agreement": reference_agreement,
        "views_evaluated": int(proposal["views_evaluated"]),
        "max_nodes_acquired": int(proposal["max_nodes_acquired"]),
        "stage_count": int(proposal["stage_count"]),
    }
    return {**body, "score_sha256": digest(body)}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["producer"]), str(row["policy"]))].append(row)
    summary: dict[str, Any] = {}
    for producer in PRODUCERS:
        for policy in POLICIES:
            group = groups[(producer, policy)]
            if not group:
                raise ValueError(f"missing summary group {producer}::{policy}")
            n = len(group)
            summary[f"{producer}::{policy}"] = {
                "episodes": n,
                "terminal_count": sum(bool(row["terminal"]) for row in group),
                "correct_count": sum(bool(row["correct_owner_diagnostic"]) for row in group),
                "false_promotion_count": sum(
                    bool(row["false_promotion_owner_diagnostic"]) for row in group
                ),
                "reference_agreement_count": sum(
                    bool(row["reference_signature_agreement"]) for row in group
                ),
                "terminal_rate": sum(bool(row["terminal"]) for row in group) / n,
                "correct_rate": sum(bool(row["correct_owner_diagnostic"]) for row in group) / n,
                "false_promotion_rate": sum(
                    bool(row["false_promotion_owner_diagnostic"]) for row in group
                )
                / n,
                "mean_views_evaluated": sum(int(row["views_evaluated"]) for row in group) / n,
                "mean_max_nodes_acquired": sum(int(row["max_nodes_acquired"]) for row in group) / n,
                "mean_stage_count": sum(int(row["stage_count"]) for row in group) / n,
            }
    return summary


def evaluate_gates(summary: Mapping[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for producer in PRODUCERS:
        seq = summary[f"{producer}::sequential_all_mode"]
        full = summary[f"{producer}::fullbank_all_mode"]
        fixed25 = summary[f"{producer}::fixed_25_m5"]
        fixed49 = summary[f"{producer}::fixed_49_m5"]
        confidence = summary[f"{producer}::confidence_max"]
        checks = {
            "false_promotion_le_1pct": seq["false_promotion_rate"] <= 0.01,
            "terminal_coverage_ge_75pct": seq["terminal_rate"] >= 0.75,
            "correct_within_5pp_of_fullbank_consensus": seq["correct_rate"] >= full["correct_rate"] - 0.05,
            "false_no_worse_than_fixed_25_m5": seq["false_promotion_rate"] <= fixed25["false_promotion_rate"],
            "false_no_worse_than_fixed_49_m5": seq["false_promotion_rate"] <= fixed49["false_promotion_rate"],
            "false_reduction_vs_confidence_ge_50pp": confidence["false_promotion_rate"] - seq["false_promotion_rate"] >= 0.50,
            "mean_views_lt_fullbank_16": seq["mean_views_evaluated"] < 16.0,
            "mean_max_nodes_lt_49": seq["mean_max_nodes_acquired"] < 49.0,
        }
        gates[producer] = {"checks": checks, "all_passed": all(checks.values())}
    return gates


def build_pilot() -> dict[str, Any]:
    manifest = source_manifest()
    public_episodes: list[dict[str, Any]] = []
    owner_episodes: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    ordinal = 0
    for producer in PRODUCERS:
        for seed in DEVELOPMENT_SEEDS:
            for family in Family:
                for noise_model in NOISE_MODELS:
                    for relative_noise in RELATIVE_NOISES:
                        public, owner = _build_episode(
                            producer=producer,
                            seed=seed,
                            family=family,
                            noise_model=noise_model,
                            relative_noise=relative_noise,
                            ordinal=ordinal,
                        )
                        ordinal += 1
                        public_episodes.append(public)
                        owner_episodes.append(owner)
                        for policy in POLICIES:
                            proposal = propose(public, policy)
                            proposals.append(proposal)
                            scored.append(score_proposal(public, owner, proposal))
    summary = summarize(scored)
    gates = evaluate_gates(summary)
    body: dict[str, Any] = {
        "schema_version": "verisigs.sequential_consensus_controller.pilot.v1",
        "status": (
            "CALIBRATION_DESIGN_ELIGIBLE"
            if all(row["all_passed"] for row in gates.values())
            else "STOP_AFTER_NEGATIVE_DEVELOPMENT"
        ),
        "design_sha256": design_sha256(),
        "source_manifest": manifest,
        "split": {
            "development_seeds": list(DEVELOPMENT_SEEDS),
            "reserved_calibration_seeds": list(RESERVED_CALIBRATION_SEEDS),
            "reserved_confirmation_seeds": list(RESERVED_CONFIRMATION_SEEDS),
            "calibration_access_count": 0,
            "confirmation_access_count": 0,
        },
        "episode_count": len(public_episodes),
        "proposal_count": len(proposals),
        "public_episodes": public_episodes,
        "owner_episodes": owner_episodes,
        "proposals": proposals,
        "scored_rows": scored,
        "summary": summary,
        "development_gates": gates,
        "claim_ceiling": (
            "two-seed finite synthetic development only; no population, physical, "
            "continuum, independence, calibrated-risk, or confirmation claim"
        ),
    }
    return {**body, "content_sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build_pilot()
    from ..adversarial_verifier_gaming_v1.io_utils import write_json

    write_json(args.output, payload)
    print(payload["status"])
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()

