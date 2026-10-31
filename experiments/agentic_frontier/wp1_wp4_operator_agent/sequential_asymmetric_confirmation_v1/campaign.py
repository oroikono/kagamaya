"""Build one immutable confirmation shard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..adversarial_verifier_gaming_v1.io_utils import write_json
from ..maintrack_suite_v41.simulator import Family
from ..sequential_asymmetric_controller_v2.policy import propose
from ..sequential_consensus_controller_v1.campaign import (
    _build_episode,
    score_proposal,
)
from .contract import (
    CONFIRMATION_SEEDS,
    NOISE_MODELS,
    POLICIES,
    PRODUCERS,
    RELATIVE_NOISES,
    design_sha256,
    digest,
    shard_ordinal,
    validate_calibration_eligibility,
    validate_freeze,
)


def summarize_shard(scores: list[dict[str, Any]], producer: str) -> dict[str, Any]:
    """Summarize the one producer that is structurally present in a shard."""
    output: dict[str, Any] = {}
    for policy in POLICIES:
        group = [
            row
            for row in scores
            if row["producer"] == producer and row["policy"] == policy
        ]
        if not group:
            raise ValueError(f"missing shard group {producer}::{policy}")
        n = len(group)
        output[f"{producer}::{policy}"] = {
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
            ) / n,
            "mean_views_evaluated": sum(int(row["views_evaluated"]) for row in group) / n,
            "mean_max_nodes_acquired": sum(
                int(row["max_nodes_acquired"]) for row in group
            ) / n,
            "mean_stage_count": sum(int(row["stage_count"]) for row in group) / n,
        }
    return output


def build_shard(producer: str, seed: int) -> dict[str, Any]:
    if producer not in PRODUCERS or seed not in CONFIRMATION_SEEDS:
        raise ValueError("producer/seed is outside frozen confirmation split")
    validate_calibration_eligibility()
    freeze = validate_freeze()
    publics: list[dict[str, Any]] = []
    owners: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    within = 0
    for family in Family:
        for noise_model in NOISE_MODELS:
            for relative_noise in RELATIVE_NOISES:
                public, owner = _build_episode(
                    producer=producer,
                    seed=seed,
                    family=family,
                    noise_model=noise_model,
                    relative_noise=relative_noise,
                    ordinal=shard_ordinal(producer, seed, within),
                )
                within += 1
                publics.append(public)
                owners.append(owner)
                for policy in POLICIES:
                    proposal = propose(public, policy)
                    proposals.append(proposal)
                    scores.append(score_proposal(public, owner, proposal))
    body = {
        "schema_version": "verisigs.sequential_asymmetric.confirmation_shard.v1",
        "status": "CONFIRMATION_SHARD_COMPLETE",
        "design_sha256": design_sha256(),
        "freeze_sha256": freeze["freeze_sha256"],
        "source_manifest": freeze["source_manifest"],
        "producer": producer,
        "seed": seed,
        "confirmation_access_count": 1,
        "episode_count": len(publics),
        "proposal_count": len(proposals),
        "public_episodes": publics,
        "owner_episodes": owners,
        "proposals": proposals,
        "scored_rows": scores,
        "shard_summary": summarize_shard(scores, producer),
        "claim_ceiling": "finite synthetic confirmation shard; repeated cells within one seed cluster",
    }
    return {**body, "content_sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", choices=PRODUCERS, required=True)
    parser.add_argument("--seed", type=int, choices=CONFIRMATION_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = build_shard(args.producer, args.seed)
    write_json(args.output, payload)
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()
