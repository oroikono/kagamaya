"""Deterministic structural and policy replay for one confirmation shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ..sequential_asymmetric_controller_v2.policy import propose
from ..sequential_consensus_controller_v1.campaign import score_proposal
from ..sequential_consensus_controller_v1.policy import validate_public_episode
from .contract import (
    CONFIRMATION_SEEDS,
    EPISODES_PER_SHARD,
    POLICIES,
    PRODUCERS,
    design_sha256,
    digest,
    validate_freeze,
)
from .campaign import summarize_shard


def validate_shard(payload: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    body = dict(payload)
    content = body.pop("content_sha256", None)
    if content != digest(body):
        errors.append("content_digest")
    try:
        freeze = validate_freeze()
    except (FileNotFoundError, ValueError) as error:
        freeze = {}
        errors.append(f"freeze:{error}")
    if payload.get("design_sha256") != design_sha256():
        errors.append("design_binding")
    if payload.get("freeze_sha256") != freeze.get("freeze_sha256"):
        errors.append("freeze_binding")
    if payload.get("source_manifest") != freeze.get("source_manifest"):
        errors.append("source_binding")
    producer = payload.get("producer")
    seed = payload.get("seed")
    if producer not in PRODUCERS or seed not in CONFIRMATION_SEEDS:
        errors.append("split_binding")
    if payload.get("confirmation_access_count") != 1:
        errors.append("confirmation_access")

    publics: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("public_episodes", ()):
        try:
            validate_public_episode(row)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"public:{error}")
            continue
        key = str(row["episode_id"])
        if key in publics:
            errors.append("duplicate_public")
        publics[key] = row
        if row.get("producer") != producer:
            errors.append("public_producer")
    owners: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("owner_episodes", ()):
        owner_body = dict(row)
        owner_hash = owner_body.pop("owner_sha256", None)
        if owner_hash != digest(owner_body):
            errors.append("owner_digest")
        key = str(row.get("episode_id"))
        if key in owners:
            errors.append("duplicate_owner")
        owners[key] = row
        try:
            if int(row["owner_episode_key"][1]) != seed:
                errors.append("owner_seed")
        except (KeyError, IndexError, TypeError, ValueError):
            errors.append("owner_key")
    if set(publics) != set(owners) or len(publics) != EPISODES_PER_SHARD:
        errors.append("episode_coverage")

    proposals: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("proposals", ()):
        proposal_body = dict(row)
        proposal_hash = proposal_body.pop("proposal_sha256", None)
        if proposal_hash != digest(proposal_body):
            errors.append("proposal_digest")
        key = (str(row.get("episode_id")), str(row.get("policy")))
        if key in proposals:
            errors.append("duplicate_proposal")
        proposals[key] = row
    scores: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("scored_rows", ()):
        score_body = dict(row)
        score_hash = score_body.pop("score_sha256", None)
        if score_hash != digest(score_body):
            errors.append("score_digest")
        key = (str(row.get("episode_id")), str(row.get("policy")))
        if key in scores:
            errors.append("duplicate_score")
        scores[key] = row
    expected = {(episode, policy) for episode in publics for policy in POLICIES}
    if set(proposals) != expected or set(scores) != expected:
        errors.append("decision_coverage")

    replay: list[Mapping[str, Any]] = []
    if not errors:
        for episode, policy in sorted(expected):
            expected_proposal = propose(publics[episode], policy)
            if expected_proposal != proposals[(episode, policy)]:
                errors.append("proposal_replay")
                break
            expected_score = score_proposal(publics[episode], owners[episode], expected_proposal)
            if expected_score != scores[(episode, policy)]:
                errors.append("score_replay")
                break
            replay.append(expected_score)
    if not errors and summarize_shard(replay, str(producer)) != payload.get("shard_summary"):
        errors.append("summary_replay")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "producer": producer,
        "seed": seed,
        "episodes_validated": len(publics),
        "decisions_validated": len(scores),
        "content_sha256": content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    report = validate_shard(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
