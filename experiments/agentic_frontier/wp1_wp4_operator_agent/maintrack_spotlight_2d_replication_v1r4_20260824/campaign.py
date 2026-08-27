"""One-seed/one-origin campaign shard with owner-isolated proposal scoring."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .authorization import validate as validate_authorization
from .attack_guard import POLICIES
from .authority import AuthorityRecord, evaluate
from .contract import (
    CONFIRMATION_SEEDS,
    DESIGN_SEEDS,
    MECHANISM_SEEDS,
    NOISE_CELLS,
    REFERENCE_VIEW,
    SCHEMA_VERSION,
    VIEW_BANK,
    adapted_public_objective,
    public_objective,
)
from .environment import OwnerLabel, PublicEpisode, score_decision
from .owner_truth import OWNER_CASES
from .trajectory import Trajectory
from .weak_evidence import build_weak_evidence, observed_arrays


SPLITS = {
    "design": DESIGN_SEEDS,
    "mechanism": MECHANISM_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
}
ORIGINS = ("spectral", "fd")


def _noise_seed(case_id: str, seed: int, origin: str,
                noise_model: str, noise_level: float) -> int:
    payload = f"{case_id}|{seed}|{origin}|{noise_model}|{noise_level:.9g}|spotlight-v1"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def _public_episode_id(trajectory: Trajectory, noise_model: str,
                       noise_level: float) -> str:
    payload = (
        f"{trajectory.episode_id}|{noise_model}|{noise_level:.9g}|"
        "spotlight-public-v1"
    )
    return "P-" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def simulate_trajectory(case_id: str, seed: int, origin: str) -> Trajectory:
    if origin == "spectral":
        from .numerical_spectral import simulate
        return simulate(case_id, seed, n=36, snapshot_count=49,
                        internal_steps=480)
    if origin == "fd":
        from .numerical_fd import simulate
        return simulate(case_id, seed, n=36, snapshot_count=49,
                        internal_steps=720)
    raise ValueError("origin must be spectral or fd")


def build_public_episode(trajectory: Trajectory, *, noise_model: str,
                         noise_level: float
                         ) -> tuple[PublicEpisode, OwnerLabel, AuthorityRecord]:
    case = next(case for case in OWNER_CASES if case.case_id == trajectory.case_id)
    noise_seed = _noise_seed(
        case.case_id, trajectory.seed, trajectory.origin,
        noise_model, noise_level,
    )
    observed_u, observed_v = observed_arrays(
        trajectory,
        noise_level=noise_level,
        noise_model=noise_model,
        noise_seed=noise_seed,
    )
    episode_id = _public_episode_id(trajectory, noise_model, noise_level)
    records: dict[str, AuthorityRecord] = {}
    for view in VIEW_BANK:
        evidence = build_weak_evidence(
            trajectory,
            view,
            observed_u=observed_u,
            observed_v=observed_v,
            episode_id=episode_id,
        )
        records[view.view_id] = evaluate(evidence)
    reference = evaluate(build_weak_evidence(
        trajectory,
        REFERENCE_VIEW,
        observed_u=observed_u,
        observed_v=observed_v,
        episode_id=episode_id,
    ))
    episode = PublicEpisode(
        episode_id=episode_id,
        origin=trajectory.origin,
        noise_model=noise_model,
        noise_level=noise_level,
        view_records=records,
    )
    episode.validate()
    label = OwnerLabel(
        episode_id=episode_id,
        family=case.family,
        truth_kind=case.truth_kind,
        truth_support=case.truth_support,
        omitted_mechanism=case.omitted_mechanism,
    )
    return episode, label, reference


def _record_payload(record: AuthorityRecord) -> dict:
    payload = record.public_dict()
    payload["coefficients"] = [list(item) for item in record.coefficients]
    payload["public_objective"] = public_objective(record)
    payload["adapted_public_objective"] = adapted_public_objective(record)
    return payload


def _reference_payload(record: AuthorityRecord) -> dict:
    payload = record.public_dict()
    payload["coefficients"] = [list(item) for item in record.coefficients]
    return payload


def _proposal_payload(episode: PublicEpisode, label: OwnerLabel,
                      reference: AuthorityRecord) -> list[dict]:
    proposals = []
    for policy_id, policy in POLICIES.items():
        proposal = policy(episode)
        # The proposal is fully locked before the owner label is joined here.
        score = score_decision(proposal.decision, label)
        signature = (proposal.decision.status, proposal.decision.support)
        reference_signature = (reference.status, reference.support)
        decisive = proposal.decision.status in {"IDENTIFIED", "OUTSIDE_LIBRARY"}
        proposals.append({
            "policy_id": policy_id,
            "selected_view_id": proposal.selected_view_id,
            "status": proposal.decision.status,
            "support": list(proposal.decision.support),
            "reason": proposal.decision.reason,
            "witnessed_view_ids": list(proposal.witnessed_view_ids),
            "realized_cost": proposal.decision.realized_cost,
            "selected_public_objective": proposal.public_objective,
            "correct": score.correct,
            "false_promotion": score.false_promotion,
            "inconclusive": score.inconclusive,
            "reference_signature_agreement": signature == reference_signature,
            "goodhart_event": decisive and signature != reference_signature,
        })
    return proposals


def build_shard(*, split: str, seed: int, origin: str,
                source_closure: str,
                authorization_digest: str | None = None,
                stage_input_bindings: dict[str, str] | None = None) -> dict:
    if split not in SPLITS or seed not in SPLITS[split]:
        raise ValueError("seed is outside the prospective split")
    if origin not in ORIGINS:
        raise ValueError("unsupported numerical origin")
    episodes = []
    labels = []
    convergence = []
    for case in OWNER_CASES:
        trajectory = simulate_trajectory(case.case_id, seed, origin)
        convergence.append({
            "case_id": case.case_id,
            "origin": trajectory.origin,
            "convergence_error": trajectory.convergence_error,
            "public_digest": trajectory.public_digest,
        })
        for noise_model, noise_level in NOISE_CELLS:
            episode, label, reference = build_public_episode(
                trajectory,
                noise_model=noise_model,
                noise_level=noise_level,
            )
            episodes.append({
                "episode_id": episode.episode_id,
                "noise_model": episode.noise_model,
                "noise_level": episode.noise_level,
                "views": [
                    _record_payload(episode.view_records[view.view_id])
                    for view in VIEW_BANK
                ],
                "reference": _reference_payload(reference),
                "proposals": _proposal_payload(episode, label, reference),
            })
            labels.append(asdict(label))
    payload = {
        "schema_version": SCHEMA_VERSION + ".shard",
        "source_closure_sha256": source_closure,
        "config": {
            "split": split,
            "seed": seed,
            "origin": origin,
            "stage_authorization_sha256": authorization_digest,
            "stage_input_bindings": dict(stage_input_bindings or {}),
        },
        "metadata": {
            "case_ids": [case.case_id for case in OWNER_CASES],
            "noise_cells": [list(cell) for cell in NOISE_CELLS],
            "view_ids": [view.view_id for view in VIEW_BANK],
            "policy_ids": list(POLICIES),
        },
        "episode_count": len(episodes),
        "view_record_count": len(episodes) * len(VIEW_BANK),
        "episodes": episodes,
        "owner_labels": labels,
        "convergence": convergence,
    }
    payload["content_sha256"] = canonical_digest(payload)
    return payload


def canonical_digest(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("content_sha256", None)
    raw = json.dumps(clone, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_write_gzip_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".tmp.", delete=False
    ) as temporary:
        temp_path = Path(temporary.name)
        with gzip.GzipFile(filename="", mode="wb", fileobj=temporary,
                           compresslevel=6, mtime=0) as compressed:
            compressed.write(raw)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--origin", choices=ORIGINS, required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument(
        "--upstream-aggregate-sha256",
        help="required only for the sealed internal confirmation stage",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = (
        "internal_confirmation" if args.split == "confirmation"
        else "internal_mechanism"
    )
    bindings = {}
    if stage == "internal_confirmation":
        if not args.upstream_aggregate_sha256:
            raise SystemExit(
                "confirmation requires --upstream-aggregate-sha256"
            )
        bindings["internal_mechanism_aggregate_sha256"] = (
            args.upstream_aggregate_sha256
        )
    authorization = validate_authorization(
        args.authorization, stage=stage, input_bindings=bindings
    )
    payload = build_shard(
        split=args.split,
        seed=args.seed,
        origin=args.origin,
        source_closure=args.source_closure,
        authorization_digest=authorization.authorization_digest,
        stage_input_bindings=bindings,
    )
    atomic_write_gzip_json(args.output, payload)
    print(json.dumps({
        "content_sha256": payload["content_sha256"],
        "episodes": payload["episode_count"],
        "views": payload["view_record_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
