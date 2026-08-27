"""Producer-free structural and scientific replay validator for campaign shards."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .authorization import validate as validate_authorization
from .attack_guard import POLICIES
from .authority import AuthorityRecord, diagnostic_from_json
from .campaign import SPLITS, _proposal_payload, canonical_digest
from .contract import (
    NOISE_CELLS, REFERENCE_VIEW, SCHEMA_VERSION, VIEW_BANK,
    adapted_public_objective,
    public_objective,
)
from .environment import OwnerLabel, PublicEpisode
from .owner_truth import OWNER_CASES


def _record(payload: dict) -> AuthorityRecord:
    return AuthorityRecord(
        episode_id=str(payload["episode_id"]),
        view_id=str(payload["view_id"]),
        origin=str(payload["origin"]),
        status=str(payload["status"]),
        support=tuple(payload.get("support") or ()),
        coefficients=tuple(
            (str(atom), float(value))
            for atom, value in payload.get("coefficients", [])
        ),
        residual=diagnostic_from_json(payload["residual"]),
        score_gap=diagnostic_from_json(payload["score_gap"]),
        condition=diagnostic_from_json(payload["condition"]),
        minimum_coefficient=diagnostic_from_json(
            payload["minimum_coefficient"]
        ),
        ambiguity_size=int(payload["ambiguity_size"]),
        registered_cost=float(payload["registered_cost"]),
        evidence_digest=str(payload["evidence_digest"]),
    )


def read_payload(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def validate_payload(payload: dict) -> dict:
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION + ".shard":
        errors.append("schema_version")
    if payload.get("content_sha256") != canonical_digest(payload):
        errors.append("content_sha256")
    config = payload.get("config", {})
    split = config.get("split")
    seed = config.get("seed")
    if split not in SPLITS or seed not in SPLITS.get(split, ()):
        errors.append("prospective_split")
    expected_episodes = len(OWNER_CASES) * len(NOISE_CELLS)
    if payload.get("episode_count") != expected_episodes:
        errors.append("episode_count")
    if payload.get("view_record_count") != expected_episodes * len(VIEW_BANK):
        errors.append("view_record_count")
    episodes = payload.get("episodes", [])
    labels = payload.get("owner_labels", [])
    if len(episodes) != expected_episodes or len(labels) != expected_episodes:
        errors.append("row_count")
        return {"valid": False, "errors": sorted(set(errors))}

    view_ids = [view.view_id for view in VIEW_BANK]
    seen: set[str] = set()
    for index, (episode_payload, label_payload) in enumerate(zip(episodes, labels)):
        episode_id = str(episode_payload.get("episode_id"))
        if episode_id in seen:
            errors.append("duplicate_episode")
        seen.add(episode_id)
        case = OWNER_CASES[index // len(NOISE_CELLS)]
        noise = NOISE_CELLS[index % len(NOISE_CELLS)]
        expected_label = {
            "episode_id": episode_id,
            "family": case.family,
            "truth_kind": case.truth_kind,
            "truth_support": list(case.truth_support),
            "omitted_mechanism": case.omitted_mechanism,
        }
        normalized_label = dict(label_payload)
        normalized_label["truth_support"] = list(
            normalized_label.get("truth_support") or ()
        )
        if normalized_label != expected_label:
            errors.append(f"owner_label:{index}")
        if (episode_payload.get("noise_model"),
                float(episode_payload.get("noise_level"))) != noise:
            errors.append(f"noise_order:{index}")
        view_payloads = episode_payload.get("views", [])
        if [item.get("view_id") for item in view_payloads] != view_ids:
            errors.append(f"view_manifest:{index}")
            continue
        records = {item["view_id"]: _record(item) for item in view_payloads}
        for item, view in zip(view_payloads, VIEW_BANK):
            record = records[view.view_id]
            if float(record.registered_cost) != float(view.registered_cost):
                errors.append(f"registered_cost:{index}:{view.view_id}")
            if item.get("public_objective") != public_objective(record):
                errors.append(f"public_objective:{index}:{view.view_id}")
            if item.get("adapted_public_objective") != (
                adapted_public_objective(record)
            ):
                errors.append(
                    f"adapted_public_objective:{index}:{view.view_id}"
                )
        if any(record.episode_id != episode_id for record in records.values()):
            errors.append(f"cross_episode_record:{index}")
        public = PublicEpisode(
            episode_id=episode_id,
            origin=next(iter(records.values())).origin,
            noise_model=str(episode_payload["noise_model"]),
            noise_level=float(episode_payload["noise_level"]),
            view_records=records,
        )
        try:
            public.validate()
        except Exception as exc:  # validator must fail closed on any interface defect
            errors.append(f"public_episode:{index}:{type(exc).__name__}")
            continue
        reference = _record(episode_payload.get("reference", {}))
        if reference.view_id != REFERENCE_VIEW.view_id:
            errors.append(f"reference_view:{index}")
        label = OwnerLabel(
            episode_id=episode_id,
            family=case.family,
            truth_kind=case.truth_kind,
            truth_support=case.truth_support,
            omitted_mechanism=case.omitted_mechanism,
        )
        recomputed = _proposal_payload(public, label, reference)
        if episode_payload.get("proposals") != recomputed:
            errors.append(f"proposal_replay:{index}")
        if {item.get("policy_id") for item in recomputed} != set(POLICIES):
            errors.append(f"policy_manifest:{index}")

    convergence = payload.get("convergence", [])
    if len(convergence) != len(OWNER_CASES):
        errors.append("convergence_count")
    if [row.get("case_id") for row in convergence] != [
        case.case_id for case in OWNER_CASES
    ]:
        errors.append("convergence_case_order")

    # Once a freeze exists, every executable shard must bind it exactly.
    freeze_path = Path(__file__).with_name("PREOUTCOME_FREEZE.json")
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if payload.get("source_closure_sha256") != freeze.get(
            "source_closure_sha256"
        ):
            errors.append("source_closure_binding")
    if config.get("origin") not in {"spectral", "fd"}:
        errors.append("origin")
    expected_origin = {
        "spectral": "spectral-rk4-v1",
        "fd": "centered-fd-rk4-v1",
    }.get(config.get("origin"))
    for episode in episodes:
        for row in episode.get("views", []):
            if row.get("origin") != expected_origin:
                errors.append("origin_binding")

    stage = (
        "internal_confirmation" if split == "confirmation"
        else "internal_mechanism"
    )
    authorization_digest = config.get("stage_authorization_sha256")
    bindings = config.get("stage_input_bindings")
    if not isinstance(bindings, dict):
        errors.append("stage_input_bindings")
        bindings = {}
    if not isinstance(authorization_digest, str) or len(authorization_digest) != 64:
        errors.append("stage_authorization")
    elif freeze_path.exists():
        auth_path = Path(__file__).with_name(
            f"root_authorization_{stage}_{authorization_digest[:16]}.json"
        )
        try:
            authorization = validate_authorization(
                auth_path, stage=stage, input_bindings=bindings
            )
            if authorization.authorization_digest != authorization_digest:
                errors.append("stage_authorization_binding")
        except Exception as exc:
            errors.append(f"stage_authorization:{type(exc).__name__}")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "episode_count": len(episodes),
        "view_record_count": len(episodes) * len(VIEW_BANK),
        "proposal_count": len(episodes) * len(POLICIES),
        "content_sha256": payload.get("content_sha256"),
    }


def validate_artifact(path: Path) -> dict:
    return validate_payload(read_payload(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_artifact(args.artifact)
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
