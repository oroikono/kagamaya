"""Producer-free replay validator for external The Well campaign shards."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .authorization import validate as validate_authorization
from .attack_guard import POLICIES
from .authority import AuthorityRecord, diagnostic_from_json
from .campaign import canonical_digest
from .contract import (
    NOISE_CELLS,
    VIEW_BANK,
    adapted_public_objective,
    public_objective,
)
from .environment import PublicEpisode
from .external_well import (
    EXTERNAL_ORIGIN,
    EXTERNAL_REFERENCE_VIEW_ID,
    EXTERNAL_SCHEMA_VERSION,
    EXPECTED_REGIMES,
    EXPECTED_SPLITS,
    LIBRARY_CONTRACTS,
    TARGETS,
    TRAJECTORIES_PER_FILE,
    _proposal_payload,
    load_manifest,
    manifest_entry,
    owner_label,
)


def read_external_payload(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


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


def _expected_cells() -> list[tuple[int, str, float, str, str]]:
    return [
        (trajectory_index, noise_model, noise_level, target, contract_id)
        for trajectory_index in range(TRAJECTORIES_PER_FILE)
        for noise_model, noise_level in NOISE_CELLS
        for target in TARGETS
        for contract_id in LIBRARY_CONTRACTS
    ]


def validate_external_payload(payload: dict, *, manifest_path: Path) -> dict:
    errors: list[str] = []
    try:
        manifest, manifest_digest = load_manifest(manifest_path)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"manifest:{type(exc).__name__}"],
        }
    if payload.get("schema_version") != EXTERNAL_SCHEMA_VERSION + ".shard":
        errors.append("schema_version")
    if payload.get("content_sha256") != canonical_digest(payload):
        errors.append("content_sha256")
    if payload.get("external_manifest_sha256") != manifest_digest:
        errors.append("external_manifest_binding")
    config = payload.get("config", {})
    split = str(config.get("split"))
    regime = str(config.get("regime"))
    if split not in EXPECTED_SPLITS:
        errors.append("split")
    if regime not in EXPECTED_REGIMES:
        errors.append("regime")
    if split in EXPECTED_SPLITS and regime in EXPECTED_REGIMES:
        entry = manifest_entry(manifest, split=split, regime=regime)
        if config.get("publisher_path") != entry["path"]:
            errors.append("publisher_path")
        if config.get("publisher_file_sha256") != entry["sha256"]:
            errors.append("publisher_file_sha256")
    authorization_digest = str(config.get("stage_authorization_sha256", ""))
    if len(authorization_digest) != 64:
        errors.append("stage_authorization_sha256")
    bindings = config.get("stage_input_bindings")
    if not isinstance(bindings, dict):
        errors.append("stage_input_bindings")
        bindings = {}
    expected_binding_keys = {
        "internal_confirmation_aggregate_sha256"
    } if split == "valid" else {
        "internal_confirmation_aggregate_sha256",
        "external_validation_aggregate_sha256",
    }
    if split in EXPECTED_SPLITS:
        if set(bindings) != expected_binding_keys:
            errors.append("stage_input_binding_keys")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in bindings.values()
        ):
            errors.append("stage_input_binding_digest")
    expected_cells = _expected_cells()
    expected_episodes = len(expected_cells)
    if payload.get("episode_count") != expected_episodes:
        errors.append("episode_count")
    if payload.get("view_record_count") != expected_episodes * len(VIEW_BANK):
        errors.append("view_record_count")
    episodes = payload.get("episodes", [])
    labels = payload.get("owner_labels", [])
    if len(episodes) != expected_episodes or len(labels) != expected_episodes:
        errors.append("row_count")
        return {
            "valid": False,
            "errors": sorted(set(errors)),
            "episode_count": len(episodes),
        }
    view_ids = [view.view_id for view in VIEW_BANK]
    seen: set[str] = set()
    for index, (episode_payload, label_payload, expected) in enumerate(
        zip(episodes, labels, expected_cells)
    ):
        trajectory_index, noise_model, noise_level, target, contract_id = expected
        episode_id = str(episode_payload.get("episode_id"))
        if episode_id in seen:
            errors.append("duplicate_episode")
        seen.add(episode_id)
        observed_cell = (
            int(episode_payload.get("trajectory_index", -1)),
            str(episode_payload.get("noise_model")),
            float(episode_payload.get("noise_level", -1.0)),
            str(episode_payload.get("target")),
            str(episode_payload.get("library_contract")),
        )
        if observed_cell != expected:
            errors.append(f"cell_order:{index}")
        expected_label = owner_label(
            episode_id, target=target, contract_id=contract_id
        )
        normalized_label = dict(label_payload)
        normalized_label["truth_support"] = tuple(
            normalized_label.get("truth_support") or ()
        )
        if normalized_label != {
            "episode_id": expected_label.episode_id,
            "family": expected_label.family,
            "truth_kind": expected_label.truth_kind,
            "truth_support": expected_label.truth_support,
            "omitted_mechanism": expected_label.omitted_mechanism,
        }:
            errors.append(f"owner_label:{index}")
        view_payloads = episode_payload.get("views", [])
        if [row.get("view_id") for row in view_payloads] != view_ids:
            errors.append(f"view_manifest:{index}")
            continue
        try:
            records = {row["view_id"]: _record(row) for row in view_payloads}
        except Exception as exc:
            errors.append(f"record_decode:{index}:{type(exc).__name__}")
            continue
        if any(record.episode_id != episode_id for record in records.values()):
            errors.append(f"cross_episode_record:{index}")
        if any(record.origin != EXTERNAL_ORIGIN for record in records.values()):
            errors.append(f"origin:{index}")
        for row in view_payloads:
            try:
                if float(row["public_objective"]) != public_objective(records[row["view_id"]]):
                    errors.append(f"public_objective:{index}")
                    break
                if float(row["adapted_public_objective"]) != adapted_public_objective(
                    records[row["view_id"]]
                ):
                    errors.append(f"adapted_public_objective:{index}")
                    break
            except Exception:
                errors.append(f"dual_public_objective:{index}")
                break
        public = PublicEpisode(
            episode_id=episode_id,
            origin=EXTERNAL_ORIGIN,
            noise_model=noise_model,
            noise_level=noise_level,
            view_records=records,
        )
        try:
            public.validate()
            reference = _record(episode_payload.get("reference", {}))
        except Exception as exc:
            errors.append(f"episode_decode:{index}:{type(exc).__name__}")
            continue
        if reference.view_id != EXTERNAL_REFERENCE_VIEW_ID:
            errors.append(f"reference_view:{index}")
        if reference.episode_id != episode_id:
            errors.append(f"reference_episode:{index}")
        recomputed = _proposal_payload(public, expected_label, reference)
        if episode_payload.get("proposals") != recomputed:
            errors.append(f"proposal_replay:{index}")
        if [row.get("policy_id") for row in recomputed] != list(POLICIES):
            errors.append(f"policy_manifest:{index}")

    metadata = payload.get("metadata", {})
    if metadata.get("feature_scaling") != (
        "published-parameter mechanism groups; expected complete-model "
        "coefficients have magnitude one; authority thresholds unchanged"
    ):
        errors.append("feature_scaling_contract")
    if metadata.get("library_contracts") != list(LIBRARY_CONTRACTS):
        errors.append("library_contracts")
    if metadata.get("targets") != list(TARGETS):
        errors.append("targets")
    freeze_path = Path(__file__).with_name("PREOUTCOME_FREEZE.json")
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if payload.get("source_closure_sha256") != freeze.get(
            "source_closure_sha256"
        ):
            errors.append("source_closure_binding")
        if len(authorization_digest) == 64 and split in EXPECTED_SPLITS:
            stage = (
                "external_validation" if split == "valid"
                else "external_confirmation"
            )
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
        "split": split,
        "regime": regime,
        "episode_count": len(episodes),
        "view_record_count": len(episodes) * len(VIEW_BANK),
        "proposal_count": len(episodes) * len(POLICIES),
        "content_sha256": payload.get("content_sha256"),
    }


def validate_external_artifact(path: Path, *, manifest_path: Path) -> dict:
    return validate_external_payload(
        read_external_payload(path), manifest_path=manifest_path
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_external_artifact(args.artifact, manifest_path=args.manifest)
    if args.report:
        if args.report.exists():
            raise FileExistsError(args.report)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
