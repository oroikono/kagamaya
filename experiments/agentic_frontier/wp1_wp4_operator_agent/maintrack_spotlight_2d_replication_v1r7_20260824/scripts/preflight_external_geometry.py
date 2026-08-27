"""Geometry-only six-file publisher-VALID preflight for the v1r7 repair.

This command is deliberately incapable of opening publisher TEST. For each
manifest-pinned VALID file it first streams the complete file bytes through
SHA-256, then decodes only coordinate/shape/boundary metadata and fixed
edge/neighbor slabs for all 20 trajectories at raw frames 20, 500, and 980.
It never decodes Gray--Scott parameters, builds weak evidence, invokes the
authority, or decodes a policy outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..external_well import (
    EXPECTED_REGIMES,
    load_manifest,
    manifest_entry,
    read_publisher_geometry_witness,
    summarize_geometry_corpus,
    verify_publisher_file,
)
from ..geometry_receipt import (
    THRESHOLD_PROVENANCE,
    validate_geometry_source_receipt,
)
from .stage_external_geometry_preflight import (
    STAGE_MANIFEST_PATH,
    stage_identity,
    stage_manifest_bytes,
)


RECEIPT_SCHEMA = "maintrack.external-well-geometry-preflight.valid.v1r7"
LANE_ROOT = Path(__file__).resolve().parents[1]
V1R6_FREEZE_DIGEST = (
    "9caa529977f33f8d5afac609944a96903b886c1b2331493c725f974b3937f239"
)
V1R6_SOURCE_CLOSURE = (
    "8e9b65b938f6b98b0aad11fdc118b4e0b4bd0ef4f39069a726220f3306915c20"
)
V1R6_EXTERNAL_VALIDATION_AUTHORIZATION = (
    "8f91ff35c7e42c1f56b9e6a6de3b89068ac03bc387cf95457d1dade017279a19"
)
INTERNAL_CONFIRMATION_AGGREGATE = (
    "929a69f0619eb5e9f5dc0993706988fb5ee262b3eb11ef1f02474eae627ec166"
)


def _content_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_upstream_v1r6_provenance(
    freeze_path: Path, authorization_path: Path, internal_digest: str,
) -> dict[str, object]:
    """Bind geometry access to the immutable v1r6 VALID authorization.

    The already-authorized v1r6 VALID attempt accessed exactly this corpus and
    failed closed before producing any shard.  This receipt is not a v1r7
    scientific-stage authorization; a fresh v1r7 authorization can only be
    created after the repaired source and this receipt are frozen.
    """

    if internal_digest != INTERNAL_CONFIRMATION_AGGREGATE:
        raise RuntimeError("upstream internal-confirmation digest drift")
    freeze = json.loads(freeze_path.read_text())
    claimed_freeze = freeze.pop("freeze_digest", None)
    if (
        claimed_freeze != V1R6_FREEZE_DIGEST
        or _content_digest(freeze) != claimed_freeze
    ):
        raise RuntimeError("immutable v1r6 freeze does not validate")
    if freeze.get("source_closure_sha256") != V1R6_SOURCE_CLOSURE:
        raise RuntimeError("immutable v1r6 source closure drift")
    authorization = json.loads(authorization_path.read_text())
    claimed_authorization = authorization.pop("authorization_digest", None)
    if (
        claimed_authorization != V1R6_EXTERNAL_VALIDATION_AUTHORIZATION
        or _content_digest(authorization) != claimed_authorization
    ):
        raise RuntimeError("immutable v1r6 authorization does not validate")
    expected = {
        "authority": "root",
        "decision": "AUTHORIZE_EXACT_STAGE",
        "stage": "external_validation",
        "freeze_digest": V1R6_FREEZE_DIGEST,
        "input_bindings": {
            "internal_confirmation_aggregate_sha256": internal_digest,
        },
        "single_stage_only": True,
        "post_outcome_tuning_permitted": False,
        "external_signature": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise RuntimeError(f"immutable v1r6 authorization drift: {key}")
    return {
        "v1r6_freeze_digest": V1R6_FREEZE_DIGEST,
        "v1r6_source_closure_sha256": V1R6_SOURCE_CLOSURE,
        "v1r6_external_validation_authorization_sha256": (
            V1R6_EXTERNAL_VALIDATION_AUTHORIZATION
        ),
        "internal_confirmation_aggregate_sha256": internal_digest,
        "failed_valid_job_ids": [
            "11661634_0",
            "11661634_1",
            "11661634_2",
            "11661634_3",
            "11661634_4",
            "11661634_5",
        ],
        "scientific_artifacts_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--upstream-v1r6-freeze", type=Path, required=True)
    parser.add_argument(
        "--upstream-v1r6-authorization", type=Path, required=True
    )
    parser.add_argument(
        "--internal-confirmation-aggregate-sha256", required=True
    )
    parser.add_argument("--preflight-stage-archive-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    provenance = _validate_upstream_v1r6_provenance(
        args.upstream_v1r6_freeze,
        args.upstream_v1r6_authorization,
        args.internal_confirmation_aggregate_sha256,
    )
    source_result = validate_geometry_source_receipt(root=LANE_ROOT)
    if not source_result["valid"]:
        raise RuntimeError(
            "geometry source receipt invalid: "
            + ",".join(str(item) for item in source_result["errors"])
        )
    stage = stage_identity(LANE_ROOT.parents[3])
    if args.preflight_stage_archive_sha256 != stage["archive_sha256"]:
        raise RuntimeError("preflight stage archive digest mismatch")
    staged_manifest_path = LANE_ROOT.parents[3] / STAGE_MANIFEST_PATH
    staged_manifest_raw = staged_manifest_path.read_bytes()
    expected_manifest_raw = stage_manifest_bytes(stage["stage_manifest"])
    if staged_manifest_raw != expected_manifest_raw:
        raise RuntimeError("preflight stage manifest content mismatch")
    manifest, manifest_digest = load_manifest(args.manifest)
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on Euler
        raise RuntimeError("h5py is required for geometry preflight") from exc

    files = []
    for regime in EXPECTED_REGIMES:
        entry = manifest_entry(manifest, split="valid", regime=regime)
        path = args.data_root / entry["path"]
        verified_digest = verify_publisher_file(path, entry)
        with h5py.File(path, "r") as handle:
            geometry = read_publisher_geometry_witness(handle)
        if geometry.get("fields_accessed") is not True:
            raise AssertionError("geometry preflight did not report field access")
        if geometry.get("outcomes_decoded") is not False:
            raise AssertionError("geometry preflight reported outcome decode")
        files.append({
            "regime": regime,
            "publisher_path": entry["path"],
            "bytes": int(entry["bytes"]),
            "publisher_file_sha256": verified_digest,
            "geometry": geometry,
        })

    corpus_geometry_witness = summarize_geometry_corpus(files)

    body = {
        "schema_version": RECEIPT_SCHEMA,
        "split": "valid",
        "purpose": "VALID-only pre-v1r7-freeze field-geometry witness",
        "not_v1r7_scientific_authorization": True,
        "upstream_v1r6_provenance": provenance,
        "external_manifest_sha256": manifest_digest,
        "preflight_stage": stage,
        "executed_source_sha256": {
            "EXTERNAL_WELL_MANIFEST.json": _file_sha256(args.manifest),
            "external_well.py": _file_sha256(LANE_ROOT / "external_well.py"),
            "geometry_receipt.py": _file_sha256(
                LANE_ROOT / "geometry_receipt.py"
            ),
            "external_sources/GEOMETRY_SOURCE_RECEIPT.json": _file_sha256(
                LANE_ROOT / "external_sources/GEOMETRY_SOURCE_RECEIPT.json"
            ),
            "scripts/preflight_external_geometry.py": _file_sha256(
                Path(__file__).resolve()
            ),
            "scripts/stage_external_geometry_preflight.py": _file_sha256(
                LANE_ROOT / "scripts/stage_external_geometry_preflight.py"
            ),
            "upstream_geometry_probe/PROBE_PROVENANCE.json": _file_sha256(
                LANE_ROOT / "upstream_geometry_probe/PROBE_PROVENANCE.json"
            ),
            "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json": (
                _file_sha256(
                    LANE_ROOT
                    / "upstream_geometry_probe/"
                    "NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
                )
            ),
        },
        "regime_count": len(files),
        "files": files,
        "corpus_geometry_witness": corpus_geometry_witness,
        "publisher_access_protocol": {
            "full_valid_hdf5_byte_stream_hashed_before_hdf5_decode": True,
            "decoded_hdf5_scope": (
                "coordinate/shape/boundary metadata and edge/neighbor slabs "
                "for the fixed VALID coverage only"
            ),
            "parameters_decoded": False,
            "evidence_or_policy_outcomes_decoded": False,
            "publisher_test_accessed": False,
        },
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "fields_accessed": True,
        "outcomes_decoded": False,
        "scientific_artifacts_created": False,
    }
    body["content_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=args.receipt.parent,
        prefix=args.receipt.name + ".tmp.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, args.receipt)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({
        "content_sha256": body["content_sha256"],
        "files": len(files),
        "fields_accessed": True,
        "outcomes_decoded": False,
        "split": "valid",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
