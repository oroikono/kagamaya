from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import tarfile

import numpy as np
import pytest

from ..aggregate_external import authority_admission
from ..authority import AuthorityRecord
from ..campaign import canonical_digest
from ..contract import (
    NOISE_CELLS,
    VIEW_BANK,
    VIEW_BY_CONFIG,
    adapted_public_objective,
    public_objective,
)
from .. import external_well as external_module
from .. import freeze as freeze_module
from .. import validate_external as validate_external_module
from ..environment import PublicEpisode
from ..external_well import (
    CANONICAL_SPATIAL_RESOLUTION,
    CONTRACT_COLUMNS,
    DELTA_A,
    EXTERNAL_ORIGIN,
    EXTERNAL_REFERENCE_VIEW_ID,
    EXTERNAL_SCHEMA_VERSION,
    EXPECTED_REGIMES,
    ExternalTrajectory,
    GEOMETRY_WITNESS_FRAME_INDICES,
    GEOMETRY_WITNESS_RATIO_FLOOR,
    GEOMETRY_WITNESS_RATIO_FLOOR_TIMING,
    PUBLISHER_COORDINATE_SHA256,
    PUBLISHER_PERIOD,
    ProjectionBundle,
    SPATIAL_RESOLUTION,
    _canonical_publisher_axis,
    _canonicalize_publisher_fields,
    _full_design,
    _low_modes,
    _periodic_length,
    _physical_wavenumber_squared,
    _proposal_payload,
    _publisher_axis,
    _record_payload,
    _spatial_projection,
    _validate_periodic_boundary_metadata,
    atoms_for_target,
    authorization_bindings,
    build_external_shard,
    build_external_evidence,
    build_projection_bundle,
    external_reference_cost,
    external_registered_cost,
    external_spatial_grid_contract,
    load_manifest,
    observed_external_arrays,
    owner_label,
    read_publisher_geometry_witness,
    read_publisher_grid_metadata,
    read_open_well_trajectory,
    summarize_geometry_corpus,
    temporal_positions,
)
from ..geometry_receipt import (
    PUBLISHER_ACCESS_PROTOCOL,
    THRESHOLD_PROVENANCE,
    V1R6_PROVENANCE,
    validate_external_geometry_preflight_receipt,
    validate_geometry_source_receipt,
    validate_upstream_v1r6_provenance,
)
from ..scripts.preflight_external_geometry import (
    INTERNAL_CONFIRMATION_AGGREGATE,
    V1R6_EXTERNAL_VALIDATION_AUTHORIZATION,
    V1R6_FREEZE_DIGEST,
    V1R6_SOURCE_CLOSURE,
    _validate_upstream_v1r6_provenance,
)
from ..scripts.check_v1r7_diff import audit as audit_v1r7_diff
from ..scripts.stage_external_geometry_preflight import (
    STAGE_FILES,
    STAGE_MANIFEST_PATH,
    build_stage,
    stage_identity,
)
from ..scripts import stage_external_geometry_preflight as preflight_stage_module
from ..scripts.stage_scientific import (
    SCIENTIFIC_STAGE_MANIFEST_PATH,
    scientific_stage_archive_bytes,
)
from ..validate_external import validate_external_payload


LANE = Path(__file__).resolve().parents[1]
SYNTHETIC_SOURCE_CLOSURE = "a" * 64
SYNTHETIC_AUTHORIZATION_DIGEST = "b" * 64
SYNTHETIC_INTERNAL_BINDING = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_receipt(body: dict) -> dict:
    payload = json.loads(json.dumps(body))
    payload["content_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    return payload


def _valid_preflight_receipt(root: Path = LANE) -> dict:
    manifest, manifest_digest = load_manifest(
        root / "EXTERNAL_WELL_MANIFEST.json"
    )
    axis = {
        "raw_dtype": "float32",
        "raw_shape": [128],
        "raw_first": -1.0,
        "raw_last": 1.0,
        "raw_sha256": PUBLISHER_COORDINATE_SHA256,
        "raw_role": "publisher_label_metadata_only",
        "raw_used_as_solver_coordinates": False,
        "override_reason": (
            "pinned generator plus Chebfun spinop2/trigpts define 128 unique "
            "endpoint-exclusive periodic solver samples"
        ),
        "canonical_nodes": 128,
        "canonical_first": -1.0,
        "canonical_last": 0.984375,
        "canonical_last_excluded": 1.0,
        "canonical_period": 2.0,
        "canonical_spacing": 0.015625,
    }
    files = []
    for regime in EXPECTED_REGIMES:
        entry = next(
            row for row in manifest["files"]
            if row["split"] == "valid" and row["regime"] == regime
        )
        fields = {}
        for field_name in ("A", "B"):
            frame_witnesses = []
            for frame_index in GEOMETRY_WITNESS_FRAME_INDICES:
                axes = {}
                for axis_name in ("x", "y"):
                    prefix = f"{regime}|{field_name}|{frame_index}|{axis_name}"
                    axes[axis_name] = {
                        "edge_neighbor_slab_shape": [20, 4, 128],
                        "edge_neighbor_slab_sha256": hashlib.sha256(
                            (prefix + "|slab").encode()
                        ).hexdigest(),
                        "first_edge_sha256": hashlib.sha256(
                            (prefix + "|first").encode()
                        ).hexdigest(),
                        "last_edge_sha256": hashlib.sha256(
                            (prefix + "|last").encode()
                        ).hexdigest(),
                        "wrap_rms": 1.0,
                        "adjacent_rms": 0.8,
                        "wrap_to_adjacent_rms_ratio": 1.25,
                        "wrap_sum_squares": 1.0 * (20 * 128),
                        "wrap_value_count": 20 * 128,
                        "adjacent_sum_squares": 0.8**2 * (20 * 2 * 128),
                        "adjacent_value_count": 20 * 2 * 128,
                        "trajectory_count": 20,
                        "per_trajectory_ratio_sha256": hashlib.sha256(
                            (prefix + "|ratios").encode()
                        ).hexdigest(),
                        "per_trajectory_edge_identity_sha256": hashlib.sha256(
                            (prefix + "|identity").encode()
                        ).hexdigest(),
                        "opposing_edges_identical_trajectory_count": 0,
                        "informative_trajectory_count": 20,
                        "degenerate_trajectory_count": 0,
                        "per_trajectory_informative_sha256": hashlib.sha256(
                            (prefix + "|informative").encode()
                        ).hexdigest(),
                        "wrap_rms_summary": {
                            "min": 1.0,
                            "median": 1.0,
                            "max": 1.0,
                        },
                        "adjacent_rms_summary": {
                            "min": 0.8,
                            "median": 0.8,
                            "max": 0.8,
                        },
                        "wrap_to_adjacent_ratio_summary": {
                            "min": 1.25,
                            "median": 1.25,
                            "max": 1.25,
                        },
                        "ratio_floor": GEOMETRY_WITNESS_RATIO_FLOOR,
                        "ratio_floor_applied_at": "corpus_axis_pool_only",
                        "cell_status": "informative",
                        "opposing_edges_identical": False,
                        "finite": True,
                    }
                frame_witnesses.append({
                    "raw_frame_index": frame_index,
                    "axes": axes,
                })
            fields[field_name] = {
                "dataset_shape": [20, 1001, 128, 128],
                "dataset_dtype": "float32",
                "trajectory_indices": list(range(20)),
                "frame_witnesses": frame_witnesses,
            }
        grid_metadata = {
            "axes": {
                name: {"name": name, **axis} for name in ("x", "y")
            },
            "boundary_conditions": {
                name: {
                    "group": f"{name}_periodic",
                    "associated_dims": [name],
                    "associated_fields": [],
                    "bc_type": "PERIODIC",
                    "sample_varying": False,
                    "time_varying": False,
                }
                for name in ("x", "y")
            },
            "fields_accessed": False,
            "outcomes_decoded": False,
        }
        files.append({
            "regime": regime,
            "publisher_path": entry["path"],
            "bytes": entry["bytes"],
            "publisher_file_sha256": entry["sha256"],
            "geometry": {
                "grid_metadata": grid_metadata,
                "field_witness": {
                    "trajectory_indices": list(range(20)),
                    "raw_frame_indices": list(GEOMETRY_WITNESS_FRAME_INDICES),
                    "ratio_floor": GEOMETRY_WITNESS_RATIO_FLOOR,
                    "ratio_floor_timing": GEOMETRY_WITNESS_RATIO_FLOOR_TIMING,
                    "fields": fields,
                    "interpretation": (
                        "individual constant, axis-invariant, or seam-symmetric "
                        "cells are recorded as degenerate and noninformative; "
                        "nonduplication is decided only by the "
                        "pooled all-VALID corpus-axis gate"
                    ),
                },
                "fields_accessed": True,
                "outcomes_decoded": False,
            },
        })
    return _seal_receipt({
        "schema_version": (
            "maintrack.external-well-geometry-preflight.valid.v1r7"
        ),
        "split": "valid",
        "purpose": "VALID-only pre-v1r7-freeze field-geometry witness",
        "not_v1r7_scientific_authorization": True,
        "upstream_v1r6_provenance": V1R6_PROVENANCE,
        "external_manifest_sha256": manifest_digest,
        "preflight_stage": stage_identity(LANE.parents[3]),
        "executed_source_sha256": {
            "EXTERNAL_WELL_MANIFEST.json": _sha256(
                root / "EXTERNAL_WELL_MANIFEST.json"
            ),
            "external_well.py": _sha256(root / "external_well.py"),
            "geometry_receipt.py": _sha256(root / "geometry_receipt.py"),
            "external_sources/GEOMETRY_SOURCE_RECEIPT.json": _sha256(
                root / "external_sources/GEOMETRY_SOURCE_RECEIPT.json"
            ),
            "scripts/preflight_external_geometry.py": _sha256(
                root / "scripts/preflight_external_geometry.py"
            ),
            "scripts/stage_external_geometry_preflight.py": _sha256(
                root / "scripts/stage_external_geometry_preflight.py"
            ),
            "upstream_geometry_probe/PROBE_PROVENANCE.json": _sha256(
                root / "upstream_geometry_probe/PROBE_PROVENANCE.json"
            ),
            "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json": (
                _sha256(
                    root
                    / "upstream_geometry_probe/"
                    "NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
                )
            ),
        },
        "regime_count": 6,
        "files": files,
        "corpus_geometry_witness": summarize_geometry_corpus(files),
        "publisher_access_protocol": PUBLISHER_ACCESS_PROTOCOL,
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "fields_accessed": True,
        "outcomes_decoded": False,
        "scientific_artifacts_created": False,
    })


def _single_episode_external_payload(
    *,
    cost_nodes: int = 128,
    source_closure: str = SYNTHETIC_SOURCE_CLOSURE,
    authorization_digest: str = SYNTHETIC_AUTHORIZATION_DIGEST,
    internal_binding: str = SYNTHETIC_INTERNAL_BINDING,
) -> dict:
    manifest, manifest_digest = load_manifest(
        LANE / "EXTERNAL_WELL_MANIFEST.json"
    )
    entry = next(
        row for row in manifest["files"]
        if row["split"] == "valid" and row["regime"] == "bubbles"
    )
    episode_id = "W-validator-mutation"
    records = {}
    for view in VIEW_BANK:
        records[view.view_id] = AuthorityRecord(
            episode_id=episode_id,
            view_id=view.view_id,
            origin=EXTERNAL_ORIGIN,
            status="OUTSIDE_LIBRARY",
            support=(),
            coefficients=(),
            residual=0.25,
            score_gap=0.01,
            condition=5.0,
            minimum_coefficient=1.0,
            ambiguity_size=1,
            registered_cost=external_registered_cost(view, cost_nodes),
            evidence_digest=hashlib.sha256(view.view_id.encode()).hexdigest(),
        )
    public = PublicEpisode(
        episode_id=episode_id,
        origin=EXTERNAL_ORIGIN,
        noise_model="iid",
        noise_level=0.0,
        view_records=records,
    )
    public.validate()
    label = owner_label(episode_id, target="A", contract_id="full")
    reference = AuthorityRecord(
        episode_id=episode_id,
        view_id=EXTERNAL_REFERENCE_VIEW_ID,
        origin=EXTERNAL_ORIGIN,
        status="OUTSIDE_LIBRARY",
        support=(),
        coefficients=(),
        residual=0.25,
        score_gap=0.01,
        condition=5.0,
        minimum_coefficient=1.0,
        ambiguity_size=1,
        registered_cost=external_reference_cost(cost_nodes),
        evidence_digest="e" * 64,
    )
    payload = {
        "schema_version": EXTERNAL_SCHEMA_VERSION + ".shard",
        "source_closure_sha256": source_closure,
        "external_manifest_sha256": manifest_digest,
        "config": {
            "split": "valid",
            "regime": "bubbles",
            "publisher_path": entry["path"],
            "publisher_file_sha256": entry["sha256"],
            "stage_authorization_sha256": authorization_digest,
            "stage_input_bindings": {
                "internal_confirmation_aggregate_sha256": internal_binding,
            },
        },
        "metadata": {
            "feature_scaling": (
                "published-parameter mechanism groups; expected complete-model "
                "coefficients have magnitude one; authority thresholds unchanged"
            ),
            "library_contracts": list(external_module.LIBRARY_CONTRACTS),
            "targets": list(external_module.TARGETS),
            "spatial_grid_contract": external_spatial_grid_contract(),
        },
        "episode_count": 1,
        "view_record_count": len(VIEW_BANK),
        "episodes": [{
            "episode_id": episode_id,
            "trajectory_index": 0,
            "target": "A",
            "library_contract": "full",
            "noise_model": "iid",
            "noise_level": 0.0,
            "views": [
                _record_payload(records[view.view_id], include_objective=True)
                for view in VIEW_BANK
            ],
            "reference": _record_payload(reference, include_objective=False),
            "proposals": _proposal_payload(public, label, reference),
        }],
        "owner_labels": [asdict(label)],
    }
    payload["content_sha256"] = canonical_digest(payload)
    return payload


def _isolate_external_validator_governance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    postfreeze_staged: bool,
) -> None:
    """Make a content-validator unit test independent of ambient lane state."""
    staged_lane = tmp_path / "staged_lane"
    staged_lane.mkdir()
    fake_module_path = staged_lane / "validate_external.py"
    monkeypatch.setattr(
        validate_external_module,
        "Path",
        lambda _value: fake_module_path,
    )

    if not postfreeze_staged:
        def reject_authorization(*_args, **_kwargs):
            raise AssertionError("prefreeze validation accessed authorization")

        monkeypatch.setattr(
            validate_external_module,
            "validate_authorization",
            reject_authorization,
        )
        return

    (staged_lane / "PREOUTCOME_FREEZE.json").write_text(json.dumps({
        "source_closure_sha256": SYNTHETIC_SOURCE_CLOSURE,
    }))

    def validate_synthetic_authorization(
        path: Path, *, stage: str, input_bindings: dict,
    ):
        assert path == staged_lane / (
            "root_authorization_external_validation_"
            f"{SYNTHETIC_AUTHORIZATION_DIGEST[:16]}.json"
        )
        assert stage == "external_validation"
        assert input_bindings == {
            "internal_confirmation_aggregate_sha256": (
                SYNTHETIC_INTERNAL_BINDING
            ),
        }

        class Authorization:
            authorization_digest = SYNTHETIC_AUTHORIZATION_DIGEST

        return Authorization()

    monkeypatch.setattr(
        validate_external_module,
        "validate_authorization",
        validate_synthetic_authorization,
    )


def _trajectory(n: int = 12) -> ExternalTrajectory:
    times = np.arange(961, dtype=float) * 10.0 + 200.0
    x = np.linspace(-1.0, 1.0, n, endpoint=False)
    y = np.linspace(-1.0, 1.0, n, endpoint=False)
    tt = np.linspace(0.0, 1.0, 961)[:, None, None]
    xx = np.arange(n)[None, :, None]
    yy = np.arange(n)[None, None, :]
    A = (0.7 + 0.05 * np.cos(2 * np.pi * xx / n)
         + 0.01 * tt + 0.0 * yy).astype(np.float32)
    B = (0.2 + 0.04 * np.sin(2 * np.pi * yy / n)
         - 0.01 * tt + 0.0 * xx).astype(np.float32)
    return ExternalTrajectory(
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        times=times,
        x=x,
        y=y,
        A=A,
        B=B,
        F=0.098,
        k=0.057,
        source_file_sha256="a" * 64,
    )


def test_pinned_manifest_is_exact_six_by_two() -> None:
    manifest, digest = load_manifest(LANE / "EXTERNAL_WELL_MANIFEST.json")
    assert len(digest) == 64
    assert {
        (row["split"], row["regime"]) for row in manifest["files"]
    } == {
        (split, regime)
        for split in ("valid", "test")
        for regime in EXPECTED_REGIMES
    }
    worms = next(
        row for row in manifest["files"]
        if row["split"] == "valid" and row["regime"] == "worms"
    )
    assert worms["sha256"] == (
        "f8df51aa04ca360113ad088f2a5038a95b366ffbb049526e889367e6d8a3ddfa"
    )


def test_geometry_preflight_binds_failed_closed_v1r6_valid_access() -> None:
    upstream = LANE / "upstream_v1r6"
    authorization = upstream / (
        "root_authorization_external_validation_"
        f"{V1R6_EXTERNAL_VALIDATION_AUTHORIZATION[:16]}.json"
    )
    provenance = _validate_upstream_v1r6_provenance(
        upstream / "PREOUTCOME_FREEZE.json",
        authorization,
        INTERNAL_CONFIRMATION_AGGREGATE,
    )
    assert provenance == V1R6_PROVENANCE
    assert provenance["v1r6_freeze_digest"] == V1R6_FREEZE_DIGEST
    assert provenance["v1r6_source_closure_sha256"] == V1R6_SOURCE_CLOSURE


def test_preflight_receipt_validates_canonical_content_and_final_sources(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_valid_preflight_receipt(), sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert result["valid"], result


def test_preflight_receipt_accepts_a_degenerate_row_when_corpus_is_informative(
    tmp_path: Path,
) -> None:
    payload = _valid_preflight_receipt()
    payload.pop("content_sha256")
    row = payload["files"][0]["geometry"]["field_witness"]["fields"]["A"][
        "frame_witnesses"
    ][0]["axes"]["x"]
    for key in (
        "wrap_rms",
        "adjacent_rms",
        "wrap_to_adjacent_rms_ratio",
        "wrap_rms_summary",
        "adjacent_rms_summary",
        "wrap_to_adjacent_ratio_summary",
    ):
        row[key] = None
    row["wrap_sum_squares"] = 0.0
    row["wrap_value_count"] = 0
    row["adjacent_sum_squares"] = 0.0
    row["adjacent_value_count"] = 0
    row["opposing_edges_identical_trajectory_count"] = 20
    row["informative_trajectory_count"] = 0
    row["degenerate_trajectory_count"] = 20
    row["cell_status"] = "degenerate_noninformative"
    row["opposing_edges_identical"] = True
    payload["corpus_geometry_witness"] = summarize_geometry_corpus(
        payload["files"]
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_seal_receipt(payload), sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert result["valid"], result


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("split", "test", "split"),
        ("fields_accessed", False, "fields_accessed"),
        ("outcomes_decoded", True, "outcomes_decoded"),
        ("scientific_artifacts_created", True, "scientific_artifacts_created"),
        (
            "not_v1r7_scientific_authorization",
            False,
            "not_v1r7_scientific_authorization",
        ),
        ("regime_count", 5, "regime_count"),
    ],
)
def test_preflight_receipt_rejects_semantic_mutation_even_with_new_digest(
    tmp_path: Path, field: str, value: object, error: str,
) -> None:
    payload = _valid_preflight_receipt()
    payload.pop("content_sha256")
    payload[field] = value
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_seal_receipt(payload), sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert not result["valid"]
    assert error in result["errors"]


def test_preflight_receipt_rejects_content_tamper_without_new_digest(
    tmp_path: Path,
) -> None:
    payload = _valid_preflight_receipt()
    payload["files"][0]["geometry"]["field_witness"]["fields"]["A"][
        "frame_witnesses"
    ][0]["axes"]["x"]["wrap_to_adjacent_rms_ratio"] = 0.0
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert not result["valid"]
    assert "content_sha256" in result["errors"]
    assert any("ratio_replay" in error for error in result["errors"])


@pytest.mark.parametrize(
    "mutation,expected_error",
    [
        ("missing_trajectory", "field_witness_trajectories:bubbles"),
        ("missing_frame", "bubbles:A:frames"),
        ("bad_slab_hash", "edge_neighbor_slab_sha256"),
        ("low_ratio", "rms_replay"),
        ("invalid_partition", "trajectory_class_partition"),
    ],
)
def test_geometry_receipt_rejects_nested_mutations_with_resealed_digest(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    payload = _valid_preflight_receipt()
    payload.pop("content_sha256")
    geometry = payload["files"][0]["geometry"]
    field = geometry["field_witness"]["fields"]["A"]
    axis = field["frame_witnesses"][0]["axes"]["x"]
    if mutation == "missing_trajectory":
        geometry["field_witness"]["trajectory_indices"] = list(range(19))
    elif mutation == "missing_frame":
        field["frame_witnesses"].pop()
    elif mutation == "bad_slab_hash":
        axis["edge_neighbor_slab_sha256"] = "not-a-digest"
    elif mutation == "low_ratio":
        axis["wrap_rms"] = 0.01
        axis["adjacent_rms"] = 1.0
        axis["wrap_to_adjacent_rms_ratio"] = 0.01
    else:
        axis["degenerate_trajectory_count"] = 1
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_seal_receipt(payload), sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert not result["valid"]
    assert any(expected_error in error for error in result["errors"])


def test_geometry_receipt_rejects_a_resealed_below_floor_corpus(
    tmp_path: Path,
) -> None:
    payload = _valid_preflight_receipt()
    payload.pop("content_sha256")
    for item in payload["files"]:
        for field in item["geometry"]["field_witness"]["fields"].values():
            for frame in field["frame_witnesses"]:
                row = frame["axes"]["x"]
                row["wrap_rms"] = 0.1
                row["adjacent_rms"] = 1.0
                row["wrap_to_adjacent_rms_ratio"] = 0.1
                row["wrap_sum_squares"] = 0.1**2 * row["wrap_value_count"]
                row["adjacent_sum_squares"] = float(
                    row["adjacent_value_count"]
                )
                row["wrap_rms_summary"] = {
                    "min": 0.1,
                    "median": 0.1,
                    "max": 0.1,
                }
                row["adjacent_rms_summary"] = {
                    "min": 1.0,
                    "median": 1.0,
                    "max": 1.0,
                }
                row["wrap_to_adjacent_ratio_summary"] = {
                    "min": 0.1,
                    "median": 0.1,
                    "max": 0.1,
                }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(_seal_receipt(payload), sort_keys=True))
    result = validate_external_geometry_preflight_receipt(path, root=LANE)
    assert not result["valid"]
    assert "corpus_geometry_witness:ValueError" in result["errors"]


def test_preflight_receipt_rejects_post_receipt_source_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    root = tmp_path / "lane"
    (root / "scripts").mkdir(parents=True)
    (root / "external_sources").mkdir(parents=True)
    (root / "upstream_geometry_probe").mkdir(parents=True)
    for relative in (
        "EXTERNAL_WELL_MANIFEST.json",
        "external_well.py",
        "geometry_receipt.py",
        "external_sources/GEOMETRY_SOURCE_RECEIPT.json",
        "scripts/preflight_external_geometry.py",
        "scripts/stage_external_geometry_preflight.py",
        "upstream_geometry_probe/PROBE_PROVENANCE.json",
        "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json",
    ):
        destination = root / relative
        destination.write_bytes((LANE / relative).read_bytes())
    payload = _valid_preflight_receipt(root)
    monkeypatch.setattr(
        preflight_stage_module,
        "stage_identity",
        lambda _root: payload["preflight_stage"],
    )
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    assert validate_external_geometry_preflight_receipt(
        path, root=root
    )["valid"]
    with (root / "external_well.py").open("ab") as handle:
        handle.write(b"\n# post-receipt drift\n")
    result = validate_external_geometry_preflight_receipt(path, root=root)
    assert not result["valid"]
    assert "executed_source_sha256" in result["errors"]


def test_freeze_creation_rejects_invalid_geometry_before_source_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(freeze_module, "FREEZE_PATH", tmp_path / "freeze.json")
    monkeypatch.setattr(
        freeze_module,
        "validate_external_geometry_preflight_receipt",
        lambda: {"valid": False, "errors": ["content_sha256"]},
    )
    with pytest.raises(RuntimeError, match="content_sha256"):
        freeze_module.create()
    assert not (tmp_path / "freeze.json").exists()


def test_v1r7_binds_exact_failed_closed_v1r6_provenance() -> None:
    result = validate_upstream_v1r6_provenance()
    assert result["valid"], result
    assert result["provenance"] == V1R6_PROVENANCE


def test_v1r7_rejects_mutated_upstream_v1r6_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lane"
    upstream = root / "upstream_v1r6"
    upstream.mkdir(parents=True)
    freeze_path = upstream / "PREOUTCOME_FREEZE.json"
    authorization_path = upstream / (
        "root_authorization_external_validation_8f91ff35c7e42c1f.json"
    )
    freeze = json.loads((LANE / "upstream_v1r6/PREOUTCOME_FREEZE.json").read_text())
    freeze["source_closure_sha256"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze))
    authorization_path.write_bytes(
        (LANE / "upstream_v1r6/root_authorization_external_validation_8f91ff35c7e42c1f.json").read_bytes()
    )
    result = validate_upstream_v1r6_provenance(root=root)
    assert not result["valid"]
    assert "v1r6_source_closure" in result["errors"]


def test_v1r7_freeze_requires_geometry_source_and_receipt_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_freeze = tmp_path / "PREOUTCOME_FREEZE.json"
    monkeypatch.setattr(freeze_module, "FREEZE_PATH", temporary_freeze)
    monkeypatch.setattr(
        freeze_module,
        "validate_geometry_source_receipt",
        lambda: {"valid": True, "errors": [], "file_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_external_geometry_preflight_receipt",
        lambda: {
            "valid": True,
            "errors": [],
            "file_sha256": "2" * 64,
            "content_sha256": "3" * 64,
        },
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_upstream_v1r6_provenance",
        lambda: {"valid": True, "errors": [], "provenance": V1R6_PROVENANCE},
    )
    monkeypatch.setattr(freeze_module, "manifest", lambda: {"x": "4" * 64})
    monkeypatch.setattr(
        freeze_module, "dependency_manifest", lambda: {"y": "5" * 64}
    )
    created = freeze_module.create()
    assert created["schema_version"] == (
        "maintrack.spotlight-2d-replication.freeze.v1r7"
    )
    assert created["upstream_v1r6_provenance"] == V1R6_PROVENANCE
    assert created["geometry_source_receipt_sha256"] == "1" * 64
    assert created["external_geometry_preflight_sha256"] == "2" * 64


def test_source_allowlist_rejects_added_code_binary_symlink_nonregular_and_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(freeze_module, "SOURCE_FILE_ALLOWLIST", ("kept.py",))
    kept = tmp_path / "kept.py"
    kept.write_text("x = 1\n")
    baseline = freeze_module.manifest(tmp_path)
    assert set(baseline) == {"kept.py"}

    for name in ("added.py", "native.so", "artifact.json"):
        extra = tmp_path / name
        extra.write_text("unmanifested\n")
        with pytest.raises(RuntimeError, match=f"unmanifested:{name}"):
            freeze_module.manifest(tmp_path)
        extra.unlink()

    test_data = tmp_path / "data" / "test"
    test_data.mkdir(parents=True)
    (test_data / "publisher.hdf5").write_text("sealed\n")
    with pytest.raises(RuntimeError, match="unmanifested:data/test"):
        freeze_module.manifest(tmp_path)
    (test_data / "publisher.hdf5").unlink()
    test_data.rmdir()
    test_data.parent.rmdir()

    link = tmp_path / "alias.py"
    link.symlink_to(kept)
    with pytest.raises(RuntimeError, match="symlink:alias.py"):
        freeze_module.manifest(tmp_path)
    link.unlink()

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="nonregular:pipe"):
        freeze_module.manifest(tmp_path)
    fifo.unlink()

    with pytest.raises(ValueError, match="unsafe manifest path"):
        freeze_module._safe_regular_bytes(tmp_path, "../outside.py")

    kept.unlink()
    with pytest.raises(FileNotFoundError):
        freeze_module.manifest(tmp_path)
    kept.write_text("x = 2\n")
    assert freeze_module.manifest(tmp_path) != baseline


def _synthetic_freeze_payload(
    source_manifest: dict[str, str], dependency_manifest: dict[str, str]
) -> dict:
    body = {
        "schema_version": freeze_module.FREEZE_SCHEMA,
        "frozen_at_utc": "2026-08-24T12:00:00+00:00",
        "status": freeze_module.FREEZE_STATUS,
        "source_manifest": source_manifest,
        "source_closure_sha256": freeze_module.content_digest(source_manifest),
        "dependency_manifest": dependency_manifest,
        "dependency_closure_sha256": freeze_module.content_digest(
            dependency_manifest
        ),
        "external_manifest_sha256": "1" * 64,
        "external_geometry_preflight_sha256": "2" * 64,
        "external_geometry_preflight_content_sha256": "3" * 64,
        "geometry_source_receipt_sha256": "4" * 64,
        "upstream_v1r6_provenance": V1R6_PROVENANCE,
        "opened_scientific_splits": freeze_module.OPENED_SCIENTIFIC_SPLITS,
        "sealed_scientific_splits": freeze_module.SEALED_SCIENTIFIC_SPLITS,
        "same_attack_and_guard": True,
        "post_outcome_tuning_permitted": False,
        "claim_ceiling": freeze_module.CLAIM_CEILING,
    }
    return {**body, "freeze_digest": freeze_module.json_digest(body)}


def _reseal_freeze(payload: dict) -> dict:
    result = json.loads(json.dumps(payload))
    result.pop("freeze_digest", None)
    return {**result, "freeze_digest": freeze_module.json_digest(result)}


def _isolate_freeze_verifier(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    sources: dict[str, str],
    dependencies: dict[str, str],
) -> None:
    monkeypatch.setattr(freeze_module, "source_files", lambda: tuple(sources))
    monkeypatch.setattr(freeze_module, "manifest", lambda _root: dict(sources))
    monkeypatch.setattr(
        freeze_module, "DEPENDENCY_RELATIVE_FILES", {name: name for name in dependencies}
    )
    monkeypatch.setattr(
        freeze_module, "dependency_manifest", lambda _repo: dict(dependencies)
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_upstream_v1r6_provenance",
        lambda **_kwargs: {"valid": True, "errors": []},
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_geometry_source_receipt",
        lambda **_kwargs: {"valid": True, "errors": [], "file_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        freeze_module,
        "validate_external_geometry_preflight_receipt",
        lambda *_args, **_kwargs: {
            "valid": True,
            "errors": [],
            "file_sha256": "2" * 64,
            "content_sha256": "3" * 64,
        },
    )
    (root / "EXTERNAL_WELL_MANIFEST.json").write_text("manifest\n")
    monkeypatch.setattr(
        freeze_module,
        "file_sha",
        lambda path: "1" * 64 if path.name == "EXTERNAL_WELL_MANIFEST.json" else "0" * 64,
    )
    (root / "EXTERNAL_GEOMETRY_PREFLIGHT_VALID.json").write_text("{}\n")


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("status", "WRONG", "semantic:status"),
        ("opened_scientific_splits", [], "semantic:opened_scientific_splits"),
        ("sealed_scientific_splits", [], "semantic:sealed_scientific_splits"),
        ("same_attack_and_guard", False, "semantic:same_attack_and_guard"),
        (
            "post_outcome_tuning_permitted",
            True,
            "semantic:post_outcome_tuning_permitted",
        ),
        ("claim_ceiling", "inflated", "semantic:claim_ceiling"),
    ],
)
def test_freeze_verify_enforces_exact_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    sources = {"kept.py": "a" * 64}
    dependencies = {"dep": "b" * 64}
    _isolate_freeze_verifier(monkeypatch, tmp_path, sources, dependencies)
    payload = _synthetic_freeze_payload(sources, dependencies)
    payload[field] = value
    payload = _reseal_freeze(payload)
    path = tmp_path / "PREOUTCOME_FREEZE.json"
    path.write_text(json.dumps(payload))
    result = freeze_module.verify(root=tmp_path, repo=tmp_path)
    assert not result["valid"]
    assert error in result["errors"]


def test_freeze_verify_uses_fresh_exact_manifest_keysets_and_root_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sources = {"kept.py": "a" * 64}
    dependencies = {"dep": "b" * 64}
    _isolate_freeze_verifier(monkeypatch, tmp_path, sources, dependencies)
    clean = _synthetic_freeze_payload(sources, dependencies)
    path = tmp_path / "PREOUTCOME_FREEZE.json"
    path.write_text(json.dumps(clean))
    assert freeze_module.verify(root=tmp_path, repo=tmp_path)["valid"]

    extra_root = dict(clean)
    extra_root["unexpected"] = True
    path.write_text(json.dumps(_reseal_freeze(extra_root)))
    result = freeze_module.verify(root=tmp_path, repo=tmp_path)
    assert "freeze_root_keys" in result["errors"]

    stale = dict(clean)
    stale["source_manifest"] = {"kept.py": "f" * 64, "added.py": "e" * 64}
    stale["source_closure_sha256"] = freeze_module.content_digest(
        stale["source_manifest"]
    )
    path.write_text(json.dumps(_reseal_freeze(stale)))
    result = freeze_module.verify(root=tmp_path, repo=tmp_path)
    assert "source_manifest_keyset" in result["errors"]
    assert "source_manifest" in result["errors"]


def test_freeze_verify_requires_a_utc_timestamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    sources = {"kept.py": "a" * 64}
    dependencies = {"dep": "b" * 64}
    _isolate_freeze_verifier(monkeypatch, tmp_path, sources, dependencies)
    payload = _synthetic_freeze_payload(sources, dependencies)
    payload["frozen_at_utc"] = "2026-08-24T14:00:00+02:00"
    path = tmp_path / "PREOUTCOME_FREEZE.json"
    path.write_text(json.dumps(_reseal_freeze(payload)))
    result = freeze_module.verify(root=tmp_path, repo=tmp_path)
    assert not result["valid"]
    assert "semantic:frozen_at_utc" in result["errors"]


def test_scientific_stage_is_manifest_only_reproducible_and_excludes_generated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    lane = repo / "lane"
    lane.mkdir(parents=True)
    (lane / "kept.py").write_text("x = 1\n")
    (repo / "dep.py").write_text("dependency = 1\n")
    for relative in (
        "added.py",
        "native.so",
        "artifacts/unmanifested.json",
        "governance_preexecution_v1r7_candidate/state.json",
        "__pycache__/cache.pyc",
    ):
        path = lane / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("must not be staged\n")
    sealed_test = repo / "data/test/publisher.hdf5"
    sealed_test.parent.mkdir(parents=True)
    sealed_test.write_text("must not be staged\n")
    sources = {"kept.py": _sha256(lane / "kept.py")}
    dependencies = {"dep": _sha256(repo / "dep.py")}
    monkeypatch.setattr(freeze_module, "source_files", lambda: ("kept.py",))
    monkeypatch.setattr(
        freeze_module, "DEPENDENCY_RELATIVE_FILES", {"dep": "dep.py"}
    )
    monkeypatch.setattr(freeze_module, "LANE_REPO_RELATIVE", PurePosixPath("lane"))
    payload = _synthetic_freeze_payload(sources, dependencies)
    (lane / "PREOUTCOME_FREEZE.json").write_text(json.dumps(payload))

    first, first_manifest = scientific_stage_archive_bytes(repo, payload)
    second, second_manifest = scientific_stage_archive_bytes(repo, payload)
    assert first == second
    assert first_manifest == second_manifest
    archive_path = tmp_path / "stage.tar.gz"
    archive_path.write_bytes(first)
    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
    assert SCIENTIFIC_STAGE_MANIFEST_PATH in names
    assert names == {
        "lane/kept.py",
        "lane/PREOUTCOME_FREEZE.json",
        "dep.py",
        SCIENTIFIC_STAGE_MANIFEST_PATH,
    }
    assert not any(
        token in name
        for name in names
        for token in (
            "governance_preexecution",
            "upstream_archival_governance",
            "__pycache__",
            "/artifacts/",
            "/data/test/",
        )
    )

    (lane / "kept.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="source hash drift"):
        scientific_stage_archive_bytes(repo, payload)
    (lane / "kept.py").unlink()
    with pytest.raises(FileNotFoundError):
        scientific_stage_archive_bytes(repo, payload)

    (lane / "kept.py").symlink_to(repo / "dep.py")
    with pytest.raises(ValueError, match="regular non-symlink"):
        scientific_stage_archive_bytes(repo, payload)


def test_prefreeze_stage_is_deterministic_and_contains_no_test_or_v1r7_freeze(
    tmp_path: Path,
) -> None:
    repo = LANE.parents[3]
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_result = build_stage(repo, first)
    second_result = build_stage(repo, second)
    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    with tarfile.open(first, "r:gz") as archive:
        names = set(archive.getnames())
        stage_manifest = json.load(archive.extractfile(STAGE_MANIFEST_PATH))
    assert set(STAGE_FILES).issubset(names)
    assert STAGE_MANIFEST_PATH in names
    assert stage_manifest["requires_v1r7_freeze"] is False
    assert stage_manifest["fields_accessed"] is True
    assert stage_manifest["outcomes_decoded"] is False
    assert stage_manifest["contains_publisher_test"] is False
    assert all("PREOUTCOME_FREEZE.json" not in name or "upstream_v1r6" in name
               for name in names)


def test_exact_geometry_sources_validate() -> None:
    result = validate_geometry_source_receipt()
    assert result["valid"], result


def test_geometry_source_receipt_binds_exact_exploratory_probe_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lane"
    relative_paths = (
        "external_sources/GEOMETRY_SOURCE_RECEIPT.json",
        "external_sources/the_well_gray_scott_README.md",
        "external_sources/the_well_endpoint_false_notebook.ipynb",
        "external_sources/THE_WELL_LICENSE",
        "external_sources/gen_gs.m",
        "external_sources/GENERATOR_LICENSE",
        "external_sources/chebfun_trigpts.m",
        "external_sources/chebfun_spinop2_getGrid.m",
        "external_sources/chebfun_spinop2_reshapeGrid.m",
        "external_sources/chebfun_spinop2_solvepde.m",
        "external_sources/CHEBFUN_LICENSE.txt",
        "upstream_geometry_probe/grid_probe_valid_bfb8966815d4.py",
        "upstream_geometry_probe/grid_probe_11663182.json",
        "upstream_geometry_probe/PROBE_PROVENANCE.json",
        "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json",
    )
    for relative in relative_paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((LANE / relative).read_bytes())
    receipt = root / "external_sources/GEOMETRY_SOURCE_RECEIPT.json"
    assert validate_geometry_source_receipt(receipt, root=root)["valid"]

    provenance_path = root / "upstream_geometry_probe/PROBE_PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["coverage"]["raw_time_indices"] = [20, 500, 980]
    provenance_path.write_text(json.dumps(provenance, sort_keys=True))
    result = validate_geometry_source_receipt(receipt, root=root)
    assert not result["valid"]
    assert "exploratory_probe_provenance_sha256" in result["errors"]
    assert "exploratory_probe_provenance_content" in result["errors"]
    provenance_path.write_bytes(
        (LANE / "upstream_geometry_probe/PROBE_PROVENANCE.json").read_bytes()
    )

    noncanonical_path = (
        root
        / "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
    )
    noncanonical_path.write_bytes(
        (
            LANE
            / "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
        ).read_bytes()
    )
    noncanonical = json.loads(noncanonical_path.read_text())
    noncanonical["coverage"]["raw_time_indices"] = [20, 500]
    noncanonical_path.write_text(json.dumps(noncanonical, sort_keys=True))
    result = validate_geometry_source_receipt(receipt, root=root)
    assert not result["valid"]
    assert "noncanonical_diagnostic_provenance_sha256" in result["errors"]
    assert "noncanonical_diagnostic_provenance_content" in result["errors"]


def test_v1r7_diff_is_confined_to_declared_geometry_surface() -> None:
    v1r6 = LANE / "upstream_v1r6/source_diff_baseline"
    result = audit_v1r7_diff(v1r6, LANE)
    assert result["valid"], result["errors"]


@pytest.mark.parametrize("count,stride", [(7, 160), (13, 80), (25, 40), (961, 1)])
def test_temporal_views_are_exact_nested_publisher_frames(
    count: int, stride: int
) -> None:
    positions = temporal_positions(count)
    assert len(positions) == count
    assert positions.tolist() == list(range(0, 961, stride))


def test_fft_projection_matches_direct_periodic_quadrature() -> None:
    n = 16
    x = np.arange(n)
    y = np.arange(n)
    x2, y2 = np.meshgrid(x, y, indexing="ij")
    values = (
        0.8 * np.cos(2 * np.pi * (2 * x2 + 3 * y2) / n + 0.31)
        + 0.1 * np.sin(2 * np.pi * (x2 + y2) / n)
    )[None, :, :]
    modes = _low_modes(values, cell_area=4.0 / n**2, chunk_size=1)
    for kind, phase in (("sin", 0.173), ("cos", -0.291)):
        z = 2 * np.pi * (2 * x2 + 3 * y2) / n + phase
        test = np.sin(z) if kind == "sin" else np.cos(z)
        direct = np.sum(values[0] * test) * 4.0 / n**2
        projected = _spatial_projection(modes, kind=kind, kx=2, ky=3)[0]
        assert projected == pytest.approx(direct, abs=1e-12)


def _bundle() -> ProjectionBundle:
    rng = np.random.default_rng(123)
    shape = (961, 6, 6)
    bundle = ProjectionBundle(
        times=np.arange(961, dtype=float) * 10.0,
        A=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        B=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        AB2=rng.normal(size=shape) + 1j * rng.normal(size=shape),
        length_x=2.0,
        length_y=2.0,
        source_digest="b" * 64,
        spatial_nodes=CANONICAL_SPATIAL_RESOLUTION,
    )
    bundle.validate()
    return bundle


@pytest.mark.parametrize("modes,rows", [(2, 32), (3, 90), (4, 192), (5, 350)])
def test_external_row_counts_match_original_mode_bank(modes: int, rows: int) -> None:
    design, target, roles = _full_design(
        _bundle(), target="A", temporal_samples=7, test_mode_count=modes,
        F=0.098, k=0.057
    )
    assert design.shape == (rows, 3)
    assert target.shape == (rows,)
    assert len(roles) == rows
    assert set(roles) == {"train", "validate"}


def test_manufactured_diffusion_mode_satisfies_weak_identity() -> None:
    times = np.linspace(0.0, 9600.0, 961)
    shape = (961, 6, 6)
    A = np.zeros(shape, dtype=np.complex128)
    physical_k2 = np.pi**2
    A[:, 1, 0] = np.exp(-DELTA_A * physical_k2 * times)
    bundle = ProjectionBundle(
        times=times,
        A=A,
        B=np.zeros(shape, dtype=np.complex128),
        AB2=np.zeros(shape, dtype=np.complex128),
        length_x=2.0,
        length_y=2.0,
        source_digest="f" * 64,
        spatial_nodes=128,
    )
    design, target, _ = _full_design(
        bundle,
        target="A",
        temporal_samples=961,
        test_mode_count=2,
        F=0.0,
        k=0.0,
    )
    mask = np.abs(design[:, 0]) > 1.0e-12
    assert np.count_nonzero(mask) > 0
    np.testing.assert_allclose(
        target[mask], design[mask, 0], rtol=2.0e-5, atol=1.0e-12
    )


@pytest.mark.parametrize("target", ["A", "B"])
def test_each_omission_removes_exactly_one_whole_scaled_mechanism(
    target: str,
) -> None:
    bundle = _bundle()
    full_components = _full_design(
        bundle, target=target, temporal_samples=7, test_mode_count=2,
        F=0.098, k=0.057
    )
    full = build_external_evidence(
        bundle, target=target, contract_id="full", episode_id="episode",
        view_id="A00", temporal_samples=7, test_mode_count=2,
        F=0.098, k=0.057, registered_cost=1.0,
        full_components=full_components,
    )
    assert full.atoms == atoms_for_target(target)
    for contract_id, columns in CONTRACT_COLUMNS.items():
        evidence = build_external_evidence(
            bundle, target=target, contract_id=contract_id,
            episode_id="episode", view_id="A00", temporal_samples=7,
            test_mode_count=2, F=0.098, k=0.057, registered_cost=1.0,
            full_components=full_components,
        )
        assert evidence.atoms == tuple(full.atoms[index] for index in columns)
        np.testing.assert_array_equal(evidence.design, full.design[:, columns])


def test_shared_noise_is_deterministic_and_view_independent() -> None:
    trajectory = _trajectory()
    first = observed_external_arrays(
        trajectory, noise_model="correlated", noise_level=0.005
    )
    second = observed_external_arrays(
        trajectory, noise_model="correlated", noise_level=0.005
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert not np.array_equal(first[0], trajectory.A)
    assert ("correlated", 0.0) not in NOISE_CELLS


class _FakeDataset:
    def __init__(self, values, **attrs):
        self.values = np.asarray(values)
        self.attrs = attrs

    def __getitem__(self, item):
        return self.values[item]

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype


class _FakeGroup(dict):
    def __init__(self, *args, attrs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {} if attrs is None else attrs


class _FakeWindowFieldDataset:
    """Return a 961-frame broadcast view for the reader's frozen window."""

    def __init__(self, frame: np.ndarray):
        self.frame = np.asarray(frame, dtype=np.float32)

    def __getitem__(self, item):
        trajectory, time_slice, x_slice, y_slice = item
        assert trajectory == 0
        assert time_slice == slice(20, 981)
        assert x_slice == slice(None)
        assert y_slice == slice(None)
        return np.broadcast_to(
            self.frame,
            (961, SPATIAL_RESOLUTION, SPATIAL_RESOLUTION),
        )


class _FakeGeometryFieldDataset:
    """Synthesize only the edge slabs requested by the geometry preflight."""

    shape = (20, 1001, 128, 128)
    dtype = np.dtype(np.float32)

    def __init__(
        self,
        phase: float,
        *,
        duplicate_edges: bool = False,
        constant: bool = False,
        axis_invariant: str | None = None,
        symmetric_trajectories: int = 0,
    ):
        self.phase = phase
        self.duplicate_edges = duplicate_edges
        self.constant = constant
        self.axis_invariant = axis_invariant
        self.symmetric_trajectories = symmetric_trajectories

    def __getitem__(self, item):
        trajectories, frame, x_index, y_index = item
        assert trajectories == slice(None)
        assert frame in GEOMETRY_WITNESS_FRAME_INDICES
        r = np.arange(20, dtype=np.float64)[:, None]
        if self.constant:
            return np.broadcast_to(
                0.1 * r + 0.001 * frame,
                (20, 128),
            ).astype(np.float32)
        if isinstance(x_index, int):
            selected_last = x_index % 128 == 127
            index = x_index % 128
            if self.duplicate_edges and index == 127:
                index = 0
            j = np.arange(128, dtype=np.float64)[None, :]
            i = np.full_like(j, float(index))
            if self.axis_invariant == "x":
                i = np.zeros_like(i)
        else:
            assert x_index == slice(None) and isinstance(y_index, int)
            selected_last = y_index % 128 == 127
            index = y_index % 128
            if self.duplicate_edges and index == 127:
                index = 0
            i = np.arange(128, dtype=np.float64)[None, :]
            j = np.full_like(i, float(index))
            if self.axis_invariant == "y":
                j = np.zeros_like(j)
        values = (
            0.1 * r
            + 0.001 * frame
            + np.sin(2.0 * np.pi * (i + r / 20.0 + self.phase) / 128.0)
            + 0.7 * np.cos(2.0 * np.pi * (j - r / 30.0 + self.phase) / 128.0)
        )
        if self.symmetric_trajectories and selected_last:
            if isinstance(x_index, int):
                reference_i = np.zeros_like(i)
                reference_j = j
            else:
                reference_i = i
                reference_j = np.zeros_like(j)
            reference = (
                0.1 * r
                + 0.001 * frame
                + np.sin(
                    2.0
                    * np.pi
                    * (reference_i + r / 20.0 + self.phase)
                    / 128.0
                )
                + 0.7
                * np.cos(
                    2.0
                    * np.pi
                    * (reference_j - r / 30.0 + self.phase)
                    / 128.0
                )
            )
            values[: self.symmetric_trajectories] = reference[
                : self.symmetric_trajectories
            ]
        return values.astype(np.float32)


def _publisher_metadata_handle(
    *, x: np.ndarray | None = None, y: np.ndarray | None = None,
) -> _FakeGroup:
    axis = np.linspace(-1.0, 1.0, SPATIAL_RESOLUTION, dtype=np.float32)
    handle = _FakeGroup(attrs={
        "n_trajectories": 20,
        "n_spatial_dims": 2,
        "dataset_name": "gray_scott_reaction_diffusion",
    })
    handle["dimensions"] = _FakeGroup({
        "x": _FakeDataset(axis if x is None else x),
        "y": _FakeDataset(axis if y is None else y),
    }, attrs={"spatial_dims": ["x", "y"]})
    handle["boundary_conditions"] = _FakeGroup({
        f"{name}_periodic": _FakeGroup(attrs={
            "associated_dims": [name],
            "associated_fields": [],
            "bc_type": "PERIODIC",
            "sample_varying": False,
            "time_varying": False,
        })
        for name in ("x", "y")
    })
    return handle


def _publisher_reader_handle() -> tuple[
    _FakeGroup, np.ndarray, np.ndarray,
]:
    handle = _publisher_metadata_handle()
    handle["dimensions"]["time"] = _FakeDataset(
        np.arange(1001, dtype=np.float32) * np.float32(10.0),
        sample_varying=False,
    )
    i = np.arange(CANONICAL_SPATIAL_RESOLUTION, dtype=np.float32)[:, None]
    j = np.arange(CANONICAL_SPATIAL_RESOLUTION, dtype=np.float32)[None, :]
    unique_A = i * np.float32(1000.0) + j
    unique_B = unique_A + np.float32(0.5)
    handle["t0_fields"] = _FakeGroup({
        "A": _FakeWindowFieldDataset(unique_A),
        "B": _FakeWindowFieldDataset(unique_B),
    }, attrs={"field_names": ["A", "B"]})
    handle["scalars"] = _FakeGroup({
        "F": _FakeDataset(np.asarray(0.098), sample_varying=False),
        "k": _FakeDataset(np.asarray(0.057), sample_varying=False),
    })
    return handle, unique_A, unique_B


def test_fake_well_schema_reader_binds_fields_coordinates_and_parameters() -> None:
    n = 12
    field_A = np.zeros((1, 1001, n, n), dtype=np.float32)
    field_B = np.ones((1, 1001, n, n), dtype=np.float32)
    handle = _FakeGroup(attrs={
        "n_trajectories": 20,
        "n_spatial_dims": 2,
        "dataset_name": "gray_scott_reaction_diffusion",
    })
    handle["dimensions"] = _FakeGroup({
        "x": _FakeDataset(np.linspace(-1.0, 1.0, n, endpoint=False)),
        "y": _FakeDataset(np.linspace(-1.0, 1.0, n, endpoint=False)),
        "time": _FakeDataset(np.arange(1001, dtype=float) * 10.0,
                             sample_varying=False),
    }, attrs={"spatial_dims": ["x", "y"]})
    handle["t0_fields"] = _FakeGroup({
        "A": _FakeDataset(field_A),
        "B": _FakeDataset(field_B),
    }, attrs={"field_names": ["A", "B"]})
    handle["scalars"] = _FakeGroup({
        "F": _FakeDataset(np.asarray(0.098), sample_varying=False),
        "k": _FakeDataset(np.asarray(0.057), sample_varying=False),
    })
    trajectory = read_open_well_trajectory(
        handle,
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        source_file_sha256="c" * 64,
        require_publisher_shape=False,
    )
    assert trajectory.A.shape == (961, n, n)
    assert trajectory.B.shape == (961, n, n)
    assert trajectory.times[[0, -1]].tolist() == [200.0, 9800.0]
    assert trajectory.F == pytest.approx(0.098)
    assert trajectory.k == pytest.approx(0.057)


def test_full_publisher_reader_preserves_all_samples_without_transpose() -> None:
    handle, unique_A, unique_B = _publisher_reader_handle()
    trajectory = read_open_well_trajectory(
        handle,
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        source_file_sha256="c" * 64,
        require_publisher_shape=True,
    )

    assert trajectory.A.shape == (
        961,
        CANONICAL_SPATIAL_RESOLUTION,
        CANONICAL_SPATIAL_RESOLUTION,
    )
    assert trajectory.B.shape == trajectory.A.shape
    np.testing.assert_array_equal(trajectory.x, _canonical_publisher_axis())
    np.testing.assert_array_equal(trajectory.y, _canonical_publisher_axis())
    np.testing.assert_array_equal(trajectory.A[0], unique_A)
    np.testing.assert_array_equal(trajectory.B[0], unique_B)
    assert trajectory.A[0, 17, 83] == unique_A[17, 83]
    assert trajectory.A[0, 17, 83] != unique_A[83, 17]
    assert trajectory.times[[0, -1]].tolist() == [200.0, 9800.0]


def test_exact_publisher_axis_maps_to_unique_periodic_lattice() -> None:
    raw = np.linspace(-1.0, 1.0, SPATIAL_RESOLUTION, dtype=np.float32)
    differences = np.diff(raw)
    assert len(np.unique(differences)) == 6
    canonical, receipt = _publisher_axis(raw, "x")
    np.testing.assert_array_equal(canonical, _canonical_publisher_axis())
    assert canonical.shape == (CANONICAL_SPATIAL_RESOLUTION,)
    assert canonical[0] == -1.0
    assert canonical[-1] == pytest.approx(1.0 - 2.0 / 128.0)
    assert receipt["raw_sha256"] == PUBLISHER_COORDINATE_SHA256
    assert receipt["raw_used_as_solver_coordinates"] is False
    assert _periodic_length(canonical) == PUBLISHER_PERIOD


def _axis_mutations() -> list[tuple[str, np.ndarray]]:
    exact = np.linspace(-1.0, 1.0, SPATIAL_RESOLUTION, dtype=np.float32)
    one_ulp = exact.copy()
    one_ulp[63] = np.nextafter(one_ulp[63], np.float32(np.inf))
    warped = exact.copy()
    warped[63] += np.float32(2.0e-5)
    shifted = exact.copy()
    shifted[0] = np.nextafter(shifted[0], np.float32(np.inf))
    return [
        ("float64", exact.astype(np.float64)),
        ("endpoint-exclusive", np.linspace(
            -1.0, 1.0, SPATIAL_RESOLUTION, endpoint=False, dtype=np.float32
        )),
        ("127 points", exact[:-1]),
        ("129 points", np.linspace(-1.0, 1.0, 129, dtype=np.float32)),
        ("reversed", exact[::-1]),
        ("one ulp", one_ulp),
        ("interior warp", warped),
        ("shifted endpoint", shifted),
    ]


@pytest.mark.parametrize("label,coordinates", _axis_mutations())
def test_exact_publisher_axis_rejects_every_geometry_mutation(
    label: str, coordinates: np.ndarray,
) -> None:
    del label
    with pytest.raises(ValueError, match="PUBLISHER_COORDINATE_CONTRACT_INVALID"):
        _publisher_axis(coordinates, "x")


def test_publisher_metadata_preflight_never_requires_field_group() -> None:
    receipt = read_publisher_grid_metadata(_publisher_metadata_handle())
    assert receipt["fields_accessed"] is False
    assert receipt["outcomes_decoded"] is False
    assert set(receipt["axes"]) == {"x", "y"}
    assert set(receipt["boundary_conditions"]) == {"x", "y"}


def _publisher_geometry_handle(
    *,
    duplicate_edges: bool = False,
    constant: bool = False,
    axis_invariant: str | None = None,
    symmetric_trajectories: int = 0,
) -> _FakeGroup:
    handle = _publisher_metadata_handle()
    handle["t0_fields"] = _FakeGroup({
        "A": _FakeGeometryFieldDataset(
            0.0,
            duplicate_edges=duplicate_edges,
            constant=constant,
            axis_invariant=axis_invariant,
            symmetric_trajectories=symmetric_trajectories,
        ),
        "B": _FakeGeometryFieldDataset(
            0.37,
            duplicate_edges=duplicate_edges,
            constant=constant,
            axis_invariant=axis_invariant,
            symmetric_trajectories=symmetric_trajectories,
        ),
    }, attrs={"field_names": ["A", "B"]})
    return handle


def test_geometry_witness_covers_all_valid_trajectories_frames_fields_axes() -> None:
    receipt = read_publisher_geometry_witness(_publisher_geometry_handle())
    assert receipt["fields_accessed"] is True
    assert receipt["outcomes_decoded"] is False
    witness = receipt["field_witness"]
    assert witness["trajectory_indices"] == list(range(20))
    assert witness["raw_frame_indices"] == [20, 500, 980]
    for field_name in ("A", "B"):
        field = witness["fields"][field_name]
        assert field["dataset_shape"] == [20, 1001, 128, 128]
        assert [row["raw_frame_index"] for row in field["frame_witnesses"]] == [
            20,
            500,
            980,
        ]
        for frame in field["frame_witnesses"]:
            for axis in ("x", "y"):
                row = frame["axes"][axis]
                assert row["edge_neighbor_slab_shape"] == [20, 4, 128]
                assert len(row["edge_neighbor_slab_sha256"]) == 64
                assert row["opposing_edges_identical"] is False
                assert row["wrap_to_adjacent_rms_ratio"] is not None
                assert row["trajectory_count"] == 20
                assert row["opposing_edges_identical_trajectory_count"] == 0
                assert row["wrap_to_adjacent_ratio_summary"]["min"] >= 0.0


@pytest.mark.parametrize(
    "kind,axis",
    [("constant", None), ("axis_invariant", "x"), ("axis_invariant", "y")],
)
def test_geometry_reader_accepts_degenerate_noninformative_cells(
    kind: str, axis: str | None,
) -> None:
    receipt = read_publisher_geometry_witness(
        _publisher_geometry_handle(
            constant=kind == "constant",
            axis_invariant=axis if kind == "axis_invariant" else None,
        )
    )
    rows = [
        frame["axes"][axis_name]
        for field in receipt["field_witness"]["fields"].values()
        for frame in field["frame_witnesses"]
        for axis_name in ("x", "y")
    ]
    assert any(row["cell_status"] == "degenerate_noninformative" for row in rows)
    assert all(row["ratio_floor_applied_at"] == "corpus_axis_pool_only" for row in rows)


def test_corpus_geometry_gate_tolerates_degenerate_cells_when_pool_is_informative(
) -> None:
    files = [
        {"geometry": read_publisher_geometry_witness(_publisher_geometry_handle())},
        {
            "geometry": read_publisher_geometry_witness(
                _publisher_geometry_handle(constant=True)
            )
        },
        {
            "geometry": read_publisher_geometry_witness(
                _publisher_geometry_handle(axis_invariant="x")
            )
        },
    ]
    summary = summarize_geometry_corpus(files)
    assert summary["passed"] is True
    assert summary["overall"]["degenerate_noninformative_cell_count"] > 0


def test_corpus_geometry_gate_tolerates_locally_symmetric_seam_cells() -> None:
    symmetric = read_publisher_geometry_witness(
        _publisher_geometry_handle(symmetric_trajectories=5)
    )
    assert any(
        row["opposing_edges_identical_trajectory_count"] == 5
        and row["degenerate_trajectory_count"] == 5
        and row["informative_trajectory_count"] == 15
        for field in symmetric["field_witness"]["fields"].values()
        for frame in field["frame_witnesses"]
        for row in frame["axes"].values()
    )
    summary = summarize_geometry_corpus([
        {"geometry": symmetric},
        {"geometry": read_publisher_geometry_witness(_publisher_geometry_handle())},
    ])
    assert summary["passed"] is True
    assert summary["overall"]["opposing_edges_identical_cell_count"] > 0


def test_corpus_geometry_gate_rejects_true_duplicated_endpoint_faces() -> None:
    files = [
        {
            "geometry": read_publisher_geometry_witness(
                _publisher_geometry_handle(duplicate_edges=True)
            )
        }
        for _ in range(6)
    ]
    with pytest.raises(
        ValueError,
        match="PUBLISHER_GEOMETRY_CORPUS_WITNESS_INVALID",
    ):
        summarize_geometry_corpus(files)


@pytest.mark.parametrize(
    "mutation", ["type", "dims", "fields", "sample", "missing"]
)
def test_periodic_boundary_contract_rejects_mutations(mutation: str) -> None:
    handle = _publisher_metadata_handle()
    if mutation == "type":
        handle["boundary_conditions"]["x_periodic"].attrs["bc_type"] = "OPEN"
    elif mutation == "dims":
        handle["boundary_conditions"]["x_periodic"].attrs["associated_dims"] = ["y"]
    elif mutation == "fields":
        handle["boundary_conditions"]["x_periodic"].attrs[
            "associated_fields"
        ] = ["A"]
    elif mutation == "sample":
        handle["boundary_conditions"]["x_periodic"].attrs["sample_varying"] = True
    else:
        del handle["boundary_conditions"]["x_periodic"]
    with pytest.raises(ValueError, match="PUBLISHER_COORDINATE_CONTRACT_INVALID"):
        _validate_periodic_boundary_metadata(handle)


def _unique_periodic_field(frames: int = 2) -> np.ndarray:
    i = np.arange(CANONICAL_SPATIAL_RESOLUTION, dtype=np.float32)[None, :, None]
    j = np.arange(CANONICAL_SPATIAL_RESOLUTION, dtype=np.float32)[None, None, :]
    t = np.arange(frames, dtype=np.float32)[:, None, None]
    return np.asarray(t * 10000.0 + i * 100.0 + j, dtype=np.float32)


def test_all_samples_and_sentinels_are_preserved_without_transpose() -> None:
    raw_A = _unique_periodic_field()
    raw_B = raw_A + np.float32(0.25)
    x, y, A, B = _canonicalize_publisher_fields(raw_A, raw_B)
    np.testing.assert_array_equal(x, _canonical_publisher_axis())
    np.testing.assert_array_equal(y, _canonical_publisher_axis())
    np.testing.assert_array_equal(A, raw_A)
    np.testing.assert_array_equal(B, raw_B)
    assert A.flags.c_contiguous and B.flags.c_contiguous
    assert A[1, 17, 83] == raw_A[1, 17, 83]
    assert A[1, 17, 83] != raw_A[1, 83, 17]
    assert A[1, -1, -1] == raw_A[1, -1, -1]
    assert A.shape == raw_A.shape


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite"])
def test_publisher_field_contract_rejects_structural_mutations(
    mutation: str,
) -> None:
    raw_A = _unique_periodic_field()
    raw_B = _unique_periodic_field()
    if mutation == "shape":
        raw_A = raw_A[:, :-1, :]
    elif mutation == "dtype":
        raw_A = raw_A.astype(np.float64)
    else:
        raw_A[:, 10, 10] = np.nan
    with pytest.raises(
        ValueError, match="PUBLISHER_COORDINATE_FIELD_CONTRACT_INVALID"
    ):
        _canonicalize_publisher_fields(raw_A, raw_B)


def test_canonical_constant_quadrature_has_area_four_and_zero_non_dc() -> None:
    n = CANONICAL_SPATIAL_RESOLUTION
    values = np.ones((1, n, n), dtype=np.float32)
    modes = _low_modes(values, cell_area=PUBLISHER_PERIOD**2 / n**2)
    assert modes[0, 0, 0] == pytest.approx(4.0, abs=1.0e-15)
    non_dc = modes.copy()
    non_dc[0, 0, 0] = 0.0
    assert np.max(np.abs(non_dc)) < 1.0e-13


@pytest.mark.parametrize("kx,ky", [(1, 0), (0, 1), (2, 3), (5, 4)])
def test_canonical_modes_through_five_match_direct_quadrature(
    kx: int, ky: int,
) -> None:
    n = CANONICAL_SPATIAL_RESOLUTION
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    values = np.cos(2 * np.pi * (kx * i + ky * j) / n + 0.31)[None]
    modes = _low_modes(values, cell_area=PUBLISHER_PERIOD**2 / n**2)
    for kind, phase in (("sin", 0.173), ("cos", -0.291)):
        angle = 2 * np.pi * (kx * i + ky * j) / n + phase
        test = np.sin(angle) if kind == "sin" else np.cos(angle)
        direct = np.sum(values[0] * test) * PUBLISHER_PERIOD**2 / n**2
        projected = _spatial_projection(
            modes, kind=kind, kx=kx, ky=ky
        )[0]
        assert projected == pytest.approx(direct, abs=2.0e-12)


def test_period_two_wavenumber_and_raw_acquisition_cost_are_separate() -> None:
    assert _physical_wavenumber_squared(
        1, 0, length_x=2.0, length_y=2.0
    ) == pytest.approx(np.pi**2, abs=1.0e-14)
    view = VIEW_BY_CONFIG[(7, 2)]
    assert external_registered_cost(view) == (
        SPATIAL_RESOLUTION**2 * 7
        + 0.1 * 2 * SPATIAL_RESOLUTION**2
    )
    assert external_reference_cost() == (
        SPATIAL_RESOLUTION**2 * 961
        + 0.1 * 5 * SPATIAL_RESOLUTION**2
    )
    assert external_registered_cost(view) == external_registered_cost(
        view, CANONICAL_SPATIAL_RESOLUTION
    )
    contract = external_spatial_grid_contract()
    assert contract["publisher"]["nodes_per_axis"] == 128
    assert contract["canonical_quadrature"]["nodes_per_axis"] == 128
    assert contract["canonical_quadrature"]["drop_samples"] is False


def test_projection_bundle_uses_period_two_and_128_squared_cell_area() -> None:
    n = CANONICAL_SPATIAL_RESOLUTION
    times = np.arange(961, dtype=float) * 10.0
    axis = _canonical_publisher_axis()
    constant = np.ones((961, n, n), dtype=np.float32)
    trajectory = ExternalTrajectory(
        split="valid",
        regime="bubbles",
        trajectory_index=0,
        times=times,
        x=axis,
        y=axis,
        A=constant,
        B=constant,
        F=0.098,
        k=0.057,
        source_file_sha256="a" * 64,
    )
    bundle = build_projection_bundle(
        trajectory, trajectory.A, trajectory.B, chunk_size=128
    )
    assert bundle.spatial_nodes == CANONICAL_SPATIAL_RESOLUTION
    assert bundle.length_x == PUBLISHER_PERIOD
    assert bundle.length_y == PUBLISHER_PERIOD
    assert bundle.A[0, 0, 0] == pytest.approx(4.0, abs=1.0e-14)


def test_invalid_coordinate_fails_before_field_group_access() -> None:
    coordinates = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    coordinates[63] += np.float32(2.0e-5)
    handle = _FakeGroup(attrs={
        "n_trajectories": 20,
        "n_spatial_dims": 2,
        "dataset_name": "gray_scott_reaction_diffusion",
    })
    handle["dimensions"] = _FakeGroup({
        "x": _FakeDataset(coordinates),
        "y": _FakeDataset(np.linspace(-1.0, 1.0, 128, dtype=np.float32)),
        "time": _FakeDataset(np.arange(1001, dtype=float) * 10.0,
                             sample_varying=False),
    }, attrs={"spatial_dims": ["x", "y"]})
    with pytest.raises(ValueError, match="PUBLISHER_COORDINATE_CONTRACT_INVALID"):
        read_open_well_trajectory(
            handle,
            split="valid",
            regime="bubbles",
            trajectory_index=0,
            source_file_sha256="c" * 64,
        )


def test_test_split_requires_validation_aggregate_binding() -> None:
    digest = "d" * 64
    stage, bindings = authorization_bindings(
        split="valid", internal_confirmation_aggregate_sha256=digest
    )
    assert stage == "external_validation"
    assert bindings == {"internal_confirmation_aggregate_sha256": digest}
    with pytest.raises(ValueError, match="validation aggregate"):
        authorization_bindings(
            split="test", internal_confirmation_aggregate_sha256=digest
        )
    stage, bindings = authorization_bindings(
        split="test",
        internal_confirmation_aggregate_sha256=digest,
        external_validation_aggregate_sha256="e" * 64,
    )
    assert stage == "external_confirmation"
    assert set(bindings) == {
        "internal_confirmation_aggregate_sha256",
        "external_validation_aggregate_sha256",
    }


def _admission_rows() -> list[dict]:
    rows = []
    for regime in EXPECTED_REGIMES:
        for trajectory_index in range(20):
            for target in ("A", "B"):
                rows.append({
                    "regime": regime,
                    "trajectory_index": trajectory_index,
                    "target": target,
                    "library_contract": "full",
                    "noise_model": "iid",
                    "noise_level": 0.0,
                    "reference_correct": True,
                    "reference_false_promotion": False,
                    "proposals": {
                        "fixed_25_m5": {
                            "false_promotion": False,
                            "inconclusive": False,
                        }
                    },
                })
    return rows


def test_authority_admission_is_frozen_and_fail_closed() -> None:
    rows = _admission_rows()
    result = authority_admission(rows)
    assert result["status"] == "ADMITTED_FOR_SEALED_TEST"
    assert result["n"] == 240
    rows[0] = {**rows[0], "reference_correct": False,
               "reference_false_promotion": True}
    result = authority_admission(rows)
    assert result["status"] == "AUTHORITY_NOT_VALID_ON_EXTERNAL_SOURCE"
    assert not result["checks"]["reference_false_promotion"]


def test_reference_id_is_not_in_selectable_bank() -> None:
    assert EXTERNAL_REFERENCE_VIEW_ID not in {
        VIEW_BY_CONFIG[(nt, modes)].view_id
        for nt in (7, 13, 25)
        for modes in (2, 3, 4, 5)
    }


def test_each_public_row_serializes_both_frozen_score_variants() -> None:
    view = VIEW_BY_CONFIG[(7, 2)]
    record = AuthorityRecord(
        episode_id="episode",
        view_id=view.view_id,
        origin="external",
        status="OUTSIDE_LIBRARY",
        support=(),
        coefficients=(),
        residual=0.25,
        score_gap=0.01,
        condition=5.0,
        minimum_coefficient=1.0,
        ambiguity_size=1,
        registered_cost=1.0,
        evidence_digest="f" * 64,
    )
    payload = _record_payload(record, include_objective=True)
    assert payload["public_objective"] == public_objective(record)
    assert payload["adapted_public_objective"] == adapted_public_objective(record)


@pytest.mark.parametrize(
    "postfreeze_staged",
    [False, True],
    ids=("prefreeze", "postfreeze-staged"),
)
def test_external_validator_accepts_exact_grid_contract_and_raw_costs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postfreeze_staged: bool,
) -> None:
    _isolate_external_validator_governance(
        monkeypatch,
        tmp_path,
        postfreeze_staged=postfreeze_staged,
    )
    monkeypatch.setattr(
        validate_external_module,
        "_expected_cells",
        lambda: [(0, "iid", 0.0, "A", "full")],
    )
    result = validate_external_payload(
        _single_episode_external_payload(),
        manifest_path=LANE / "EXTERNAL_WELL_MANIFEST.json",
    )
    assert result["valid"], result


def test_active_postfreeze_staged_validator_binding_if_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze_path = LANE / "PREOUTCOME_FREEZE.json"
    if not freeze_path.is_file():
        pytest.skip("active v1r7 freeze is intentionally absent pre-freeze")
    freeze = json.loads(freeze_path.read_text())
    candidates = sorted(LANE.glob(
        "root_authorization_external_validation_*.json"
    ))
    if not candidates:
        pytest.skip("v1r7 external-validation authorization not yet published")
    authorization_path = candidates[-1]
    authorization = json.loads(authorization_path.read_text())
    binding = authorization["input_bindings"][
        "internal_confirmation_aggregate_sha256"
    ]
    payload = _single_episode_external_payload(
        source_closure=freeze["source_closure_sha256"],
        authorization_digest=authorization["authorization_digest"],
        internal_binding=binding,
    )
    monkeypatch.setattr(
        validate_external_module,
        "_expected_cells",
        lambda: [(0, "iid", 0.0, "A", "full")],
    )
    result = validate_external_payload(
        payload,
        manifest_path=LANE / "EXTERNAL_WELL_MANIFEST.json",
    )
    assert result["valid"], result


def test_external_validator_rejects_tampered_spatial_grid_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_external_validator_governance(
        monkeypatch,
        tmp_path,
        postfreeze_staged=False,
    )
    monkeypatch.setattr(
        validate_external_module,
        "_expected_cells",
        lambda: [(0, "iid", 0.0, "A", "full")],
    )
    payload = _single_episode_external_payload()
    payload["metadata"]["spatial_grid_contract"]["canonical_quadrature"][
        "nodes_per_axis"
    ] = 126
    payload["content_sha256"] = canonical_digest(payload)
    result = validate_external_payload(
        payload, manifest_path=LANE / "EXTERNAL_WELL_MANIFEST.json"
    )
    assert not result["valid"]
    assert "spatial_grid_contract" in result["errors"]
    assert "source_closure_binding" not in result["errors"]
    assert not any(
        error.startswith("stage_authorization") for error in result["errors"]
    )


def test_external_validator_rejects_serialized_127_based_costs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_external_validator_governance(
        monkeypatch,
        tmp_path,
        postfreeze_staged=False,
    )
    monkeypatch.setattr(
        validate_external_module,
        "_expected_cells",
        lambda: [(0, "iid", 0.0, "A", "full")],
    )
    result = validate_external_payload(
        _single_episode_external_payload(cost_nodes=127),
        manifest_path=LANE / "EXTERNAL_WELL_MANIFEST.json",
    )
    assert not result["valid"]
    assert any(error.startswith("registered_cost:0:") for error in result["errors"])
    assert "reference_registered_cost:0" in result["errors"]
    assert "source_closure_binding" not in result["errors"]
    assert not any(
        error.startswith("stage_authorization") for error in result["errors"]
    )


def test_shard_builder_validates_authorization_before_source_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []

    def reject(*args, **kwargs):
        events.append("authorization")
        raise RuntimeError("stop before source access")

    def manifest_access(*args, **kwargs):
        events.append("manifest")
        raise AssertionError("source metadata accessed before authorization")

    monkeypatch.setattr(external_module, "validate_authorization", reject)
    monkeypatch.setattr(external_module, "load_manifest", manifest_access)
    with pytest.raises(RuntimeError, match="stop before source access"):
        build_external_shard(
            split="test",
            regime="bubbles",
            data_path=tmp_path / "sealed.hdf5",
            manifest_path=tmp_path / "manifest.json",
            source_closure="a" * 64,
            authorization_path=tmp_path / "authorization.json",
            authorization_stage="external_confirmation",
            authorization_input_bindings={
                "internal_confirmation_aggregate_sha256": "b" * 64,
                "external_validation_aggregate_sha256": "c" * 64,
            },
        )
    assert events == ["authorization"]
