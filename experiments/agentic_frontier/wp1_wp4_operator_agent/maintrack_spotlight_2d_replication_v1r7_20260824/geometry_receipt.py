"""Fail-closed validators for the v1r7 geometry sources and VALID receipt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parent
EXTERNAL_GEOMETRY_PREFLIGHT_PATH = (
    ROOT / "EXTERNAL_GEOMETRY_PREFLIGHT_VALID.json"
)
EXTERNAL_GEOMETRY_PREFLIGHT_SCHEMA = (
    "maintrack.external-well-geometry-preflight.valid.v1r7"
)
GEOMETRY_SOURCE_RECEIPT_PATH = (
    ROOT / "external_sources/GEOMETRY_SOURCE_RECEIPT.json"
)
GEOMETRY_SOURCE_RECEIPT_SHA256 = (
    "a03e95cb86392134e7889e12d13456d8e022bf1743d1ab878b2887c4473765b8"
)
EXPECTED_REGIMES = ("bubbles", "gliders", "maze", "spirals", "spots", "worms")
PUBLISHER_COORDINATE_SHA256 = (
    "798b0189563db91b33b49e69f5407fb6845506aed58bfcc0d36bc819b3556492"
)
INTERNAL_CONFIRMATION_AGGREGATE = (
    "929a69f0619eb5e9f5dc0993706988fb5ee262b3eb11ef1f02474eae627ec166"
)
V1R6_PROVENANCE: dict[str, Any] = {
    "v1r6_freeze_digest": (
        "9caa529977f33f8d5afac609944a96903b886c1b2331493c725f974b3937f239"
    ),
    "v1r6_source_closure_sha256": (
        "8e9b65b938f6b98b0aad11fdc118b4e0b4bd0ef4f39069a726220f3306915c20"
    ),
    "v1r6_external_validation_authorization_sha256": (
        "8f91ff35c7e42c1f56b9e6a6de3b89068ac03bc387cf95457d1dade017279a19"
    ),
    "internal_confirmation_aggregate_sha256": INTERNAL_CONFIRMATION_AGGREGATE,
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
EXPLORATORY_PROBE: dict[str, Any] = {
    "job_id": "11663182",
    "script_sha256": (
        "bfb8966815d4cf159238dfd77c865c7e5e7eca513f322ba47ee1075f0b0a7c98"
    ),
    "output_sha256": (
        "b18a803ef2f64b3fd938facc64b8652f310e5dfe8e0a91af5ff707b3240f3af9"
    ),
    "stderr_sha256": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "provenance_sha256": (
        "2f47b8dfcf297fc02ad320095ea08781d8bc7f8903054f7269d686d94c58a621"
    ),
}
NONCANONICAL_DIAGNOSTIC: dict[str, Any] = {
    "successful_call_id": "call_fuLA4VZfL7zj1gBuQcvOWpSy",
    "successful_tool_input_sha256": (
        "a1dbd3055abcf78017346de911f7f5c132aa49c2a790fb3d74fc750a743de8a9"
    ),
    "successful_raw_stdout_text_sha256": (
        "a4ccbdfaf5ff0cbd1e44cb89d210745a667052c43b20b3531951d614fa47fbc3"
    ),
    "failed_call_id": "call_oKzzQEsZZmy3odPvwIiHQhbe",
    "failed_tool_input_sha256": (
        "7cda826fa36f1c7728baced7052f4fa8fbf3f8bb20724b94cf31584290fdbefa"
    ),
    "failed_stderr_text_sha256": (
        "ae106f202631882826723ebef58f5331ade43f3dcea5728e8635ca7112e23f21"
    ),
    "provenance_sha256": (
        "dad93644f7afa458066b80a43aac9d43ad27b0c23867a6ca07c53832f4470a84"
    ),
}
EXPLORATORY_PROBE_THRESHOLD_HISTORY = (
    "The 0.25 pooled wrap-to-adjacent RMS floor was fixed after this archived "
    "outcome-blind VALID geometry diagnostic and the retrospectively recovered "
    "noncanonical VALID diagnostic call_fuLA4VZfL7zj1gBuQcvOWpSy, but before "
    "any v1r7 scientific evaluation and before the formal all-VALID geometry "
    "receipt."
)
NONCANONICAL_DIAGNOSTIC_THRESHOLD_HISTORY = (
    "The 0.25 pooled wrap-to-adjacent RMS floor was fixed after both this "
    "noncanonical outcome-blind VALID diagnostic and archived job 11663182, "
    "but before any v1r7 scientific evaluation and before the formal "
    "all-VALID geometry receipt."
)
EXPLORATORY_PROBE_LIMITATIONS = (
    "This exploratory probe inspected only trajectory 0 in each VALID file "
    "and five frames. It is archived for honesty and source archaeology; it "
    "is not the formal corpus-level witness, a scientific result, or evidence "
    "about policy outcomes."
)
THRESHOLD_PROVENANCE: dict[str, Any] = {
    "ratio_floor": 0.25,
    "selection_timing": (
        "fixed after outcome-blind VALID geometry diagnostics but before v1r7 "
        "scientific evaluation and the formal all-VALID geometry receipt"
    ),
    "exploratory_probe_job_id": EXPLORATORY_PROBE["job_id"],
    "exploratory_probe_script_sha256": EXPLORATORY_PROBE["script_sha256"],
    "exploratory_probe_output_sha256": EXPLORATORY_PROBE["output_sha256"],
    "exploratory_probe_stderr_sha256": EXPLORATORY_PROBE["stderr_sha256"],
    "exploratory_probe_provenance_sha256": EXPLORATORY_PROBE[
        "provenance_sha256"
    ],
    "noncanonical_diagnostic_successful_call_id": NONCANONICAL_DIAGNOSTIC[
        "successful_call_id"
    ],
    "noncanonical_diagnostic_tool_input_sha256": NONCANONICAL_DIAGNOSTIC[
        "successful_tool_input_sha256"
    ],
    "noncanonical_diagnostic_stdout_sha256": NONCANONICAL_DIAGNOSTIC[
        "successful_raw_stdout_text_sha256"
    ],
    "noncanonical_diagnostic_failed_call_id": NONCANONICAL_DIAGNOSTIC[
        "failed_call_id"
    ],
    "noncanonical_diagnostic_failed_input_sha256": NONCANONICAL_DIAGNOSTIC[
        "failed_tool_input_sha256"
    ],
    "noncanonical_diagnostic_failed_stderr_sha256": NONCANONICAL_DIAGNOSTIC[
        "failed_stderr_text_sha256"
    ],
    "noncanonical_diagnostic_provenance_sha256": NONCANONICAL_DIAGNOSTIC[
        "provenance_sha256"
    ],
}
PUBLISHER_ACCESS_PROTOCOL: dict[str, Any] = {
    "full_valid_hdf5_byte_stream_hashed_before_hdf5_decode": True,
    "decoded_hdf5_scope": (
        "coordinate/shape/boundary metadata and edge/neighbor slabs for the "
        "fixed VALID coverage only"
    ),
    "parameters_decoded": False,
    "evidence_or_policy_outcomes_decoded": False,
    "publisher_test_accessed": False,
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_upstream_v1r6_provenance(
    *, root: Path = ROOT,
) -> dict[str, object]:
    errors: list[str] = []
    freeze_path = root / "upstream_v1r6/PREOUTCOME_FREEZE.json"
    authorization_path = root / (
        "upstream_v1r6/"
        "root_authorization_external_validation_8f91ff35c7e42c1f.json"
    )
    try:
        freeze = json.loads(freeze_path.read_text())
        body = dict(freeze)
        claimed = body.pop("freeze_digest", None)
        if claimed != json_digest(body):
            errors.append("v1r6_freeze_content")
        if claimed != V1R6_PROVENANCE["v1r6_freeze_digest"]:
            errors.append("v1r6_freeze_digest")
        if freeze.get("source_closure_sha256") != V1R6_PROVENANCE[
            "v1r6_source_closure_sha256"
        ]:
            errors.append("v1r6_source_closure")
        if freeze.get("schema_version") != (
            "maintrack.spotlight-2d-replication.freeze.v1r6"
        ):
            errors.append("v1r6_freeze_schema")
    except Exception as exc:
        errors.append(f"v1r6_freeze:{type(exc).__name__}")
    try:
        authorization = json.loads(authorization_path.read_text())
        body = dict(authorization)
        claimed = body.pop("authorization_digest", None)
        if claimed != json_digest(body):
            errors.append("v1r6_authorization_content")
        if claimed != V1R6_PROVENANCE[
            "v1r6_external_validation_authorization_sha256"
        ]:
            errors.append("v1r6_authorization_digest")
        expected = {
            "stage": "external_validation",
            "freeze_digest": V1R6_PROVENANCE["v1r6_freeze_digest"],
            "input_bindings": {
                "internal_confirmation_aggregate_sha256": (
                    INTERNAL_CONFIRMATION_AGGREGATE
                ),
            },
            "decision": "AUTHORIZE_EXACT_STAGE",
            "single_stage_only": True,
            "post_outcome_tuning_permitted": False,
            "external_signature": False,
        }
        for key, value in expected.items():
            if authorization.get(key) != value:
                errors.append(f"v1r6_authorization:{key}")
    except Exception as exc:
        errors.append(f"v1r6_authorization:{type(exc).__name__}")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "provenance": V1R6_PROVENANCE,
    }


def validate_geometry_source_receipt(
    path: Path = GEOMETRY_SOURCE_RECEIPT_PATH,
    *,
    root: Path = ROOT,
) -> dict[str, object]:
    errors: list[str] = []
    if not path.is_file():
        return {"valid": False, "errors": ["source_receipt_absent"]}
    if file_sha(path) != GEOMETRY_SOURCE_RECEIPT_SHA256:
        errors.append("source_receipt_sha256")
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"source_receipt_decode:{type(exc).__name__}"],
        }
    if payload.get("schema_version") != (
        "maintrack.external-well-geometry-source-receipt.v1r7"
    ):
        errors.append("source_receipt_schema")
    expected_files = {
        "external_sources/the_well_gray_scott_README.md": (
            "7103571e4bba9c2cbd7b3ea41bf188df6ebc73f2020ec7025f018328c91b2242"
        ),
        "external_sources/the_well_endpoint_false_notebook.ipynb": (
            "4a37804ba4ec621718a065203e8e05962ee2e010796c73ca067d28461e43b058"
        ),
        "external_sources/THE_WELL_LICENSE": (
            "714e789efb1453e42fafd50d4f02e6f73d516e345a93b4c9238cf8f203f34caf"
        ),
        "external_sources/gen_gs.m": (
            "029832b3e06b485341c5f2ad04b388f1d80c22f0fbc8f858154934b6dd1f49e3"
        ),
        "external_sources/GENERATOR_LICENSE": (
            "dc66dae34d9a48b40e0afa8a4f4eb1b13af02fa184f7d5d2283e8b567f6e34e1"
        ),
        "external_sources/chebfun_trigpts.m": (
            "2c3400e98b129669f544a201aa277419d9cf5846af7d013ddef05568c01e919d"
        ),
        "external_sources/chebfun_spinop2_getGrid.m": (
            "67ebbdb97db94f33e7f4447d2e72f39fb6cd1026fb3cdaa4ddedf407bc09412e"
        ),
        "external_sources/chebfun_spinop2_reshapeGrid.m": (
            "16653bebb6ad905000c35ce3fe9c5f443dd61f21f41aecb9eb71a184b0d89370"
        ),
        "external_sources/chebfun_spinop2_solvepde.m": (
            "8f3e7e5d38dd99a3e875a7e9f1e4431f4ae629198249ff977e3462883ac6ec29"
        ),
        "external_sources/CHEBFUN_LICENSE.txt": (
            "7f575b348352c86b31d6a27fff0d31becacc305c005df1aee0ce64a7f7148077"
        ),
        "upstream_geometry_probe/grid_probe_valid_bfb8966815d4.py": (
            EXPLORATORY_PROBE["script_sha256"]
        ),
        "upstream_geometry_probe/grid_probe_11663182.json": (
            EXPLORATORY_PROBE["output_sha256"]
        ),
        "upstream_geometry_probe/PROBE_PROVENANCE.json": (
            EXPLORATORY_PROBE["provenance_sha256"]
        ),
        "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json": (
            NONCANONICAL_DIAGNOSTIC["provenance_sha256"]
        ),
    }
    for name, expected in expected_files.items():
        try:
            observed = file_sha(root / name)
        except FileNotFoundError:
            errors.append(f"source_witness_missing:{name}")
            continue
        if observed != expected:
            errors.append(f"source_witness_sha256:{name}")
    canonical = {
        "nodes_per_axis": 128,
        "coordinate_formula": "x_j=-1+2j/128, j=0,...,127",
        "domain": "[-1,1)^2",
        "period_per_axis": 2.0,
        "spacing_per_axis": 0.015625,
        "cell_area": 0.000244140625,
        "first_wavenumber": "pi",
        "field_operation": (
            "preserve all values; no endpoint drop; no transpose; no interpolation"
        ),
    }
    if payload.get("canonical_geometry") != canonical:
        errors.append("canonical_geometry")
    chebfun = payload.get("chebfun_semantics_witness", {})
    if chebfun.get("historical_revision") != (
        "db207bc9f48278ca4def15bf90591bfa44d0801d"
    ):
        errors.append("chebfun_historical_revision")
    notebook = payload.get("earliest_public_intermediate_hdf5_notebook_witness", {})
    if notebook.get("revision") != (
        "5160c65f12792691cb3b66975d00bbcc2571c002"
    ):
        errors.append("notebook_revision")
    limitation = str(payload.get("limitation", ""))
    for phrase in (
        "exact Chebfun commit",
        "MAT-to-HDF5 converter",
        "not a claim",
    ):
        if phrase not in limitation:
            errors.append("source_limitation")
    expected_probe_receipt = {
        "job_id": EXPLORATORY_PROBE["job_id"],
        "script_path": (
            "upstream_geometry_probe/grid_probe_valid_bfb8966815d4.py"
        ),
        "script_sha256": EXPLORATORY_PROBE["script_sha256"],
        "output_path": "upstream_geometry_probe/grid_probe_11663182.json",
        "output_sha256": EXPLORATORY_PROBE["output_sha256"],
        "stderr_sha256": EXPLORATORY_PROBE["stderr_sha256"],
        "provenance_path": "upstream_geometry_probe/PROBE_PROVENANCE.json",
        "provenance_sha256": EXPLORATORY_PROBE["provenance_sha256"],
        "coverage": (
            "six VALID files; trajectory 0 only; raw times "
            "0,20,500,980,1000; complete 128x128 A/B frames; coordinate, "
            "shape, chunk, and root-attribute metadata; x/y wrap and "
            "adjacent-edge summaries"
        ),
        "access_boundary": (
            "The probe decoded published F/k root attributes but did not "
            "access TEST, labels, scores, weak-form evidence, authority "
            "outputs, policy outputs, or scientific terminals."
        ),
        "ratio_floor_fixed_after_probe": 0.25,
        "threshold_history": EXPLORATORY_PROBE_THRESHOLD_HISTORY,
        "limitations": EXPLORATORY_PROBE_LIMITATIONS,
    }
    if payload.get("exploratory_valid_geometry_probe") != expected_probe_receipt:
        errors.append("exploratory_probe")
    expected_noncanonical_receipt = {
        "status": (
            "retrospectively_recovered_unarchived_noncanonical_valid_access"
        ),
        "successful_call_id": NONCANONICAL_DIAGNOSTIC["successful_call_id"],
        "successful_tool_input_sha256": NONCANONICAL_DIAGNOSTIC[
            "successful_tool_input_sha256"
        ],
        "successful_raw_stdout_text_sha256": NONCANONICAL_DIAGNOSTIC[
            "successful_raw_stdout_text_sha256"
        ],
        "failed_parse_attempt_call_id": NONCANONICAL_DIAGNOSTIC[
            "failed_call_id"
        ],
        "failed_parse_attempt_tool_input_sha256": NONCANONICAL_DIAGNOSTIC[
            "failed_tool_input_sha256"
        ],
        "failed_parse_attempt_stderr_text_sha256": NONCANONICAL_DIAGNOSTIC[
            "failed_stderr_text_sha256"
        ],
        "provenance_path": (
            "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
        ),
        "provenance_sha256": NONCANONICAL_DIAGNOSTIC["provenance_sha256"],
        "coverage": (
            "six VALID files; all 20 trajectories; raw times 20,500,980; "
            "complete 128x128 A/B frames; 24 file-field-axis rows of pooled "
            "x/y wrap RMS and all-adjacent RMS"
        ),
        "access_boundary": (
            "The diagnostic did not access TEST, coordinate/root attributes, "
            "labels, scores, weak-form evidence, authority outputs, policy "
            "outputs, or scientific terminals."
        ),
        "observed_wrap_to_adjacent_ratio_range": [
            0.8568654255754081,
            1.0298739196438143,
        ],
        "threshold_history": NONCANONICAL_DIAGNOSTIC_THRESHOLD_HISTORY,
        "limitations": (
            "This was an ad hoc diagnostic rather than a frozen receipt. It "
            "is disclosed retrospectively and is not a scientific endpoint, "
            "policy result, or substitute for the formal no-overwrite corpus "
            "receipt."
        ),
    }
    if payload.get(
        "retrospectively_recovered_noncanonical_valid_geometry_diagnostic"
    ) != expected_noncanonical_receipt:
        errors.append("noncanonical_diagnostic")
    provenance_path = root / "upstream_geometry_probe/PROBE_PROVENANCE.json"
    try:
        if file_sha(provenance_path) != EXPLORATORY_PROBE["provenance_sha256"]:
            errors.append("exploratory_probe_provenance_sha256")
        provenance = json.loads(provenance_path.read_text())
    except Exception as exc:
        errors.append(f"exploratory_probe_provenance:{type(exc).__name__}")
    else:
        expected_provenance = {
            "schema_version": (
                "maintrack.external-well-geometry-exploratory-probe.v1r7"
            ),
            "job_id": EXPLORATORY_PROBE["job_id"],
            "split": "valid",
            "script": {
                "path": expected_probe_receipt["script_path"],
                "sha256": EXPLORATORY_PROBE["script_sha256"],
            },
            "stdout": {
                "path": expected_probe_receipt["output_path"],
                "sha256": EXPLORATORY_PROBE["output_sha256"],
            },
            "stderr": {
                "empty": True,
                "sha256": EXPLORATORY_PROBE["stderr_sha256"],
            },
            "coverage": {
                "publisher_files": 6,
                "publisher_split": "valid",
                "trajectory_indices_per_file": [0],
                "raw_time_indices": [0, 20, 500, 980, 1000],
                "fields": ["A", "B"],
                "axes": ["x", "y"],
                "decoded_values": (
                    "one complete 128x128 frame at each selected "
                    "trajectory/time/field cell"
                ),
                "reported_diagnostics": [
                    "coordinate and dataset shape metadata",
                    "first-versus-last edge differences",
                    "first-versus-second neighbor differences",
                    "last-versus-penultimate neighbor differences",
                ],
            },
            "access_boundary": {
                "publisher_test_accessed": False,
                "scores_decoded": False,
                "labels_decoded": False,
                "authority_invoked": False,
                "policies_invoked": False,
                "scientific_terminals_decoded": False,
                "root_metadata_including_published_F_and_k_reported": True,
            },
            "threshold_history": EXPLORATORY_PROBE_THRESHOLD_HISTORY,
            "limitations": EXPLORATORY_PROBE_LIMITATIONS,
        }
        if provenance != expected_provenance:
            errors.append("exploratory_probe_provenance_content")
    noncanonical_path = root / (
        "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
    )
    try:
        if file_sha(noncanonical_path) != NONCANONICAL_DIAGNOSTIC[
            "provenance_sha256"
        ]:
            errors.append("noncanonical_diagnostic_provenance_sha256")
        noncanonical = json.loads(noncanonical_path.read_text())
    except Exception as exc:
        errors.append(f"noncanonical_diagnostic_provenance:{type(exc).__name__}")
    else:
        expected_noncanonical = {
            "schema_version": (
                "maintrack.external-well-geometry-noncanonical-diagnostic.v1r7"
            ),
            "status": (
                "retrospectively_recovered_unarchived_noncanonical_valid_access"
            ),
            "source_log": {
                "codex_session_relative_path": (
                    "2026/08/24/rollout-2026-08-24T20-30-44-"
                    "01a0350a-06a7-7d91-bb65-bc8136d86d05.jsonl"
                ),
                "successful_call_lines": [45489, 45490],
                "successful_call_id": NONCANONICAL_DIAGNOSTIC[
                    "successful_call_id"
                ],
                "successful_call_timestamp_utc": "2026-08-24T18:37:41.494Z",
                "successful_output_timestamp_utc": "2026-08-24T18:38:05.373Z",
                "successful_tool_input_sha256": NONCANONICAL_DIAGNOSTIC[
                    "successful_tool_input_sha256"
                ],
                "successful_raw_stdout_text_sha256": NONCANONICAL_DIAGNOSTIC[
                    "successful_raw_stdout_text_sha256"
                ],
                "execution": (
                    "inline SSH command using Euler and "
                    "/cluster/home/ooikonomou/miniconda3/bin/python"
                ),
                "raw_command_and_stdout_bundled": False,
                "nonbundling_reason": (
                    "The containing Codex session also holds unrelated private "
                    "conversation state; exact line pointers, call IDs, hashes, "
                    "scope, and aggregate values are retained instead."
                ),
            },
            "failed_parse_attempt": {
                "call_lines": [45481, 45482],
                "call_id": NONCANONICAL_DIAGNOSTIC["failed_call_id"],
                "timestamp_utc": "2026-08-24T18:37:20.491Z",
                "tool_input_sha256": NONCANONICAL_DIAGNOSTIC[
                    "failed_tool_input_sha256"
                ],
                "stderr_text_sha256": NONCANONICAL_DIAGNOSTIC[
                    "failed_stderr_text_sha256"
                ],
                "failure": "Python SyntaxError before HDF5 open or data access",
            },
            "coverage": {
                "publisher_files": 6,
                "publisher_split": "valid",
                "trajectory_indices_per_file": "all 20",
                "raw_time_indices": [20, 500, 980],
                "fields": ["A", "B"],
                "axes": ["x", "y"],
                "decoded_values": (
                    "complete 128x128 frames for the fixed "
                    "trajectory/time/field cells"
                ),
                "reported_diagnostics": (
                    "24 file-field-axis rows of pooled wrap RMS and "
                    "all-adjacent RMS"
                ),
                "observed_wrap_to_adjacent_ratio_range": [
                    0.8568654255754081,
                    1.0298739196438143,
                ],
            },
            "access_boundary": {
                "publisher_test_accessed": False,
                "coordinate_or_root_attributes_decoded": False,
                "scores_decoded": False,
                "labels_decoded": False,
                "weak_evidence_built": False,
                "authority_invoked": False,
                "policies_invoked": False,
                "scientific_terminals_decoded": False,
            },
            "threshold_history": NONCANONICAL_DIAGNOSTIC_THRESHOLD_HISTORY,
            "limitations": (
                "This was an ad hoc diagnostic rather than a frozen receipt. "
                "It is disclosed retrospectively and is not a scientific "
                "endpoint, policy result, or substitute for the formal "
                "no-overwrite corpus receipt."
            ),
        }
        if noncanonical != expected_noncanonical:
            errors.append("noncanonical_diagnostic_provenance_content")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "file_sha256": file_sha(path),
    }


def _expected_axis(axis: str) -> dict[str, object]:
    return {
        "name": axis,
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


def _validate_boundary(
    boundary: object, *, axis: str, prefix: str, errors: list[str]
) -> None:
    if not isinstance(boundary, Mapping):
        errors.append(prefix)
        return
    expected = {
        "group": f"{axis}_periodic",
        "associated_dims": [axis],
        "associated_fields": [],
        "bc_type": "PERIODIC",
        "sample_varying": False,
        "time_varying": False,
    }
    for key, value in expected.items():
        if boundary.get(key) != value:
            errors.append(f"{prefix}:{key}")


def _validate_axis_witness(
    row: object, *, prefix: str, errors: list[str]
) -> None:
    if not isinstance(row, Mapping):
        errors.append(prefix)
        return
    if row.get("edge_neighbor_slab_shape") != [20, 4, 128]:
        errors.append(f"{prefix}:slab_shape")
    for key in (
        "edge_neighbor_slab_sha256",
        "first_edge_sha256",
        "last_edge_sha256",
        "per_trajectory_ratio_sha256",
        "per_trajectory_edge_identity_sha256",
        "per_trajectory_informative_sha256",
    ):
        if not is_sha256(row.get(key)):
            errors.append(f"{prefix}:{key}")
    if row.get("trajectory_count") != 20:
        errors.append(f"{prefix}:trajectory_count")
    informative_count = row.get("informative_trajectory_count")
    degenerate_count = row.get("degenerate_trajectory_count")
    identical_count = row.get("opposing_edges_identical_trajectory_count")
    if not all(
        isinstance(value, int) and 0 <= value <= 20
        for value in (informative_count, degenerate_count, identical_count)
    ):
        errors.append(f"{prefix}:trajectory_class_counts")
    elif informative_count + degenerate_count != 20:
        errors.append(f"{prefix}:trajectory_class_partition")
    if not isinstance(row.get("opposing_edges_identical"), bool):
        errors.append(f"{prefix}:edge_identity_type")
    elif isinstance(identical_count, int) and (
        row["opposing_edges_identical"] != (identical_count == 20)
    ):
        errors.append(f"{prefix}:edge_identity_replay")
    if row.get("finite") is not True:
        errors.append(f"{prefix}:finite")
    if row.get("ratio_floor") != 0.25:
        errors.append(f"{prefix}:ratio_floor")
    if row.get("ratio_floor_applied_at") != "corpus_axis_pool_only":
        errors.append(f"{prefix}:ratio_floor_scope")
    if row.get("cell_status") not in {
        "informative",
        "degenerate_noninformative",
    }:
        errors.append(f"{prefix}:cell_status")
    for key in ("wrap_sum_squares", "adjacent_sum_squares"):
        value = row.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            errors.append(f"{prefix}:{key}")
    expected_informative = (
        informative_count if isinstance(informative_count, int) else -1
    )
    if row.get("wrap_value_count") != expected_informative * 128:
        errors.append(f"{prefix}:wrap_value_count")
    if row.get("adjacent_value_count") != expected_informative * 2 * 128:
        errors.append(f"{prefix}:adjacent_value_count")
    summaries: dict[str, tuple[float, float, float]] = {}
    for key in (
        "wrap_rms_summary",
        "adjacent_rms_summary",
        "wrap_to_adjacent_ratio_summary",
    ):
        summary = row.get(key)
        if summary is None:
            if informative_count != 0:
                errors.append(f"{prefix}:{key}:unexpected_null")
            continue
        if not isinstance(summary, Mapping) or set(summary) != {
            "min",
            "median",
            "max",
        }:
            errors.append(f"{prefix}:{key}")
            continue
        try:
            values = tuple(float(summary[name]) for name in ("min", "median", "max"))
        except (TypeError, ValueError):
            errors.append(f"{prefix}:{key}:decode")
            continue
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{prefix}:{key}:finite")
        ordered = 0.0 <= values[0] <= values[1] <= values[2]
        if not ordered:
            errors.append(f"{prefix}:{key}:order")
        summaries[key] = values
    try:
        if informative_count:
            wrap_rms = float(row["wrap_rms"])
            adjacent_rms = float(row["adjacent_rms"])
            expected_wrap_rms = math.sqrt(
                float(row["wrap_sum_squares"])
                / int(row["wrap_value_count"])
            )
            expected_adjacent_rms = math.sqrt(
                float(row["adjacent_sum_squares"])
                / int(row["adjacent_value_count"])
            )
            if not math.isclose(wrap_rms, expected_wrap_rms, rel_tol=1e-12):
                errors.append(f"{prefix}:wrap_rms_replay")
            if not math.isclose(
                adjacent_rms, expected_adjacent_rms, rel_tol=1e-12
            ):
                errors.append(f"{prefix}:adjacent_rms_replay")
            ratio = float(row.get("wrap_to_adjacent_rms_ratio"))
            if not math.isfinite(ratio):
                errors.append(f"{prefix}:ratio_finite")
            if not math.isclose(ratio, wrap_rms / adjacent_rms, rel_tol=1e-12):
                errors.append(f"{prefix}:ratio_replay")
            if row.get("cell_status") != "informative":
                errors.append(f"{prefix}:informative_status")
        else:
            ratio = None
            wrap_rms = None
            adjacent_rms = None
            if row.get("wrap_rms") is not None:
                errors.append(f"{prefix}:degenerate_wrap_rms")
            if row.get("adjacent_rms") is not None:
                errors.append(f"{prefix}:degenerate_adjacent_rms")
            if row.get("wrap_to_adjacent_rms_ratio") is not None:
                errors.append(f"{prefix}:degenerate_ratio")
            if any(
                float(row[key]) != 0.0
                for key in ("wrap_sum_squares", "adjacent_sum_squares")
            ):
                errors.append(f"{prefix}:degenerate_sum_squares")
            if row.get("cell_status") != "degenerate_noninformative":
                errors.append(f"{prefix}:degenerate_status")
        if informative_count:
            wrap_summary = summaries["wrap_rms_summary"]
            adjacent_summary = summaries["adjacent_rms_summary"]
            if not wrap_summary[0] <= wrap_rms <= wrap_summary[2]:
                errors.append(f"{prefix}:wrap_summary_bounds")
            if not adjacent_summary[0] <= adjacent_rms <= adjacent_summary[2]:
                errors.append(f"{prefix}:adjacent_summary_bounds")
            ratio_summary = summaries["wrap_to_adjacent_ratio_summary"]
            if not ratio_summary[0] <= ratio <= ratio_summary[2]:
                errors.append(f"{prefix}:ratio_summary_bounds")
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        errors.append(f"{prefix}:ratio_decode")


def _validate_geometry_file(
    item: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    regime: str,
    errors: list[str],
) -> None:
    for key, manifest_key in (
        ("publisher_path", "path"),
        ("bytes", "bytes"),
        ("publisher_file_sha256", "sha256"),
    ):
        if item.get(key) != expected.get(manifest_key):
            errors.append(f"{key}:{regime}")
    geometry = item.get("geometry")
    if not isinstance(geometry, Mapping):
        errors.append(f"geometry:{regime}")
        return
    if geometry.get("fields_accessed") is not True:
        errors.append(f"geometry_fields_accessed:{regime}")
    if geometry.get("outcomes_decoded") is not False:
        errors.append(f"geometry_outcomes_decoded:{regime}")
    grid = geometry.get("grid_metadata")
    if not isinstance(grid, Mapping):
        errors.append(f"grid_metadata:{regime}")
        return
    if grid.get("fields_accessed") is not False:
        errors.append(f"grid_fields_accessed:{regime}")
    if grid.get("outcomes_decoded") is not False:
        errors.append(f"grid_outcomes_decoded:{regime}")
    axes = grid.get("axes")
    if not isinstance(axes, Mapping) or set(axes) != {"x", "y"}:
        errors.append(f"grid_axes:{regime}")
        axes = {}
    for axis in ("x", "y"):
        if axes.get(axis) != _expected_axis(axis):
            errors.append(f"grid_axis:{regime}:{axis}")
    boundaries = grid.get("boundary_conditions")
    if not isinstance(boundaries, Mapping) or set(boundaries) != {"x", "y"}:
        errors.append(f"grid_boundaries:{regime}")
        boundaries = {}
    for axis in ("x", "y"):
        _validate_boundary(
            boundaries.get(axis),
            axis=axis,
            prefix=f"grid_boundary:{regime}:{axis}",
            errors=errors,
        )
    witness = geometry.get("field_witness")
    if not isinstance(witness, Mapping):
        errors.append(f"field_witness:{regime}")
        return
    if witness.get("trajectory_indices") != list(range(20)):
        errors.append(f"field_witness_trajectories:{regime}")
    if witness.get("raw_frame_indices") != [20, 500, 980]:
        errors.append(f"field_witness_frames:{regime}")
    if witness.get("ratio_floor") != 0.25:
        errors.append(f"field_witness_ratio_floor:{regime}")
    if witness.get("ratio_floor_timing") != THRESHOLD_PROVENANCE[
        "selection_timing"
    ]:
        errors.append(f"field_witness_ratio_floor_timing:{regime}")
    if witness.get("interpretation") != (
        "individual constant, axis-invariant, or seam-symmetric cells are "
        "recorded as degenerate and noninformative; nonduplication "
        "is decided only by the pooled all-VALID corpus-axis gate"
    ):
        errors.append(f"field_witness_interpretation:{regime}")
    fields = witness.get("fields")
    if not isinstance(fields, Mapping) or set(fields) != {"A", "B"}:
        errors.append(f"field_witness_fields:{regime}")
        fields = {}
    for field_name in ("A", "B"):
        field = fields.get(field_name)
        prefix = f"field_witness:{regime}:{field_name}"
        if not isinstance(field, Mapping):
            errors.append(prefix)
            continue
        if field.get("dataset_shape") != [20, 1001, 128, 128]:
            errors.append(f"{prefix}:dataset_shape")
        if field.get("dataset_dtype") != "float32":
            errors.append(f"{prefix}:dataset_dtype")
        if field.get("trajectory_indices") != list(range(20)):
            errors.append(f"{prefix}:trajectories")
        frames = field.get("frame_witnesses")
        if not isinstance(frames, list) or len(frames) != 3:
            errors.append(f"{prefix}:frames")
            frames = []
        if [frame.get("raw_frame_index") for frame in frames] != [20, 500, 980]:
            errors.append(f"{prefix}:frame_order")
        for frame in frames:
            frame_index = frame.get("raw_frame_index")
            axis_rows = frame.get("axes")
            if not isinstance(axis_rows, Mapping) or set(axis_rows) != {"x", "y"}:
                errors.append(f"{prefix}:axes:{frame_index}")
                axis_rows = {}
            for axis in ("x", "y"):
                _validate_axis_witness(
                    axis_rows.get(axis),
                    prefix=f"{prefix}:{frame_index}:{axis}",
                    errors=errors,
                )


def validate_external_geometry_preflight_receipt(
    path: Path = EXTERNAL_GEOMETRY_PREFLIGHT_PATH,
    *,
    root: Path = ROOT,
) -> dict[str, object]:
    errors: list[str] = []
    if not path.is_file():
        return {"valid": False, "errors": ["receipt_absent"]}
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"receipt_decode:{type(exc).__name__}"],
        }
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["receipt_root_type"]}
    expected_keys = {
        "schema_version",
        "split",
        "purpose",
        "not_v1r7_scientific_authorization",
        "upstream_v1r6_provenance",
        "external_manifest_sha256",
        "preflight_stage",
        "executed_source_sha256",
        "regime_count",
        "files",
        "corpus_geometry_witness",
        "publisher_access_protocol",
        "threshold_provenance",
        "fields_accessed",
        "outcomes_decoded",
        "scientific_artifacts_created",
        "content_sha256",
    }
    if set(payload) != expected_keys:
        errors.append("root_keys")
    claimed_content = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if not is_sha256(claimed_content) or json_digest(body) != claimed_content:
        errors.append("content_sha256")
    exact = {
        "schema_version": EXTERNAL_GEOMETRY_PREFLIGHT_SCHEMA,
        "split": "valid",
        "purpose": "VALID-only pre-v1r7-freeze field-geometry witness",
        "not_v1r7_scientific_authorization": True,
        "upstream_v1r6_provenance": V1R6_PROVENANCE,
        "regime_count": 6,
        "fields_accessed": True,
        "outcomes_decoded": False,
        "scientific_artifacts_created": False,
    }
    for key, value in exact.items():
        if payload.get(key) != value:
            errors.append(key)
    manifest_path = root / "EXTERNAL_WELL_MANIFEST.json"
    try:
        manifest_digest = file_sha(manifest_path)
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        errors.append(f"manifest:{type(exc).__name__}")
        manifest_digest = ""
        manifest = {}
    if payload.get("external_manifest_sha256") != manifest_digest:
        errors.append("external_manifest_sha256")
    try:
        from .scripts.stage_external_geometry_preflight import stage_identity

        expected_stage = stage_identity(root.parents[3])
    except Exception as exc:
        errors.append(f"preflight_stage_replay:{type(exc).__name__}")
        expected_stage = {}
    if payload.get("preflight_stage") != expected_stage:
        errors.append("preflight_stage")
    try:
        source_hashes = {
            "EXTERNAL_WELL_MANIFEST.json": manifest_digest,
            "external_well.py": file_sha(root / "external_well.py"),
            "geometry_receipt.py": file_sha(root / "geometry_receipt.py"),
            "external_sources/GEOMETRY_SOURCE_RECEIPT.json": file_sha(
                root / "external_sources/GEOMETRY_SOURCE_RECEIPT.json"
            ),
            "scripts/preflight_external_geometry.py": file_sha(
                root / "scripts/preflight_external_geometry.py"
            ),
            "scripts/stage_external_geometry_preflight.py": file_sha(
                root / "scripts/stage_external_geometry_preflight.py"
            ),
            "upstream_geometry_probe/PROBE_PROVENANCE.json": file_sha(
                root / "upstream_geometry_probe/PROBE_PROVENANCE.json"
            ),
            "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json": (
                file_sha(
                    root
                    / "upstream_geometry_probe/"
                    "NONCANONICAL_DIAGNOSTIC_PROVENANCE.json"
                )
            ),
        }
    except Exception as exc:
        errors.append(f"executed_source:{type(exc).__name__}")
        source_hashes = {}
    if payload.get("executed_source_sha256") != source_hashes:
        errors.append("executed_source_sha256")
    if payload.get("publisher_access_protocol") != PUBLISHER_ACCESS_PROTOCOL:
        errors.append("publisher_access_protocol")
    if payload.get("threshold_provenance") != THRESHOLD_PROVENANCE:
        errors.append("threshold_provenance")
    valid_rows = {
        str(item.get("regime")): item
        for item in manifest.get("files", [])
        if item.get("split") == "valid"
    }
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 6:
        errors.append("files")
        files = []
    observed: list[str] = []
    for item in files:
        if not isinstance(item, Mapping):
            errors.append("file_type")
            continue
        regime = str(item.get("regime"))
        observed.append(regime)
        _validate_geometry_file(
            item,
            expected=valid_rows.get(regime, {}),
            regime=regime,
            errors=errors,
        )
    if tuple(observed) != EXPECTED_REGIMES:
        errors.append("regime_order")
    try:
        from .external_well import summarize_geometry_corpus

        replayed_corpus = summarize_geometry_corpus(files)
    except Exception as exc:
        errors.append(f"corpus_geometry_witness:{type(exc).__name__}")
    else:
        if payload.get("corpus_geometry_witness") != replayed_corpus:
            errors.append("corpus_geometry_witness")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "content_sha256": claimed_content,
        "file_sha256": file_sha(path),
    }
