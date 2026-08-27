"""Fail closed unless v1r7 changes only the declared geometry surface.

The comparison is intentionally structural for ``external_well.py``: Python
AST nodes are compared without source locations, so comments and formatting do
not hide a scientific change.  Files that encode the frozen attack, guard,
authority, action bank, seeds, noise, trajectories, and manifest must remain
byte-identical to immutable v1r6.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


V1R6_LANE = "maintrack_spotlight_2d_replication_v1r6_20260824"
V1R7_LANE = "maintrack_spotlight_2d_replication_v1r7_20260824"

BYTE_IDENTICAL_FILES = (
    "EXTERNAL_WELL_MANIFEST.json",
    "__init__.py",
    "aggregate.py",
    "attack_guard.py",
    "authority.py",
    "authorization.py",
    "campaign.py",
    "capture_external_manifest.py",
    "contract.py",
    "environment.py",
    "numerical_fd.py",
    "numerical_spectral.py",
    "owner_truth.py",
    "trajectory.py",
    "validate.py",
    "validate_external.py",
    "weak_evidence.py",
)

EXPECTED_BYTE_SHA256 = {
    "EXTERNAL_WELL_MANIFEST.json": (
        "5c5e33ca0eb3be3fe532966e4b22071bde394506a29724f06ac94ff6e2047b36"
    ),
    "__init__.py": (
        "dd40056c00fd476ca1b340c4c328200da25c9f891531f16853f53f67eaa40e38"
    ),
    "aggregate.py": (
        "d1d2a98a272e5e224f61cbb4014e0c7ea4332c366170d5c59321a243f36f5178"
    ),
    "attack_guard.py": (
        "7b808ded38fa730713e7926a21de907baf0c16be85cca907e247b3bdc269b212"
    ),
    "authority.py": (
        "2fbc2d57238041e0ca0800d0c283e480df1a09cc7e27d07e0b781a752db2e294"
    ),
    "authorization.py": (
        "eeccb1ced92861a79a4a054de8315dc65f352e455ecc25751e2d87f589ec4ee7"
    ),
    "campaign.py": (
        "2121d5033dd38a8b3930e8a744104638d7c4845d0cc10a486200a94fc4354329"
    ),
    "capture_external_manifest.py": (
        "b480aa69856505fe51b944398d537a45a172f70bd1364e2f11159f31957592af"
    ),
    "contract.py": (
        "b889b1544c2ac959cc2d0ab33ab7305fcb30389d394cc5166ec4476ad9f56602"
    ),
    "environment.py": (
        "0370ce4cff1dd49b0b97c43d0ecc1d3581666f5bc480c71ec7d2a298d6afaad2"
    ),
    "numerical_fd.py": (
        "17f9914913a5e49677582a934c73a6d5eabc45a101fee6e1584cadaed5ee0e68"
    ),
    "numerical_spectral.py": (
        "7f1dbf0bea8c58f4660394b8ccc0cf06c257e689f2248a714b232b84b176d417"
    ),
    "owner_truth.py": (
        "52315218b6821c6f511a4824277ab4dca9cabc7b53f3f92479df5f48b9b56739"
    ),
    "trajectory.py": (
        "35d0cc362432eea889328cf262a4f276003a2babae4d8a85d34093ff5355d769"
    ),
    "validate.py": (
        "ac2a24ef35fa0c0b66397709eb1918b9cea9a4ef169272035f1edafa3deb8a77"
    ),
    "validate_external.py": (
        "1a387858b97b2878be10b1ce60858aae55eb3d20af128e91e354d6a27d0ff320"
    ),
    "weak_evidence.py": (
        "97a98107a7cc53094e8b02c77af7ecef8e4b2062b99138efc8502f092b7a3b56"
    ),
}

EXPECTED_BASELINE_SOURCE_SHA256 = {
    "aggregate_external.py": (
        "e3251f658c390cf98fabd4c00707b6cde54c845af8ab37c38e58a8a29318f24d"
    ),
    "external_well.py": (
        "ad01b5de309c6019b1fb015e983e42a831a38ed245fea37b6fb1967e66944914"
    ),
}

FROZEN_ASSIGNMENTS = (
    "CONTRACT_COLUMNS",
    "DATASET_ID",
    "DATASET_REVISION",
    "DELTA_A",
    "DELTA_B",
    "EXPECTED_REGIMES",
    "EXPECTED_SPLITS",
    "GENERATOR_REVISION",
    "LIBRARY_CONTRACTS",
    "MAX_MODE",
    "RAW_STEPS",
    "REGIME_PARAMETERS",
    "TARGETS",
    "TRAJECTORIES_PER_FILE",
    "WINDOW_FIRST_INDEX",
    "WINDOW_LAST_INDEX",
    "WINDOW_LENGTH",
)

FROZEN_FUNCTIONS = (
    "_full_design",
    "_proposal_payload",
    "_temporal_test",
    "atoms_for_target",
    "authorization_bindings",
    "build_external_episodes",
    "build_external_evidence",
    "external_reference_cost",
    "external_registered_cost",
    "noise_seed",
    "observed_external_arrays",
    "omitted_mechanism",
    "owner_label",
    "temporal_positions",
)

EXPECTED_EXTERNAL_WELL_CHANGE_SURFACE = {
    "added": [
        "FunctionDef:_array_sha256",
        "FunctionDef:_finite_summary",
        "FunctionDef:_iter_geometry_axis_rows",
        "FunctionDef:_pool_geometry_rows",
        "FunctionDef:_validate_publisher_field",
        "FunctionDef:read_publisher_geometry_witness",
        "FunctionDef:summarize_geometry_corpus",
        "assign:CANONICAL_CELL_AREA",
        "assign:CANONICAL_FIRST_WAVENUMBER",
        "assign:CANONICAL_SPACING",
        "assign:CHEBFUN_TRIGPTS_STABLE_SINCE",
        "assign:CHEBFUN_WITNESS_REVISION",
        "assign:EXPECTED_PERIODIC_ASSOCIATED_FIELDS",
        "assign:GEOMETRY_WITNESS_FRAME_INDICES",
        "assign:GEOMETRY_WITNESS_RATIO_FLOOR",
        "assign:GEOMETRY_WITNESS_RATIO_FLOOR_TIMING",
        "assign:THE_WELL_SOURCE_REVISION",
    ],
    "changed": [
        "FunctionDef:_canonicalize_publisher_fields",
        "FunctionDef:_publisher_axis",
        "FunctionDef:_validate_periodic_boundary_metadata",
        "FunctionDef:external_spatial_grid_contract",
        "assign:CANONICAL_SPATIAL_RESOLUTION",
        "assign:EXTERNAL_SCHEMA_VERSION",
    ],
    "removed": [
        "FunctionDef:_validate_periodic_face_closure",
        "assign:PUBLISHER_PERIODIC_CLOSURE_ULPS",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ast_nodes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text())
    nodes: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[f"{type(node).__name__}:{node.name}"] = ast.dump(
                node, include_attributes=False
            )
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                nodes[f"assign:{target.id}"] = ast.dump(
                    node.value, include_attributes=False
                )
    return nodes


def _change_surface(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(after.keys() - before.keys()),
        "changed": sorted(
            key for key in before.keys() & after.keys() if before[key] != after[key]
        ),
        "removed": sorted(before.keys() - after.keys()),
    }


def audit(v1r6_root: Path, v1r7_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    baseline_source_hashes = {
        relative: _sha256(v1r6_root / relative)
        for relative in EXPECTED_BASELINE_SOURCE_SHA256
    }
    for relative, expected_hash in EXPECTED_BASELINE_SOURCE_SHA256.items():
        if baseline_source_hashes[relative] != expected_hash:
            errors.append(f"baseline_source_drift:{relative}")
    byte_rows: dict[str, dict[str, Any]] = {}
    for relative in BYTE_IDENTICAL_FILES:
        before_hash = _sha256(v1r6_root / relative)
        after_hash = _sha256(v1r7_root / relative)
        expected_hash = EXPECTED_BYTE_SHA256[relative]
        matches = before_hash == after_hash == expected_hash
        byte_rows[relative] = {
            "expected_sha256": expected_hash,
            "v1r6_sha256": before_hash,
            "v1r7_sha256": after_hash,
            "byte_identical": matches,
        }
        if not matches:
            errors.append(f"byte_drift:{relative}")

    before_nodes = _ast_nodes(v1r6_root / "external_well.py")
    after_nodes = _ast_nodes(v1r7_root / "external_well.py")
    observed_surface = _change_surface(before_nodes, after_nodes)
    if observed_surface != EXPECTED_EXTERNAL_WELL_CHANGE_SURFACE:
        errors.append("external_well_change_surface")

    frozen_nodes: dict[str, bool] = {}
    for name in FROZEN_ASSIGNMENTS:
        key = f"assign:{name}"
        frozen_nodes[key] = before_nodes.get(key) == after_nodes.get(key)
    for name in FROZEN_FUNCTIONS:
        key = f"FunctionDef:{name}"
        frozen_nodes[key] = before_nodes.get(key) == after_nodes.get(key)
    for key, matches in frozen_nodes.items():
        if not matches:
            errors.append(f"frozen_ast_drift:{key}")

    before_aggregate = (v1r6_root / "aggregate_external.py").read_text()
    after_aggregate = (v1r7_root / "aggregate_external.py").read_text()
    aggregate_schema_only = after_aggregate.replace(
        "maintrack.external-well-gray-scott.aggregate.v1r7",
        "maintrack.external-well-gray-scott.aggregate.v1r5",
    ) == before_aggregate
    if not aggregate_schema_only:
        errors.append("aggregate_external_non_schema_drift")

    return {
        "schema_version": "maintrack.v1r7-source-diff-audit.v1",
        "baseline": {
            "lane": V1R6_LANE,
            "freeze_digest": (
                "9caa529977f33f8d5afac609944a96903b886c1b2331493c725f974b3937f239"
            ),
            "source_closure_sha256": (
                "8e9b65b938f6b98b0aad11fdc118b4e0b4bd0ef4f39069a726220f3306915c20"
            ),
        },
        "candidate_lane": V1R7_LANE,
        "valid": not errors,
        "errors": errors,
        "baseline_source_sha256": baseline_source_hashes,
        "byte_identical_files": byte_rows,
        "external_well_change_surface": observed_surface,
        "frozen_external_well_ast_nodes": frozen_nodes,
        "aggregate_external_schema_only": aggregate_schema_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("v1r6_root", type=Path)
    parser.add_argument("v1r7_root", type=Path)
    args = parser.parse_args()
    result = audit(args.v1r6_root, args.v1r7_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
