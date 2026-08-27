"""Create once and verify the v1r7 prospective source closure."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

from .geometry_receipt import (
    EXTERNAL_GEOMETRY_PREFLIGHT_PATH,
    V1R6_PROVENANCE,
    file_sha,
    json_digest,
    validate_external_geometry_preflight_receipt,
    validate_geometry_source_receipt,
    validate_upstream_v1r6_provenance,
)


ROOT = Path(__file__).resolve().parent
FREEZE_PATH = ROOT / "PREOUTCOME_FREEZE.json"
REPO = ROOT.parents[3]
LANE_REPO_RELATIVE = PurePosixPath(
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "maintrack_spotlight_2d_replication_v1r7_20260824"
)

DEPENDENCY_RELATIVE_FILES: dict[str, str] = {
    "wp1_wp4_operator_agent.__init__": "experiments/agentic_frontier/wp1_wp4_operator_agent/__init__.py",
    "wp1_wp4_operator_agent.config": "experiments/agentic_frontier/wp1_wp4_operator_agent/config.py",
    "wp1_wp4_operator_agent.engine": "experiments/agentic_frontier/wp1_wp4_operator_agent/engine.py",
    "wp1_wp4_operator_agent.manufactured": "experiments/agentic_frontier/wp1_wp4_operator_agent/manufactured.py",
    "wp1_wp4_operator_agent.proposals": "experiments/agentic_frontier/wp1_wp4_operator_agent/proposals.py",
    "wp1_wp4_operator_agent.run_demo": "experiments/agentic_frontier/wp1_wp4_operator_agent/run_demo.py",
    "wp1_wp4_operator_agent.serialization": "experiments/agentic_frontier/wp1_wp4_operator_agent/serialization.py",
    "wp1_wp4_operator_agent.susano_candidates": "experiments/agentic_frontier/wp1_wp4_operator_agent/susano_candidates.py",
    "wp1_wp4_operator_agent.validate_artifacts": "experiments/agentic_frontier/wp1_wp4_operator_agent/validate_artifacts.py",
    "wp1_wp4_operator_agent.weak_form": "experiments/agentic_frontier/wp1_wp4_operator_agent/weak_form.py",
    "maintrack_2d_physical_v1.__init__": "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/__init__.py",
    "adversarial_verifier_gaming_v1.__init__": "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/__init__.py",
    "adversarial_verifier_gaming_v1.contract": "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py",
    "exact_2d_oracle": "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_oracle.py",
    "exact_2d_coupled_oracle": "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_2d_physical_v1/exact_2d_coupled_oracle.py",
}

SOURCE_FILE_ALLOWLIST = (
    "DECISIONS.md",
    "EXTERNAL_GEOMETRY_PREFLIGHT_VALID.json",
    "EXTERNAL_WELL_MANIFEST.json",
    "PROSPECTIVE_PROTOCOL.md",
    "README.md",
    "SUPERSESSION.md",
    "__init__.py",
    "aggregate.py",
    "aggregate_external.py",
    "attack_guard.py",
    "authority.py",
    "authorization.py",
    "campaign.py",
    "capture_external_manifest.py",
    "contract.py",
    "docs/PREOUTCOME_GEOMETRY_REPAIR_AUDIT.md",
    "docs/V1R7_SOURCE_DIFF_CLASSIFICATION.md",
    "docs/ambiguities.md",
    "docs/branch_registry.md",
    "docs/conflict_register.md",
    "docs/data_contract.md",
    "docs/evaluation_protocol.md",
    "docs/external_well_protocol.md",
    "docs/interface_contract.md",
    "docs/knowledge.md",
    "docs/project_plan.md",
    "docs/reproducibility_protocol.md",
    "docs/source_truth_register.md",
    "environment.py",
    "external_sources/CHEBFUN_LICENSE.txt",
    "external_sources/GENERATOR_LICENSE",
    "external_sources/GEOMETRY_SOURCE_RECEIPT.json",
    "external_sources/THE_WELL_LICENSE",
    "external_sources/chebfun_spinop2_getGrid.m",
    "external_sources/chebfun_spinop2_reshapeGrid.m",
    "external_sources/chebfun_spinop2_solvepde.m",
    "external_sources/chebfun_trigpts.m",
    "external_sources/gen_gs.m",
    "external_sources/gray_scott_reaction_diffusion.yaml",
    "external_sources/hf_api_response.json",
    "external_sources/init_fourier.m",
    "external_sources/random_gaussians.m",
    "external_sources/the_well_endpoint_false_notebook.ipynb",
    "external_sources/the_well_gray_scott_README.md",
    "external_well.py",
    "freeze.py",
    "geometry_receipt.py",
    "numerical_fd.py",
    "numerical_spectral.py",
    "owner_truth.py",
    "schemas/internal_shard.schema.json",
    "schemas/stage_authorization.schema.json",
    "scripts/__init__.py",
    "scripts/check_v1r7_diff.py",
    "scripts/download_external_well.py",
    "scripts/external_geometry_preflight.sbatch",
    "scripts/external_well_aggregate.sbatch",
    "scripts/external_well_array.sbatch",
    "scripts/internal_aggregate.sbatch",
    "scripts/internal_array.sbatch",
    "scripts/preflight_external_geometry.py",
    "scripts/stage_external_geometry_preflight.py",
    "scripts/stage_minimal.sh",
    "scripts/stage_scientific.py",
    "tests/__init__.py",
    "tests/test_external_well.py",
    "tests/test_internal_protocol.py",
    "trajectory.py",
    "upstream_geometry_probe/PROBE_PROVENANCE.json",
    "upstream_geometry_probe/NONCANONICAL_DIAGNOSTIC_PROVENANCE.json",
    "upstream_geometry_probe/grid_probe_11663182.json",
    "upstream_geometry_probe/grid_probe_valid_bfb8966815d4.py",
    "upstream_v1r4/PREOUTCOME_FREEZE.json",
    "upstream_v1r4/root_authorization_external_validation_76bb15c4063b0f76.json",
    "upstream_v1r5/PREOUTCOME_FREEZE.json",
    "upstream_v1r5/root_authorization_external_validation_c22b282702cfd64a.json",
    "upstream_v1r6/EXTERNAL_COORDINATE_PREFLIGHT_VALID.json",
    "upstream_v1r6/PREOUTCOME_FREEZE.json",
    "upstream_v1r6/root_authorization_external_validation_8f91ff35c7e42c1f.json",
    "upstream_v1r6/source_diff_baseline/EXTERNAL_WELL_MANIFEST.json",
    "upstream_v1r6/source_diff_baseline/__init__.py",
    "upstream_v1r6/source_diff_baseline/aggregate.py",
    "upstream_v1r6/source_diff_baseline/aggregate_external.py",
    "upstream_v1r6/source_diff_baseline/attack_guard.py",
    "upstream_v1r6/source_diff_baseline/authority.py",
    "upstream_v1r6/source_diff_baseline/authorization.py",
    "upstream_v1r6/source_diff_baseline/campaign.py",
    "upstream_v1r6/source_diff_baseline/capture_external_manifest.py",
    "upstream_v1r6/source_diff_baseline/contract.py",
    "upstream_v1r6/source_diff_baseline/environment.py",
    "upstream_v1r6/source_diff_baseline/external_well.py",
    "upstream_v1r6/source_diff_baseline/numerical_fd.py",
    "upstream_v1r6/source_diff_baseline/numerical_spectral.py",
    "upstream_v1r6/source_diff_baseline/owner_truth.py",
    "upstream_v1r6/source_diff_baseline/trajectory.py",
    "upstream_v1r6/source_diff_baseline/validate.py",
    "upstream_v1r6/source_diff_baseline/validate_external.py",
    "upstream_v1r6/source_diff_baseline/weak_evidence.py",
    "validate.py",
    "validate_external.py",
    "weak_evidence.py",
)

FREEZE_SCHEMA = "maintrack.spotlight-2d-replication.freeze.v1r7"
FREEZE_STATUS = "SOURCE_FROZEN_AFTER_SOURCE_SUPPORTED_GEOMETRY_REPAIR_NO_SCIENTIFIC_ARTIFACTS"
OPENED_SCIENTIFIC_SPLITS = [
    "external_validation_publisher_valid_v1r6_failed_closed_no_artifacts",
    "external_validation_publisher_valid_v1r7_geometry_witness_only",
]
SEALED_SCIENTIFIC_SPLITS = ["external_confirmation_publisher_test"]
CLAIM_CEILING = (
    "prospective finite 2-D/coupled replication; external arm is an "
    "independent numerical origin, not physical validation"
)
FREEZE_ROOT_KEYS = {
    "schema_version",
    "frozen_at_utc",
    "status",
    "source_manifest",
    "source_closure_sha256",
    "dependency_manifest",
    "dependency_closure_sha256",
    "external_manifest_sha256",
    "external_geometry_preflight_sha256",
    "external_geometry_preflight_content_sha256",
    "geometry_source_receipt_sha256",
    "upstream_v1r6_provenance",
    "opened_scientific_splits",
    "sealed_scientific_splits",
    "same_attack_and_guard",
    "post_outcome_tuning_permitted",
    "claim_ceiling",
    "freeze_digest",
}
_DYNAMIC_ROOT_AUTHORIZATION = re.compile(
    r"root_authorization_(?:external_validation|external_confirmation)_[0-9a-f]{16}\.json"
)
_IGNORED_GENERATED_PREFIXES = (
    "governance_preexecution_",
    "upstream_archival_governance/",
)
_IGNORED_CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def source_files() -> tuple[str, ...]:
    if len(SOURCE_FILE_ALLOWLIST) != len(set(SOURCE_FILE_ALLOWLIST)):
        raise RuntimeError("duplicate path in source allowlist")
    return tuple(sorted(SOURCE_FILE_ALLOWLIST))


def _safe_regular_bytes(root: Path, relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe manifest path: {relative}")
    path = root / relative
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"manifest path is not a regular non-symlink file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest path escapes root: {relative}") from exc
    return resolved.read_bytes()


def _file_digest(root: Path, relative: str) -> str:
    return hashlib.sha256(_safe_regular_bytes(root, relative)).hexdigest()


def source_tree_errors(root: Path = ROOT) -> list[str]:
    root = root.resolve(strict=True)
    allowed = set(source_files())
    errors: list[str] = []
    for path in root.rglob("*"):
        relative_path = path.relative_to(root)
        relative = relative_path.as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            errors.append(f"symlink:{relative}")
            continue
        if any(part in _IGNORED_CACHE_PARTS for part in relative_path.parts):
            continue
        if any(
            relative == prefix.removesuffix("/") or relative.startswith(prefix)
            for prefix in _IGNORED_GENERATED_PREFIXES
        ):
            continue
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            errors.append(f"nonregular:{relative}")
            continue
        if relative in {"STATUS.md", "PREOUTCOME_FREEZE.json"}:
            continue
        if relative_path.parent == Path(".") and _DYNAMIC_ROOT_AUTHORIZATION.fullmatch(
            relative_path.name
        ):
            continue
        if relative not in allowed:
            errors.append(f"unmanifested:{relative}")
    return sorted(errors)


def manifest(root: Path = ROOT) -> dict[str, str]:
    resolved = root.resolve(strict=True)
    errors = source_tree_errors(resolved)
    if errors:
        raise RuntimeError("source tree outside explicit allowlist: " + ",".join(errors))
    return {name: _file_digest(resolved, name) for name in source_files()}


def dependency_manifest(repo: Path = REPO) -> dict[str, str]:
    resolved = repo.resolve(strict=True)
    return {
        name: _file_digest(resolved, relative)
        for name, relative in DEPENDENCY_RELATIVE_FILES.items()
    }


def content_digest(value: Mapping[str, str]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _require_valid(result: Mapping[str, object], label: str) -> None:
    if not result["valid"]:
        raise RuntimeError(
            f"{label} does not validate: "
            + ",".join(str(item) for item in result["errors"])
        )


def _semantic_errors(payload: Mapping[str, Any]) -> list[str]:
    expected = {
        "schema_version": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "opened_scientific_splits": OPENED_SCIENTIFIC_SPLITS,
        "sealed_scientific_splits": SEALED_SCIENTIFIC_SPLITS,
        "same_attack_and_guard": True,
        "post_outcome_tuning_permitted": False,
        "claim_ceiling": CLAIM_CEILING,
    }
    errors = [
        f"semantic:{key}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    try:
        parsed = datetime.fromisoformat(str(payload.get("frozen_at_utc")))
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ValueError("UTC timezone missing")
    except (TypeError, ValueError):
        errors.append("semantic:frozen_at_utc")
    return errors


def create() -> dict[str, object]:
    if FREEZE_PATH.exists():
        raise FileExistsError("one-time pre-outcome freeze already exists")
    geometry_source = validate_geometry_source_receipt()
    preflight = validate_external_geometry_preflight_receipt()
    upstream_v1r6 = validate_upstream_v1r6_provenance()
    _require_valid(geometry_source, "geometry source receipt")
    _require_valid(preflight, "external geometry preflight")
    _require_valid(upstream_v1r6, "upstream v1r6 provenance")
    files = manifest()
    dependencies = dependency_manifest()
    body: dict[str, object] = {
        "schema_version": FREEZE_SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": FREEZE_STATUS,
        "source_manifest": files,
        "source_closure_sha256": content_digest(files),
        "dependency_manifest": dependencies,
        "dependency_closure_sha256": content_digest(dependencies),
        "external_manifest_sha256": file_sha(ROOT / "EXTERNAL_WELL_MANIFEST.json"),
        "external_geometry_preflight_sha256": preflight["file_sha256"],
        "external_geometry_preflight_content_sha256": preflight["content_sha256"],
        "geometry_source_receipt_sha256": geometry_source["file_sha256"],
        "upstream_v1r6_provenance": V1R6_PROVENANCE,
        "opened_scientific_splits": OPENED_SCIENTIFIC_SPLITS,
        "sealed_scientific_splits": SEALED_SCIENTIFIC_SPLITS,
        "same_attack_and_guard": True,
        "post_outcome_tuning_permitted": False,
        "claim_ceiling": CLAIM_CEILING,
    }
    payload = {**body, "freeze_digest": json_digest(body)}
    FREEZE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify(
    *, root: Path = ROOT, repo: Path = REPO, freeze_path: Path | None = None
) -> dict[str, object]:
    root = root.resolve(strict=True)
    repo = repo.resolve(strict=True)
    path = root / "PREOUTCOME_FREEZE.json" if freeze_path is None else freeze_path
    if not path.is_file():
        return {"valid": False, "errors": ["freeze_absent"]}
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {"valid": False, "errors": [f"freeze_decode:{type(exc).__name__}"]}
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["freeze_root_type"]}
    if set(payload) != FREEZE_ROOT_KEYS:
        errors.append("freeze_root_keys")
    body = dict(payload)
    claimed = body.pop("freeze_digest", None)
    if claimed != json_digest(body):
        errors.append("freeze_digest")
    errors.extend(_semantic_errors(payload))
    upstream = validate_upstream_v1r6_provenance(root=root)
    if not upstream["valid"]:
        errors.extend(f"upstream_v1r6:{item}" for item in upstream["errors"])
    if payload.get("upstream_v1r6_provenance") != V1R6_PROVENANCE:
        errors.append("upstream_v1r6_provenance")
    try:
        actual = manifest(root)
    except Exception as exc:
        actual = {}
        errors.append(f"source_manifest_replay:{type(exc).__name__}:{exc}")
    recorded = payload.get("source_manifest")
    if not isinstance(recorded, dict):
        errors.append("source_manifest_type")
        recorded = {}
    if set(recorded) != set(source_files()):
        errors.append("source_manifest_keyset")
    if recorded != actual:
        errors.append("source_manifest")
    if payload.get("source_closure_sha256") != content_digest(actual):
        errors.append("source_closure")
    try:
        dependencies = dependency_manifest(repo)
    except Exception as exc:
        dependencies = {}
        errors.append(f"dependency_manifest_replay:{type(exc).__name__}:{exc}")
    recorded_dependencies = payload.get("dependency_manifest")
    if not isinstance(recorded_dependencies, dict):
        errors.append("dependency_manifest_type")
        recorded_dependencies = {}
    if set(recorded_dependencies) != set(DEPENDENCY_RELATIVE_FILES):
        errors.append("dependency_manifest_keyset")
    if recorded_dependencies != dependencies:
        errors.append("dependency_manifest")
    if payload.get("dependency_closure_sha256") != content_digest(dependencies):
        errors.append("dependency_closure")
    try:
        external_manifest_sha = file_sha(root / "EXTERNAL_WELL_MANIFEST.json")
    except Exception as exc:
        external_manifest_sha = ""
        errors.append(f"external_manifest_replay:{type(exc).__name__}")
    if payload.get("external_manifest_sha256") != external_manifest_sha:
        errors.append("external_manifest")
    source_result = validate_geometry_source_receipt(root=root)
    if not source_result["valid"]:
        errors.extend(f"geometry_source:{item}" for item in source_result["errors"])
    elif payload.get("geometry_source_receipt_sha256") != source_result[
        "file_sha256"
    ]:
        errors.append("geometry_source_receipt")
    preflight_path = root / EXTERNAL_GEOMETRY_PREFLIGHT_PATH.name
    if not preflight_path.is_file():
        errors.append("external_geometry_preflight_missing")
        preflight: Mapping[str, object] = {
            "valid": False,
            "errors": ["receipt_absent"],
        }
    else:
        preflight = validate_external_geometry_preflight_receipt(
            preflight_path, root=root
        )
    if not preflight["valid"]:
        errors.extend(
            f"external_geometry_preflight:{item}" for item in preflight["errors"]
        )
    else:
        if payload.get("external_geometry_preflight_sha256") != preflight[
            "file_sha256"
        ]:
            errors.append("external_geometry_preflight")
        if payload.get("external_geometry_preflight_content_sha256") != preflight[
            "content_sha256"
        ]:
            errors.append("external_geometry_preflight_content")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "freeze_digest": claimed,
        "source_closure_sha256": content_digest(actual),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    args = parser.parse_args()
    result = create() if args.command == "create" else verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "verify" and not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
