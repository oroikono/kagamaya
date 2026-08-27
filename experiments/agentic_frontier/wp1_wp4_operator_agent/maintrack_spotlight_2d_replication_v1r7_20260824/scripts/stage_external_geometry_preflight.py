"""Build a deterministic pre-freeze source archive for geometry preflight.

Unlike the scientific ``stage_minimal.sh``, this staging path neither requires
nor reads an active v1r7 ``PREOUTCOME_FREEZE.json``. It contains only the import
closure needed to produce the VALID-only geometry receipt plus immutable v1r6
provenance and the pinned source-semantics witnesses.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import gzip
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import tarfile
import tempfile


LANE = (
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "maintrack_spotlight_2d_replication_v1r7_20260824"
)
STAGE_FILES = (
    "experiments/agentic_frontier/wp1_wp4_operator_agent/__init__.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/config.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/engine.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/manufactured.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/proposals.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/run_demo.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/serialization.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/susano_candidates.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/validate_artifacts.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/weak_form.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "adversarial_verifier_gaming_v1/__init__.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "adversarial_verifier_gaming_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "maintrack_2d_physical_v1/__init__.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "maintrack_2d_physical_v1/exact_2d_oracle.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/"
    "maintrack_2d_physical_v1/exact_2d_coupled_oracle.py",
    f"{LANE}/__init__.py",
    f"{LANE}/attack_guard.py",
    f"{LANE}/authority.py",
    f"{LANE}/authorization.py",
    f"{LANE}/campaign.py",
    f"{LANE}/contract.py",
    f"{LANE}/environment.py",
    f"{LANE}/external_well.py",
    f"{LANE}/geometry_receipt.py",
    f"{LANE}/owner_truth.py",
    f"{LANE}/trajectory.py",
    f"{LANE}/weak_evidence.py",
    f"{LANE}/EXTERNAL_WELL_MANIFEST.json",
    f"{LANE}/scripts/__init__.py",
    f"{LANE}/external_sources/CHEBFUN_LICENSE.txt",
    f"{LANE}/external_sources/GENERATOR_LICENSE",
    f"{LANE}/external_sources/GEOMETRY_SOURCE_RECEIPT.json",
    f"{LANE}/external_sources/THE_WELL_LICENSE",
    f"{LANE}/external_sources/chebfun_spinop2_getGrid.m",
    f"{LANE}/external_sources/chebfun_spinop2_reshapeGrid.m",
    f"{LANE}/external_sources/chebfun_spinop2_solvepde.m",
    f"{LANE}/external_sources/chebfun_trigpts.m",
    f"{LANE}/external_sources/gen_gs.m",
    f"{LANE}/external_sources/the_well_endpoint_false_notebook.ipynb",
    f"{LANE}/external_sources/the_well_gray_scott_README.md",
    f"{LANE}/scripts/external_geometry_preflight.sbatch",
    f"{LANE}/scripts/preflight_external_geometry.py",
    f"{LANE}/scripts/stage_external_geometry_preflight.py",
    f"{LANE}/upstream_v1r6/PREOUTCOME_FREEZE.json",
    f"{LANE}/upstream_v1r6/"
    "root_authorization_external_validation_8f91ff35c7e42c1f.json",
    f"{LANE}/upstream_geometry_probe/PROBE_PROVENANCE.json",
    f"{LANE}/upstream_geometry_probe/"
    "NONCANONICAL_DIAGNOSTIC_PROVENANCE.json",
    f"{LANE}/upstream_geometry_probe/grid_probe_11663182.json",
    f"{LANE}/upstream_geometry_probe/grid_probe_valid_bfb8966815d4.py",
)
STAGE_MANIFEST_PATH = f"{LANE}/PREOUTCOME_PREFLIGHT_STAGE_MANIFEST.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, BytesIO(data))


def _regular_file_bytes(repo_root: Path, relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe stage path: {relative}")
    path = repo_root / relative
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"stage input is not a regular non-symlink file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"stage input escapes repository root: {relative}") from exc
    return resolved.read_bytes()


def collect_stage_files(repo_root: Path) -> dict[str, bytes]:
    root = repo_root.resolve(strict=True)
    if len(STAGE_FILES) != len(set(STAGE_FILES)):
        raise ValueError("duplicate path in geometry-preflight stage allowlist")
    return {
        relative: _regular_file_bytes(root, relative)
        for relative in sorted(STAGE_FILES)
    }


def _stage_manifest_from_files(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "schema_version": "maintrack.external-geometry-preflight-stage.v1r7",
        "purpose": "VALID-only pre-freeze publisher field-geometry witness",
        "requires_v1r7_freeze": False,
        "contains_publisher_test": False,
        "fields_accessed": True,
        "outcomes_decoded": False,
        "files": {name: _sha256(data) for name, data in files.items()},
    }


def stage_manifest(repo_root: Path) -> dict[str, object]:
    return _stage_manifest_from_files(collect_stage_files(repo_root))


def stage_manifest_bytes(value: dict[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def stage_archive_bytes(repo_root: Path) -> tuple[bytes, dict[str, object], bytes]:
    files = collect_stage_files(repo_root)
    manifest = _stage_manifest_from_files(files)
    manifest_bytes = stage_manifest_bytes(manifest)
    stream = BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(
            fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name, data in sorted(files.items()):
                _add_bytes(archive, name, data)
            _add_bytes(archive, STAGE_MANIFEST_PATH, manifest_bytes)
    return stream.getvalue(), manifest, manifest_bytes


def stage_identity(repo_root: Path) -> dict[str, object]:
    archive, manifest, manifest_bytes = stage_archive_bytes(repo_root)
    return {
        "archive_sha256": _sha256(archive),
        "stage_manifest_sha256": _sha256(manifest_bytes),
        "stage_manifest": manifest,
    }


def build_stage(repo_root: Path, output: Path) -> dict[str, object]:
    archive_bytes, stage_manifest_value, manifest_bytes = stage_archive_bytes(
        repo_root
    )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".tmp.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(archive_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    archive_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "archive": str(output),
        "archive_sha256": archive_sha,
        "file_count": len(stage_manifest_value["files"]),
        "stage_manifest_sha256": _sha256(manifest_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_stage(args.repo_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
