"""Build the deterministic post-freeze scientific source archive.

The archive is assembled only from the exact source and dependency manifests
inside a verified v1r7 freeze. No directory is added recursively. Stage
authorization remains a separate content-addressed runtime input.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
from typing import Any, Mapping

from .. import freeze


SCIENTIFIC_STAGE_MANIFEST_PATH = "V1R7_SCIENTIFIC_STAGE_MANIFEST.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_file_bytes(repo_root: Path, relative: str) -> bytes:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise ValueError(f"unsafe scientific stage path: {relative}")
    path = repo_root / relative
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            f"scientific stage input is not a regular non-symlink file: {relative}"
        )
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(
            f"scientific stage input escapes repository root: {relative}"
        ) from exc
    return resolved.read_bytes()


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


def _validate_frozen_manifest_shape(payload: Mapping[str, Any]) -> None:
    if set(payload) != freeze.FREEZE_ROOT_KEYS:
        raise ValueError("scientific stage freeze root-key drift")
    if freeze._semantic_errors(payload):
        raise ValueError("scientific stage freeze semantic drift")
    source_manifest = payload.get("source_manifest")
    dependency_manifest = payload.get("dependency_manifest")
    if not isinstance(source_manifest, dict) or set(source_manifest) != set(
        freeze.source_files()
    ):
        raise ValueError("scientific stage source manifest keyset drift")
    if not isinstance(dependency_manifest, dict) or set(dependency_manifest) != set(
        freeze.DEPENDENCY_RELATIVE_FILES
    ):
        raise ValueError("scientific stage dependency manifest keyset drift")
    if payload.get("source_closure_sha256") != freeze.content_digest(source_manifest):
        raise ValueError("scientific stage source closure drift")
    if payload.get("dependency_closure_sha256") != freeze.content_digest(
        dependency_manifest
    ):
        raise ValueError("scientific stage dependency closure drift")
    body = dict(payload)
    claimed = body.pop("freeze_digest", None)
    if claimed != freeze.json_digest(body):
        raise ValueError("scientific stage freeze digest drift")


def collect_scientific_stage_files(
    repo_root: Path, payload: Mapping[str, Any]
) -> dict[str, bytes]:
    repo = repo_root.resolve(strict=True)
    _validate_frozen_manifest_shape(payload)
    lane = freeze.LANE_REPO_RELATIVE.as_posix()
    files: dict[str, bytes] = {}
    source_manifest = payload["source_manifest"]
    for relative in sorted(source_manifest):
        archive_name = f"{lane}/{relative}"
        data = _regular_file_bytes(repo, archive_name)
        if _sha256(data) != source_manifest[relative]:
            raise ValueError(f"scientific stage source hash drift: {relative}")
        files[archive_name] = data
    dependency_manifest = payload["dependency_manifest"]
    for logical_name in sorted(dependency_manifest):
        relative = freeze.DEPENDENCY_RELATIVE_FILES[logical_name]
        data = _regular_file_bytes(repo, relative)
        if _sha256(data) != dependency_manifest[logical_name]:
            raise ValueError(
                f"scientific stage dependency hash drift: {logical_name}"
            )
        if relative in files and files[relative] != data:
            raise ValueError(f"scientific stage path collision: {relative}")
        files[relative] = data
    freeze_name = f"{lane}/PREOUTCOME_FREEZE.json"
    freeze_data = _regular_file_bytes(repo, freeze_name)
    if json.loads(freeze_data) != payload:
        raise ValueError("scientific stage freeze file differs from payload")
    files[freeze_name] = freeze_data
    forbidden_parts = {
        "__pycache__",
        ".pytest_cache",
        "artifacts",
        "results",
        "governance_preexecution_v1r7_candidate",
        "upstream_archival_governance",
    }
    for name in files:
        parts = set(PurePosixPath(name).parts)
        if parts & forbidden_parts or "/data/test/" in f"/{name}/":
            raise ValueError(f"forbidden scientific stage member: {name}")
    return dict(sorted(files.items()))


def scientific_stage_archive_bytes(
    repo_root: Path, payload: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    files = collect_scientific_stage_files(repo_root, payload)
    manifest: dict[str, Any] = {
        "schema_version": "maintrack.scientific-stage.v1r7",
        "freeze_digest": payload["freeze_digest"],
        "source_closure_sha256": payload["source_closure_sha256"],
        "dependency_closure_sha256": payload["dependency_closure_sha256"],
        "authorization_included": False,
        "authorization_contract": (
            "supply one separately validated content-addressed stage authorization"
        ),
        "publisher_test_data_included": False,
        "recursive_directory_membership": False,
        "files": {name: _sha256(data) for name, data in files.items()},
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    stream = BytesIO()
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with tarfile.open(
            fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
        ) as archive:
            for name, data in files.items():
                _add_bytes(archive, name, data)
            _add_bytes(archive, SCIENTIFIC_STAGE_MANIFEST_PATH, manifest_bytes)
    return stream.getvalue(), manifest


def build_stage(repo_root: Path, output: Path) -> dict[str, Any]:
    repo = repo_root.resolve(strict=True)
    lane_root = repo / freeze.LANE_REPO_RELATIVE.as_posix()
    verification = freeze.verify(root=lane_root, repo=repo)
    if not verification["valid"]:
        raise RuntimeError(
            "cannot stage an invalid freeze: "
            + ",".join(str(item) for item in verification["errors"])
        )
    payload = json.loads((lane_root / "PREOUTCOME_FREEZE.json").read_text())
    archive_bytes, manifest = scientific_stage_archive_bytes(repo, payload)
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
    return {
        "archive": str(output),
        "archive_sha256": _sha256(archive_bytes),
        "file_count": len(manifest["files"]),
        "stage_manifest_sha256": _sha256(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_stage(args.repo_root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
