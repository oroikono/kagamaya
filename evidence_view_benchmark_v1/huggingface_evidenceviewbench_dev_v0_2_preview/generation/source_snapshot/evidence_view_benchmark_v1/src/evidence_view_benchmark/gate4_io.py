"""Append-only raw-record publication and Gate-4 manifest helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return payload + (b"\n" if newline else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def publish_exclusive(path: Path, value: Any) -> None:
    """Create one artifact once; an existing path is always an error."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value, newline=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_manifest(paths: Iterable[Path], *, root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        payload = path.read_bytes()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    return rows
