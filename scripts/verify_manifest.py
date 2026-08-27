#!/usr/bin/env python3
"""Verify every file in the release manifest without external dependencies."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "reproduction"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = [line.rstrip("\n") for line in MANIFEST.read_text().splitlines() if line.strip()]
    failures = []
    expected_paths = set()
    for row in rows:
        expected, relative = row.split("  ", 1)
        if relative in expected_paths:
            failures.append(f"duplicate manifest entry: {relative}")
            continue
        expected_paths.add(relative)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif digest(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
    }
    for relative in sorted(actual_paths - expected_paths):
        failures.append(f"unmanifested: {relative}")
    for relative in sorted(expected_paths - actual_paths):
        if (ROOT / relative).is_file():
            failures.append(f"excluded but manifested: {relative}")
    if failures:
        raise SystemExit("MANIFEST_FAIL\n" + "\n".join(failures))
    print(f"MANIFEST_PASS files={len(rows)}")


if __name__ == "__main__":
    main()
