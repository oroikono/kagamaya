#!/usr/bin/env python3
"""Verify every file in the release manifest without external dependencies."""
from __future__ import annotations

from pathlib import Path

if __package__:
    from .manifest_common import digest, inventory, parse_manifest, regular_file
else:
    from manifest_common import digest, inventory, parse_manifest, regular_file

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"


def main() -> None:
    failures = []
    try:
        manifest = regular_file(ROOT, MANIFEST.relative_to(ROOT).as_posix())
        rows = parse_manifest(manifest.read_text(encoding="utf-8"))
        # Validate the entire inventory and every entry before hashing any
        # payload. Malformed rows cannot trigger reads outside the root.
        actual = inventory(ROOT, MANIFEST.relative_to(ROOT).as_posix())
        expected_files = {relative: regular_file(ROOT, relative) for relative in rows}
        for relative in sorted(actual.keys() - rows.keys()):
            failures.append(f"unmanifested: {relative}")
        for relative in sorted(rows.keys() - actual.keys()):
            failures.append(f"excluded but manifested: {relative}")
        if not failures:
            for relative, path in expected_files.items():
                if digest(path) != rows[relative]:
                    failures.append(f"hash mismatch: {relative}")
    except (OSError, ValueError) as error:
        failures.append(str(error))
    if failures:
        raise SystemExit("MANIFEST_FAIL\n" + "\n".join(failures))
    print(f"MANIFEST_PASS files={len(rows)}")


if __name__ == "__main__":
    main()
