#!/usr/bin/env python3
"""Remove workstation prefixes from an upstream source manifest.

File bytes and SHA-256 values are unchanged. The upstream source-closure hash
is retained as an external binding, but is not claimed to be the content hash
of the path-redacted release copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


MARKERS = ("evidence_view_benchmark_v1", "experiments", "src", "cases")


def sanitized_path(raw: str) -> str:
    path = PurePosixPath(raw)
    parts = path.parts
    if not path.is_absolute():
        return raw
    for marker in MARKERS:
        if marker in parts:
            index = parts.index(marker)
            return str(PurePosixPath(*parts[index:]))
    return f"<redacted-project-root>/{path.name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for entry in payload.get("files", []):
        entry["path"] = sanitized_path(str(entry["path"]))
    payload["release_redaction"] = {
        "absolute_workstation_prefixes_removed": True,
        "file_bytes_and_sha256_unchanged": True,
        "source_closure_sha256_is_the_upstream_unredacted_closure": True
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
