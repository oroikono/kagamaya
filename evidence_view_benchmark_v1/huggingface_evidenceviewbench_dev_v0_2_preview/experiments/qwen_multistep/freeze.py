"""Create or validate the immutable v2 source closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contract import LANE_ROOT, canonical_json, digest, file_sha256, source_paths


MANIFEST = LANE_ROOT / "SOURCE_MANIFEST.json"


def expected() -> dict:
    rows = []
    for path in source_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    body = {
        "schema_version": "evidence_view_benchmark.metadata_blind_source_freeze.v2",
        "files": rows,
    }
    body["source_closure_sha256"] = digest(body)
    return body


def create() -> dict:
    value = expected()
    with MANIFEST.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
    return value


def validate() -> dict:
    actual = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wanted = expected()
    if actual != wanted:
        raise ValueError("v2 source closure mismatch")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "validate"))
    args = parser.parse_args()
    value = create() if args.mode == "create" else validate()
    print(json.dumps({"files": len(value["files"]), "source_closure_sha256": value["source_closure_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

