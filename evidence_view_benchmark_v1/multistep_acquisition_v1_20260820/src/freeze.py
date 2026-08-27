"""Create or validate the one-time source closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import LANE_ROOT, canonical_json, digest, file_sha256, source_paths


MANIFEST_PATH = LANE_ROOT / "SOURCE_MANIFEST.json"


def build() -> dict[str, Any]:
    files = []
    for path in source_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    body = {
        "schema_version": "evidence_view_benchmark.multistep_source_freeze.v1",
        "files": files,
    }
    body["source_closure_sha256"] = digest(body)
    return body


def validate(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = build()
    if value != expected:
        raise ValueError("source closure differs from frozen manifest")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "validate"))
    args = parser.parse_args()
    if args.command == "create":
        value = build()
        with MANIFEST_PATH.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(value) + "\n")
    else:
        value = validate()
    print(json.dumps({"files": len(value["files"]), "source_closure_sha256": value["source_closure_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

