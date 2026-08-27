#!/usr/bin/env python3
"""Seal Gate-1 source, generate M0/M1 artifacts, and run read-only replay."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


LANE_ROOT = Path(__file__).resolve().parents[1]
SRC = LANE_ROOT / "src"
sys.path.insert(0, str(SRC))

from evidence_view_benchmark.canonical import digest, file_sha256, publish_json  # noqa: E402
from evidence_view_benchmark.case_sampler import build  # noqa: E402
from evidence_view_benchmark.replay import run as replay  # noqa: E402


def source_manifest() -> dict:
    roots = [
        LANE_ROOT / "GATE1_DESIGN_FREEZE.md",
        LANE_ROOT / "schema/token_migration.json",
        *sorted((LANE_ROOT / "prompts").glob("*.txt")),
        *sorted((LANE_ROOT / "src").rglob("*.py")),
        *sorted((LANE_ROOT / "tests").glob("*.py")),
        LANE_ROOT / "tools/build_gate1.py",
        LANE_ROOT / "tools/validate_gate1.py",
    ]
    rows = []
    for path in roots:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT)),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "evidence_view_benchmark.gate1_source_manifest.v1",
        "files": rows,
        "closure_sha256": digest(rows),
    }


def main() -> None:
    manifest = source_manifest()
    publish_json(LANE_ROOT / "SOURCE_MANIFEST.json", manifest)
    dataset = build()
    replay_report = replay()
    print(manifest["closure_sha256"])
    print(dataset["cases_file_sha256"])
    print(replay_report["status"])


if __name__ == "__main__":
    main()

