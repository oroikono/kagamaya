#!/usr/bin/env python3
"""Build the public/owner split for the 2-D coupled design preview.

The upstream scientific artifacts remain immutable.  This utility copies only
selector-visible episode/view records into the public table and writes owner
labels and convergence diagnostics to separate evaluator-side tables.  The
release is development-only: owner tables are downloadable and therefore are
not a hidden test set.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_payload(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return sha256_bytes(path.read_bytes())


def build(upstream: Path, output: Path) -> dict[str, Any]:
    public_rows: list[dict[str, Any]] = []
    owner_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []

    for origin_name, filename in (("spectral", "spectral.json.gz"), ("fd", "fd.json.gz")):
        source = upstream / filename
        payload = read_payload(source)
        source_files.append({
            "origin_component": origin_name,
            "filename": filename,
            "file_sha256": sha256_bytes(source.read_bytes()),
            "schema_version": payload["schema_version"],
        })
        for episode in payload["episodes"]:
            row = {
                "episode_id": episode["episode_id"],
                "origin_component": origin_name,
                "noise_model": episode["noise_model"],
                "noise_level": episode["noise_level"],
                "views": episode["views"],
            }
            row["public_row_sha256"] = sha256_bytes(canonical_bytes(row))
            public_rows.append(row)
        for label in payload["owner_labels"]:
            row = {"origin_component": origin_name, **label}
            row["owner_row_sha256"] = sha256_bytes(canonical_bytes(row))
            owner_rows.append(row)
        for item in payload["convergence"]:
            row = {"origin_component": origin_name, **item}
            row["convergence_row_sha256"] = sha256_bytes(canonical_bytes(row))
            convergence_rows.append(row)

    public_rows.sort(key=lambda row: (row["origin_component"], row["episode_id"]))
    owner_rows.sort(key=lambda row: (row["origin_component"], row["episode_id"]))
    convergence_rows.sort(key=lambda row: (row["origin_component"], row["case_id"]))

    data_dir = output / "data"
    public_path = data_dir / "two_d_design_public.jsonl.gz"
    owner_path = data_dir / "two_d_design_owner.jsonl.gz"
    convergence_path = data_dir / "two_d_design_convergence.jsonl.gz"
    file_hashes = {
        public_path.name: write_jsonl_gz(public_path, public_rows),
        owner_path.name: write_jsonl_gz(owner_path, owner_rows),
        convergence_path.name: write_jsonl_gz(convergence_path, convergence_rows),
    }
    resources_dir = output / "evidenceviewbench" / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(public_path, resources_dir / public_path.name)
    shutil.copyfile(owner_path, resources_dir / owner_path.name)
    manifest = {
        "schema_version": "evidenceviewbench.2d-preview-manifest.v1",
        "release_status": "private_development_preview_not_hidden_test",
        "public_rows": len(public_rows),
        "owner_rows": len(owner_rows),
        "convergence_rows": len(convergence_rows),
        "views_per_public_row": 72,
        "factorial_rows_per_origin": 90,
        "primary_unique_observation_rows_per_origin": 75,
        "duplicate_rule": "correlated noise at level 0 duplicates iid noise at level 0 and is excluded from primary analysis",
        "source_files": source_files,
        "output_file_sha256": file_hashes,
        "claim_ceiling": [
            "one design seed only",
            "synthetic 2-D periodic manufactured systems",
            "two numerical producer components share analytic owner and forcing",
            "owner labels are public development labels, not a hidden authority",
            "no adaptive-policy advantage was demonstrated",
        ],
    }
    manifest_path = data_dir / "two_d_design_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.upstream, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
