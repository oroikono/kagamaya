"""Create the full-run R2 closure while preserving the R1 prompt inventory."""

from __future__ import annotations

import json
import os

from .contract import LANE_ROOT, canonical_json, digest, file_sha256, source_paths


def main() -> None:
    inventory = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    r1 = json.loads((LANE_ROOT / "SOURCE_MANIFEST.json").read_text())
    if file_sha256(inventory) != r1["prompt_inventory_sha256"]:
        raise ValueError("R1 prompt inventory changed")
    rows = []
    for path in source_paths() + [inventory]:
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT.parent)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    value = {
        "schema_version": "evidence_view_benchmark.claude_full_source_manifest.r2",
        "supersedes_for_full_execution": r1["source_closure_sha256"],
        "transport_smoke_raw_sha256": file_sha256(LANE_ROOT / "results/transport_smoke.jsonl"),
        "prompt_inventory_sha256": r1["prompt_inventory_sha256"],
        "files": rows,
        "source_closure_sha256": digest(rows),
    }
    path = LANE_ROOT / "SOURCE_MANIFEST_FULL_R2.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
