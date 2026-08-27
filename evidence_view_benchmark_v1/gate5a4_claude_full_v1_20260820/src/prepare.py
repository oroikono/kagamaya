"""Prepare immutable prompt inventory and source closure before provider access."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .contract import LANE_ROOT, canonical_json, digest, file_sha256, prompt_inventory, read_authorization, source_paths


def exclusive_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    read_authorization()
    inventory = prompt_inventory()
    inventory_path = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    exclusive_text(inventory_path, "".join(canonical_json(row) + "\n" for row in inventory))
    rows = []
    for path in source_paths() + [inventory_path]:
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT.parent) if LANE_ROOT.parent in path.parents else path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema_version": "evidence_view_benchmark.claude_full_source_manifest.v1",
        "files": rows,
        "prompt_count": len(inventory),
        "prompt_inventory_sha256": file_sha256(inventory_path),
    }
    manifest["source_closure_sha256"] = digest(rows)
    exclusive_text(LANE_ROOT / "SOURCE_MANIFEST.json", canonical_json(manifest) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
