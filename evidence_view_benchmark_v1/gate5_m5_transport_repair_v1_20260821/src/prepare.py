"""Create the outcome-blind prompt inventory, bridge selection, and source seal."""

from __future__ import annotations

import json
from pathlib import Path

from .contract import (
    BENCHMARK_ROOT,
    LANE_ROOT,
    PARENT_BINDINGS,
    bridge_case_ids,
    canonical_json,
    digest,
    file_sha256,
    prompt_inventory,
    source_paths,
    verify_parent_bindings,
)


def write_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    verify_parent_bindings()
    inventory = prompt_inventory()
    inventory_path = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    write_exclusive(inventory_path, "".join(canonical_json(row) + "\n" for row in inventory))
    selection = {
        "schema_version": "evidence_view_benchmark.m5_bridge_selection.v1",
        "selection_rule": "first 24 case_ids after sorting by SHA256(case_id)",
        "case_ids": list(bridge_case_ids()),
    }
    selection["content_sha256"] = digest(selection)
    write_exclusive(LANE_ROOT / "CASE_SELECTION.json", canonical_json(selection) + "\n")
    rows = [
        {"path": str(path.relative_to(LANE_ROOT)), "sha256": file_sha256(path)}
        for path in source_paths()
    ]
    parent = [
        {"path": relative, "sha256": expected}
        for relative, expected in sorted(PARENT_BINDINGS.items())
    ]
    manifest = {
        "schema_version": "evidence_view_benchmark.m5_repair_source_manifest.v1",
        "files": rows,
        "parent_bindings": parent,
        "prompt_inventory_sha256": file_sha256(inventory_path),
        "case_selection_sha256": file_sha256(LANE_ROOT / "CASE_SELECTION.json"),
    }
    manifest["source_closure_sha256"] = digest(manifest)
    write_exclusive(LANE_ROOT / "SOURCE_MANIFEST.json", canonical_json(manifest) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
