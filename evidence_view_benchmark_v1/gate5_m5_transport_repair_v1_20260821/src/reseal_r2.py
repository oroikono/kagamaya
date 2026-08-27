"""Write the immutable R2 source receipt without touching the R1 receipt."""

from __future__ import annotations

import json

from .contract import (
    LANE_ROOT,
    PARENT_BINDINGS,
    SOURCE_MANIFEST_PATH,
    canonical_json,
    digest,
    file_sha256,
    prompt_inventory,
    source_paths,
    verify_parent_bindings,
)


def main() -> None:
    verify_parent_bindings()
    inventory_path = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    observed = [json.loads(line) for line in inventory_path.read_text().splitlines() if line]
    if canonical_json(observed) != canonical_json(prompt_inventory()):
        raise ValueError("prompt inventory replay mismatch")
    manifest = {
        "schema_version": "evidence_view_benchmark.m5_repair_source_manifest.v2",
        "files": [
            {"path": str(path.relative_to(LANE_ROOT)), "sha256": file_sha256(path)}
            for path in source_paths()
        ],
        "parent_bindings": [
            {"path": relative, "sha256": expected}
            for relative, expected in sorted(PARENT_BINDINGS.items())
        ],
        "prompt_inventory_sha256": file_sha256(inventory_path),
        "case_selection_sha256": file_sha256(LANE_ROOT / "CASE_SELECTION.json"),
        "supersedes_source_manifest_sha256": file_sha256(LANE_ROOT / "SOURCE_MANIFEST.json"),
    }
    manifest["source_closure_sha256"] = digest(manifest)
    with SOURCE_MANIFEST_PATH.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(manifest) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
