"""Read-only replay/round-trip audit over every inventoried predecessor artifact."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import digest, publish_json
from .migration import load_migration


LANE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = LANE_ROOT.parent
PREDECESSOR = (
    REPO_ROOT
    / "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1"
)
INVENTORY = REPO_ROOT / "GATE0_STATUS_TOKEN_INVENTORY.tsv"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    ).encode("utf-8")


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_exact(value: Any, tokens: set[str]) -> int:
    if isinstance(value, Mapping):
        return sum(_count_exact(key, tokens) + _count_exact(item, tokens) for key, item in value.items())
    if isinstance(value, list):
        return sum(_count_exact(item, tokens) for item in value)
    return int(isinstance(value, str) and value in tokens)


def _semantic_rows(value: Any, current_terminal: set[str]) -> int:
    count = 0
    if isinstance(value, Mapping):
        if "public_status" in value and "public_terminal" in value:
            if bool(value["public_status"] in current_terminal) != bool(value["public_terminal"]):
                raise ValueError("terminality changed under migration")
            count += 1
        count += sum(_semantic_rows(item, current_terminal) for item in value.values())
    elif isinstance(value, list):
        count += sum(_semantic_rows(item, current_terminal) for item in value)
    return count


def run() -> dict[str, Any]:
    migration = load_migration(LANE_ROOT / "schema/token_migration.json")
    before = {}
    parsed = 0
    roundtripped = 0
    migrated_exact_tokens = 0
    semantic_rows = 0
    categories: dict[str, int] = {}
    with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        inventory_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(inventory_rows) != 197:
        raise ValueError("Gate 0 inventory size changed")
    legacy_tokens = set(migration.authority_forward) | set(migration.execution_forward)
    for row in inventory_rows:
        category = row["category"]
        categories[category] = categories.get(category, 0) + 1
        path = PREDECESSOR / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        before[str(row["path"])] = hashlib.sha256(path.read_bytes()).hexdigest()
        if category != "stored_result":
            continue
        value = _read_json(path)
        migrated = migration.migrate(value)
        restored = migration.restore_legacy(migrated)
        if _json_bytes(restored) != _json_bytes(value):
            raise ValueError(f"legacy round-trip changed {path}")
        parsed += 1
        roundtripped += 1
        migrated_exact_tokens += _count_exact(value, legacy_tokens)
        semantic_rows += _semantic_rows(migrated, set(migration.current_terminal_statuses))
    after = {
        row["path"]: hashlib.sha256((PREDECESSOR / row["path"]).read_bytes()).hexdigest()
        for row in inventory_rows
    }
    if before != after:
        raise ValueError("a predecessor artifact changed during replay")
    report = {
        "schema_version": "evidence_view_benchmark.legacy_replay.v1",
        "inventory_sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        "inventoried_files": len(inventory_rows),
        "categories": categories,
        "stored_json_artifacts_parsed": parsed,
        "stored_json_artifacts_roundtripped": roundtripped,
        "exact_legacy_tokens_migrated": migrated_exact_tokens,
        "scored_rows_terminality_replayed": semantic_rows,
        "all_file_hashes_unchanged": before == after,
        "artifact_hash_set_sha256": digest(before),
        "status": "PASS",
    }
    publish_json(LANE_ROOT / "results/legacy_replay_report.json", report)
    return report

