"""Producer-free structural validator for compact campaign artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .contract import TERMINAL_STATUSES, VIEW_BANK


def validate_artifact(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw)
    errors: list[str] = []
    if payload.get("schema_version") != "maintrack.2d-coupled-bank.v2":
        errors.append("schema_version")
    view_ids = [view.view_id for view in VIEW_BANK]
    episodes = payload.get("episodes", [])
    labels = {item["episode_id"]: item for item in payload.get("owner_labels", [])}
    seen: set[str] = set()
    for episode in episodes:
        episode_id = episode.get("episode_id")
        if episode_id in seen:
            errors.append(f"duplicate_episode:{episode_id}")
        seen.add(episode_id)
        if episode_id not in labels:
            errors.append(f"missing_label:{episode_id}")
        views = episode.get("views", [])
        if [item.get("view_id") for item in views] != view_ids:
            errors.append(f"view_manifest:{episode_id}")
        for record in views:
            if record.get("episode_id") != episode_id:
                errors.append(f"cross_episode_record:{episode_id}")
            if record.get("status") not in TERMINAL_STATUSES:
                errors.append(f"terminal_status:{episode_id}")
            if record.get("status") != "IDENTIFIED" and record.get("support"):
                errors.append(f"nonidentified_support:{episode_id}")
            for key in ("residual", "condition", "minimum_coefficient",
                        "registered_cost"):
                value = record.get(key)
                if value is None or not isinstance(value, (int, float)):
                    errors.append(f"numeric:{episode_id}:{key}")
    if set(labels) != seen:
        errors.append("label_episode_set")
    expected = (len(payload.get("metadata", {}).get("seeds", []))
                * len(payload.get("metadata", {}).get("origins", []))
                * len(payload.get("metadata", {}).get("case_ids", []))
                * len(payload.get("metadata", {}).get("noise_cells", [])))
    if expected != len(episodes):
        errors.append(f"episode_count:{len(episodes)}!={expected}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "valid": not errors,
        "errors": errors,
        "episode_count": len(episodes),
        "view_record_count": len(episodes) * len(VIEW_BANK),
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    result = validate_artifact(args.artifact)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

