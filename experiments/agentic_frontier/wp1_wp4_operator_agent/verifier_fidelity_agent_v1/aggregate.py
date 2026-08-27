"""Aggregate independently validated single-seed confirmation shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import CONFIRMATION_SEEDS, _summarize, design_sha, digest, source_sha
from .validate import validate


def aggregate(shard_dir: Path) -> dict:
    shards = []
    for path in sorted(shard_dir.glob("*.json")):
        report = validate(path)
        payload = json.loads(path.read_text())
        if payload["config"]["split"] != "confirmation":
            raise ValueError(f"non-confirmation shard: {path}")
        shards.append((path, payload, report))
    if len(shards) != 100:
        raise ValueError(f"expected 100 shards, found {len(shards)}")
    grouped: dict[str, list[dict]] = {"spectral_etdrk4": [], "fd_bdf": []}
    seen = set()
    manifest = []
    for path, payload, report in shards:
        producer = payload["config"]["producer"]
        seeds = payload["config"]["seeds"]
        if len(seeds) != 1: raise ValueError("each shard must contain one seed")
        key = (producer, seeds[0])
        if key in seen: raise ValueError(f"duplicate shard {key}")
        seen.add(key); grouped[producer].extend(payload["rows"])
        manifest.append({"file": path.name, "producer": producer, "seed": seeds[0],
                         "content_sha256": report["sha"]})
    expected = {(producer, seed) for producer in grouped for seed in CONFIRMATION_SEEDS}
    if seen != expected: raise ValueError("producer/seed coverage mismatch")
    summaries = {producer: _summarize(rows) for producer, rows in grouped.items()}
    positive = all(summaries[p]["modes_5"]["positive_gate"] for p in summaries)
    payload = {
        "status": "CONFIRMATION_PASS" if positive else "CONFIRMATION_FAIL",
        "design_freeze_sha256": design_sha(), "source_closure_sha256": source_sha(),
        "confirmation_seeds": list(CONFIRMATION_SEEDS), "shards": manifest,
        "producer_summaries": summaries, "positive_gate": positive,
    }
    payload["content_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    result = aggregate(args.shard_dir); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "positive_gate": result["positive_gate"],
                      "sha": result["content_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
