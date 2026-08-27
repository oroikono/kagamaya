"""Independent structural validator for fidelity-agent artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import (
    CONFIRMATION_SEEDS, NESTED_NODES, POLICIES, SMOKE_SEEDS, TEMPORAL_MODES,
    design_sha, digest, select, signature, source_sha,
)


def validate(path: Path) -> dict:
    payload = json.loads(path.read_text())
    claimed = payload["content_sha256"]
    bare = {key: value for key, value in payload.items() if key != "content_sha256"}
    if claimed != digest(bare): raise ValueError("content hash mismatch")
    if payload["design_freeze_sha256"] != design_sha(): raise ValueError("design hash mismatch")
    if payload["source_closure_sha256"] != source_sha(): raise ValueError("source hash mismatch")
    split = payload["config"]["split"]
    allowed = set(CONFIRMATION_SEEDS if split == "confirmation" else SMOKE_SEEDS)
    seeds = set(payload["config"]["seeds"])
    if not seeds <= allowed: raise ValueError("reserved seed violation")
    expected_episodes = 6 * len(seeds) * 2 * 3
    expected_rows = expected_episodes * len(TEMPORAL_MODES) * len(NESTED_NODES)
    if len(payload["rows"]) != expected_rows: raise ValueError("row count mismatch")
    cells = set()
    for row in payload["rows"]:
        key = (tuple(row["episode_key"]), row["temporal_modes"], row["nodes"])
        if key in cells: raise ValueError("duplicate cell")
        cells.add(key)
    by_episode = {}
    for row in payload["rows"]:
        if row["temporal_modes"] == 5:
            by_episode.setdefault(tuple(row["episode_key"]), {})[row["nodes"]] = row
    if len(by_episode) != expected_episodes: raise ValueError("episode count mismatch")
    for episode_cells in by_episode.values():
        if set(episode_cells) != set(NESTED_NODES): raise ValueError("incomplete ladder")
        mutated = {n: {**r, "truth_support": ["MUTATED"]} for n, r in episode_cells.items()}
        for policy in POLICIES:
            if (select(episode_cells, policy)["nodes"], signature(select(episode_cells, policy))) != (select(mutated, policy)["nodes"], signature(select(mutated, policy))):
                raise ValueError("truth mutation changed policy")
    recomputed = payload["summary"]["modes_5"]["policies"]
    for policy in POLICIES:
        chosen = [select(cells, policy) for cells in by_episode.values()]
        if sum(row["nodes"] for row in chosen) / len(chosen) != recomputed[policy]["mean_nodes"]:
            raise ValueError("summary mismatch")
    return {"rows": expected_rows, "episodes": expected_episodes,
            "hidden_mutation_invariant": True, "positive_gate": payload["positive_gate"],
            "sha": claimed}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path)
    args = parser.parse_args(); print(json.dumps(validate(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__": main()
