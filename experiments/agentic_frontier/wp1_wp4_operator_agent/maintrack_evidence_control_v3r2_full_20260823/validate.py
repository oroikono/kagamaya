"""Producer-free shard validation and semantic mutation checks."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .contract import (CALIBRATION_SEEDS, CONFIRMATION_SEEDS, NOISE_CELLS,
                       ORIGINS, TRAIN_SEEDS, VIEW_BANK)
from .owner_truth import OWNER_CASES, OWNER_CASE_BY_ID


FREEZE = Path(__file__).resolve().parent / "PREOUTCOME_FREEZE.json"


def _canonical_digest(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("content_sha256", None)
    return hashlib.sha256(json.dumps(
        clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_payload(payload: dict) -> dict:
    errors: list[str] = []
    if payload.get("schema_version") != "maintrack.evidence-control.shard.v3r2":
        errors.append("schema version mismatch")
    if payload.get("content_sha256") != _canonical_digest(payload):
        errors.append("semantic digest mismatch")
    metadata = payload.get("metadata", {})
    split = metadata.get("split")
    seed = metadata.get("seed")
    origin = metadata.get("origin")
    split_seeds = {"train": TRAIN_SEEDS, "calibration": CALIBRATION_SEEDS,
                   "confirmation": CONFIRMATION_SEEDS}
    if split not in split_seeds or seed not in split_seeds.get(split, ()):
        errors.append("split/seed contract mismatch")
    if origin not in ORIGINS:
        errors.append("origin contract mismatch")
    frozen_closure = json.loads(FREEZE.read_text()).get("source_closure") if FREEZE.exists() else None
    if not isinstance(metadata.get("source_closure"), str) or not metadata.get("source_closure"):
        errors.append("source closure missing")
    elif frozen_closure is not None and metadata.get("source_closure") != frozen_closure:
        errors.append("source closure mismatch")
    auth_sha = metadata.get("confirmation_authorization_sha256")
    if (split == "confirmation") != isinstance(auth_sha, str):
        errors.append("confirmation authorization metadata")
    if metadata.get("view_count") != len(VIEW_BANK):
        errors.append("metadata view count mismatch")
    if metadata.get("noise_cells") != [list(cell) for cell in NOISE_CELLS]:
        errors.append("metadata noise cross mismatch")
    episodes = payload.get("episodes", [])
    labels = payload.get("owner_labels", [])
    expected = len(OWNER_CASES) * len(NOISE_CELLS)
    if len(episodes) != expected or len(labels) != expected:
        errors.append("episode/owner count mismatch")
    expected_views = [view.view_id for view in VIEW_BANK]
    ids = []
    noise_counts = {cell: 0 for cell in NOISE_CELLS}
    for episode in episodes:
        ids.append(episode.get("episode_id"))
        cell = (episode.get("noise_model"), episode.get("noise_level"))
        if cell not in noise_counts:
            errors.append("unknown noise cell")
        else:
            noise_counts[cell] += 1
        if episode.get("origin") != origin:
            errors.append("episode origin mismatch")
        views = episode.get("views", [])
        if [row.get("view_id") for row in views] != expected_views:
            errors.append("view cross/order mismatch")
        if len({row.get("view_id") for row in views}) != len(VIEW_BANK):
            errors.append("duplicate view")
        if any(row.get("episode_id") != episode.get("episode_id") for row in views):
            errors.append("cross-episode record")
        for row, spec in zip(views, VIEW_BANK):
            if row.get("status") not in {"IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE"}:
                errors.append("unknown terminal status")
            if row.get("registered_cost") != spec.registered_cost:
                errors.append("registered cost mismatch")
    if any(count != len(OWNER_CASES) for count in noise_counts.values()):
        errors.append("noise factorial count mismatch")
    if len(ids) != len(set(ids)):
        errors.append("duplicate episode nonce")
    if {row.get("episode_id") for row in labels} != set(ids):
        errors.append("owner/public join mismatch")
    case_counts = {case.case_id: 0 for case in OWNER_CASES}
    for row in labels:
        case = OWNER_CASE_BY_ID.get(row.get("owner_case_id"))
        if case is None:
            errors.append("unknown owner case")
            continue
        case_counts[case.case_id] += 1
        expected_owner = {
            "family": case.family, "truth_kind": case.truth_kind,
            "truth_support": list(case.truth_support),
            "omitted_mechanism": case.omitted_mechanism,
        }
        if any(row.get(key) != value for key, value in expected_owner.items()):
            errors.append("owner truth mismatch")
    if any(count != len(NOISE_CELLS) for count in case_counts.values()):
        errors.append("owner case factorial mismatch")
    return {"valid": not errors, "errors": sorted(set(errors)),
            "episodes": len(episodes), "views": len(episodes) * len(VIEW_BANK)}


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_payload(load(args.artifact))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
