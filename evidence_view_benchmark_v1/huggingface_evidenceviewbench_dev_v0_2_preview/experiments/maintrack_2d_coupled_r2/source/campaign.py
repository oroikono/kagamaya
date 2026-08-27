"""Outcome-isolated campaign runner and compact artifact writer."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .authority import AuthorityRecord, evaluate
from .contract import (
    CALIBRATION_SEEDS,
    CONFIRMATION_SEEDS,
    DESIGN_SEEDS,
    DEVELOPMENT_SEEDS,
    VIEW_BANK,
)
from .environment import OwnerLabel, PublicEpisode
from .owner_truth import OWNER_CASES
from .trajectory import Trajectory
from .weak_evidence import build_weak_evidence, observed_arrays


NOISE_CELLS = (
    ("iid", 0.0), ("iid", 0.005), ("iid", 0.02),
    ("correlated", 0.0), ("correlated", 0.005), ("correlated", 0.02),
)

SPLITS = {
    "design": DESIGN_SEEDS,
    "calibration": CALIBRATION_SEEDS,
    "development": DEVELOPMENT_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
}


def _noise_seed(case_id: str, seed: int, origin: str,
                noise_model: str, noise_level: float) -> int:
    payload = f"{case_id}|{seed}|{origin}|{noise_model}|{noise_level:.9g}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _public_episode_id(trajectory: Trajectory, noise_model: str, noise_level: float) -> str:
    payload = f"{trajectory.episode_id}|{noise_model}|{noise_level:.9g}|public-v2".encode()
    return "P-" + hashlib.sha256(payload).hexdigest()[:20]


def simulate_trajectory(case_id: str, seed: int, origin: str, *,
                        base_nodes: int, snapshot_count: int) -> Trajectory:
    if origin == "spectral":
        from .numerical_spectral import simulate
        return simulate(case_id, seed, n=base_nodes, snapshot_count=snapshot_count,
                        internal_steps=480)
    if origin == "fd":
        from .numerical_fd import simulate
        return simulate(case_id, seed, n=base_nodes, snapshot_count=snapshot_count,
                        internal_steps=720)
    raise ValueError("origin must be spectral or fd")


def build_public_episode(trajectory: Trajectory, *, noise_model: str,
                         noise_level: float) -> tuple[PublicEpisode, OwnerLabel]:
    case = next(case for case in OWNER_CASES if case.case_id == trajectory.case_id)
    noise_seed = _noise_seed(case.case_id, trajectory.seed, trajectory.origin,
                             noise_model, noise_level)
    observed_u, observed_v = observed_arrays(
        trajectory, noise_level=noise_level, noise_model=noise_model,
        noise_seed=noise_seed,
    )
    episode_id = _public_episode_id(trajectory, noise_model, noise_level)
    records: dict[str, AuthorityRecord] = {}
    for view in VIEW_BANK:
        evidence = build_weak_evidence(
            trajectory, view, observed_u=observed_u, observed_v=observed_v,
            episode_id=episode_id,
        )
        records[view.view_id] = evaluate(evidence)
    episode = PublicEpisode(
        episode_id=episode_id, origin=trajectory.origin,
        noise_model=noise_model, noise_level=noise_level,
        view_records=records,
    )
    episode.validate()
    label = OwnerLabel(
        episode_id=episode_id, family=case.family, truth_kind=case.truth_kind,
        truth_support=case.truth_support, omitted_mechanism=case.omitted_mechanism,
    )
    return episode, label


def run_bank(*, seeds: tuple[int, ...], origins: tuple[str, ...],
             case_ids: tuple[str, ...], noise_cells=NOISE_CELLS,
             base_nodes: int = 72, snapshot_count: int = 49,
             progress=None) -> tuple[list[PublicEpisode], dict[str, OwnerLabel], list[dict]]:
    episodes: list[PublicEpisode] = []
    labels: dict[str, OwnerLabel] = {}
    convergence: list[dict] = []
    for origin in origins:
        for seed in seeds:
            for case_id in case_ids:
                trajectory = simulate_trajectory(
                    case_id, seed, origin, base_nodes=base_nodes,
                    snapshot_count=snapshot_count,
                )
                convergence.append({
                    "case_id": case_id, "seed": seed, "origin": trajectory.origin,
                    "convergence_error": trajectory.convergence_error,
                    "public_digest": trajectory.public_digest,
                })
                for noise_model, noise_level in noise_cells:
                    episode, label = build_public_episode(
                        trajectory, noise_model=noise_model, noise_level=noise_level)
                    episodes.append(episode)
                    labels[label.episode_id] = label
                if progress is not None:
                    progress(origin, seed, case_id, len(episodes))
    return episodes, labels, convergence


def _record_payload(record: AuthorityRecord) -> dict:
    payload = record.public_dict()
    payload["coefficients"] = [[atom, value] for atom, value in record.coefficients]
    return payload


def artifact_payload(episodes: list[PublicEpisode], labels: dict[str, OwnerLabel],
                     convergence: list[dict], metadata: dict) -> dict:
    return {
        "schema_version": "maintrack.2d-coupled-bank.v2",
        "metadata": metadata,
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "origin": episode.origin,
                "noise_model": episode.noise_model,
                "noise_level": episode.noise_level,
                "views": [_record_payload(episode.view_records[view.view_id])
                          for view in VIEW_BANK],
            }
            for episode in episodes
        ],
        "owner_labels": [asdict(labels[key]) for key in sorted(labels)],
        "convergence": convergence,
    }


def atomic_write_gzip_json(path: Path, payload: dict) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.",
                                     delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        with gzip.open(temp_path, "wb", compresslevel=6) as handle:
            handle.write(canonical)
        os.link(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return digest, len(canonical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--origins", default="spectral,fd")
    parser.add_argument("--case-ids", default=",".join(case.case_id for case in OWNER_CASES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args()
    seeds = SPLITS[args.split]
    if args.max_seeds is not None:
        seeds = seeds[:args.max_seeds]
    if args.split == "confirmation":
        raise SystemExit("confirmation remains sealed; no authorization interface exists")

    def progress(origin, seed, case_id, count):
        print(json.dumps({"origin": origin, "seed": seed,
                          "case_id": case_id, "episodes": count}), flush=True)

    episodes, labels, convergence = run_bank(
        seeds=seeds, origins=tuple(args.origins.split(",")),
        case_ids=tuple(args.case_ids.split(",")), progress=progress,
    )
    payload = artifact_payload(episodes, labels, convergence, {
        "split": args.split, "seeds": list(seeds),
        "origins": args.origins.split(","), "case_ids": args.case_ids.split(","),
        "noise_cells": NOISE_CELLS, "view_count": len(VIEW_BANK),
    })
    digest, size = atomic_write_gzip_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "content_sha256": digest,
                      "uncompressed_bytes": size, "episodes": len(episodes)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
