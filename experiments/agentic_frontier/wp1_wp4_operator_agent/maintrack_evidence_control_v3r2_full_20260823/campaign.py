"""One-seed/one-origin shard generator for the full campaign."""

from __future__ import annotations

from dataclasses import asdict
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .authority import AuthorityRecord, evaluate
from .contract import (CALIBRATION_SEEDS, CONFIRMATION_SEEDS, NOISE_CELLS,
                       ORIGINS, TRAIN_SEEDS, VIEW_BANK)
from .owner_truth import OWNER_CASES
from .trajectory import Trajectory
from .weak_evidence import build_weak_evidence, observed_arrays


SPLITS = {
    "train": TRAIN_SEEDS,
    "calibration": CALIBRATION_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
}


def _digest_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def simulate(case_id: str, seed: int, origin: str) -> Trajectory:
    if origin == "spectral":
        from .numerical_spectral import simulate as solver
        return solver(case_id, seed, n=72, snapshot_count=49, internal_steps=480)
    if origin == "fd":
        from .numerical_fd import simulate as solver
        return solver(case_id, seed, n=72, snapshot_count=49, internal_steps=720)
    raise ValueError("unregistered origin")


def _public_id(trajectory: Trajectory, noise_model: str, noise_level: float) -> str:
    raw = (f"{trajectory.episode_id}|{noise_model}|{noise_level:.9g}|v3r2-full")
    return "P-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _record(record: AuthorityRecord) -> dict:
    payload = record.public_dict()
    payload["coefficients"] = [[atom, value] for atom, value in record.coefficients]
    return payload


def run_shard(seed: int, origin: str) -> dict:
    episodes: list[dict] = []
    labels: list[dict] = []
    convergence: list[dict] = []
    for case in OWNER_CASES:
        trajectory = simulate(case.case_id, seed, origin)
        convergence.append({
            "case_id": case.case_id,
            "seed": seed,
            "origin": origin,
            "convergence_error": trajectory.convergence_error,
            "public_digest": trajectory.public_digest,
        })
        for noise_model, noise_level in NOISE_CELLS:
            noise_seed = _digest_int(
                f"{case.case_id}|{seed}|{origin}|{noise_model}|{noise_level:.9g}")
            observed_u, observed_v = observed_arrays(
                trajectory, noise_level=noise_level, noise_model=noise_model,
                noise_seed=noise_seed)
            episode_id = _public_id(trajectory, noise_model, noise_level)
            views = []
            for view in VIEW_BANK:
                evidence = build_weak_evidence(
                    trajectory, view, observed_u=observed_u,
                    observed_v=observed_v, episode_id=episode_id)
                views.append(_record(evaluate(evidence)))
            episodes.append({
                "episode_id": episode_id,
                "origin": origin,
                "noise_model": noise_model,
                "noise_level": noise_level,
                "views": views,
            })
            labels.append({
                "episode_id": episode_id,
                "owner_case_id": case.case_id,
                "family": case.family,
                "truth_kind": case.truth_kind,
                "truth_support": list(case.truth_support),
                "omitted_mechanism": case.omitted_mechanism,
            })
    return {"episodes": episodes, "owner_labels": labels,
            "convergence": convergence}


def canonical_digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def authorization_digest(payload: dict) -> str:
    clone = dict(payload)
    clone.pop("authorization_sha256", None)
    return hashlib.sha256(json.dumps(
        clone, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_gzip(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".tmp.",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with gzip.open(temporary, "wb", compresslevel=6) as handle:
            handle.write(raw)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--origin", choices=ORIGINS, required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed not in SPLITS[args.split]:
        raise SystemExit("seed is outside the prospective split")
    freeze_path = Path(__file__).resolve().parent / "PREOUTCOME_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    if args.source_closure != freeze.get("source_closure"):
        raise SystemExit("source closure does not match the immutable freeze")
    confirmation_authorization_sha = None
    if args.split == "confirmation":
        if (args.authorization is None or not args.authorization.is_file()
                or args.authorization_sha is None):
            raise SystemExit("confirmation requires a content-addressed authorization")
        authorization = json.loads(args.authorization.read_text())
        if (authorization.get("stage") != "confirmation"
                or authorization.get("source_closure") != args.source_closure
                or not authorization.get("calibration_gates_passed")
                or authorization_digest(authorization) != args.authorization_sha
                or authorization.get("authorization_sha256") != args.authorization_sha):
            raise SystemExit("confirmation authorization is invalid or failing")
        confirmation_authorization_sha = args.authorization_sha
    scientific = run_shard(args.seed, args.origin)
    payload = {
        "schema_version": "maintrack.evidence-control.shard.v3r2",
        "metadata": {
            "split": args.split,
            "seed": args.seed,
            "origin": args.origin,
            "source_closure": args.source_closure,
            "view_count": len(VIEW_BANK),
            "noise_cells": [list(cell) for cell in NOISE_CELLS],
            "confirmation_authorization_sha256": confirmation_authorization_sha,
        },
        **scientific,
    }
    payload["content_sha256"] = canonical_digest(payload)
    atomic_gzip(args.output, payload)
    print(json.dumps({"output": str(args.output),
                      "content_sha256": payload["content_sha256"],
                      "episodes": len(payload["episodes"])}, sort_keys=True))


if __name__ == "__main__":
    main()
