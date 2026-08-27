"""Structural and semantic-hash validator for finite-difference producer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .campaign import design_sha, ladder_summary, paired_contrasts, summarize
from ..maintrack_suite_v41.weakform import SUPPORTS

EXPECTED_CELLS = {(nodes, modes) for nodes in (7, 13, 25, 49) for modes in (2, 5)}
EXPECTED_PRODUCER = {
    "space": "periodic_centered_nine_point_finite_difference",
    "time": "scipy_solve_ivp_BDF",
    "rtol": 1e-8,
    "atol": 1e-10,
    "max_step": 0.001,
    "independence_claim": "same_repository_cross_discretization_only",
}


def _expected_labels(row: dict) -> tuple[bool, bool]:
    final = row["final"]
    truth = row["truth_support"]
    correct = (
        truth is not None
        and final["status"] == "IDENTIFIED"
        and final.get("support") == truth
    ) or (truth is None and final["status"] == "OUT_OF_GRAMMAR")
    false = final["status"] in {"IDENTIFIED", "OUT_OF_GRAMMAR"} and not correct
    return bool(correct), bool(false)


def validate_payload(payload: dict) -> None:
    if payload["status"] != "CROSS_DISCRETIZATION_CONFIRMATION_AUDIT_ONLY":
        raise ValueError("invalid status/claim ceiling")
    if payload.get("design_freeze_sha256") != design_sha():
        raise ValueError("design freeze hash mismatch")
    if payload.get("producer") != EXPECTED_PRODUCER:
        raise ValueError("producer contract changed")
    if payload["grammar_size"] != 92:
        raise ValueError("grammar size changed")
    config = payload["config"]
    seeds = tuple(config["seeds"])
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed")
    expected_episode_keys = {
        (family, seed, noise_model, noise)
        for family in config["families"]
        for seed in seeds
        for noise_model in config["noise_models"]
        for noise in config["relative_noises"]
    }
    expected_episodes = len(expected_episode_keys)
    expected_rows = expected_episodes * 8
    if len(payload["rows"]) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(payload['rows'])}")
    episodes: dict[tuple, set[tuple[int, int]]] = {}
    row_keys = []
    ordinals: dict[tuple, set[int]] = {}
    for row in payload["rows"]:
        key = tuple(row["episode_key"])
        episodes.setdefault(key, set()).add((row["nodes"], row["temporal_modes"]))
        row_keys.append((key, row["nodes"], row["temporal_modes"]))
        ordinals.setdefault(key, set()).add(int(row["episode"]))
        if row["final"]["status"] not in {"IDENTIFIED", "OUT_OF_GRAMMAR", "UNIDENTIFIED", "FAILED"}:
            raise ValueError("unknown terminal status")
        truth = row.get("truth_support")
        if truth is not None and tuple(truth) not in SUPPORTS:
            raise ValueError("truth support outside frozen grammar")
        support = row["final"].get("support")
        if row["final"]["status"] == "IDENTIFIED":
            if not isinstance(support, list) or tuple(support) not in SUPPORTS:
                raise ValueError("identified support outside frozen grammar")
        elif support is not None:
            raise ValueError("non-identified terminal carries a support")
        expected_correct, expected_false = _expected_labels(row)
        if row.get("correct") is not expected_correct:
            raise ValueError("correct label mismatch")
        if row.get("false_promotion") is not expected_false:
            raise ValueError("false-promotion label mismatch")
        if row["nodes"] == 49 and (abs(row["channel_design_error_to_49"]) > 1e-12 or abs(row["channel_target_error_to_49"]) > 1e-12):
            raise ValueError("49-node self-reference error is nonzero")
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("duplicate factorial key")
    if set(episodes) != expected_episode_keys or any(cells != EXPECTED_CELLS for cells in episodes.values()):
        raise ValueError("episode cell coverage failure")
    if any(len(values) != 1 for values in ordinals.values()):
        raise ValueError("episode ordinal changed within matched cells")
    observed_ordinals = {next(iter(values)) for values in ordinals.values()}
    global_bijection = observed_ordinals == set(range(expected_episodes))
    episodes_per_seed = (
        len(config["families"])
        * len(config["noise_models"])
        * len(config["relative_noises"])
    )
    shard_local_bijections = all(
        {
            next(iter(ordinals[key]))
            for key in ordinals
            if key[1] == seed
        }
        == set(range(episodes_per_seed))
        for seed in seeds
    )
    if not (global_bijection or shard_local_bijections):
        raise ValueError("episode ordinals match neither global nor shard-local bijections")
    if payload.get("summary") != summarize(payload["rows"]):
        raise ValueError("summary mismatch")
    if payload.get("paired_contrasts") != paired_contrasts(payload["rows"]):
        raise ValueError("paired contrast mismatch")
    if payload.get("ladder") != ladder_summary(payload["rows"]):
        raise ValueError("ladder mismatch")
    claimed = payload["content_sha256"]
    body = dict(payload)
    del body["content_sha256"]
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(raw).hexdigest() != claimed:
        raise ValueError("semantic content hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    payload = json.loads(Path(args.path).read_text())
    validate_payload(payload)
    print(f"VALID rows={len(payload['rows'])} sha256={payload['content_sha256']}")


if __name__ == "__main__":
    main()
