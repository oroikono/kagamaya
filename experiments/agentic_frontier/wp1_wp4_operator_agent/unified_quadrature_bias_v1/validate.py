"""Structural and semantic validator for the frozen quadrature-bias payload."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .campaign import design_sha, paired_contrasts, stratified, summarize
from ..maintrack_suite_v41.weakform import SUPPORTS


EXPECTED_CELLS = {(nodes, modes) for nodes in (7, 13, 25, 49) for modes in (2, 5)}


def _expected_labels(row: dict) -> tuple[bool, bool]:
    """Reconstruct post-hoc endpoints without trusting serialized labels."""

    final = row["final"]
    truth = row["truth_support"]
    support = final.get("support")
    correct = (
        truth is not None
        and final["status"] == "IDENTIFIED"
        and support == truth
    ) or (truth is None and final["status"] == "OUT_OF_GRAMMAR")
    false = final["status"] in {"IDENTIFIED", "OUT_OF_GRAMMAR"} and not correct
    return bool(correct), bool(false)


def validate_row(row: dict) -> None:
    if (row["nodes"], row["temporal_modes"]) not in EXPECTED_CELLS:
        raise AssertionError("unexpected factorial cell")
    if abs(row["action_cost"] - 1.0) > 1e-12:
        raise AssertionError("action cost changed across cells")
    if row["final"]["status"] not in {
        "IDENTIFIED",
        "OUT_OF_GRAMMAR",
        "UNIDENTIFIED",
    }:
        raise AssertionError("invalid terminal status")
    truth = row.get("truth_support")
    if truth is not None and tuple(truth) not in SUPPORTS:
        raise AssertionError("truth support outside frozen grammar")
    support = row["final"].get("support")
    if row["final"]["status"] == "IDENTIFIED":
        if not isinstance(support, list) or tuple(support) not in SUPPORTS:
            raise AssertionError("identified support outside frozen grammar")
    elif support is not None:
        raise AssertionError("non-identified terminal carries a support")
    expected_correct, expected_false = _expected_labels(row)
    if row.get("correct") is not expected_correct:
        raise AssertionError("correct label does not match truth and terminal")
    if row.get("false_promotion") is not expected_false:
        raise AssertionError("false-promotion label does not match truth and terminal")
    if row["channel_design_error_to_49"] < 0 or row["channel_target_error_to_49"] < 0:
        raise AssertionError("negative evidence error")
    if row["nodes"] == 49 and (
        row["channel_design_error_to_49"] > 1e-14
        or row["channel_target_error_to_49"] > 1e-14
    ):
        raise AssertionError("reference cell is not exact")


def validate_payload(payload: dict) -> None:
    if payload["status"] != "MATCHED_DEVELOPMENT_NOT_CONFIRMATORY":
        raise AssertionError("claim ceiling changed")
    if payload["design_freeze_sha256"] != design_sha():
        raise AssertionError("design freeze hash mismatch")
    config = payload["config"]
    seeds = tuple(config["seeds"])
    if len(seeds) != len(set(seeds)):
        raise AssertionError("duplicate seed")
    if any(seed >= 9500 for seed in seeds):
        raise AssertionError("reserved seed executed")
    raw = dict(payload)
    claimed = raw.pop("content_sha256")
    actual = hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if claimed != actual:
        raise AssertionError("content hash mismatch")
    expected_episodes = {
        (family, seed, noise_model, noise)
        for family in config["families"]
        for seed in seeds
        for noise_model in config["noise_models"]
        for noise in config["relative_noises"]
    }
    row_keys = []
    episode_ordinals: dict[tuple, set[int]] = {}
    for row in payload["rows"]:
        validate_row(row)
        episode = tuple(row["episode_key"])
        row_keys.append((episode, row["nodes"], row["temporal_modes"]))
        episode_ordinals.setdefault(episode, set()).add(int(row["episode"]))
    if len(row_keys) != len(set(row_keys)):
        raise AssertionError("duplicate factorial key")
    episodes = {tuple(row["episode_key"]) for row in payload["rows"]}
    if episodes != expected_episodes:
        raise AssertionError("episode set does not match configuration")
    if any(len(values) != 1 for values in episode_ordinals.values()):
        raise AssertionError("episode ordinal changed within matched cells")
    ordinals = {next(iter(values)) for values in episode_ordinals.values()}
    if ordinals != set(range(len(expected_episodes))):
        raise AssertionError("episode ordinals are not a complete bijection")
    for episode in episodes:
        cells = {
            (row["nodes"], row["temporal_modes"])
            for row in payload["rows"]
            if tuple(row["episode_key"]) == episode
        }
        if cells != EXPECTED_CELLS:
            raise AssertionError("unmatched or incomplete episode")
    if payload["summary"] != summarize(payload["rows"]):
        raise AssertionError("summary mismatch")
    if payload["paired_contrasts"] != paired_contrasts(payload["rows"]):
        raise AssertionError("paired contrast mismatch")
    if payload["stratified"] != stratified(payload["rows"]):
        raise AssertionError("stratified summary mismatch")


def main(path: str) -> None:
    payload = json.loads(Path(path).read_text())
    validate_payload(payload)
    print(f"validated_rows={len(payload['rows'])}")
    print(payload["content_sha256"])


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
