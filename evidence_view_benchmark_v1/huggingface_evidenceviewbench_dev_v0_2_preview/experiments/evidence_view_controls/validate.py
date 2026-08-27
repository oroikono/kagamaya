"""Producer-free structural and exact-summary validator for A10--A12."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .campaign import _a10_summary, _a11_summary, _a12_summary
from .contract import (
    A10_SEEDS, A11_SEEDS, A12_SEEDS, HIGH_NOISES, NOISE_MODELS,
    PRODUCERS, STANDARD_NOISES, digest, load_and_validate_freeze,
    publish_exclusive,
)


EXPECTED_ROWS = {"a10": 720, "a11": 720, "a12": 720}
EXPECTED_SEEDS = {"a10": A10_SEEDS, "a11": A11_SEEDS, "a12": A12_SEEDS}
EXPECTED_NOISES = {"a10": STANDARD_NOISES, "a11": HIGH_NOISES, "a12": STANDARD_NOISES}


def _body_digest(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("content_sha256", None)
    return digest(body)


def validate_result(path: Path) -> dict[str, Any]:
    freeze = load_and_validate_freeze()
    payload = json.loads(path.read_text())
    lane = payload.get("lane")
    if lane not in EXPECTED_ROWS:
        raise ValueError("unknown lane in result")
    if payload.get("content_sha256") != _body_digest(payload):
        raise ValueError("result digest mismatch")
    if payload.get("preoutcome_freeze_sha256") != freeze["freeze_sha256"]:
        raise ValueError("result is not bound to current preoutcome freeze")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS[lane]:
        raise ValueError(f"{lane} row cardinality mismatch")
    if payload.get("row_count") != len(rows):
        raise ValueError("declared row count mismatch")
    if set(row["producer"] for row in rows) != set(PRODUCERS):
        raise ValueError("producer coverage mismatch")
    if set(int(row["seed"]) for row in rows) != set(EXPECTED_SEEDS[lane]):
        raise ValueError("reserved seed coverage mismatch")
    if set(row["noise_model"] for row in rows) != set(NOISE_MODELS):
        raise ValueError("noise-model coverage mismatch")
    if set(float(row["relative_noise"]) for row in rows) != set(EXPECTED_NOISES[lane]):
        raise ValueError("noise-level coverage mismatch")
    counts = Counter((row["producer"], row["family"], row["seed"], row["noise_model"], row["relative_noise"]) for row in rows)
    if set(counts.values()) != {1}:
        raise ValueError("cross contains duplicate/missing cells")
    recomputed = {"a10": _a10_summary, "a11": _a11_summary, "a12": _a12_summary}[lane](rows)
    if recomputed != payload.get("summary"):
        raise ValueError("stored summary does not exactly recompute")
    if lane == "a12":
        if any(row["final49"]["best_residual"] < 0 or row["final97"]["best_residual"] < 0 for row in rows):
            raise ValueError("negative residual")
    return {
        "schema_version": "verisigs.evidence_view_controls.validation.v1",
        "lane": lane, "valid": True, "errors": [],
        "row_count": len(rows), "seed_clusters": len(EXPECTED_SEEDS[lane]),
        "producer_counts": dict(sorted(Counter(row["producer"] for row in rows).items())),
        "result_content_sha256": payload["content_sha256"],
        "result_file_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "preoutcome_freeze_sha256": freeze["freeze_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate_result(args.input)
    body = report | {"validation_sha256": digest(report)}
    publish_exclusive(args.output, body)
    print(json.dumps(body, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

