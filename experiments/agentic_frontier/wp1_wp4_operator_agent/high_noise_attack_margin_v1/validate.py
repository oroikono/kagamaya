"""Independent structural and arithmetic validation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .campaign import FAMILIES, summarize
from .contract import NOISES, NOISE_MODELS, POLICIES, PRODUCERS, SEEDS, digest, load_freeze


def validate(path: Path) -> dict:
    payload = json.loads(path.read_text())
    claimed = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if claimed != digest(body):
        raise ValueError("result content hash mismatch")
    freeze = load_freeze()
    if payload["freeze_sha256"] != freeze["freeze_sha256"]:
        raise ValueError("result is not bound to current freeze")
    producer = payload["producer"]
    if producer not in PRODUCERS:
        raise ValueError("invalid producer")
    rows = payload["rows"]
    expected_episodes = len(FAMILIES) * len(SEEDS) * len(NOISE_MODELS) * len(NOISES)
    if len(rows) != expected_episodes * len(POLICIES):
        raise ValueError("wrong row cardinality")
    cells = Counter((r["family"], r["seed"], r["noise_model"], r["relative_noise"], r["policy"])
                    for r in rows)
    if len(cells) != expected_episodes * len(POLICIES) or set(cells.values()) != {1}:
        raise ValueError("factorial or policy cells are duplicated/missing")
    if set(r["seed"] for r in rows) != set(SEEDS):
        raise ValueError("seed registry mismatch")
    if set(r["relative_noise"] for r in rows) != set(NOISES):
        raise ValueError("noise registry mismatch")
    for row in rows:
        row_claim = row.get("row_sha256")
        row_body = dict(row)
        row_body.pop("row_sha256", None)
        if row_claim != digest(row_body):
            raise ValueError("row digest mismatch")
        if row["producer"] != producer:
            raise ValueError("producer mismatch")
        if bool(row["correct"]) and bool(row["false_promotion"]):
            raise ValueError("row cannot be both correct and false")
        if bool(row["abstained"]) != (row["status"] == "UNIDENTIFIED"):
            raise ValueError("abstention/status mismatch")
    recomputed = summarize(rows)
    if recomputed != payload["summary"]:
        raise ValueError("summary does not recompute")
    return {
        "valid": True,
        "producer": producer,
        "episodes": expected_episodes,
        "rows": len(rows),
        "content_sha256": claimed,
        "freeze_sha256": freeze["freeze_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.path), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
