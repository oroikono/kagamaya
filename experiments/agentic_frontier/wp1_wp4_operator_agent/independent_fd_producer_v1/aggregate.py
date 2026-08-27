"""Aggregate structurally validated one-seed Euler shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .campaign import CampaignConfig, ladder_summary, paired_contrasts, summarize
from .validate import validate_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.shards).glob("seed-*.json"))
    if len(paths) != 20:
        raise ValueError(f"expected 20 shards, found {len(paths)}")
    payloads = [json.loads(path.read_text()) for path in paths]
    for payload in payloads:
        validate_payload(payload)
    seeds = tuple(sorted(int(seed) for payload in payloads for seed in payload["config"]["seeds"]))
    if seeds != tuple(range(9600, 9620)):
        raise ValueError(f"unexpected seed coverage: {seeds}")
    rows = [row for payload in payloads for row in payload["rows"]]
    first = payloads[0]
    result = {
        "status": first["status"],
        "design_freeze_sha256": first["design_freeze_sha256"],
        "config": asdict(CampaignConfig(seeds)),
        "producer": first["producer"],
        "grammar_size": first["grammar_size"],
        "admission_gate": first["admission_gate"],
        "summary": summarize(rows),
        "paired_contrasts": paired_contrasts(rows),
        "ladder": ladder_summary(rows),
        "rows": rows,
        "shard_content_sha256": [payload["content_sha256"] for payload in payloads],
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["content_sha256"] = hashlib.sha256(raw).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    validate_payload(result)
    print(json.dumps({"paired": result["paired_contrasts"], "ladder": result["ladder"]}, indent=2))
    print(result["content_sha256"])


if __name__ == "__main__":
    main()
