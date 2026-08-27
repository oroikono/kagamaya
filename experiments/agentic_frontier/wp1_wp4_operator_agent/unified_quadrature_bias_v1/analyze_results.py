"""Recompute terminal error types from a validated quadrature-bias payload."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .validate import validate_payload


def analyze(payload: dict) -> dict:
    result = {}
    for modes in (2, 5):
        for nodes in (7, 13, 25, 49):
            rows = [
                row
                for row in payload["rows"]
                if row["nodes"] == nodes and row["temporal_modes"] == modes
            ]
            errors = Counter()
            for row in rows:
                if not row["false_promotion"]:
                    continue
                if row["final"]["status"] == "IDENTIFIED":
                    errors["false_identified_support"] += 1
                elif row["final"]["status"] == "OUT_OF_GRAMMAR":
                    errors["false_out_of_grammar"] += 1
                else:
                    raise AssertionError("an abstention cannot be a false promotion")
            result[f"nodes_{nodes}_modes_{modes}"] = {
                "episodes": len(rows),
                "false_identified_support": errors["false_identified_support"],
                "false_out_of_grammar": errors["false_out_of_grammar"],
                "false_promotion_total": sum(errors.values()),
            }
    return result


def main(path: str) -> None:
    payload = json.loads(Path(path).read_text())
    validate_payload(payload)
    print(json.dumps(analyze(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    import sys

    main(sys.argv[1])

