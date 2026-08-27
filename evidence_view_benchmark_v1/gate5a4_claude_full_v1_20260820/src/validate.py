"""Independent structural validation for smoke and full public runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .contract import LANE_ROOT, MODELS, digest, file_sha256, prompt_inventory, validate_tool_arguments


def validate(path: Path, stage: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    expected_count = 6 if stage == "smoke" else 3240
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(rows)}")
    prompts = {row["prompt_sha256"]: row for row in prompt_inventory()}
    seen = set()
    errors = []
    for row in rows:
        if row.get("schema_version") != "evidence_view_benchmark.claude_full_raw.v1":
            errors.append(f"ordinal {row.get('ordinal')}: incident/non-raw row")
            continue
        claimed = row.get("record_sha256")
        unsigned = dict(row)
        unsigned.pop("record_sha256", None)
        if claimed != digest(unsigned):
            errors.append(f"ordinal {row['ordinal']}: record hash")
        prompt = prompts.get(row.get("prompt_sha256"))
        if prompt is None or prompt["case_id"] != row.get("case_id"):
            errors.append(f"ordinal {row['ordinal']}: prompt binding")
        key = (row["model_requested"], row["case_id"], row["module"], row["arm"])
        if key in seen:
            errors.append(f"duplicate {key}")
        seen.add(key)
        if row.get("model_requested") not in MODELS:
            errors.append(f"ordinal {row['ordinal']}: model")
        if row.get("tool_call_valid") is True:
            try:
                validate_tool_arguments(row["module"], row["parsed_tool_arguments"])
            except ValueError as exc:
                errors.append(f"ordinal {row['ordinal']}: {exc}")
    if errors:
        raise ValueError("; ".join(errors[:20]))
    summary = {
        "schema_version": "evidence_view_benchmark.claude_full_validation.v1",
        "stage": stage,
        "valid": True,
        "rows": len(rows),
        "tool_valid": sum(row["tool_call_valid"] for row in rows),
        "by_model": dict(Counter(row["model_requested"] for row in rows)),
        "raw_sha256": file_sha256(path),
    }
    summary["content_sha256"] = digest(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = validate(args.path, args.stage)
    if args.output:
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
