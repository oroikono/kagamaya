"""Producer-free structural and deterministic replay validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import ARMS, PRODUCERS, digest, read_cases, validate_first, validate_second
from .protocol import counterfactual_second, execute_public
from .qwen_runner import strict_json


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("content_sha256", None)
    if claimed != digest(value):
        raise ValueError("Qwen artifact content hash mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 216 * len(ARMS):
        raise ValueError("Qwen row cardinality mismatch")
    cases = {row["case_id"]: row for row in read_cases()}
    seen = set()
    for row in rows:
        unsigned = dict(row)
        row_digest = unsigned.pop("record_sha256", None)
        if row_digest != digest(unsigned):
            raise ValueError("Qwen row hash mismatch")
        key = (row["case_id"], row["arm"])
        if key in seen or row["arm"] not in ARMS:
            raise ValueError("duplicate or invalid Qwen schedule")
        seen.add(key)
        case = cases[row["case_id"]]
        if row["case_sha256"] != case["case_sha256"]:
            raise ValueError("case binding mismatch")
        if row["first_valid"]:
            first = validate_first(strict_json(row["first_generation"]["raw_response"]))
            if first != row["first"]:
                raise ValueError("first parse replay mismatch")
            if row["second_valid"]:
                second = validate_second(strict_json(row["second_generation"]["raw_response"]))
            else:
                second = counterfactual_second("ABSTAIN")
            if second != row["second"]:
                raise ValueError("second parse replay mismatch")
            for name, decision in (
                ("actual", second),
                ("forced_commit", counterfactual_second("COMMIT")),
                ("forced_challenge", counterfactual_second("REQUEST_CHALLENGE")),
            ):
                replay = execute_public(case, first, decision)
                if replay != row[name]:
                    raise ValueError(f"public execution replay mismatch: {name}")
        elif any(row[name] is not None for name in ("actual", "forced_commit", "forced_challenge")):
            raise ValueError("invalid first proposal acquired evidence")
    result = {
        "schema_version": "evidence_view_benchmark.multistep_qwen_validation.v1",
        "valid": True,
        "rows": len(rows),
        "first_valid": sum(row["first_valid"] for row in rows),
        "second_valid": sum(row["second_valid"] for row in rows),
        "content_sha256": claimed,
    }
    result["validation_sha256"] = digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.path)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

