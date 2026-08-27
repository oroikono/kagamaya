"""Independently replay the deterministic public artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contract import ARMS, VIEW_TOKENS, digest, initial_prompt, read_cases, second_prompt
from .deterministic_campaign import _compact, deterministic_decision
from .protocol import counterfactual_second, execute_public


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("content_sha256", None)
    if claimed != digest(value):
        raise ValueError("deterministic artifact hash mismatch")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 216 * len(ARMS) * 42:
        raise ValueError("deterministic schedule cardinality mismatch")
    cases = {row["case_id"]: row for row in read_cases()}
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        public = row["public"]
        case = cases[public["case_id"]]
        key = (row["kind"], row["policy"], row["arm"], public["case_id"])
        if key in seen:
            raise ValueError("duplicate deterministic schedule row")
        seen.add(key)
        if row["kind"] == "policy":
            first, second = deterministic_decision(row["policy"], row["arm"], case)
        elif row["kind"] == "enumeration":
            token, decision = row["policy"].split(":")
            if token not in VIEW_TOKENS or decision not in {"COMMIT", "REQUEST_CHALLENGE"}:
                raise ValueError("invalid enumeration identifier")
            first = {f"{producer}_action": token for producer in ("spectral_etdrk4", "fd_bdf")}
            second = counterfactual_second(decision)
        else:
            raise ValueError("invalid deterministic row kind")
        replay_full = execute_public(case, first, second)
        if _compact(replay_full) != public:
            raise ValueError("deterministic public replay mismatch")
        if row["kind"] == "policy":
            p1 = hashlib.sha256(initial_prompt(case, row["arm"]).encode()).hexdigest()
            p2 = hashlib.sha256(second_prompt(case, row["arm"], first, replay_full["scout_observations"]).encode()).hexdigest()
            if row["first_prompt_sha256"] != p1 or row["second_prompt_sha256"] != p2:
                raise ValueError("deterministic prompt binding mismatch")
    result = {
        "schema_version": "evidence_view_benchmark.multistep_deterministic_validation.v1",
        "valid": True,
        "rows": len(rows),
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

