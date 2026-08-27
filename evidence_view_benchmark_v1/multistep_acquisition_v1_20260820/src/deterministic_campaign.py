"""Frozen deterministic policies and exhaustive public schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contract import ARMS, PRODUCERS, VIEW_TOKENS, canonical_json, digest, initial_prompt, read_cases, second_prompt
from .protocol import counterfactual_second, execute_public


def _same_action(token: str) -> dict[str, str]:
    return {f"{producer}_action": token for producer in PRODUCERS}


def _compact(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "case_id": record["case_id"],
        "case_sha256": record["case_sha256"],
        "first": record["first"],
        "second": record["second"],
        "producers": {
            producer: {
                key: value
                for key, value in record["producers"][producer].items()
                if key != "trace"
            }
            | {"trace_sha256": record["producers"][producer]["trace"]["trace_sha256"]}
            for producer in PRODUCERS
        },
    }


def deterministic_decision(policy: str, arm: str, case: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    if policy == "low_mode5_commit":
        return _same_action("A03"), counterfactual_second("COMMIT")
    if policy == "high_fidelity_challenge":
        return _same_action("N49M5"), counterfactual_second("REQUEST_CHALLENGE")
    if policy == "accuracy_guard":
        return _same_action("A08"), counterfactual_second("REQUEST_CHALLENGE")
    if policy == "decisiveness_commit":
        return _same_action("A03"), counterfactual_second("COMMIT")
    if policy == "novelty_commit":
        return _same_action("A03"), counterfactual_second("COMMIT")
    if policy.startswith("random_"):
        replica = int(policy.split("_")[-1])
        first: dict[str, str] = {}
        second: dict[str, str] = {}
        for producer in PRODUCERS:
            seed = hashlib.sha256(f"multistep:{replica}:{arm}:{case['case_id']}:{producer}".encode()).digest()
            first[f"{producer}_action"] = VIEW_TOKENS[int.from_bytes(seed[:8], "big") % len(VIEW_TOKENS)]
            second[f"{producer}_decision"] = ("COMMIT", "REQUEST_CHALLENGE", "ABSTAIN")[int.from_bytes(seed[8:16], "big") % 3]
        return first, second
    raise ValueError("unknown deterministic policy")


def run(output: Path) -> dict[str, Any]:
    policies = (
        "low_mode5_commit",
        "high_fidelity_challenge",
        "accuracy_guard",
        "decisiveness_commit",
        "novelty_commit",
        *(f"random_{i}" for i in range(5)),
    )
    rows = []
    for case in read_cases():
        for arm in ARMS:
            for policy in policies:
                first, second = deterministic_decision(policy, arm, case)
                record = execute_public(case, first, second)
                rows.append({
                    "kind": "policy",
                    "policy": policy,
                    "arm": arm,
                    "first_prompt_sha256": hashlib.sha256(initial_prompt(case, arm).encode()).hexdigest(),
                    "second_prompt_sha256": hashlib.sha256(second_prompt(case, arm, first, record["scout_observations"]).encode()).hexdigest(),
                    "public": _compact(record),
                })
            for token in VIEW_TOKENS:
                first = _same_action(token)
                for decision in ("COMMIT", "REQUEST_CHALLENGE"):
                    record = execute_public(case, first, counterfactual_second(decision))
                    rows.append({"kind": "enumeration", "policy": f"{token}:{decision}", "arm": arm, "public": _compact(record)})
    body = {
        "schema_version": "evidence_view_benchmark.multistep_deterministic.v1",
        "cases": 216,
        "arms": list(ARMS),
        "rows": rows,
    }
    body["content_sha256"] = digest(body)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(body) + "\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.output)
    print(json.dumps({"rows": len(value["rows"]), "content_sha256": value["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
