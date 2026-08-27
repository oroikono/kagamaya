"""Run metadata-blind factual and permuted-observation Qwen controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from multistep_acquisition_v1_20260820.src.protocol import counterfactual_second, execute_public
from multistep_acquisition_v1_20260820.src.qwen_runner import (
    _generate,
    strict_json,
    validate_model_identity,
)

from .contract import (
    ARMS,
    BENCHMARK_ROOT,
    CONDITIONS,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    canonical_json,
    digest,
    donor_map,
    initial_prompt,
    read_cases,
    second_prompt,
    validate_first,
    validate_second,
)


PROVIDER_ROOT = BENCHMARK_ROOT / "src"
if str(PROVIDER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROVIDER_ROOT))

from evidence_view_benchmark.gate4_providers import QwenTransformersProvider  # noqa: E402


def _signed(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = digest(result)
    return result


def run(model_path: Path, identity_path: Path, output: Path) -> dict[str, Any]:  # pragma: no cover - Euler GPU
    identity = validate_model_identity(model_path, identity_path)
    provider = QwenTransformersProvider(model_path)
    cases = read_cases()
    by_id = {row["case_id"]: row for row in cases}
    donors = donor_map()

    initial_generations: dict[str, Any] = {}
    first_by_arm: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        prompt = initial_prompt(arm)
        generation = _generate(provider, prompt)
        first = validate_first(strict_json(generation["raw_response"]))
        initial_generations[arm] = generation
        first_by_arm[arm] = first

    scouts: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ARMS:
        first = first_by_arm[arm]
        for case in cases:
            scout_only = execute_public(case, first, counterfactual_second("ABSTAIN"))
            scouts[(arm, case["case_id"])] = scout_only["scout_observations"]

    rows = []
    for case in cases:
        case_id = case["case_id"]
        for arm in ARMS:
            first = first_by_arm[arm]
            for condition in CONDITIONS:
                donor_id = case_id if condition == "FACTUAL" else donors[case_id]
                observations = scouts[(arm, donor_id)]
                prompt = second_prompt(arm, first, observations)
                generation = _generate(provider, prompt)
                valid = False
                error = None
                try:
                    second = validate_second(strict_json(generation["raw_response"]))
                    valid = True
                except (ValueError, json.JSONDecodeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    second = counterfactual_second("ABSTAIN")
                row = {
                    "case_id": case_id,
                    "case_sha256": case["case_sha256"],
                    "arm": arm,
                    "condition": condition,
                    "observation_donor_case_id": donor_id,
                    "first": first,
                    "second_generation": generation,
                    "second_valid": valid,
                    "second_error": error,
                    "second": second,
                    "actual": execute_public(case, first, second),
                    "forced_commit": execute_public(case, first, counterfactual_second("COMMIT")),
                    "forced_challenge": execute_public(case, first, counterfactual_second("REQUEST_CHALLENGE")),
                }
                rows.append(_signed(row))

    body = {
        "schema_version": "evidence_view_benchmark.metadata_blind_intervention.v2",
        "model_identity": identity,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "conditions": list(CONDITIONS),
        "permutation": {"kind": "sorted_case_cyclic_shift", "shift": 37},
        "initial_generations": initial_generations,
        "first_by_arm": first_by_arm,
        "rows": rows,
    }
    body["content_sha256"] = digest(body)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(body) + "\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.model_path, args.model_identity, args.output)
    print(json.dumps({"rows": len(value["rows"]), "content_sha256": value["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

