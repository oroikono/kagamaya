"""Independent prompt, model, response, and public-transition replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multistep_acquisition_v1_20260820.src.protocol import counterfactual_second, execute_public
from multistep_acquisition_v1_20260820.src.qwen_runner import strict_json

from .contract import (
    ARMS,
    CONDITIONS,
    EXPECTED_ROWS,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    digest,
    donor_map,
    expected_model_identity,
    initial_prompt,
    read_cases,
    second_prompt,
    validate_first,
    validate_second,
    verify_generation_binding,
)


def validate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(value)
    claimed = unsigned.pop("content_sha256", None)
    if claimed != digest(unsigned):
        raise ValueError("artifact content hash mismatch")
    if value.get("schema_version") != "evidence_view_benchmark.metadata_blind_intervention.v2":
        raise ValueError("artifact schema mismatch")
    if value.get("model_identity") != expected_model_identity():
        raise ValueError("embedded model identity differs from frozen identity")
    if value.get("model_id") != MODEL_ID or value.get("model_revision") != MODEL_REVISION:
        raise ValueError("top-level model binding mismatch")
    if value.get("temperature") != 0 or value.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise ValueError("decoding contract mismatch")
    if value.get("conditions") != list(CONDITIONS):
        raise ValueError("condition contract mismatch")
    if value.get("permutation") != {"kind": "sorted_case_cyclic_shift", "shift": 37}:
        raise ValueError("intervention contract mismatch")
    if set(value.get("initial_generations", {})) != set(ARMS) or set(value.get("first_by_arm", {})) != set(ARMS):
        raise ValueError("initial arm coverage mismatch")

    first_by_arm: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        generation = value["initial_generations"][arm]
        prompt = initial_prompt(arm)
        verify_generation_binding(generation, prompt)
        first = validate_first(strict_json(generation["raw_response"]))
        if first != value["first_by_arm"][arm]:
            raise ValueError("initial decision replay mismatch")
        first_by_arm[arm] = first

    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROWS:
        raise ValueError("row cardinality mismatch")
    cases = {row["case_id"]: row for row in read_cases()}
    donors = donor_map()
    scouts: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ARMS:
        for case in cases.values():
            scout = execute_public(case, first_by_arm[arm], counterfactual_second("ABSTAIN"))
            scouts[(arm, case["case_id"])] = scout["scout_observations"]

    seen = set()
    valid_seconds = 0
    for row in rows:
        row_unsigned = dict(row)
        record_sha = row_unsigned.pop("record_sha256", None)
        if record_sha != digest(row_unsigned):
            raise ValueError("row content hash mismatch")
        key = (row["case_id"], row["arm"], row["condition"])
        if key in seen:
            raise ValueError("duplicate schedule row")
        seen.add(key)
        if row["arm"] not in ARMS or row["condition"] not in CONDITIONS:
            raise ValueError("invalid schedule coordinate")
        case = cases[row["case_id"]]
        if row["case_sha256"] != case["case_sha256"]:
            raise ValueError("case binding mismatch")
        expected_donor = row["case_id"] if row["condition"] == "FACTUAL" else donors[row["case_id"]]
        if row["observation_donor_case_id"] != expected_donor:
            raise ValueError("observation intervention binding mismatch")
        first = first_by_arm[row["arm"]]
        if row["first"] != first:
            raise ValueError("row initial action mismatch")
        observations = scouts[(row["arm"], expected_donor)]
        prompt = second_prompt(row["arm"], first, observations)
        verify_generation_binding(row["second_generation"], prompt)
        parsed_second = None
        parse_error = None
        try:
            parsed_second = validate_second(strict_json(row["second_generation"]["raw_response"]))
        except (ValueError, json.JSONDecodeError) as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        if row["second_valid"]:
            if parsed_second is None or parse_error is not None or row["second_error"] is not None:
                raise ValueError("valid second response has inconsistent parse fields")
            second = parsed_second
            valid_seconds += 1
        else:
            if parsed_second is not None or row["second_error"] != parse_error:
                raise ValueError("invalid second response has inconsistent parse fields")
            second = counterfactual_second("ABSTAIN")
        if row["second"] != second:
            raise ValueError("second decision replay mismatch")
        for name, decision in (
            ("actual", second),
            ("forced_commit", counterfactual_second("COMMIT")),
            ("forced_challenge", counterfactual_second("REQUEST_CHALLENGE")),
        ):
            if row[name] != execute_public(case, first, decision):
                raise ValueError(f"public transition replay mismatch: {name}")
    if len(seen) != EXPECTED_ROWS:
        raise ValueError("schedule coverage mismatch")
    result = {
        "schema_version": "evidence_view_benchmark.metadata_blind_validation.v2",
        "valid": True,
        "rows": len(rows),
        "valid_second_responses": valid_seconds,
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
