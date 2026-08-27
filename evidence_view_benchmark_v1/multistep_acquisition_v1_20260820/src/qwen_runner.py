"""Run the frozen two-call Qwen acquisition policy on Euler."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    ARMS,
    BENCHMARK_ROOT,
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PRODUCERS,
    canonical_json,
    digest,
    file_sha256,
    initial_prompt,
    read_cases,
    second_prompt,
    validate_first,
    validate_second,
)
from .protocol import counterfactual_second, execute_public

PROVIDER_ROOT = BENCHMARK_ROOT / "src"
if str(PROVIDER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROVIDER_ROOT))

from evidence_view_benchmark.gate4_providers import QwenTransformersProvider  # noqa: E402


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def strict_json(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    value, end = decoder.raw_decode(text)
    if text[end:].strip() or not isinstance(value, Mapping):
        raise ValueError("response is not exactly one JSON object")
    return value


def validate_model_identity(model_path: Path, identity_path: Path) -> dict[str, Any]:
    value = json.loads(identity_path.read_text(encoding="utf-8"))
    if value.get("repo_id") != MODEL_ID or value.get("revision") != MODEL_REVISION:
        raise ValueError("model identity mismatch")
    for name, expected in value.get("file_sha256", {}).items():
        path = model_path / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"model runtime file mismatch: {name}")
    return value


def _generate(provider: Any, prompt: str) -> dict[str, Any]:
    generation = provider.generate(model=MODEL_ID, prompt=prompt, max_output_tokens=MAX_OUTPUT_TOKENS)
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": generation.raw_response,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "latency_seconds": generation.latency_seconds,
        "resolved_model_id": generation.resolved_model_id,
        "stop_reason": generation.stop_reason,
    }


def run(model_path: Path, identity_path: Path, output: Path) -> dict[str, Any]:  # pragma: no cover - Euler GPU
    identity = validate_model_identity(model_path, identity_path)
    provider = QwenTransformersProvider(model_path)
    rows = []
    for case in read_cases():
        for arm in ARMS:
            first_prompt = initial_prompt(case, arm)
            first_generation = _generate(provider, first_prompt)
            first_valid = False
            first_error = None
            first = None
            try:
                first = validate_first(strict_json(first_generation["raw_response"]))
                first_valid = True
            except (ValueError, json.JSONDecodeError) as exc:
                first_error = f"{type(exc).__name__}: {exc}"
            second_generation = None
            second_valid = False
            second_error = "first proposal invalid"
            second = None
            actual = forced_commit = forced_challenge = None
            if first_valid and first is not None:
                scout_only = execute_public(case, first, counterfactual_second("ABSTAIN"))
                prompt2 = second_prompt(case, arm, first, scout_only["scout_observations"])
                second_generation = _generate(provider, prompt2)
                try:
                    second = validate_second(strict_json(second_generation["raw_response"]))
                    second_valid = True
                    second_error = None
                except (ValueError, json.JSONDecodeError) as exc:
                    second_error = f"{type(exc).__name__}: {exc}"
                    second = counterfactual_second("ABSTAIN")
                actual = execute_public(case, first, second)
                forced_commit = execute_public(case, first, counterfactual_second("COMMIT"))
                forced_challenge = execute_public(case, first, counterfactual_second("REQUEST_CHALLENGE"))
            row = {
                "case_id": case["case_id"],
                "case_sha256": case["case_sha256"],
                "arm": arm,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "first_generation": first_generation,
                "first_valid": first_valid,
                "first_error": first_error,
                "first": first,
                "second_generation": second_generation,
                "second_valid": second_valid,
                "second_error": second_error,
                "second": second,
                "actual": actual,
                "forced_commit": forced_commit,
                "forced_challenge": forced_challenge,
            }
            row["record_sha256"] = digest(row)
            rows.append(row)
    body = {
        "schema_version": "evidence_view_benchmark.multistep_qwen_raw.v1",
        "model_identity": identity,
        "cases": 216,
        "arms": list(ARMS),
        "temperature": 0,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
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
    result = run(args.model_path, args.model_identity, args.output)
    print(json.dumps({"rows": len(result["rows"]), "content_sha256": result["content_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

