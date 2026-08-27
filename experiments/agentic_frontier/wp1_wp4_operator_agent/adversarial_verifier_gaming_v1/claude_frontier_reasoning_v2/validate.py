"""Independent structural and arithmetic replay for the frontier arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..contract import digest
from ..io_utils import read_json, write_json
from ..llm_contract import decode_response, system_prompt, user_prompt
from .analyze import case_from_custom_id, model_from_custom_id
from .contract import CAMPAIGN_SHA256, MODEL_SPECS, SCHEMA, cost


KEY = re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")


def check_bound(payload: dict[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("content_sha256", None)
    if claimed != digest(body):
        raise ValueError("content hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    campaign = read_json(args.campaign)
    archive = read_json(args.run_dir / "prompt_archive.json.gz")
    responses = read_json(args.run_dir / "responses.json.gz")
    analysis = read_json(args.analysis)
    for payload in (archive, responses, analysis):
        check_bound(payload)
    if campaign["campaign_sha256"] != CAMPAIGN_SHA256:
        raise ValueError("campaign hash mismatch")
    cases = {case["case_id"]: case for case in campaign["cases"]}
    archived = {row["case_id"]: row for row in archive["rows"]}
    if set(cases) != set(archived) or len(cases) != 24:
        raise ValueError("prompt archive case mismatch")
    for case_id, case in cases.items():
        row = archived[case_id]
        if row["system"] != system_prompt() or row["user"] != user_prompt(case["request"]):
            raise ValueError("prompt archive cannot replay")
        if row["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError("prompt hash mismatch")
    rows = responses["rows"]
    if len(rows) != 72 or len({row["custom_id"] for row in rows}) != 72:
        raise ValueError("response cardinality mismatch")
    text_blob = json.dumps({"archive": archive, "responses": responses, "analysis": analysis}, sort_keys=True)
    if KEY.search(text_blob):
        raise ValueError("credential material appears in artifacts")
    by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_SPECS}
    for row in rows:
        model = model_from_custom_id(row["custom_id"])
        case_id = case_from_custom_id(row["custom_id"])
        if case_id not in cases:
            raise ValueError("unknown response case")
        by_model[model].append(row)
        if row["result_type"] == "succeeded":
            if row["text_sha256"] != hashlib.sha256(row["text"].encode("utf-8")).hexdigest():
                raise ValueError("text hash mismatch")
            try:
                decoded = decode_response(row["text"])
            except ValueError:
                decoded = None
            if decoded != row["strict_decoded"] or bool(decoded) != row["strict_valid"]:
                raise ValueError("strict parser replay mismatch")
            for block in row["content"]:
                if "signature" in block:
                    raise ValueError("raw encrypted thinking signature was stored")
    for model, model_rows in by_model.items():
        if len(model_rows) != 24:
            raise ValueError("model response count mismatch")
        expected_input = sum(int(row.get("usage", {}).get("input_tokens", 0)) for row in model_rows)
        expected_output = sum(int(row.get("usage", {}).get("output_tokens", 0)) for row in model_rows)
        expected_cost = sum(cost(model, int(row.get("usage", {}).get("input_tokens", 0)), int(row.get("usage", {}).get("output_tokens", 0))) for row in model_rows)
        recorded = analysis["models"][model]
        if recorded["input_tokens"] != expected_input or recorded["output_tokens_including_thinking"] != expected_output:
            raise ValueError("token aggregate mismatch")
        if abs(float(recorded["estimated_batch_cost_usd"]) - expected_cost) > 1e-12:
            raise ValueError("cost aggregate mismatch")
    output = {
        "schema_version": f"{SCHEMA}.independent_validation",
        "status": "valid",
        "models": list(MODEL_SPECS),
        "responses": 72,
        "campaign_sha256": CAMPAIGN_SHA256,
        "analysis_sha256": analysis["content_sha256"],
        "credential_pattern_matches": 0,
    }
    output["content_sha256"] = digest(output)
    write_json(args.output, output)


if __name__ == "__main__":
    main()

