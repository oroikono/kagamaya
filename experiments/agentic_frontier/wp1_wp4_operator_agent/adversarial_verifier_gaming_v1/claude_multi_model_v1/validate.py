"""Independent structural and semantic validator for the Claude study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .. import llm_runner
from ..contract import digest
from ..io_utils import read_json, write_json
from ..llm_contract import decode_response
from ..score import score
from .analyze import guarded_rows, summarize
from .parser_sensitivity import MODEL_DIRS, recovered_campaign
from .runner import CAMPAIGN_SHA256, MODEL_SPECS as V1_SPECS
from .runner_r2 import MODEL_SPECS_R2


SPECS = {
    "claude-haiku-4-5-20251001": V1_SPECS["claude-haiku-4-5-20251001"],
    **MODEL_SPECS_R2,
}


def without_hash(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = dict(row)
    claimed = body.pop(field, None)
    if claimed != digest(body):
        raise ValueError(f"{field} mismatch")
    return body


def validate_model(
    *, model: str, policy: str, root: Path, campaign: Mapping[str, Any], public_root: Path
) -> dict[str, Any]:
    spec = SPECS[model]
    model_dir = root / policy
    preflight = read_json(model_dir / "preflight.json")
    without_hash(preflight, "preflight_sha256")
    if (
        preflight["model"] != model
        or preflight["policy"] != policy
        or preflight["campaign_sha256"] != CAMPAIGN_SHA256
        or preflight["call_count"] != 24
        or preflight["retry_count"] != 0
        or preflight["max_output_tokens"] != 64
        or preflight["total_exact_input_tokens"] != spec["expected_total_input_tokens"]
        or preflight["maximum_authorized_cost_usd"] != spec["maximum_authorized_cost_usd"]
    ):
        raise ValueError(f"preflight contract mismatch for {model}")
    expected_prompts = {
        row["case_id"]: row["prompt_sha256"] for row in campaign["cases"]
    }
    if {row["case_id"]: row["prompt_sha256"] for row in preflight["entries"]} != expected_prompts:
        raise ValueError(f"preflight prompt binding mismatch for {model}")

    authorization = read_json(model_dir / "authorization.json")
    expected_auth = llm_runner.authorization_template(preflight)
    expected_auth.update(
        {
            "billable_calls_authorized": True,
            "authorization_reference": authorization.get("authorization_reference"),
        }
    )
    if authorization != expected_auth or not authorization["authorization_reference"]:
        raise ValueError(f"authorization mismatch for {model}")

    records = []
    all_proposals = []
    for index, case in enumerate(campaign["cases"]):
        case_id = f"case_{index:02d}"
        if case["case_id"] != case_id:
            raise ValueError("campaign case ordering mismatch")
        lock = read_json(model_dir / "responses" / f"{case_id}.attempt.json")
        if lock != {
            "case_id": case_id,
            "preflight_sha256": preflight["preflight_sha256"],
            "attempt_ordinal": 0,
            "retry_authorized": False,
        }:
            raise ValueError(f"attempt lock mismatch for {model}/{case_id}")
        record = read_json(model_dir / "responses" / f"{case_id}.json")
        without_hash(record, "record_sha256")
        if record["response_sha256"] != digest({"response": record["raw_response"]}):
            raise ValueError(f"response hash mismatch for {model}/{case_id}")
        try:
            decoded = decode_response(record["raw_response"])
        except ValueError:
            decoded = None
        if bool(record["proposal_valid"]) != (decoded is not None):
            raise ValueError(f"strict decode status mismatch for {model}/{case_id}")
        if (
            record["model"] != model
            or record["policy"] != policy
            or record["retry_count"] != 0
            or record["api_key_recorded_or_hashed"] is not False
            or record["scientific_authority"] is not False
            or len(record["proposals"]) != 2
        ):
            raise ValueError(f"record contract mismatch for {model}/{case_id}")
        for proposal in record["proposals"]:
            without_hash(proposal, "proposal_sha256")
            if proposal["hidden_reference_accessed"] or proposal["scientific_authority"]:
                raise ValueError("proposal crossed the authority boundary")
            expected = None
            if decoded is not None:
                key = (
                    "spectral_etdrk4_action"
                    if proposal["producer"] == "spectral_etdrk4"
                    else "fd_bdf_action"
                )
                expected = decoded[key]
            if proposal["selected_action_token"] != expected:
                raise ValueError(f"proposal/decoder mismatch for {model}/{case_id}")
        records.append(record)
        all_proposals.extend(record["proposals"])

    proposals = read_json(model_dir / "proposals.json.gz")
    without_hash(proposals, "campaign_sha256")
    if (
        proposals["proposal_count"] != 48
        or proposals["proposals"] != all_proposals
        or proposals["llm_records_sha256"] != digest(records)
        or abs(
            proposals["estimated_cost_usd"]
            - sum(float(row["estimated_cost_usd"]) for row in records)
        )
        > 1e-12
    ):
        raise ValueError(f"proposal campaign mismatch for {model}")

    producer_summaries = {}
    for producer, stem in (("spectral_etdrk4", "spectral"), ("fd_bdf", "fd")):
        public = read_json(public_root / f"{stem}_public.json.gz")
        owner = read_json(public_root / f"{stem}_owner.json.gz")
        recomputed = score(public, owner, proposals)
        stored = read_json(model_dir / f"{stem}_score.json.gz")
        if recomputed != stored:
            raise ValueError(f"score replay mismatch for {model}/{producer}")
        producer_summaries[producer] = summarize(
            recomputed, guarded_rows(public, owner, recomputed)
        )
    return {
        "records": len(records),
        "strict_valid": sum(bool(row["proposal_valid"]) for row in records),
        "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in records),
        "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in records),
        "estimated_cost_usd": sum(float(row["estimated_cost_usd"]) for row in records),
        "producers": producer_summaries,
    }


def validate_incident(root: Path) -> dict[str, Any]:
    model_dir = root / "claude_sonnet_5"
    files = sorted(
        str(path.relative_to(model_dir)) for path in model_dir.rglob("*") if path.is_file()
    )
    expected = [
        "authorization.json",
        "preflight.json",
        "responses/case_00.attempt.json",
    ]
    if files != expected:
        raise ValueError("Sonnet 5 incident directory contains an unexpected output")
    return {
        "model": "claude-sonnet-5",
        "status": "provider_request_rejected_before_generation_record",
        "files": files,
        "scientific_outcome_available": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--response-root", required=True, type=Path)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--parser-sensitivity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign = read_json(args.campaign)
    without_hash(campaign, "campaign_sha256")
    if campaign["campaign_sha256"] != CAMPAIGN_SHA256:
        raise ValueError("frozen campaign mismatch")
    observed = {
        model: validate_model(
            model=model,
            policy=policy,
            root=args.response_root,
            campaign=campaign,
            public_root=args.public_root,
        )
        for model, policy in MODEL_DIRS.items()
    }
    stored_analysis = read_json(args.analysis)
    expected_analysis = {
        model: {
            "strict_valid": row["strict_valid"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "estimated_cost_usd": row["estimated_cost_usd"],
            "producers": row["producers"],
        }
        for model, row in observed.items()
    }
    for model, expected in expected_analysis.items():
        actual = stored_analysis["models"][model]
        if (
            actual["valid_responses"] != expected["strict_valid"]
            or actual["input_tokens"] != expected["input_tokens"]
            or actual["output_tokens"] != expected["output_tokens"]
            or abs(actual["estimated_cost_usd"] - expected["estimated_cost_usd"]) > 1e-12
            or actual["producers"] != expected["producers"]
        ):
            raise ValueError(f"analysis summary mismatch for {model}")

    sensitivity = read_json(args.parser_sensitivity)
    for model, policy in MODEL_DIRS.items():
        records = [
            read_json(args.response_root / policy / "responses" / f"case_{index:02d}.json")
            for index in range(24)
        ]
        recovered, classes = recovered_campaign(campaign, records, model, policy)
        if recovered != read_json(args.response_root / policy / "fence_sensitivity_proposals.json.gz"):
            raise ValueError(f"parser-sensitivity proposal mismatch for {model}")
        if sensitivity["models"][model]["response_classes"] != dict(sorted(classes.items())):
            raise ValueError(f"parser-sensitivity class mismatch for {model}")

    output: dict[str, Any] = {
        "schema_version": "verisigs.claude_multi_model.v1.independent_validation",
        "status": "valid",
        "campaign_sha256": CAMPAIGN_SHA256,
        "models": observed,
        "failed_model_incident": validate_incident(args.response_root),
        "api_key_recorded_or_hashed": False,
    }
    output["content_sha256"] = digest(output)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
