"""Credential-safe, price-correct Claude runner over the frozen 24-case table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .. import llm_runner as parent_runner
from ..contract import MAX_LLM_OUTPUT_TOKENS, SCHEMA_PREFIX, digest
from ..io_utils import read_json, write_json
from ..llm_contract import system_prompt, user_prompt
from ..providers import ClaudeFileProvider


CAMPAIGN_SHA256 = "75e3c47142478c513cb12b605517d8c368aa35a6c11dc0d121855f59b6cb2bfd"
PRICING_SNAPSHOT = "2026-08-18"
PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
MODEL_SPECS: dict[str, dict[str, float | int | str]] = {
    "claude-haiku-4-5-20251001": {
        "policy": "claude_haiku_4_5",
        "input_usd_per_million": 1.0,
        "output_usd_per_million": 5.0,
        "expected_total_input_tokens": 100_309,
        "maximum_authorized_cost_usd": 0.12,
    },
    "claude-sonnet-5": {
        "policy": "claude_sonnet_5",
        "input_usd_per_million": 2.0,
        "output_usd_per_million": 10.0,
        "expected_total_input_tokens": 129_379,
        "maximum_authorized_cost_usd": 0.30,
    },
    "claude-opus-5": {
        "policy": "claude_opus_5",
        "input_usd_per_million": 5.0,
        "output_usd_per_million": 25.0,
        "expected_total_input_tokens": 129_379,
        "maximum_authorized_cost_usd": 0.72,
    },
}


def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    spec = MODEL_SPECS[model]
    return float(
        (
            input_tokens * float(spec["input_usd_per_million"])
            + output_tokens * float(spec["output_usd_per_million"])
        )
        / 1_000_000
    )


def validate_campaign(campaign: Mapping[str, Any]) -> None:
    body = dict(campaign)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body) or claimed != CAMPAIGN_SHA256:
        raise ValueError("official campaign hash mismatch")
    if campaign.get("call_count") != 24 or len(campaign.get("cases", [])) != 24:
        raise ValueError("official Claude study requires exactly 24 cases")


def build_preflight(
    campaign: Mapping[str, Any], provider: ClaudeFileProvider, model: str
) -> dict[str, Any]:
    validate_campaign(campaign)
    if model not in MODEL_SPECS:
        raise ValueError("model is outside the frozen three-tier registry")
    entries = []
    for case in campaign["cases"]:
        count = int(
            provider.count_tokens(
                model=model,
                system=system_prompt(),
                user=user_prompt(case["request"]),
            )
        )
        if not 1 <= count <= parent_runner.MAX_INPUT_TOKENS_PER_CALL:
            raise RuntimeError("provider token count is outside frozen bounds")
        entries.append(
            {
                "case_id": case["case_id"],
                "prompt_sha256": case["prompt_sha256"],
                "exact_input_tokens": count,
            }
        )
    total_input = sum(row["exact_input_tokens"] for row in entries)
    spec = MODEL_SPECS[model]
    if total_input != int(spec["expected_total_input_tokens"]):
        raise RuntimeError("token count changed after the pre-freeze provider count")
    maximum_output = len(entries) * MAX_LLM_OUTPUT_TOKENS
    hard_cap = model_cost(model, total_input, maximum_output)
    authorized = float(spec["maximum_authorized_cost_usd"])
    if hard_cap > authorized:
        raise RuntimeError("model hard-cap cost exceeds frozen authorization")
    body: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.claude_multimodel_preflight",
        "campaign_sha256": campaign["campaign_sha256"],
        "model": model,
        "policy": spec["policy"],
        "call_count": len(entries),
        "retry_count": 0,
        "max_output_tokens": MAX_LLM_OUTPUT_TOKENS,
        "total_exact_input_tokens": total_input,
        "maximum_total_output_tokens": maximum_output,
        "hard_cap_estimated_cost_usd": hard_cap,
        "maximum_authorized_cost_usd": authorized,
        "pricing": {
            "input_usd_per_million_tokens": spec["input_usd_per_million"],
            "output_usd_per_million_tokens": spec["output_usd_per_million"],
            "snapshot_date": PRICING_SNAPSHOT,
            "source_url": PRICING_URL,
            "provider_invoice_is_authoritative": True,
        },
        "entries": entries,
    }
    body["preflight_sha256"] = digest(body)
    return body


def build_authorization(preflight: Mapping[str, Any], reference: str) -> dict[str, Any]:
    if not reference.strip():
        raise ValueError("authorization reference cannot be empty")
    row = parent_runner.authorization_template(preflight)
    row.update(
        {
            "billable_calls_authorized": True,
            "authorization_reference": reference,
        }
    )
    return row


def run(
    *,
    campaign_path: Path,
    credential_path: Path,
    model: str,
    output_root: Path,
    authorization_reference: str,
) -> None:
    campaign = read_json(campaign_path)
    validate_campaign(campaign)
    provider = ClaudeFileProvider(credential_path)
    preflight = build_preflight(campaign, provider, model)
    authorization = build_authorization(preflight, authorization_reference)
    model_dir = output_root / str(MODEL_SPECS[model]["policy"])
    model_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = model_dir / "preflight.json"
    authorization_path = model_dir / "authorization.json"
    if preflight_path.exists() or authorization_path.exists():
        raise FileExistsError("preflight or authorization already exists")
    write_json(preflight_path, preflight)
    write_json(authorization_path, authorization)

    original_cost = parent_runner._claude_cost
    parent_runner._claude_cost = lambda inp, out: model_cost(model, inp, out)
    try:
        proposals = parent_runner.execute(
            campaign,
            preflight,
            authorization,
            provider,
            model,
            str(MODEL_SPECS[model]["policy"]),
            model_dir / "responses",
        )
    finally:
        parent_runner._claude_cost = original_cost
    write_json(model_dir / "proposals.json.gz", proposals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--credential", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    run(
        campaign_path=args.campaign,
        credential_path=args.credential,
        model=args.model,
        output_root=args.output_root,
        authorization_reference=args.authorization_reference,
    )


if __name__ == "__main__":
    main()

