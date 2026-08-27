"""Preflight-bound, zero-retry runner for either Claude or local Qwen."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from .contract import (
    CLAUDE_INPUT_USD_PER_MILLION,
    CLAUDE_MAX_CAMPAIGN_COST_USD,
    CLAUDE_OUTPUT_USD_PER_MILLION,
    CLAUDE_PRICING_SNAPSHOT,
    CLAUDE_PRICING_URL,
    MAX_LLM_CALLS,
    MAX_LLM_OUTPUT_TOKENS,
    SCHEMA_PREFIX,
    digest,
)
from .io_utils import read_json, write_json
from .llm_contract import decode_response, system_prompt, user_prompt
from .providers import ClaudeFileProvider, ProviderResponse, QwenTransformersProvider


MAX_INPUT_TOKENS_PER_CALL = 12000


class Provider(Protocol):
    def count_tokens(self, *, model: str, system: str, user: str) -> int: ...
    def generate(self, *, model: str, system: str, user: str, max_output_tokens: int) -> ProviderResponse: ...


def _is_billable_claude(model: str) -> bool:
    return model.lower().startswith("claude-")


def _claude_cost(input_tokens: int, output_tokens: int) -> float:
    return float(
        (
            input_tokens * CLAUDE_INPUT_USD_PER_MILLION
            + output_tokens * CLAUDE_OUTPUT_USD_PER_MILLION
        )
        / 1_000_000
    )


def _validated_campaign(campaign: Mapping[str, Any], require_full: bool = True) -> None:
    body = dict(campaign)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError("LLM campaign hash mismatch")
    if require_full and campaign["call_count"] != MAX_LLM_CALLS:
        raise ValueError("official LLM campaign must have 24 calls")
    if len(campaign["cases"]) != campaign["call_count"]:
        raise ValueError("LLM call count mismatch")


def preflight(campaign: Mapping[str, Any], provider: Provider, model: str, *, require_full: bool = True) -> dict[str, Any]:
    _validated_campaign(campaign, require_full=require_full)
    entries = []
    for case in campaign["cases"]:
        system, user = system_prompt(), user_prompt(case["request"])
        count = int(provider.count_tokens(model=model, system=system, user=user))
        if not 1 <= count <= MAX_INPUT_TOKENS_PER_CALL:
            raise RuntimeError("provider input token count is outside frozen bounds")
        entries.append(
            {
                "case_id": case["case_id"],
                "prompt_sha256": case["prompt_sha256"],
                "exact_input_tokens": count,
            }
        )
    total_input_tokens = sum(row["exact_input_tokens"] for row in entries)
    maximum_output_tokens = len(entries) * MAX_LLM_OUTPUT_TOKENS
    hard_cap_cost = (
        _claude_cost(total_input_tokens, maximum_output_tokens)
        if _is_billable_claude(model)
        else 0.0
    )
    if hard_cap_cost > CLAUDE_MAX_CAMPAIGN_COST_USD:
        raise RuntimeError("frozen Claude campaign exceeds its monetary cost ceiling")
    body: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.llm_preflight",
        "campaign_sha256": campaign["campaign_sha256"],
        "model": model,
        "call_count": len(entries),
        "retry_count": 0,
        "max_output_tokens": MAX_LLM_OUTPUT_TOKENS,
        "total_exact_input_tokens": total_input_tokens,
        "maximum_total_output_tokens": maximum_output_tokens,
        "hard_cap_estimated_cost_usd": hard_cap_cost,
        "maximum_authorized_cost_usd": (
            CLAUDE_MAX_CAMPAIGN_COST_USD if _is_billable_claude(model) else 0.0
        ),
        "pricing": {
            "input_usd_per_million_tokens": (
                CLAUDE_INPUT_USD_PER_MILLION if _is_billable_claude(model) else 0.0
            ),
            "output_usd_per_million_tokens": (
                CLAUDE_OUTPUT_USD_PER_MILLION if _is_billable_claude(model) else 0.0
            ),
            "snapshot_date": CLAUDE_PRICING_SNAPSHOT,
            "source_url": CLAUDE_PRICING_URL,
            "provider_invoice_is_authoritative": True,
        },
        "entries": entries,
    }
    body["preflight_sha256"] = digest(body)
    return body


def authorization_template(preflight_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": f"{SCHEMA_PREFIX}.llm_authorization",
        "preflight_sha256": preflight_row["preflight_sha256"],
        "maximum_billable_calls": preflight_row["call_count"],
        "maximum_authorized_cost_usd": preflight_row["maximum_authorized_cost_usd"],
        "billable_calls_authorized": False,
        "authorization_reference": None,
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def execute(
    campaign: Mapping[str, Any],
    preflight_row: Mapping[str, Any],
    authorization: Mapping[str, Any],
    provider: Provider,
    model: str,
    policy: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validated_campaign(campaign)
    expected_auth = authorization_template(preflight_row)
    expected_auth.update(
        {
            "billable_calls_authorized": True,
            "authorization_reference": authorization.get("authorization_reference"),
        }
    )
    if dict(authorization) != expected_auth or not authorization.get("authorization_reference"):
        raise PermissionError("generation requires exact preflight-bound authorization")
    if model != preflight_row["model"] or campaign["campaign_sha256"] != preflight_row["campaign_sha256"]:
        raise ValueError("runtime does not match preflight")
    by_case = {row["case_id"]: row for row in preflight_row["entries"]}
    records, proposals = [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in campaign["cases"]:
        case_id = case["case_id"]
        result_path = output_dir / f"{case_id}.json"
        if result_path.exists():
            record = read_json(result_path)
            records.append(record)
            proposals.extend(record["proposals"])
            continue
        lock_path = output_dir / f"{case_id}.attempt.json"
        if lock_path.exists():
            raise RuntimeError("attempt lock without result; zero-retry policy blocks a new call")
        system, user = system_prompt(), user_prompt(case["request"])
        exact = int(provider.count_tokens(model=model, system=system, user=user))
        if exact != by_case[case_id]["exact_input_tokens"]:
            raise RuntimeError("provider token count changed after preflight")
        _write_exclusive(
            lock_path,
            {
                "case_id": case_id,
                "preflight_sha256": preflight_row["preflight_sha256"],
                "attempt_ordinal": 0,
                "retry_authorized": False,
            },
        )
        response = provider.generate(
            model=model,
            system=system,
            user=user,
            max_output_tokens=MAX_LLM_OUTPUT_TOKENS,
        )
        if response.input_tokens > MAX_INPUT_TOKENS_PER_CALL or response.output_tokens > MAX_LLM_OUTPUT_TOKENS:
            raise RuntimeError("provider usage exceeded the frozen cap")
        try:
            decoded = decode_response(response.text)
            error = None
        except ValueError as exc:
            decoded, error = None, f"{type(exc).__name__}: {exc}"
        case_proposals = []
        for producer, response_key in (
            ("spectral_etdrk4", "spectral_etdrk4_action"),
            ("fd_bdf", "fd_bdf_action"),
        ):
            binding = case["bindings"][producer]
            proposal: dict[str, Any] = {
                "schema_version": f"{SCHEMA_PREFIX}.proposal",
                "episode_id": binding["episode_id"],
                "producer": producer,
                "public_episode_sha256": binding["public_episode_sha256"],
                "policy": policy,
                "proposal_valid": decoded is not None,
                "selected_action_token": None if decoded is None else decoded[response_key],
                "scientific_authority": False,
                "hidden_reference_accessed": False,
                "metadata": {
                    "case_id": case_id,
                    "model": model,
                    "prompt_sha256": case["prompt_sha256"],
                    "response_sha256": digest({"response": response.text}),
                },
            }
            proposal["proposal_sha256"] = digest(proposal)
            case_proposals.append(proposal)
        record: dict[str, Any] = {
            "schema_version": f"{SCHEMA_PREFIX}.llm_result",
            "case_id": case_id,
            "model": model,
            "policy": policy,
            "raw_response": response.text,
            "response_sha256": digest({"response": response.text}),
            "proposal_valid": decoded is not None,
            "decode_error": error,
            "usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
            "estimated_cost_usd": (
                _claude_cost(response.input_tokens, response.output_tokens)
                if _is_billable_claude(model)
                else 0.0
            ),
            "latency_seconds": response.latency_seconds,
            "provider_response_id": response.response_id,
            "stop_reason": response.stop_reason,
            "retry_count": 0,
            "scientific_authority": False,
            "api_key_recorded_or_hashed": False,
            "proposals": case_proposals,
        }
        record["record_sha256"] = digest(record)
        _write_exclusive(result_path, record)
        records.append(record)
        proposals.extend(case_proposals)
    total_estimated_cost = sum(float(row["estimated_cost_usd"]) for row in records)
    if total_estimated_cost > float(preflight_row["maximum_authorized_cost_usd"]):
        raise RuntimeError("observed token usage exceeded the authorized monetary ceiling")
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.proposal_campaign",
        "public_campaign_sha256": None,
        "source_public_campaigns": campaign["source_public_campaigns"],
        "policies": [policy],
        "proposal_count": len(proposals),
        "proposals": proposals,
        "llm_records_sha256": digest(records),
        "estimated_cost_usd": total_estimated_cost,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def main() -> None:  # pragma: no cover - provider CLIs
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("preflight")
    prepare.add_argument("--campaign", required=True, type=Path)
    prepare.add_argument("--provider", required=True, choices=("claude", "qwen"))
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--credential", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--authorization", required=True, type=Path)
    run = sub.add_parser("run")
    run.add_argument("--campaign", required=True, type=Path)
    run.add_argument("--provider", required=True, choices=("claude", "qwen"))
    run.add_argument("--model", required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--credential", type=Path)
    run.add_argument("--preflight", required=True, type=Path)
    run.add_argument("--authorization", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--proposal-output", required=True, type=Path)
    args = parser.parse_args()
    provider: Provider
    if args.provider == "claude":
        if args.credential is None:
            raise ValueError("Claude requires --credential")
        provider = ClaudeFileProvider(args.credential)
    else:
        provider = QwenTransformersProvider(args.model)
    campaign = read_json(args.campaign)
    if args.command == "preflight":
        row = preflight(campaign, provider, args.model)
        write_json(args.output, row)
        write_json(args.authorization, authorization_template(row))
    else:
        result = execute(
            campaign,
            read_json(args.preflight),
            read_json(args.authorization),
            provider,
            args.model,
            args.policy,
            args.output_dir,
        )
        write_json(args.proposal_output, result)


if __name__ == "__main__":
    main()
