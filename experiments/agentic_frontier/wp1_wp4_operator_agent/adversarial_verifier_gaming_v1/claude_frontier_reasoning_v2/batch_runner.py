"""Credential-safe preflight, submission, polling, and collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..io_utils import read_json, write_json
from ..llm_contract import decode_response, system_prompt, user_prompt
from ..providers import read_anthropic_key
from .contract import (
    BATCH_URL,
    CALLS_PER_MODEL,
    CAMPAIGN_SHA256,
    EFFORT,
    MAX_AUTHORIZED_COST_USD,
    MAX_OUTPUT_TOKENS,
    MODEL_SPECS,
    PRICING_SNAPSHOT,
    PRICING_URL,
    SCHEMA,
    THINKING,
    bind,
    cost,
)


def _client(credential: Path):
    from anthropic import Anthropic

    return Anthropic(api_key=read_anthropic_key(credential))


def _exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_campaign(campaign: Mapping[str, Any]) -> None:
    body = dict(campaign)
    claimed = body.pop("campaign_sha256", None)
    from ..contract import digest

    if claimed != digest(body) or claimed != CAMPAIGN_SHA256:
        raise ValueError("frozen campaign hash mismatch")
    if campaign.get("call_count") != CALLS_PER_MODEL or len(campaign.get("cases", [])) != CALLS_PER_MODEL:
        raise ValueError("frontier arm requires exactly 24 cases")


def prompt_archive(campaign: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    system = system_prompt()
    for case in campaign["cases"]:
        user = user_prompt(case["request"])
        rows.append(
            {
                "case_id": case["case_id"],
                "prompt_sha256": case["prompt_sha256"],
                "system": system,
                "user": user,
            }
        )
    return bind({"schema_version": f"{SCHEMA}.prompt_archive", "campaign_sha256": CAMPAIGN_SHA256, "rows": rows})


def prepare(campaign_path: Path, credential: Path, output_dir: Path, authorization_reference: str) -> None:
    if not authorization_reference.strip():
        raise ValueError("authorization reference cannot be empty")
    campaign = read_json(campaign_path)
    validate_campaign(campaign)
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = prompt_archive(campaign)
    write_json(output_dir / "prompt_archive.json.gz", archive)
    client = _client(credential)
    entries: list[dict[str, Any]] = []
    totals: dict[str, int] = {}
    for model in MODEL_SPECS:
        total = 0
        for row in archive["rows"]:
            counted = client.messages.count_tokens(
                model=model,
                system=row["system"],
                messages=[{"role": "user", "content": row["user"]}],
                thinking=THINKING,
                output_config={"effort": EFFORT},
            )
            n = int(counted.input_tokens)
            if not 1 <= n <= 20_000:
                raise RuntimeError("token count outside frozen bounds")
            entries.append({"model": model, "case_id": row["case_id"], "prompt_sha256": row["prompt_sha256"], "input_tokens": n})
            total += n
        totals[model] = total
    hard_cap = sum(cost(model, totals[model], CALLS_PER_MODEL * MAX_OUTPUT_TOKENS) for model in MODEL_SPECS)
    if hard_cap > MAX_AUTHORIZED_COST_USD:
        raise RuntimeError("batch hard-cap cost exceeds authorization ceiling")
    preflight = bind(
        {
            "schema_version": f"{SCHEMA}.preflight",
            "campaign_sha256": CAMPAIGN_SHA256,
            "models": list(MODEL_SPECS),
            "call_count": len(entries),
            "calls_per_model": CALLS_PER_MODEL,
            "retry_count": 0,
            "fallbacks": [],
            "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
            "thinking": THINKING,
            "output_config": {"effort": EFFORT},
            "sampling_parameters_omitted": ["temperature", "top_p", "top_k"],
            "input_tokens_by_model": totals,
            "maximum_output_tokens_by_model": {model: CALLS_PER_MODEL * MAX_OUTPUT_TOKENS for model in MODEL_SPECS},
            "hard_cap_estimated_batch_cost_usd": hard_cap,
            "maximum_authorized_cost_usd": MAX_AUTHORIZED_COST_USD,
            "pricing_snapshot": PRICING_SNAPSHOT,
            "pricing_url": PRICING_URL,
            "batch_url": BATCH_URL,
            "provider_invoice_is_authoritative": True,
            "entries": entries,
        }
    )
    write_json(output_dir / "preflight.json", preflight)
    authorization = bind(
        {
            "schema_version": f"{SCHEMA}.authorization",
            "preflight_sha256": preflight["content_sha256"],
            "maximum_billable_calls": len(entries),
            "maximum_authorized_cost_usd": MAX_AUTHORIZED_COST_USD,
            "billable_calls_authorized": True,
            "authorization_reference": authorization_reference,
        }
    )
    write_json(output_dir / "authorization.json", authorization)


def _expected_authorization(preflight: Mapping[str, Any], authorization: Mapping[str, Any]) -> bool:
    body = dict(authorization)
    claimed = body.pop("content_sha256", None)
    from ..contract import digest

    return bool(
        claimed == digest(body)
        and body.get("preflight_sha256") == preflight.get("content_sha256")
        and body.get("billable_calls_authorized") is True
        and body.get("maximum_billable_calls") == 72
        and body.get("maximum_authorized_cost_usd") == MAX_AUTHORIZED_COST_USD
        and str(body.get("authorization_reference", "")).strip()
    )


def submit(campaign_path: Path, credential: Path, run_dir: Path) -> None:
    campaign = read_json(campaign_path)
    validate_campaign(campaign)
    preflight = read_json(run_dir / "preflight.json")
    authorization = read_json(run_dir / "authorization.json")
    if not _expected_authorization(preflight, authorization):
        raise PermissionError("invalid content-addressed authorization")
    if (run_dir / "batch_receipt.json").exists():
        raise FileExistsError("batch was already submitted")
    archive = read_json(run_dir / "prompt_archive.json.gz")
    by_case = {row["case_id"]: row for row in archive["rows"]}
    requests = []
    for model in MODEL_SPECS:
        for case in campaign["cases"]:
            row = by_case[case["case_id"]]
            requests.append(
                {
                    "custom_id": f"{model.replace('claude-', '').replace('.', '-')}-{case['case_id']}",
                    "params": {
                        "model": model,
                        "max_tokens": MAX_OUTPUT_TOKENS,
                        "system": row["system"],
                        "messages": [{"role": "user", "content": row["user"]}],
                        "thinking": THINKING,
                        "output_config": {"effort": EFFORT},
                    },
                }
            )
    if len(requests) != 72:
        raise AssertionError("batch request count changed")
    batch = _client(credential).messages.batches.create(requests=requests)
    payload = bind(
        {
            "schema_version": f"{SCHEMA}.batch_receipt",
            "batch_id": str(batch.id),
            "processing_status": str(batch.processing_status),
            "request_count": len(requests),
            "preflight_sha256": preflight["content_sha256"],
            "authorization_sha256": authorization["content_sha256"],
        }
    )
    _exclusive(run_dir / "batch_receipt.json", payload)


def status(credential: Path, run_dir: Path) -> None:
    receipt = read_json(run_dir / "batch_receipt.json")
    batch = _client(credential).messages.batches.retrieve(receipt["batch_id"])
    print(json.dumps(batch.model_dump(mode="json"), sort_keys=True))


def _sanitized_content(blocks: list[Any]) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in blocks:
        row = block.model_dump(mode="json")
        block_type = str(row.get("type"))
        if block_type in {"thinking", "redacted_thinking"}:
            signature = str(row.pop("signature", ""))
            row["signature_sha256"] = hashlib.sha256(signature.encode("utf-8")).hexdigest() if signature else None
        if block_type == "text":
            texts.append(str(row.get("text", "")))
        output.append(row)
    if len(texts) != 1:
        raise RuntimeError("frontier response must contain exactly one text block")
    return output, texts[0]


def collect(campaign_path: Path, credential: Path, run_dir: Path) -> None:
    if (run_dir / "responses.json.gz").exists():
        raise FileExistsError("responses already collected")
    campaign = read_json(campaign_path)
    validate_campaign(campaign)
    receipt = read_json(run_dir / "batch_receipt.json")
    client = _client(credential)
    batch = client.messages.batches.retrieve(receipt["batch_id"])
    if str(batch.processing_status) != "ended":
        raise RuntimeError(f"batch has not ended: {batch.processing_status}")
    rows = []
    for item in client.messages.batches.results(receipt["batch_id"]):
        result_type = str(item.result.type)
        row: dict[str, Any] = {"custom_id": str(item.custom_id), "result_type": result_type}
        if result_type == "succeeded":
            message = item.result.message
            content, text = _sanitized_content(list(message.content))
            try:
                decoded = decode_response(text)
                error = None
            except ValueError as exc:
                decoded = None
                error = f"{type(exc).__name__}: {exc}"
            usage = message.usage.model_dump(mode="json")
            model = str(message.model)
            row.update(
                {
                    "model": model,
                    "message_id": str(message.id),
                    "stop_reason": str(message.stop_reason),
                    "content": content,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "strict_decoded": decoded,
                    "strict_valid": decoded is not None,
                    "decode_error": error,
                    "usage": usage,
                    "estimated_batch_cost_usd": cost(model, int(usage["input_tokens"]), int(usage["output_tokens"])),
                }
            )
        else:
            row["error"] = item.result.model_dump(mode="json")
        rows.append(row)
    if len(rows) != 72 or len({row["custom_id"] for row in rows}) != 72:
        raise RuntimeError("batch result cardinality mismatch")
    payload = bind(
        {
            "schema_version": f"{SCHEMA}.responses",
            "batch_id": receipt["batch_id"],
            "campaign_sha256": CAMPAIGN_SHA256,
            "rows": sorted(rows, key=lambda row: row["custom_id"]),
            "api_key_recorded_or_hashed": False,
        }
    )
    write_json(run_dir / "responses.json.gz", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "submit", "status", "collect"):
        p = sub.add_parser(name)
        p.add_argument("--credential", required=True, type=Path)
        p.add_argument("--run-dir", required=True, type=Path)
        if name in {"prepare", "submit", "collect"}:
            p.add_argument("--campaign", required=True, type=Path)
        if name == "prepare":
            p.add_argument("--authorization-reference", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.campaign, args.credential, args.run_dir, args.authorization_reference)
    elif args.command == "submit":
        submit(args.campaign, args.credential, args.run_dir)
    elif args.command == "status":
        status(args.credential, args.run_dir)
    else:
        collect(args.campaign, args.credential, args.run_dir)


if __name__ == "__main__":
    main()

