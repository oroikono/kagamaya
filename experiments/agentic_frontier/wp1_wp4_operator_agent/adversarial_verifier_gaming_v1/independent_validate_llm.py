"""Stdlib-only validator for a paired-producer LLM proposal campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bound(payload: Mapping[str, Any], field: str) -> None:
    body = dict(payload)
    claimed = body.pop(field, None)
    if claimed != _digest(body):
        raise ValueError(f"invalid {field}")


def validate(
    campaign: Mapping[str, Any],
    preflight: Mapping[str, Any],
    authorization: Mapping[str, Any],
    proposals: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    attempts: list[Mapping[str, Any]],
    public: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Mapping[str, Any]],
    model_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _bound(campaign, "campaign_sha256")
    _bound(preflight, "preflight_sha256")
    _bound(proposals, "campaign_sha256")
    for payload in (*public.values(), *scores.values()):
        _bound(payload, "campaign_sha256")
    if campaign["call_count"] != 24 or len(campaign["cases"]) != 24:
        raise ValueError("LLM campaign is not exactly 24 calls")
    if preflight["campaign_sha256"] != campaign["campaign_sha256"]:
        raise ValueError("preflight/campaign binding mismatch")
    if preflight["call_count"] != 24 or preflight["retry_count"] != 0:
        raise ValueError("preflight call/retry budget changed")
    if preflight["max_output_tokens"] != 64:
        raise ValueError("preflight output cap changed")
    if authorization != {
        "schema_version": "verisigs.adversarial_verifier_gaming.v1.llm_authorization",
        "preflight_sha256": preflight["preflight_sha256"],
        "maximum_billable_calls": 24,
        "maximum_authorized_cost_usd": 0.0,
        "billable_calls_authorized": True,
        "authorization_reference": "explicit-user-bounded-qwen-2026-08-17",
    }:
        raise ValueError("authorization is not the exact Qwen authorization")
    entry_by_case = {row["case_id"]: row for row in preflight["entries"]}
    if set(entry_by_case) != {row["case_id"] for row in campaign["cases"]}:
        raise ValueError("preflight case coverage mismatch")
    for case in campaign["cases"]:
        if entry_by_case[case["case_id"]]["prompt_sha256"] != case["prompt_sha256"]:
            raise ValueError("preflight prompt binding mismatch")
    if proposals["source_public_campaigns"] != campaign["source_public_campaigns"]:
        raise ValueError("proposal/campaign public-source mismatch")
    if proposals["proposal_count"] != 48 or len(proposals["proposals"]) != 48:
        raise ValueError("proposal count is not exactly 48")
    expected_public = {
        producer: payload["campaign_sha256"] for producer, payload in public.items()
    }
    if campaign["source_public_campaigns"] != expected_public:
        raise ValueError("LLM campaign is not bound to supplied public artifacts")
    if len(records) != len(attempts) or len(records) != 24:
        raise ValueError("record/attempt count is not exactly 24")
    record_by_case = {row["case_id"]: row for row in records}
    attempt_by_case = {row["case_id"]: row for row in attempts}
    if set(record_by_case) != set(entry_by_case) or set(attempt_by_case) != set(entry_by_case):
        raise ValueError("record/attempt case coverage mismatch")
    proposal_hashes = set()
    for case_id in sorted(entry_by_case):
        record, attempt = record_by_case[case_id], attempt_by_case[case_id]
        _bound(record, "record_sha256")
        if attempt != {
            "case_id": case_id,
            "preflight_sha256": preflight["preflight_sha256"],
            "attempt_ordinal": 0,
            "retry_authorized": False,
        }:
            raise ValueError("attempt lock changed")
        if record["retry_count"] != 0 or record["api_key_recorded_or_hashed"] is not False:
            raise ValueError("record retry/credential boundary changed")
        if record["response_sha256"] != _digest({"response": record["raw_response"]}):
            raise ValueError("response hash mismatch")
        if record["usage"]["output_tokens"] > 64:
            raise ValueError("record exceeded output-token cap")
        if record["usage"]["input_tokens"] != entry_by_case[case_id]["exact_input_tokens"]:
            raise ValueError("record/preflight input-token mismatch")
        if len(record["proposals"]) != 2:
            raise ValueError("record did not emit two producer proposals")
        for proposal in record["proposals"]:
            _bound(proposal, "proposal_sha256")
            proposal_hashes.add(proposal["proposal_sha256"])
            if proposal["scientific_authority"] is not False or proposal[
                "hidden_reference_accessed"
            ] is not False:
                raise ValueError("LLM proposal crossed the authority boundary")
    if proposal_hashes != {row["proposal_sha256"] for row in proposals["proposals"]}:
        raise ValueError("record/proposal campaign coverage mismatch")
    if proposals["llm_records_sha256"] != _digest(records):
        raise ValueError("proposal campaign record-list hash mismatch")
    for producer, score in scores.items():
        if score["public_campaign_sha256"] != public[producer]["campaign_sha256"]:
            raise ValueError("score/public binding mismatch")
        if score["proposal_campaign_sha256"] != proposals["campaign_sha256"]:
            raise ValueError("score/proposal binding mismatch")
        if len(score["rows"]) != 24:
            raise ValueError("producer LLM score does not contain 24 rows")
    identity = dict(model_manifest["model_identity"])
    claimed_identity = identity.pop("identity_sha256")
    if claimed_identity != _digest(identity):
        raise ValueError("model identity hash mismatch")
    return {
        "status": "valid",
        "call_count": 24,
        "proposal_count": 48,
        "valid_record_count": sum(bool(row["proposal_valid"]) for row in records),
        "retry_count": sum(int(row["retry_count"]) for row in records),
        "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in records),
        "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in records),
        "generation_latency_seconds": sum(float(row["latency_seconds"]) for row in records),
        "campaign_sha256": campaign["campaign_sha256"],
        "preflight_sha256": preflight["preflight_sha256"],
        "proposal_campaign_sha256": proposals["campaign_sha256"],
        "model_identity_sha256": claimed_identity,
        "api_key_recorded_or_hashed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--response-dir", required=True, type=Path)
    parser.add_argument("--spectral-public", required=True, type=Path)
    parser.add_argument("--fd-public", required=True, type=Path)
    parser.add_argument("--spectral-score", required=True, type=Path)
    parser.add_argument("--fd-score", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = [_load(path) for path in sorted(args.response_dir.glob("case_[0-9][0-9].json"))]
    attempts = [
        _load(path) for path in sorted(args.response_dir.glob("case_[0-9][0-9].attempt.json"))
    ]
    result = validate(
        _load(args.campaign),
        _load(args.preflight),
        _load(args.authorization),
        _load(args.proposals),
        records,
        attempts,
        {
            "spectral_etdrk4": _load(args.spectral_public),
            "fd_bdf": _load(args.fd_public),
        },
        {
            "spectral_etdrk4": _load(args.spectral_score),
            "fd_bdf": _load(args.fd_score),
        },
        _load(args.model_manifest),
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
