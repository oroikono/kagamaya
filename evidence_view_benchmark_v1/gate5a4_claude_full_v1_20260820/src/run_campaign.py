"""Run the authorized smoke or full public campaign with hard spend reservations."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contract import (
    CONCURRENCY,
    HARD_CAP_USD,
    LANE_ROOT,
    MODEL_SPECS,
    MODELS,
    SMOKE_CASES,
    canonical_json,
    digest,
    file_sha256,
    read_authorization,
)
from .provider import ForcedToolProvider


class BudgetLedger:
    def __init__(self, cap: float, prior_spend: float = 0.0) -> None:
        if prior_spend < 0 or prior_spend > cap:
            raise ValueError("prior spend is outside the authorized grant")
        self.cap = cap
        self.prior_spend = prior_spend
        self.observed = prior_spend
        self.reserved = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self, upper: float) -> None:
        async with self.lock:
            if self.observed + self.reserved + upper > self.cap + 1e-12:
                raise RuntimeError("hard paid-API cap would be exceeded")
            self.reserved += upper

    async def settle(self, upper: float, actual: float) -> None:
        async with self.lock:
            self.reserved -= upper
            self.observed += actual
            if self.observed > self.cap + 1e-12:
                raise RuntimeError("observed spend exceeded hard cap")

    async def retain_failed_reservation(self, upper: float) -> None:
        async with self.lock:
            self.reserved -= upper
            self.observed += upper


class JsonlJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = asyncio.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)

    async def append(self, row: dict[str, Any]) -> None:
        payload = (canonical_json(row) + "\n").encode("utf-8")
        async with self.lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def load_inventory() -> list[dict[str, Any]]:
    path = LANE_ROOT / "PROMPT_INVENTORY.jsonl"
    expected = json.loads((LANE_ROOT / "SOURCE_MANIFEST.json").read_text())
    if file_sha256(path) != expected["prompt_inventory_sha256"]:
        raise ValueError("prompt inventory hash mismatch")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def work_items(stage: str) -> list[dict[str, Any]]:
    inventory = load_inventory()
    if stage == "smoke":
        selected = [row for row in inventory if row["module"] == "M3" and row["arm"] == "A1" and row["case_id"] in SMOKE_CASES]
    elif stage == "full":
        selected = inventory
    else:
        raise ValueError("unknown stage")
    rows = []
    for model in MODELS:
        for request in selected:
            rows.append({"model": model, "request": request})
    expected = 6 if stage == "smoke" else 3240
    if len(rows) != expected:
        raise AssertionError(f"wrong {stage} cardinality")
    return rows


async def execute(args: argparse.Namespace) -> None:
    authorization = read_authorization()
    provider = ForcedToolProvider(args.credential)
    items = work_items(args.stage)
    ledger = BudgetLedger(
        float(authorization["max_paid_api_usd"]), prior_spend=float(args.prior_spend_usd)
    )
    journal = JsonlJournal(args.output)
    semaphore = asyncio.Semaphore(args.concurrency)
    manifest_name = "SOURCE_MANIFEST_FULL_R2.json" if args.stage == "full" else "SOURCE_MANIFEST.json"
    source_manifest = json.loads((LANE_ROOT / manifest_name).read_text())

    async def one(ordinal: int, item: dict[str, Any]) -> None:
        async with semaphore:
            model = item["model"]
            request = item["request"]
            spec = MODEL_SPECS[model]
            count = await provider.count_tokens(spec=spec, prompt=request["prompt_text"], module=request["module"])
            upper = (
                count * spec.input_usd_per_million
                + request["max_output_tokens"] * spec.output_usd_per_million
            ) / 1_000_000
            await ledger.reserve(upper)
            try:
                response = await provider.generate(
                    spec=spec,
                    prompt=request["prompt_text"],
                    module=request["module"],
                    max_output_tokens=request["max_output_tokens"],
                )
            except Exception as exc:
                await ledger.retain_failed_reservation(upper)
                incident = {
                    "schema_version": "evidence_view_benchmark.claude_full_incident.v1",
                    "stage": args.stage,
                    "ordinal": ordinal,
                    "model_requested": model,
                    "case_id": request["case_id"],
                    "module": request["module"],
                    "arm": request["arm"],
                    "prompt_sha256": request["prompt_sha256"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "cost_upper_retained_usd": upper,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                incident["record_sha256"] = digest(incident)
                await journal.append(incident)
                return
            if response["resolved_model_id"] not in spec.accepted_resolved_ids:
                response["tool_call_valid"] = False
                response["tool_parse_error"] = (
                    f"resolved identity {response['resolved_model_id']!r} not in frozen accepted identities"
                )
            usage = response["usage"]
            if usage["input_tokens"] != count:
                response["tool_call_valid"] = False
                response["tool_parse_error"] = "generation input usage differs from token-count plan"
            actual = (
                usage["input_tokens"] * spec.input_usd_per_million
                + usage["output_tokens"] * spec.output_usd_per_million
            ) / 1_000_000
            await ledger.settle(upper, actual)
            record = {
                "schema_version": "evidence_view_benchmark.claude_full_raw.v1",
                "stage": args.stage,
                "scored": False,
                "owner_side_opened": False,
                "ordinal": ordinal,
                "model_requested": model,
                "case_id": request["case_id"],
                "case_sha256": request["case_sha256"],
                "module": request["module"],
                "arm": request["arm"],
                "prompt_sha256": request["prompt_sha256"],
                "tool_name": request["tool_name"],
                "max_output_tokens": request["max_output_tokens"],
                "token_count_plan": count,
                "list_price_upper_usd": upper,
                "observed_list_price_usd": actual,
                "source_closure_sha256": source_manifest["source_closure_sha256"],
                "credential_recorded_or_hashed": False,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                **response,
            }
            record["record_sha256"] = digest(record)
            await journal.append(record)

    await asyncio.gather(*(one(index, item) for index, item in enumerate(items)))
    rows = [json.loads(line) for line in args.output.read_text().splitlines() if line]
    summary = {
        "schema_version": "evidence_view_benchmark.claude_full_run_summary.v1",
        "stage": args.stage,
        "expected_records": len(items),
        "observed_records": len(rows),
        "valid_tool_calls": sum(row.get("tool_call_valid") is True for row in rows),
        "incident_records": sum(row.get("schema_version", "").endswith("incident.v1") for row in rows),
        "prior_grant_spend_usd": ledger.prior_spend,
        "observed_or_reserved_spend_usd": ledger.observed,
        "hard_cap_usd": HARD_CAP_USD,
        "raw_sha256": file_sha256(args.output),
    }
    summary["content_sha256"] = digest(summary)
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(summary) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--prior-spend-usd", type=float, default=0.0)
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
