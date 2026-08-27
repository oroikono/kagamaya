"""Plan and execute the authorized M5 repair under the cumulative spend cap."""

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
    PRIOR_SPEND_USD,
    SOURCE_MANIFEST_PATH,
    canonical_json,
    digest,
    file_sha256,
    read_authorization,
    schedule,
)
from .provider import ForcedToolProvider


class BudgetLedger:
    def __init__(self, cap: float, prior: float) -> None:
        if not 0 <= prior <= cap:
            raise ValueError("prior spend outside cap")
        self.cap = cap
        self.prior = prior
        self.observed = prior
        self.reserved = 0.0
        self.lock = asyncio.Lock()

    async def reserve(self, upper: float) -> None:
        async with self.lock:
            if self.observed + self.reserved + upper > self.cap + 1e-12:
                raise RuntimeError("hard cumulative paid-API cap would be exceeded")
            self.reserved += upper

    async def settle(self, upper: float, actual: float) -> None:
        async with self.lock:
            self.reserved -= upper
            self.observed += actual
            if self.observed > self.cap + 1e-12:
                raise RuntimeError("observed spend exceeded cap")

    async def retain(self, upper: float) -> None:
        async with self.lock:
            self.reserved -= upper
            self.observed += upper


class Journal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = asyncio.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)

    async def append(self, value: dict[str, Any]) -> None:
        data = (canonical_json(value) + "\n").encode()
        async with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)


async def execute(args: argparse.Namespace) -> None:
    read_authorization()
    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text())
    provider = ForcedToolProvider(args.credential)
    items = schedule()
    semaphore = asyncio.Semaphore(args.concurrency)

    async def count_one(ordinal: int, item: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            spec = MODEL_SPECS[item["model"]]
            request = item["request"]
            count = await provider.count_tokens(spec=spec, prompt=request["prompt_text"])
            upper = (
                count * spec.input_usd_per_million
                + request["max_output_tokens"] * spec.output_usd_per_million
            ) / 1_000_000
            return {
                "ordinal": ordinal,
                "model": item["model"],
                "case_id": request["case_id"],
                "prompt_sha256": request["prompt_sha256"],
                "input_tokens": count,
                "full_output_ceiling_tokens": request["max_output_tokens"],
                "list_price_upper_usd": upper,
            }

    plan = await asyncio.gather(*(count_one(i, item) for i, item in enumerate(items)))
    plan_upper = sum(row["list_price_upper_usd"] for row in plan)
    if PRIOR_SPEND_USD + plan_upper > HARD_CAP_USD + 1e-12:
        raise RuntimeError("complete token plan exceeds cumulative hard cap; no generation started")
    plan_path = LANE_ROOT / "results/TOKEN_PLAN.jsonl"
    with plan_path.open("x", encoding="utf-8") as handle:
        handle.write("".join(canonical_json(row) + "\n" for row in plan))
    plan_summary = {
        "schema_version": "evidence_view_benchmark.m5_repair_token_plan.v1",
        "call_count": len(plan),
        "prior_spend_usd": PRIOR_SPEND_USD,
        "incremental_upper_usd": plan_upper,
        "cumulative_upper_usd": PRIOR_SPEND_USD + plan_upper,
        "hard_cap_usd": HARD_CAP_USD,
        "plan_sha256": file_sha256(plan_path),
    }
    plan_summary["content_sha256"] = digest(plan_summary)
    with (LANE_ROOT / "results/TOKEN_PLAN_SUMMARY.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(plan_summary) + "\n")

    ledger = BudgetLedger(HARD_CAP_USD, PRIOR_SPEND_USD)
    raw_path = LANE_ROOT / "results/public_raw.jsonl"
    journal = Journal(raw_path)

    async def one(item: dict[str, Any], plan_row: dict[str, Any]) -> None:
        async with semaphore:
            model = item["model"]
            request = item["request"]
            spec = MODEL_SPECS[model]
            upper = float(plan_row["list_price_upper_usd"])
            await ledger.reserve(upper)
            try:
                response = await provider.generate(
                    spec=spec,
                    prompt=request["prompt_text"],
                    max_output_tokens=request["max_output_tokens"],
                )
            except Exception as exc:
                await ledger.retain(upper)
                incident = {
                    "schema_version": "evidence_view_benchmark.m5_repair_incident.v1",
                    "ordinal": plan_row["ordinal"],
                    "model_requested": model,
                    "case_id": request["case_id"],
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
                response["tool_parse_error"] = "resolved model identity outside frozen set"
            if response["usage"]["input_tokens"] != plan_row["input_tokens"]:
                response["tool_call_valid"] = False
                response["tool_parse_error"] = "generation input tokens differ from plan"
            actual = (
                response["usage"]["input_tokens"] * spec.input_usd_per_million
                + response["usage"]["output_tokens"] * spec.output_usd_per_million
            ) / 1_000_000
            await ledger.settle(upper, actual)
            record = {
                "schema_version": "evidence_view_benchmark.m5_repair_raw.v1",
                "ordinal": plan_row["ordinal"],
                "model_requested": model,
                "case_id": request["case_id"],
                "case_sha256": request["case_sha256"],
                "prompt_sha256": request["prompt_sha256"],
                "max_output_tokens": request["max_output_tokens"],
                "token_count_plan": plan_row["input_tokens"],
                "list_price_upper_usd": upper,
                "observed_list_price_usd": actual,
                "source_closure_sha256": manifest["source_closure_sha256"],
                "credential_recorded_or_hashed": False,
                "owner_side_opened": False,
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                **response,
            }
            record["record_sha256"] = digest(record)
            await journal.append(record)

    await asyncio.gather(*(one(item, row) for item, row in zip(items, plan)))
    rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    summary = {
        "schema_version": "evidence_view_benchmark.m5_repair_run_summary.v1",
        "expected_records": 456,
        "observed_records": len(rows),
        "valid_tool_calls": sum(row.get("tool_call_valid") is True for row in rows),
        "incident_records": sum(row.get("schema_version", "").endswith("incident.v1") for row in rows),
        "prior_spend_usd": PRIOR_SPEND_USD,
        "incremental_observed_or_reserved_usd": ledger.observed - PRIOR_SPEND_USD,
        "cumulative_observed_or_reserved_usd": ledger.observed,
        "hard_cap_usd": HARD_CAP_USD,
        "raw_sha256": file_sha256(raw_path),
    }
    summary["content_sha256"] = digest(summary)
    with (LANE_ROOT / "results/RUN_SUMMARY.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(summary) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = parser.parse_args()
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
