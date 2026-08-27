"""Frozen constants for the frontier adaptive-thinking batch."""

from __future__ import annotations

from typing import Any

from ..contract import digest


CAMPAIGN_SHA256 = "75e3c47142478c513cb12b605517d8c368aa35a6c11dc0d121855f59b6cb2bfd"
PRICING_SNAPSHOT = "2026-08-18"
PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
BATCH_URL = "https://platform.claude.com/docs/en/build-with-claude/batch-processing"
MAX_OUTPUT_TOKENS = 512
EFFORT = "medium"
THINKING = {"type": "adaptive", "display": "summarized"}
MAX_AUTHORIZED_COST_USD = 1.75
CALLS_PER_MODEL = 24
SCHEMA = "verisigs.claude_frontier_reasoning.v2"

MODEL_SPECS: dict[str, dict[str, str | float]] = {
    "claude-sonnet-5": {
        "policy": "claude_sonnet_5_adaptive_medium",
        "batch_input_usd_per_million": 1.0,
        "batch_output_usd_per_million": 5.0,
    },
    "claude-opus-5": {
        "policy": "claude_opus_5_adaptive_medium",
        "batch_input_usd_per_million": 2.5,
        "batch_output_usd_per_million": 12.5,
    },
    "claude-fable-5": {
        "policy": "claude_fable_5_adaptive_medium",
        "batch_input_usd_per_million": 5.0,
        "batch_output_usd_per_million": 25.0,
    },
}


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    spec = MODEL_SPECS[model]
    return (
        input_tokens * float(spec["batch_input_usd_per_million"])
        + output_tokens * float(spec["batch_output_usd_per_million"])
    ) / 1_000_000


def bind(payload: dict[str, Any], field: str = "content_sha256") -> dict[str, Any]:
    body = dict(payload)
    body[field] = digest(body)
    return body

