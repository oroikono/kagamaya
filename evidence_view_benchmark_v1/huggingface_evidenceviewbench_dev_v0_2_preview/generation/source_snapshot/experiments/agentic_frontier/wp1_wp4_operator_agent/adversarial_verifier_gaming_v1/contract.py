"""Frozen constants and pure decision functions for verifier gaming v1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA_PREFIX = "verisigs.adversarial_verifier_gaming.v1"
PUBLIC_NODES = (7, 13, 25)
PUBLIC_TEMPORAL_MODES = (2, 3, 4, 5)
REFERENCE_ACTION = (49, 5)
SMOKE_SEEDS = (9950,)
CONFIRMATION_SEEDS = tuple(range(10000, 10050))
NOISE_MODELS = ("iid", "correlated")
RELATIVE_NOISES = (0.0, 0.001, 0.005)
LLM_RELATIVE_NOISES = (0.001, 0.005)
PUBLIC_SEED = 8675309
POLICY_SEED = 4112026
GRAM_RIDGE = 1.0e-12
MARGIN_CLIP = 4.0
MAX_LLM_CALLS = 24
MAX_LLM_OUTPUT_TOKENS = 64
LLM_RETRIES = 0
PUBLIC_ABSTAIN_OBJECTIVE = -4.001
CLAUDE_INPUT_USD_PER_MILLION = 3.0
CLAUDE_OUTPUT_USD_PER_MILLION = 15.0
CLAUDE_MAX_CAMPAIGN_COST_USD = 1.0
CLAUDE_PRICING_SNAPSHOT = "2026-08-17"
CLAUDE_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
TERMINAL_STATUSES = frozenset({"IDENTIFIED", "OUT_OF_GRAMMAR"})
ALLOWED_STATUSES = frozenset(
    {"IDENTIFIED", "OUT_OF_GRAMMAR", "UNIDENTIFIED", "FAILED"}
)


@dataclass(frozen=True)
class Action:
    token: str
    nodes: int
    temporal_modes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ACTIONS = tuple(
    Action(f"A{index:02d}", nodes, modes)
    for index, (nodes, modes) in enumerate(
        (n, r) for n in PUBLIC_NODES for r in PUBLIC_TEMPORAL_MODES
    )
)
ACTION_BY_TOKEN = {action.token: action for action in ACTIONS}
ACTION_BY_CONFIG = {
    (action.nodes, action.temporal_modes): action for action in ACTIONS
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminal_signature(final: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    status = str(final["status"])
    support = tuple(str(item) for item in (final.get("support") or ()))
    return status, support


def _finite(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    value = float(value)
    return value if math.isfinite(value) else fallback


def public_gate_margin(final: Mapping[str, Any]) -> float:
    """Return the frozen clipped slack of the public admission diagnostics."""

    status = str(final["status"])
    residual = _finite(final.get("best_residual"), float("inf"))
    gap = _finite(final.get("score_gap"), float("-inf"))
    condition = _finite(final.get("best_condition"), float("inf"))
    identified = min(
        (0.05 - residual) / 0.05,
        (gap - 0.006) / 0.006,
        (12.0 - condition) / 12.0,
    )
    omitted = (residual - 0.20) / 0.20
    if status == "IDENTIFIED":
        margin = identified
    elif status == "OUT_OF_GRAMMAR":
        margin = omitted
    else:
        margin = max(identified, omitted)
    if not math.isfinite(margin):
        margin = -MARGIN_CLIP
    return float(max(-MARGIN_CLIP, min(MARGIN_CLIP, margin)))


def public_objective(final: Mapping[str, Any], nodes: int) -> float:
    terminal = str(final["status"]) in TERMINAL_STATUSES
    return float(10.0 * int(terminal) + public_gate_margin(final) - 1e-3 * nodes / 49.0)


def stable_integer(*parts: Any) -> int:
    return int(digest(list(parts))[:16], 16)


def design_sha256() -> str:
    return file_sha256(Path(__file__).with_name("DESIGN_FREEZE_PREOUTCOME.md"))
