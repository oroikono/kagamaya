"""Outcome-blind Gate-5 v1.2 serialization and accounting contract."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = LANE_ROOT / "prompts"
ARMS = ("A1", "A2", "A3", "A4", "A5")
VISIBILITIES = ("all_verdicts", "diagnostics_only", "none")
PRODUCERS = ("fd_bdf", "spectral_etdrk4")
ALLOWED_ACTIONS = tuple(f"A{index:02d}" for index in range(12))
OBJECTIVE_ARMS = frozenset({"A1", "A4"})
FORBIDDEN_PUBLIC_FRAGMENTS = (
    "truth",
    "held_out",
    "owner",
    "oracle",
    "reference",
    "error_to_49",
    "correct",
    "false_promotion",
)
DIAGNOSTIC_FIELDS = frozenset(
    {
        "best_condition",
        "best_residual",
        "gate_margin",
        "gram_condition",
        "gram_lambda_min",
        "gram_logdet",
        "node_cost",
        "score_gap",
        "weak_rows",
    }
)
VERDICT_FIELDS = frozenset({"terminal_status", "terminal_support"})
SOURCE_PRODUCER_FIELDS = DIAGNOSTIC_FIELDS | VERDICT_FIELDS | frozenset({"public_objective"})
SIGNAL_TOKENS = frozenset(
    {
        "terminal_disagreement",
        "cross_producer_disagreement",
        "residual",
        "score_gap",
        "conditioning",
        "resolution_test_space",
        "cost",
        "other",
    }
)
_KEY_PATTERN = re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _assert_public(value: Any) -> None:
    lowered = canonical_json(value).lower()
    for fragment in FORBIDDEN_PUBLIC_FRAGMENTS:
        if fragment in lowered:
            raise ValueError(f"owner-side fragment reached public serialization: {fragment}")


def serialize_rows(case: Mapping[str, Any], *, arm: str, visibility: str) -> list[dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if visibility not in VISIBILITIES:
        raise ValueError(f"unknown visibility: {visibility}")
    source_rows = case.get("public_rows")
    if not isinstance(source_rows, list) or len(source_rows) != 12:
        raise ValueError("expected twelve public rows")
    output: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        if source.get("action_token") != ALLOWED_ACTIONS[index]:
            raise ValueError("action table is not in canonical A00-A11 order")
        if set(source) != {"action_token", "nodes", "temporal_modes", *PRODUCERS}:
            raise ValueError("public source row contains an unexpected field")
        for producer in PRODUCERS:
            if set(source[producer]) != SOURCE_PRODUCER_FIELDS:
                raise ValueError(f"{producer} source payload contains an unexpected field")
        row: dict[str, Any] = {
            "action_token": source["action_token"],
            "nodes": source["nodes"],
            "temporal_modes": source["temporal_modes"],
        }
        if visibility != "none":
            for producer in PRODUCERS:
                source_payload = source[producer]
                allowed = set(DIAGNOSTIC_FIELDS)
                if visibility == "all_verdicts":
                    allowed.update(VERDICT_FIELDS)
                if arm in OBJECTIVE_ARMS:
                    allowed.add("public_objective")
                row[producer] = {key: source_payload[key] for key in sorted(allowed)}
        elif arm in OBJECTIVE_ARMS:
            for producer in PRODUCERS:
                row[producer] = {"public_objective": source[producer]["public_objective"]}
        _assert_public(row)
        output.append(row)
    return output


def _read_prompt(name: str) -> str:
    mapping = {
        "preamble": "shared_preamble.txt",
        "A1": "goal_A1_obedient.txt",
        "A2": "goal_A2_neutral.txt",
        "A3": "goal_A3_incentivized.txt",
        "A4": "goal_A4_proxy_incentive.txt",
        "A5": "goal_A5_novelty_seeking.txt",
        "m3_tail": "oneshot_tail.txt",
        "m5_tail": "gate5/capability_probe_v1_2.txt",
    }
    return (PROMPT_ROOT / mapping[name]).read_text(encoding="utf-8")


def build_m3_prompt(case: Mapping[str, Any], *, arm: str, visibility: str) -> dict[str, Any]:
    prompt = (
        _read_prompt("preamble")
        + _read_prompt(arm)
        + canonical_json(serialize_rows(case, arm=arm, visibility=visibility))
        + "\n"
        + _read_prompt("m3_tail")
    )
    return {
        "module": "M3",
        "arm": arm,
        "visibility": visibility,
        "prompt_text": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
    }


def build_m5_prompt(case: Mapping[str, Any]) -> dict[str, Any]:
    prompt = (
        _read_prompt("preamble")
        + canonical_json(serialize_rows(case, arm="A2", visibility="all_verdicts"))
        + "\n"
        + _read_prompt("m5_tail")
    )
    return {
        "module": "M5",
        "arm": None,
        "visibility": "all_verdicts",
        "prompt_text": prompt,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
    }


def validate_m5_response(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "spectral_etdrk4_action",
        "fd_bdf_action",
        "signals",
        "explanation",
    }:
        raise ValueError("M5 response keys do not match frozen schema")
    for key in ("spectral_etdrk4_action", "fd_bdf_action"):
        if value[key] not in ALLOWED_ACTIONS:
            raise ValueError("M5 response contains invalid action")
    signals = value["signals"]
    if not isinstance(signals, list) or not 1 <= len(signals) <= 3:
        raise ValueError("M5 signals must contain one to three values")
    if len(set(signals)) != len(signals) or any(signal not in SIGNAL_TOKENS for signal in signals):
        raise ValueError("M5 signal token is invalid or duplicated")
    explanation = value["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("M5 explanation must be nonempty")
    if len(explanation.split()) > 80:
        raise ValueError("M5 explanation exceeds 80 words")


def read_cases(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or LANE_ROOT / "cases/cases.jsonl"
    with source.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_roster(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or LANE_ROOT / "model_roster_candidate.json").read_text(encoding="utf-8"))


def gate5_source_bindings() -> dict[str, Any]:
    """Hash every file that defines Gate-5 v1.2 prompts, contracts, and checks."""
    paths = {
        LANE_ROOT / "PROTOCOL-v1_2.md",
        LANE_ROOT / "GATE5_PREFLIGHT_DESIGN.md",
        LANE_ROOT / "model_roster_candidate.json",
    }
    paths.update((LANE_ROOT / "prompts").rglob("*.txt"))
    paths.update((LANE_ROOT / "src/evidence_view_benchmark").glob("*.py"))
    paths.update((LANE_ROOT / "tools").glob("*gate5*.py"))
    paths.update((LANE_ROOT / "tests").glob("test_gate5*.py"))
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path.relative_to(LANE_ROOT)),
                "sha256": sha256_bytes(path.read_bytes()),
            }
        )
    return {
        "files": rows,
        "source_closure_sha256": sha256_bytes(canonical_json(rows).encode("utf-8")),
    }


def validate_roster(roster: Mapping[str, Any]) -> None:
    models = roster.get("models")
    if not isinstance(models, list) or len(models) != 5:
        raise ValueError("Gate 5 requires exactly five frozen candidate models")
    if len({row["vendor"] for row in models}) < 4:
        raise ValueError("Gate 5 requires at least four vendors")
    identities = {(row["vendor"], row["model_id"], row["revision"]) for row in models}
    if len(identities) != 5:
        raise ValueError("model identities are not unique")


def credential_status(anthropic_credential: Path | None = None) -> dict[str, str]:
    anthropic = "ABSENT"
    if anthropic_credential is not None and anthropic_credential.is_file():
        payload = anthropic_credential.read_bytes()
        matches = _KEY_PATTERN.findall(payload)
        anthropic = "PRESENT_VALID_SHAPE" if len(matches) == 1 else "INVALID_OR_AMBIGUOUS"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        anthropic = "PRESENT_ENV_UNINSPECTED"
    return {
        "qwen_local": "REACHABLE_PRIOR_GATE4_PINNED_REVISION",
        "anthropic": anthropic,
        "openai": "PRESENT_ENV_UNINSPECTED" if os.environ.get("OPENAI_API_KEY") else "ABSENT",
        "google": (
            "PRESENT_ENV_UNINSPECTED"
            if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            else "ABSENT"
        ),
    }


def structural_accounting(cases: list[dict[str, Any]], roster: Mapping[str, Any]) -> dict[str, Any]:
    validate_roster(roster)
    if len(cases) != 216:
        raise ValueError("Gate 5 requires exactly 216 cases")
    m3_unique = []
    m5_unique = []
    for case in cases:
        for arm in ARMS:
            for visibility in VISIBILITIES:
                m3_unique.append(build_m3_prompt(case, arm=arm, visibility=visibility))
        m5_unique.append(build_m5_prompt(case))
    m3_calls_per_model = len(m3_unique) * 3
    m5_calls_per_model = len(m5_unique)
    input_bytes_per_model = 3 * sum(
        len(row["prompt_text"].encode("utf-8")) for row in m3_unique
    ) + sum(len(row["prompt_text"].encode("utf-8")) for row in m5_unique)
    output_ceiling_per_model = m3_calls_per_model * 512 + m5_calls_per_model * 384
    model_costs = []
    for model in roster["models"]:
        input_rate = model["input_usd_per_mtoken"]
        output_rate = model["output_usd_per_mtoken"]
        byte_upper_cost = None
        if input_rate is not None and output_rate is not None:
            byte_upper_cost = (
                input_bytes_per_model * input_rate
                + output_ceiling_per_model * output_rate
            ) / 1_000_000
        model_costs.append(
            {
                "vendor": model["vendor"],
                "model_id": model["model_id"],
                "calls": m3_calls_per_model + m5_calls_per_model,
                "input_bytes_token_upper_bound": input_bytes_per_model,
                "output_token_ceiling": output_ceiling_per_model,
                "api_cost_byte_upper_usd": (
                    None if byte_upper_cost is None else round(byte_upper_cost, 6)
                ),
            }
        )
    api_upper = sum(row["api_cost_byte_upper_usd"] or 0.0 for row in model_costs)
    return {
        "case_count": len(cases),
        "unique_m3_prompt_count": len(m3_unique),
        "unique_m5_prompt_count": len(m5_unique),
        "m3_calls_per_model": m3_calls_per_model,
        "m5_calls_per_model": m5_calls_per_model,
        "total_calls_all_models": (m3_calls_per_model + m5_calls_per_model) * 5,
        "input_bytes_token_upper_bound_per_model": input_bytes_per_model,
        "output_token_ceiling_per_model": output_ceiling_per_model,
        "model_costs": model_costs,
        "api_cost_byte_upper_total_usd": round(api_upper, 6),
        "cost_bound_note": (
            "Input bytes are a conservative token upper bound for these ASCII prompts; "
            "output assumes every response reaches its cap. This is a spend ceiling, not "
            "an expected invoice. Provider token preflight will replace it before launch."
        ),
        "prompt_manifest_sha256": sha256_bytes(
            canonical_json(
                [
                    (row["module"], row["arm"], row["visibility"], row["prompt_sha256"])
                    for row in m3_unique + m5_unique
                ]
            ).encode("utf-8")
        ),
    }
