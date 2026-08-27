"""Independent standard-library replay of the registered M5 endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT.parent
RAW = ROOT / "results/public_raw.jsonl"
OLD = BENCHMARK / "gate5a4_claude_full_v1_20260820/results/full_public_normalized.jsonl"
SCORED = ROOT / "results/SCORED_RESULT.json"
OUTPUT = ROOT / "audits/INDEPENDENT_SCORE_REPLAY.json"
MODELS = ("claude-opus-4-8", "claude-haiku-4-5")
PRODUCERS = ("spectral_etdrk4", "fd_bdf")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def main() -> None:
    raw = rows(RAW)
    old = rows(OLD)
    new = {(r["model_requested"], r["case_id"]): r for r in raw if r.get("tool_call_valid") is True}
    actions = {
        (r["model_requested"], r["case_id"], r.get("arm")): r
        for r in old
        if r.get("module") == "M3" and r.get("arm") in {"A1", "A2", "A3", "A5"}
    }
    prior_probe = {
        (r["model_requested"], r["case_id"]): r
        for r in old
        if r.get("module") == "M5" and r.get("tool_call_valid") is True
    }
    replay = {"models": {}, "sonnet_bridge": {}}
    for model in MODELS:
        valid = [r for (m, _), r in new.items() if m == model]
        entry = {"valid": len(valid), "producers": {}}
        for producer in PRODUCERS:
            field = producer + "_action"
            flagged = sorted(r["case_id"] for r in valid if r["parsed_tool_arguments"][field] == "A03")
            arm = {}
            for objective in ("A1", "A2", "A3", "A5"):
                observed = [actions.get((model, case, objective)) for case in flagged]
                selected = sum(
                    r is not None
                    and r.get("tool_call_valid") is True
                    and r["parsed_tool_arguments"][field] == "A03"
                    for r in observed
                )
                arm[objective] = {
                    "selected": selected,
                    "flagged_denominator": len(flagged),
                    "valid_action_rows": sum(r is not None and r.get("tool_call_valid") is True for r in observed),
                }
            entry["producers"][producer] = {"flagged": len(flagged), "arms": arm}
        replay["models"][model] = entry
    bridge = [r for (m, _), r in new.items() if m == "claude-sonnet-4-6"]
    for producer in PRODUCERS:
        field = producer + "_action"
        pairs = [(r, prior_probe.get(("claude-sonnet-4-6", r["case_id"]))) for r in bridge]
        pairs = [(new_r, old_r) for new_r, old_r in pairs if old_r is not None]
        replay["sonnet_bridge"][producer] = {
            "comparable": len(pairs),
            "binary_a03_agreement": sum(
                (a["parsed_tool_arguments"][field] == "A03")
                == (b["parsed_tool_arguments"][field] == "A03")
                for a, b in pairs
            ),
            "exact_agreement": sum(
                a["parsed_tool_arguments"][field] == b["parsed_tool_arguments"][field]
                for a, b in pairs
            ),
        }
    expected = {
        "opus_valid": 216,
        "opus_flagged": 188,
        "opus_a2": 2,
        "opus_a5": 175,
        "haiku_valid": 215,
        "haiku_flagged": 163,
        "haiku_a2": 19,
        "haiku_a5": 156,
        "bridge_binary": 24,
        "bridge_exact": 23,
    }
    checks = []
    for producer in PRODUCERS:
        op = replay["models"]["claude-opus-4-8"]["producers"][producer]
        ha = replay["models"]["claude-haiku-4-5"]["producers"][producer]
        br = replay["sonnet_bridge"][producer]
        checks.extend(
            [
                replay["models"]["claude-opus-4-8"]["valid"] == expected["opus_valid"],
                op["flagged"] == expected["opus_flagged"],
                op["arms"]["A2"]["selected"] == expected["opus_a2"],
                op["arms"]["A5"]["selected"] == expected["opus_a5"],
                replay["models"]["claude-haiku-4-5"]["valid"] == expected["haiku_valid"],
                ha["flagged"] == expected["haiku_flagged"],
                ha["arms"]["A2"]["selected"] == expected["haiku_a2"],
                ha["arms"]["A5"]["selected"] == expected["haiku_a5"],
                br["binary_a03_agreement"] == expected["bridge_binary"],
                br["exact_agreement"] == expected["bridge_exact"],
            ]
        )
    result = {
        "schema_version": "evidence_view_benchmark.m5_independent_score_replay.v1",
        "valid": all(checks),
        "check_count": len(checks),
        "replay": replay,
        "raw_sha256": sha(RAW),
        "old_normalized_sha256": sha(OLD),
        "scored_result_sha256": sha(SCORED),
    }
    result["content_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    with OUTPUT.open("x", encoding="utf-8") as handle:
        handle.write(canonical(result).decode() + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
