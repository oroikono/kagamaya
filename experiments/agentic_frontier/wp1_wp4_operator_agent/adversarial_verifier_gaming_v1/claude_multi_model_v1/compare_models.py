"""Produce a compact, claim-safe comparison across Qwen and Claude policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..contract import digest
from ..io_utils import read_json, write_json
from .analyze import guarded_rows, summarize


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    raw, guard = summary["raw"], summary["governed"]
    return {
        "valid": raw["valid"],
        "terminal": raw["terminal"],
        "correct_owner_diagnostic": raw["correct"],
        "false_promotion_owner_diagnostic": raw["false"],
        "reference_change": raw["reference_change"],
        "mean_public_objective_valid": raw["mean_public_objective_valid"],
        "exact_proxy_maximizer_A03": raw["exact_optimizer_A03"],
        "guarded_terminal": guard["terminal"],
        "guarded_correct_owner_diagnostic": guard["correct"],
        "guarded_false_promotion_owner_diagnostic": guard["false"],
        "guarded_abstain": guard["abstain"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-root", required=True, type=Path)
    parser.add_argument("--claude-analysis", required=True, type=Path)
    parser.add_argument("--parser-sensitivity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    claude = read_json(args.claude_analysis)
    sensitivity = read_json(args.parser_sensitivity)
    result: dict[str, Any] = {
        "schema_version": "verisigs.claude_multi_model.v1.comparison",
        "scope": {
            "cases": 24,
            "producers": ["spectral_etdrk4", "fd_bdf"],
            "strict_parser_is_primary": True,
            "fence_only_parser_is_posthoc_sensitivity": True,
            "qwen_guard_is_retrospective": True,
            "claude_guard_was_frozen_before_claude_outcomes": True,
            "model_ranking_generalization_authorized": False,
        },
        "strict_primary": {},
        "fence_only_sensitivity": {},
    }
    for model, row in claude["models"].items():
        result["strict_primary"][model] = {
            producer: compact(summary)
            for producer, summary in row["producers"].items()
        }
    qwen_dir = args.confirmation_root / "llm" / "qwen"
    qwen_by_producer = {}
    for producer, stem in (("spectral_etdrk4", "spectral"), ("fd_bdf", "fd")):
        public = read_json(args.confirmation_root / f"{stem}_public.json.gz")
        owner = read_json(args.confirmation_root / f"{stem}_owner.json.gz")
        raw = read_json(qwen_dir / f"{stem}_score.json.gz")
        qwen_by_producer[producer] = compact(
            summarize(raw, guarded_rows(public, owner, raw))
        )
    result["strict_primary"]["Qwen/Qwen2.5-7B-Instruct@a09a35458c70"] = qwen_by_producer
    for model, row in sensitivity["models"].items():
        result["fence_only_sensitivity"][model] = {
            "response_classes": row["response_classes"],
            "producers": {
                producer: compact(summary)
                for producer, summary in row["producers"].items()
            },
        }
    result["content_sha256"] = digest(result)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
