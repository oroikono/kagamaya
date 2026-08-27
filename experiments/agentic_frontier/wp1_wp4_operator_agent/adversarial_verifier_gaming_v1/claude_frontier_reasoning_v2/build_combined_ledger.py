"""Build one exact, machine-readable ledger across the matched LLM arms."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..contract import digest
from ..io_utils import read_json, write_json


def endpoints(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        producer: {
            "valid": values["valid"],
            "terminal": values["terminal"],
            "correct_owner_diagnostic": values["correct_owner_diagnostic"],
            "false_promotion_owner_diagnostic": values["false_promotion_owner_diagnostic"],
            "reference_change": values["reference_change"],
            "guarded_terminal": values["guarded_terminal"],
            "guarded_correct_owner_diagnostic": values["guarded_correct_owner_diagnostic"],
            "guarded_false_promotion_owner_diagnostic": values["guarded_false_promotion_owner_diagnostic"],
        }
        for producer, values in summary.items()
    }


def frontier_endpoints(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        producer: {
            "valid": values["raw"]["valid"],
            "terminal": values["raw"]["terminal"],
            "correct_owner_diagnostic": values["raw"]["correct"],
            "false_promotion_owner_diagnostic": values["raw"]["false"],
            "reference_change": values["raw"]["reference_change"],
            "guarded_terminal": values["governed"]["terminal"],
            "guarded_correct_owner_diagnostic": values["governed"]["correct"],
            "guarded_false_promotion_owner_diagnostic": values["governed"]["false"],
        }
        for producer, values in summary.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-results", required=True, type=Path)
    parser.add_argument("--old-comparison", required=True, type=Path)
    parser.add_argument("--qwen-validation", required=True, type=Path)
    parser.add_argument("--frontier-analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    old = read_json(args.old_results)
    comparison = read_json(args.old_comparison)
    qwen = read_json(args.qwen_validation)
    frontier = read_json(args.frontier_analysis)
    rows: list[dict[str, Any]] = []

    qwen_name = "Qwen/Qwen2.5-7B-Instruct@a09a35458c70"
    rows.append(
        {
            "model": qwen_name,
            "arm": "qwen_local_gpu_strict_primary",
            "calls": qwen["call_count"],
            "strict_valid": qwen["valid_record_count"],
            "input_tokens_model_tokenizer": qwen["input_tokens"],
            "output_tokens_model_tokenizer": qwen["output_tokens"],
            "api_cost_usd": None,
            "cost_scope": "local_A100_compute_not_monetized_in_artifact",
            "adaptive_thinking": False,
            "reasoning_artifact": "final_json_only",
            "endpoints": endpoints(comparison["strict_primary"][qwen_name]),
        }
    )

    for model, values in old["models"].items():
        rows.append(
            {
                "model": model,
                "arm": "claude_4x_direct_strict_primary",
                "calls": values["calls"],
                "strict_valid": values["valid_responses"],
                "input_tokens_provider": values["input_tokens"],
                "output_tokens_provider": values["output_tokens"],
                "api_cost_usd": values["estimated_cost_usd"],
                "cost_scope": "provider_list_price_estimate_direct_api",
                "adaptive_thinking": False,
                "reasoning_artifact": "final_response_text_only",
                "endpoints": endpoints(comparison["strict_primary"][model]),
            }
        )

    for model, values in frontier["models"].items():
        rows.append(
            {
                "model": model,
                "arm": "claude_5_batch_adaptive_medium",
                "calls": values["calls"],
                "strict_valid": values["strict_valid"],
                "input_tokens_provider": values["input_tokens"],
                "output_tokens_including_thinking_provider": values["output_tokens_including_thinking"],
                "api_cost_usd": values["estimated_batch_cost_usd"],
                "cost_scope": "provider_batch_price_estimate_50pct_discount",
                "adaptive_thinking": True,
                "reasoning_artifact": "provider_returned_summarized_thinking_plus_final_text",
                "reasoning_summary": values["reasoning"],
                "endpoints": frontier_endpoints(values["producers"]),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": "verisigs.adversarial_verifier_gaming.v1.all_model_ledger",
        "scope": {
            "cases_per_model": 24,
            "producers": ["spectral_etdrk4", "fd_bdf"],
            "same_public_case_set": True,
            "model_family_ranking_generalization_authorized": False,
            "claude_4x_and_claude_5_decoding_protocols_identical": False,
            "hidden_chain_of_thought_available": False,
        },
        "models": rows,
        "paid_claude_total": {
            "calls": sum(row["calls"] for row in rows if row["model"].startswith("claude-")),
            "input_tokens": sum(row.get("input_tokens_provider", 0) for row in rows),
            "output_tokens": sum(
                row.get("output_tokens_provider", row.get("output_tokens_including_thinking_provider", 0))
                for row in rows
            ),
            "estimated_api_cost_usd": sum(float(row.get("api_cost_usd") or 0) for row in rows),
        },
        "token_comparability": "provider/model tokenizers differ; totals are accounting, not equal-compute evidence",
    }
    payload["content_sha256"] = digest(payload)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
