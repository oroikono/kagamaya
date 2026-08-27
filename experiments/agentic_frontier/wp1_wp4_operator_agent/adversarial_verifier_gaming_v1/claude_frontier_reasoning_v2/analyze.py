"""Owner-isolated scoring and exact token/cost summaries."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ..contract import digest
from ..io_utils import read_json, write_json
from ..score import score
from ..claude_multi_model_v1.analyze import guarded_rows, summarize
from ..claude_multi_model_v1.parser_sensitivity import fence_only_decode
from .contract import CAMPAIGN_SHA256, MODEL_SPECS, SCHEMA, bind, cost


def model_from_custom_id(custom_id: str) -> str:
    for model in MODEL_SPECS:
        prefix = model.removeprefix("claude-") + "-"
        if custom_id.startswith(prefix):
            return model
    raise ValueError("unknown custom-id model prefix")


def case_from_custom_id(custom_id: str) -> str:
    marker = "-case_"
    if marker not in custom_id:
        raise ValueError("custom id lacks case marker")
    return "case_" + custom_id.rsplit(marker, 1)[1]


def proposals_for_model(campaign: dict[str, Any], rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    by_case = {case["case_id"]: case for case in campaign["cases"]}
    policy = str(MODEL_SPECS[model]["policy"])
    proposals = []
    for row in rows:
        case = by_case[case_from_custom_id(row["custom_id"])]
        decoded = row.get("strict_decoded") if row.get("result_type") == "succeeded" else None
        for producer, key in (("spectral_etdrk4", "spectral_etdrk4_action"), ("fd_bdf", "fd_bdf_action")):
            binding = case["bindings"][producer]
            proposal: dict[str, Any] = {
                "schema_version": "verisigs.adversarial_verifier_gaming.v1.proposal",
                "episode_id": binding["episode_id"],
                "producer": producer,
                "public_episode_sha256": binding["public_episode_sha256"],
                "policy": policy,
                "proposal_valid": decoded is not None,
                "selected_action_token": None if decoded is None else decoded[key],
                "scientific_authority": False,
                "hidden_reference_accessed": False,
                "metadata": {
                    "case_id": case["case_id"],
                    "model": model,
                    "prompt_sha256": case["prompt_sha256"],
                    "response_sha256": row.get("text_sha256"),
                    "adaptive_thinking": True,
                    "effort": "medium",
                },
            }
            proposal["proposal_sha256"] = digest(proposal)
            proposals.append(proposal)
    payload = {
        "schema_version": f"{SCHEMA}.proposal_campaign",
        "source_public_campaigns": campaign["source_public_campaigns"],
        "policies": [policy],
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def reasoning_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    thinking = [block for row in rows for block in row.get("content", []) if block.get("type") in {"thinking", "redacted_thinking"}]
    summaries = [str(block.get("thinking", "")) for block in thinking]
    return {
        "returned_thinking_blocks": len(thinking),
        "nonempty_thinking_summaries": sum(bool(text) for text in summaries),
        "thinking_summary_characters": sum(len(text) for text in summaries),
        "thinking_signatures_stored_raw": False,
        "thinking_signature_hashes_present": sum(bool(block.get("signature_sha256")) for block in thinking),
        "full_hidden_chain_of_thought_available": False,
        "usage_does_not_split_thinking_from_other_output_tokens": True,
    }


def parser_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = 0
    actions: Counter[str] = Counter()
    for row in rows:
        if row.get("strict_valid") or row.get("result_type") != "succeeded":
            continue
        try:
            decoded = fence_only_decode(str(row["text"]))
        except ValueError:
            continue
        recovered += 1
        actions.update(decoded.values())
    return {"strict_invalid_recovered_by_exact_one_fence": recovered, "recovered_action_counts": dict(sorted(actions.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--spectral-public", required=True, type=Path)
    parser.add_argument("--spectral-owner", required=True, type=Path)
    parser.add_argument("--fd-public", required=True, type=Path)
    parser.add_argument("--fd-owner", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    campaign = read_json(args.campaign)
    responses = read_json(args.run_dir / "responses.json.gz")
    if campaign["campaign_sha256"] != CAMPAIGN_SHA256 or responses["campaign_sha256"] != CAMPAIGN_SHA256:
        raise ValueError("campaign binding mismatch")
    public_owner = {
        "spectral_etdrk4": (read_json(args.spectral_public), read_json(args.spectral_owner)),
        "fd_bdf": (read_json(args.fd_public), read_json(args.fd_owner)),
    }
    result: dict[str, Any] = {"schema_version": f"{SCHEMA}.analysis", "campaign_sha256": CAMPAIGN_SHA256, "models": {}}
    for model in MODEL_SPECS:
        rows = [row for row in responses["rows"] if model_from_custom_id(row["custom_id"]) == model]
        proposals = proposals_for_model(campaign, rows, model)
        model_dir = args.run_dir / str(MODEL_SPECS[model]["policy"])
        write_json(model_dir / "proposals.json.gz", proposals)
        succeeded = [row for row in rows if row["result_type"] == "succeeded"]
        model_result: dict[str, Any] = {
            "model": model,
            "policy": MODEL_SPECS[model]["policy"],
            "calls": len(rows),
            "succeeded": len(succeeded),
            "errored_or_refused": len(rows) - len(succeeded),
            "strict_valid": sum(bool(row.get("strict_valid")) for row in rows),
            "input_tokens": sum(int(row.get("usage", {}).get("input_tokens", 0)) for row in rows),
            "output_tokens_including_thinking": sum(int(row.get("usage", {}).get("output_tokens", 0)) for row in rows),
            "estimated_batch_cost_usd": sum(float(row.get("estimated_batch_cost_usd", 0)) for row in rows),
            "reasoning": reasoning_summary(rows),
            "parser_sensitivity": parser_sensitivity(rows),
            "producers": {},
        }
        for producer, (public, owner) in public_owner.items():
            raw = score(public, owner, proposals)
            guarded = guarded_rows(public, owner, raw)
            stem = "spectral" if producer == "spectral_etdrk4" else "fd"
            write_json(model_dir / f"{stem}_score.json.gz", raw)
            write_json(model_dir / f"{stem}_guarded.json.gz", bind({"schema_version": f"{SCHEMA}.guarded", "rows": guarded}))
            model_result["producers"][producer] = summarize(raw, guarded)
        result["models"][model] = model_result
    result["total"] = {
        "calls": sum(row["calls"] for row in result["models"].values()),
        "input_tokens": sum(row["input_tokens"] for row in result["models"].values()),
        "output_tokens_including_thinking": sum(row["output_tokens_including_thinking"] for row in result["models"].values()),
        "estimated_batch_cost_usd": sum(row["estimated_batch_cost_usd"] for row in result["models"].values()),
    }
    result = bind(result)
    write_json(args.output, result)


if __name__ == "__main__":
    main()

