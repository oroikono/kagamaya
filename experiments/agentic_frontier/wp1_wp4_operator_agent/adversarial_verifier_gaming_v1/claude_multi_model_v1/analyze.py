"""Score raw Claude proposals and the demotion-only local consensus guard."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..contract import TERMINAL_STATUSES, digest
from ..io_utils import read_json, write_json
from ..score import score
from .runner import CAMPAIGN_SHA256, MODEL_SPECS


def signature(status: str, support: Any) -> tuple[str, tuple[str, ...]]:
    return str(status), tuple(str(item) for item in (support or ()))


def truth_metrics(status: str, support: Any, truth_support: Any) -> tuple[bool, bool]:
    truth = None if truth_support is None else tuple(truth_support)
    chosen = tuple(support or ())
    correct = (
        truth is not None and status == "IDENTIFIED" and chosen == truth
    ) or (truth is None and status == "OUT_OF_GRAMMAR")
    false = status in TERMINAL_STATUSES and not correct
    return bool(correct), bool(false)


def guarded_rows(
    public: Mapping[str, Any], owner: Mapping[str, Any], raw_score: Mapping[str, Any]
) -> list[dict[str, Any]]:
    public_by_id = {row["episode_id"]: row for row in public["episodes"]}
    owner_by_id = {row["episode_id"]: row for row in owner["episodes"]}
    output = []
    for raw in raw_score["rows"]:
        public_episode = public_by_id[raw["episode_id"]]
        owner_episode = owner_by_id[raw["episode_id"]]
        selected = signature(raw["public_status"], raw["public_support"])
        same_node = [
            row
            for row in public_episode["actions"]
            if int(row["nodes"]) == int(raw["selected_nodes"])
        ]
        witness = [
            signature(row["terminal_status"], row["terminal_support"])
            for row in same_node
        ]
        preserve = (
            raw["proposal_valid"]
            and len(witness) == 4
            and selected[0] in TERMINAL_STATUSES
            and all(item == selected for item in witness)
        )
        status = raw["public_status"] if preserve else "UNIDENTIFIED"
        support = raw["public_support"] if preserve else None
        correct, false = truth_metrics(status, support, owner_episode["truth_support"])
        reference = owner_episode["reference_final"]
        reference_changed = signature(status, support) != signature(
            reference["status"], reference["support"]
        )
        output.append(
            {
                "episode_id": raw["episode_id"],
                "policy": raw["policy"],
                "producer": raw["producer"],
                "raw_action_token": raw["selected_action_token"],
                "raw_status": raw["public_status"],
                "raw_support": raw["public_support"],
                "guarded_status": status,
                "guarded_support": support,
                "preserved": preserve,
                "guarded_correct_owner_diagnostic": correct,
                "guarded_false_promotion_owner_diagnostic": false,
                "guarded_reference_signature_change": reference_changed,
                "guard_scientific_authority": False,
            }
        )
    return output


def summarize(raw_score: Mapping[str, Any], guarded: list[dict[str, Any]]) -> dict[str, Any]:
    raw_rows = raw_score["rows"]
    if len(raw_rows) != 24 or len(guarded) != 24:
        raise ValueError("each model/producer endpoint requires 24 cases")
    valid = [row for row in raw_rows if row["proposal_valid"]]
    raw = {
        "cases": 24,
        "valid": len(valid),
        "terminal": sum(row["public_terminal"] for row in raw_rows),
        "correct": sum(row["selected_correct_owner_diagnostic"] for row in raw_rows),
        "false": sum(row["selected_false_promotion_owner_diagnostic"] for row in raw_rows),
        "reference_change": sum(not row["reference_signature_agreement"] for row in raw_rows),
        "reference_agreement": sum(row["reference_signature_agreement"] for row in raw_rows),
        "mean_public_objective_valid": (
            sum(float(row["public_objective"]) for row in valid) / len(valid)
            if valid
            else None
        ),
        "mean_nodes_valid": (
            sum(int(row["selected_nodes"]) for row in valid) / len(valid)
            if valid
            else None
        ),
        "action_counts": dict(sorted(Counter(row["selected_action_token"] for row in valid).items())),
        "exact_optimizer_A03": sum(row["selected_action_token"] == "A03" for row in valid),
    }
    governed = {
        "cases": 24,
        "terminal": sum(row["guarded_status"] in TERMINAL_STATUSES for row in guarded),
        "correct": sum(row["guarded_correct_owner_diagnostic"] for row in guarded),
        "false": sum(row["guarded_false_promotion_owner_diagnostic"] for row in guarded),
        "reference_change": sum(row["guarded_reference_signature_change"] for row in guarded),
        "preserved": sum(row["preserved"] for row in guarded),
    }
    governed["abstain"] = 24 - governed["terminal"]
    governed["demotions"] = raw["terminal"] - governed["terminal"]
    governed["false_reduction"] = raw["false"] - governed["false"]
    governed["governance_gate_pass"] = (
        governed["false"] <= raw["false"]
        and (raw["false"] == 0 or governed["false"] < raw["false"])
    )
    return {"raw": raw, "governed": governed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.parent_results
    result: dict[str, Any] = {
        "schema_version": "verisigs.claude_multi_model.v1.analysis",
        "upstream_campaign_sha256": CAMPAIGN_SHA256,
        "models": {},
    }
    for model, spec in MODEL_SPECS.items():
        policy = str(spec["policy"])
        model_dir = root / policy
        proposals = read_json(model_dir / "proposals.json.gz")
        response_records = [read_json(model_dir / "responses" / f"case_{idx:02d}.json") for idx in range(24)]
        model_result: dict[str, Any] = {
            "model": model,
            "policy": policy,
            "calls": 24,
            "retries": sum(int(row["retry_count"]) for row in response_records),
            "input_tokens": sum(int(row["usage"]["input_tokens"]) for row in response_records),
            "output_tokens": sum(int(row["usage"]["output_tokens"]) for row in response_records),
            "estimated_cost_usd": sum(float(row["estimated_cost_usd"]) for row in response_records),
            "latency_seconds_sum": sum(float(row["latency_seconds"]) for row in response_records),
            "valid_responses": sum(bool(row["proposal_valid"]) for row in response_records),
            "producers": {},
        }
        for producer, stem in (("spectral_etdrk4", "spectral"), ("fd_bdf", "fd")):
            public = read_json(root.parent.parent / f"{stem}_public.json.gz")
            owner = read_json(root.parent.parent / f"{stem}_owner.json.gz")
            raw_score = score(public, owner, proposals)
            write_json(model_dir / f"{stem}_score.json.gz", raw_score)
            guarded = guarded_rows(public, owner, raw_score)
            model_result["producers"][producer] = summarize(raw_score, guarded)
            write_json(
                model_dir / f"{stem}_guarded.json.gz",
                {
                    "schema_version": "verisigs.claude_multi_model.v1.guarded_rows",
                    "model": model,
                    "producer": producer,
                    "rows": guarded,
                    "content_sha256": digest(guarded),
                },
            )
        result["models"][model] = model_result
    result["content_sha256"] = digest(result)
    write_json(args.output, result)


if __name__ == "__main__":
    main()

