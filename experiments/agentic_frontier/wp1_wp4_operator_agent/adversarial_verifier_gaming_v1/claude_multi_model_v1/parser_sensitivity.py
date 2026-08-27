"""Post-hoc parser sensitivity without new provider calls.

The primary endpoint remains the predeclared strict parser.  This diagnostic
admits exactly one otherwise-valid JSON object wrapped by a single
`````json`` fence.  It never extracts JSON from prose or repairs a truncated
response.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..contract import SCHEMA_PREFIX, digest
from ..io_utils import read_json, write_json
from ..llm_contract import decode_response
from ..score import score
from .analyze import guarded_rows, summarize
from .runner import CAMPAIGN_SHA256


MODEL_DIRS = {
    "claude-haiku-4-5-20251001": "claude_haiku_4_5",
    "claude-sonnet-4-6": "claude_sonnet_4_6",
    "claude-opus-4-6": "claude_opus_4_6",
}


def strip_one_json_fence(text: str) -> str:
    """Strip one exact Markdown JSON fence or leave the text untouched."""

    prefix = "```json\n"
    suffix = "\n```"
    if text.startswith(prefix) and text.endswith(suffix):
        inner = text[len(prefix) : -len(suffix)]
        if "```" in inner:
            raise ValueError("nested fence is not admissible")
        return inner
    return text


def fence_only_decode(text: str) -> dict[str, str]:
    return decode_response(strip_one_json_fence(text))


def recovered_campaign(
    campaign: Mapping[str, Any], records: list[Mapping[str, Any]], model: str, policy: str
) -> tuple[dict[str, Any], Counter[str]]:
    if campaign["campaign_sha256"] != CAMPAIGN_SHA256:
        raise ValueError("campaign does not match the frozen 24-case study")
    by_case = {row["case_id"]: row for row in records}
    if set(by_case) != {row["case_id"] for row in campaign["cases"]}:
        raise ValueError("response coverage mismatch")

    proposals: list[dict[str, Any]] = []
    classes: Counter[str] = Counter()
    for case in campaign["cases"]:
        record = by_case[case["case_id"]]
        raw = str(record["raw_response"])
        strict_valid = bool(record["proposal_valid"])
        try:
            decoded = fence_only_decode(raw)
            sensitivity_valid = True
        except ValueError:
            decoded = None
            sensitivity_valid = False
        if strict_valid:
            classes["strict_valid"] += 1
        elif sensitivity_valid:
            classes["fence_only_recovered"] += 1
        else:
            classes["still_invalid"] += 1

        for producer, response_key in (
            ("spectral_etdrk4", "spectral_etdrk4_action"),
            ("fd_bdf", "fd_bdf_action"),
        ):
            binding = case["bindings"][producer]
            proposal: dict[str, Any] = {
                "schema_version": f"{SCHEMA_PREFIX}.proposal",
                "episode_id": binding["episode_id"],
                "producer": producer,
                "public_episode_sha256": binding["public_episode_sha256"],
                "policy": f"{policy}_fence_sensitivity",
                "proposal_valid": sensitivity_valid,
                "selected_action_token": None if decoded is None else decoded[response_key],
                "scientific_authority": False,
                "hidden_reference_accessed": False,
                "metadata": {
                    "case_id": case["case_id"],
                    "model": model,
                    "response_sha256": record["response_sha256"],
                    "strict_primary_valid": strict_valid,
                    "posthoc_parser_sensitivity": True,
                    "parser": "strip_exact_one_json_fence_then_frozen_strict_decoder",
                },
            }
            proposal["proposal_sha256"] = digest(proposal)
            proposals.append(proposal)

    body: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.proposal_campaign",
        "public_campaign_sha256": None,
        "source_public_campaigns": campaign["source_public_campaigns"],
        "policies": [f"{policy}_fence_sensitivity"],
        "proposal_count": len(proposals),
        "proposals": proposals,
        "llm_records_sha256": digest(records),
        "estimated_cost_usd": 0.0,
        "posthoc_parser_sensitivity": True,
    }
    body["campaign_sha256"] = digest(body)
    return body, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--response-root", required=True, type=Path)
    parser.add_argument("--public-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    campaign = read_json(args.campaign)
    result: dict[str, Any] = {
        "schema_version": "verisigs.claude_multi_model.v1.parser_sensitivity",
        "primary_endpoint_unchanged": True,
        "new_provider_calls": 0,
        "parser": "strip_exact_one_json_fence_then_frozen_strict_decoder",
        "models": {},
    }
    for model, policy in MODEL_DIRS.items():
        model_dir = args.response_root / policy
        records = [
            read_json(model_dir / "responses" / f"case_{index:02d}.json")
            for index in range(24)
        ]
        proposals, classes = recovered_campaign(campaign, records, model, policy)
        write_json(model_dir / "fence_sensitivity_proposals.json.gz", proposals)
        model_result: dict[str, Any] = {
            "response_classes": dict(sorted(classes.items())),
            "producers": {},
        }
        for producer, stem in (("spectral_etdrk4", "spectral"), ("fd_bdf", "fd")):
            public = read_json(args.public_root / f"{stem}_public.json.gz")
            owner = read_json(args.public_root / f"{stem}_owner.json.gz")
            raw_score = score(public, owner, proposals)
            guarded = guarded_rows(public, owner, raw_score)
            write_json(model_dir / f"{stem}_fence_sensitivity_score.json.gz", raw_score)
            model_result["producers"][producer] = summarize(raw_score, guarded)
        result["models"][model] = model_result
    result["content_sha256"] = digest(result)
    write_json(args.output, result)


if __name__ == "__main__":
    main()

# End of module.
