"""Stdlib-only structural validator, independent of producer/scorer helpers."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TOKENS = tuple(f"A{index:02d}" for index in range(12))
POLICIES = (
    "random",
    "fixed_25_m5",
    "confidence_max",
    "d_optimal",
    "e_optimal",
    "evolution",
)
FORBIDDEN_PUBLIC_KEYS = {
    "truth",
    "truth_support",
    "family",
    "family_name",
    "seed",
    "owner",
    "owner_episode_key",
    "reference",
    "reference_final",
    "reference_correct",
    "reference_false_promotion",
    "correct",
    "false_promotion",
}


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path} is not a JSON object")
    return result


def _digest(value: Any) -> str:
    material = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _hash_bound(payload: Mapping[str, Any], field: str = "campaign_sha256") -> None:
    body = dict(payload)
    claimed = body.pop(field, None)
    if claimed != _digest(body):
        raise ValueError(f"invalid {field}")


def _all_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def validate(
    public: Mapping[str, Any],
    owner: Mapping[str, Any],
    proposals: Mapping[str, Any],
    scored: Mapping[str, Any],
) -> dict[str, Any]:
    for payload in (public, owner, proposals, scored):
        _hash_bound(payload)
    if FORBIDDEN_PUBLIC_KEYS & set(_all_keys(public)):
        raise ValueError("public artifact contains an owner-only key")
    if public["episode_count"] != len(public["episodes"]):
        raise ValueError("public episode count mismatch")
    public_by_id = {}
    for episode in public["episodes"]:
        _hash_bound(episode, "episode_sha256")
        episode_id = episode["episode_id"]
        if episode_id in public_by_id:
            raise ValueError("duplicate public episode ID")
        public_by_id[episode_id] = episode
        if tuple(row["action_token"] for row in episode["actions"]) != TOKENS:
            raise ValueError("public action language/order mismatch")
        if len(episode["actions"]) != 12:
            raise ValueError("public episode does not contain twelve actions")
    owner_by_id = {row["episode_id"]: row for row in owner["episodes"]}
    if len(owner_by_id) != len(owner["episodes"]) or set(owner_by_id) != set(public_by_id):
        raise ValueError("owner/public episode coverage mismatch")
    if owner["episode_count"] != len(owner_by_id):
        raise ValueError("owner episode count mismatch")
    if owner["public_campaign_sha256"] != public["campaign_sha256"]:
        raise ValueError("owner/public hash binding mismatch")
    if any(row["reference_action"] != [49, 5] for row in owner["episodes"]):
        raise ValueError("owner reference action changed")
    if owner["design_sha256"] != public["design_sha256"] or owner[
        "source_closure_sha256"
    ] != public["source_closure_sha256"]:
        raise ValueError("owner/public design or source closure mismatch")
    if proposals["public_campaign_sha256"] != public["campaign_sha256"]:
        raise ValueError("proposal/public hash binding mismatch")
    if tuple(proposals["policies"]) != POLICIES:
        raise ValueError("proposal policy family changed")
    if proposals["proposal_count"] != len(proposals["proposals"]):
        raise ValueError("proposal count mismatch")
    seen = set()
    for proposal in proposals["proposals"]:
        _hash_bound(proposal, "proposal_sha256")
        key = (proposal["episode_id"], proposal["policy"])
        if key in seen:
            raise ValueError("duplicate proposal episode/policy")
        seen.add(key)
        episode = public_by_id[proposal["episode_id"]]
        if proposal["public_episode_sha256"] != episode["episode_sha256"]:
            raise ValueError("proposal/public episode binding mismatch")
        if proposal["scientific_authority"] is not False or proposal[
            "hidden_reference_accessed"
        ] is not False:
            raise ValueError("proposal crossed the authority boundary")
    expected = {(episode_id, policy) for episode_id in public_by_id for policy in POLICIES}
    if seen != expected:
        raise ValueError("proposal coverage mismatch")
    if scored["public_campaign_sha256"] != public["campaign_sha256"]:
        raise ValueError("score/public hash binding mismatch")
    if scored["owner_campaign_sha256"] != owner["campaign_sha256"]:
        raise ValueError("score/owner hash binding mismatch")
    if scored["proposal_campaign_sha256"] != proposals["campaign_sha256"]:
        raise ValueError("score/proposal hash binding mismatch")
    if len(scored["rows"]) != len(proposals["proposals"]):
        raise ValueError("scored/proposal row count mismatch")
    proposal_hashes = {row["proposal_sha256"] for row in proposals["proposals"]}
    scored_hashes = {row["proposal_sha256"] for row in scored["rows"]}
    if proposal_hashes != scored_hashes:
        raise ValueError("scored proposal-hash coverage mismatch")
    for row in scored["rows"]:
        if bool(row["goodhart_event"]) != bool(
            row["public_terminal"] and not row["reference_signature_agreement"]
        ):
            raise ValueError("Goodhart event is inconsistent with terminal/signature fields")
        if (row["reference_demotion"] or row["reference_contradiction"]) and not row[
            "goodhart_event"
        ]:
            raise ValueError("change decomposition is inconsistent")
    return {
        "status": "valid",
        "producer": public["config"]["producer"],
        "episode_count": len(public_by_id),
        "proposal_count": len(proposals["proposals"]),
        "score_row_count": len(scored["rows"]),
        "design_sha256": public["design_sha256"],
        "source_closure_sha256": public["source_closure_sha256"],
        "public_campaign_sha256": public["campaign_sha256"],
        "owner_campaign_sha256": owner["campaign_sha256"],
        "proposal_campaign_sha256": proposals["campaign_sha256"],
        "score_campaign_sha256": scored["campaign_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--owner", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--score", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(
        _load(args.public),
        _load(args.owner),
        _load(args.proposals),
        _load(args.score),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
