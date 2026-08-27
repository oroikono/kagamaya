"""Independent structural validators for fresh consensus artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..contract import digest
from ..io_utils import read_json
from .confirmation import CONFIRMATION_SEEDS, POLICIES, PRODUCERS, design_sha256, source_sha256


def _campaign_hash(payload: dict[str, Any]) -> bool:
    body = dict(payload)
    claimed = body.pop("campaign_sha256")
    return claimed == digest(body)


def validate_shard(payload: dict[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("content_sha256")
    if claimed != digest(body):
        raise ValueError("shard content hash mismatch")
    config = payload["config"]
    producer, seed = config["producer"], int(config["seed"])
    if producer not in PRODUCERS or seed not in CONFIRMATION_SEEDS:
        raise ValueError("unfrozen producer/seed")
    if payload["design_sha256"] != design_sha256() or payload["source_closure_sha256"] != source_sha256():
        raise ValueError("source/design closure mismatch")
    public, owner, proposals, scored = (payload[key] for key in ("public", "owner", "proposals", "scored"))
    if not all(_campaign_hash(item) for item in (public, owner, proposals, scored)):
        raise ValueError("nested campaign hash mismatch")
    if public["episode_count"] != 36 or owner["episode_count"] != 36:
        raise ValueError("episode coverage mismatch")
    if owner["public_campaign_sha256"] != public["campaign_sha256"]:
        raise ValueError("owner/public binding mismatch")
    if proposals["public_campaign_sha256"] != public["campaign_sha256"]:
        raise ValueError("proposal/public binding mismatch")
    if len(proposals["proposals"]) != 36 * len(POLICIES) or len(scored["rows"]) != 36 * len(POLICIES):
        raise ValueError("proposal/score coverage mismatch")
    policy_counts = {policy: 0 for policy in POLICIES}
    for row in proposals["proposals"]:
        policy_counts[row["policy"]] += 1
        if row["scientific_authority"] is not False or row["hidden_reference_accessed"] is not False:
            raise ValueError("proposal authority violation")
    if set(policy_counts.values()) != {36}:
        raise ValueError("policy coverage mismatch")


def validate_aggregate(payload: dict[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("content_sha256")
    if claimed != digest(body):
        raise ValueError("aggregate content hash mismatch")
    if payload["design_sha256"] != design_sha256() or payload["source_closure_sha256"] != source_sha256():
        raise ValueError("aggregate source/design mismatch")
    if len(payload["shards"]) != len(PRODUCERS) * len(CONFIRMATION_SEEDS):
        raise ValueError("aggregate shard count mismatch")
    expected_status = "CONFIRMATION_PASS" if all(v["all_passed"] for v in payload["gates"].values()) else "CONFIRMATION_FAIL"
    if payload["status"] != expected_status:
        raise ValueError("aggregate verdict mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    payload = read_json(args.artifact)
    (validate_aggregate if args.aggregate else validate_shard)(payload)
    print(f"validated {args.artifact}")


if __name__ == "__main__":
    main()
