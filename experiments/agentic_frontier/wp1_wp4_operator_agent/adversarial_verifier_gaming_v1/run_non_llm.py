"""Lock non-LLM proposals from a public campaign, without owner access."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .contract import SCHEMA_PREFIX, digest
from .io_utils import read_json, write_json
from .proposers import NON_LLM_POLICIES, propose_non_llm


def run(public: dict[str, Any]) -> dict[str, Any]:
    body = dict(public)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError("public campaign hash mismatch")
    proposals = [
        propose_non_llm(episode, policy)
        for episode in public["episodes"]
        for policy in NON_LLM_POLICIES
    ]
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.proposal_campaign",
        "public_campaign_sha256": public["campaign_sha256"],
        "policies": list(NON_LLM_POLICIES),
        "proposal_count": len(proposals),
        "proposals": proposals,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(read_json(args.public))
    write_json(args.output, result)
    print(result["campaign_sha256"])


if __name__ == "__main__":
    main()

