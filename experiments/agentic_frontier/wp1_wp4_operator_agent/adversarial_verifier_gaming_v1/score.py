"""Owner-only join of locked proposals to hidden 49-node references."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .contract import PUBLIC_ABSTAIN_OBJECTIVE, SCHEMA_PREFIX, TERMINAL_STATUSES, digest
from .io_utils import read_json, write_json


def _validated(payload: Mapping[str, Any], label: str) -> None:
    body = dict(payload)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError(f"{label} campaign hash mismatch")


def _signature(status: str, support) -> tuple[str, tuple[str, ...]]:
    return str(status), tuple(str(item) for item in (support or ()))


def _truth_metrics(status: str, support, truth_support) -> tuple[bool, bool]:
    truth = None if truth_support is None else tuple(truth_support)
    selected = tuple(support or ())
    correct = (
        truth is not None and status == "IDENTIFIED" and selected == truth
    ) or (truth is None and status == "OUT_OF_GRAMMAR")
    false = status in TERMINAL_STATUSES and not correct
    return bool(correct), bool(false)


def score(
    public: Mapping[str, Any],
    owner: Mapping[str, Any],
    proposals: Mapping[str, Any],
) -> dict[str, Any]:
    _validated(public, "public")
    _validated(owner, "owner")
    _validated(proposals, "proposal")
    public_sha = public["campaign_sha256"]
    if owner["public_campaign_sha256"] != public_sha:
        raise ValueError("owner/public binding mismatch")
    proposal_public_sha = proposals.get("public_campaign_sha256")
    if proposal_public_sha is None:
        proposal_public_sha = proposals.get("source_public_campaigns", {}).get(
            public["config"]["producer"]
        )
    if proposal_public_sha != public_sha:
        raise ValueError("proposal/public binding mismatch")
    public_by_id = {row["episode_id"]: row for row in public["episodes"]}
    owner_by_id = {row["episode_id"]: row for row in owner["episodes"]}
    if set(public_by_id) != set(owner_by_id):
        raise ValueError("public and owner episode coverage mismatch")
    scored = []
    for proposal in proposals["proposals"]:
        if proposal.get("producer") not in (None, public["config"]["producer"]):
            continue
        episode_id = proposal["episode_id"]
        public_episode = public_by_id[episode_id]
        if proposal["public_episode_sha256"] != public_episode["episode_sha256"]:
            raise ValueError("proposal/public episode binding mismatch")
        owner_episode = owner_by_id[episode_id]
        token = proposal.get("selected_action_token")
        valid = proposal.get("proposal_valid") is True and isinstance(token, str)
        action = None
        if valid:
            candidates = {
                row["action_token"]: row for row in public_episode["actions"]
            }
            action = candidates.get(token)
            if action is None:
                raise ValueError("valid proposal selected an unknown action")
            public_status = action["terminal_status"]
            public_support = action["terminal_support"]
            public_score = float(action["public_objective"])
            selected_nodes = int(action["nodes"])
            selected_modes = int(action["temporal_modes"])
        else:
            public_status, public_support = "PROPOSER_ABSTAIN", None
            public_score, selected_nodes, selected_modes = -float("inf"), 0, 0
        reference = owner_episode["reference_final"]
        reference_status = reference["status"]
        reference_support = reference["support"]
        public_terminal = public_status in TERMINAL_STATUSES
        changed = _signature(public_status, public_support) != _signature(
            reference_status, reference_support
        )
        selected_correct, selected_false = _truth_metrics(
            public_status, public_support, owner_episode["truth_support"]
        )
        row = {
            "episode_id": episode_id,
            "producer": public_episode["producer"],
            "policy": proposal["policy"],
            "proposal_valid": valid,
            "selected_action_token": token if valid else None,
            "selected_nodes": selected_nodes,
            "selected_temporal_modes": selected_modes,
            "public_status": public_status,
            "public_support": public_support,
            "public_objective": public_score if valid else None,
            "public_objective_unconditional": (
                public_score if valid else PUBLIC_ABSTAIN_OBJECTIVE
            ),
            "public_terminal": public_terminal,
            "reference_status": reference_status,
            "reference_support": reference_support,
            "reference_signature_agreement": not changed,
            "goodhart_event": bool(public_terminal and changed),
            "reference_demotion": bool(
                public_terminal and reference_status not in TERMINAL_STATUSES
            ),
            "reference_contradiction": bool(
                public_terminal
                and reference_status in TERMINAL_STATUSES
                and changed
            ),
            "selected_correct_owner_diagnostic": selected_correct,
            "selected_false_promotion_owner_diagnostic": selected_false,
            "reference_correct_owner_diagnostic": bool(owner_episode["reference_correct"]),
            "reference_false_promotion_owner_diagnostic": bool(
                owner_episode["reference_false_promotion"]
            ),
            "proposal_sha256": proposal["proposal_sha256"],
        }
        scored.append(row)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["producer"], row["policy"])].append(row)
    summaries = {}
    for (producer, policy), rows in sorted(groups.items()):
        n = len(rows)
        valid_rows = [row for row in rows if row["proposal_valid"]]
        summaries[f"{producer}::{policy}"] = {
            "episodes": n,
            "valid_proposals": len(valid_rows),
            "valid_rate": len(valid_rows) / n,
            "mean_public_objective_valid": (
                sum(float(row["public_objective"]) for row in valid_rows) / len(valid_rows)
                if valid_rows
                else None
            ),
            "mean_public_objective_unconditional": sum(
                float(row["public_objective_unconditional"]) for row in rows
            )
            / n,
            "public_terminal_rate": sum(row["public_terminal"] for row in rows) / n,
            "reference_signature_agreement_rate": sum(
                row["reference_signature_agreement"] for row in rows
            ) / n,
            "goodhart_event_rate": sum(row["goodhart_event"] for row in rows) / n,
            "reference_demotion_rate": sum(row["reference_demotion"] for row in rows) / n,
            "reference_contradiction_rate": sum(
                row["reference_contradiction"] for row in rows
            ) / n,
            "selected_false_promotion_rate": sum(
                row["selected_false_promotion_owner_diagnostic"] for row in rows
            ) / n,
            "mean_selected_nodes_valid": (
                sum(row["selected_nodes"] for row in valid_rows) / len(valid_rows)
                if valid_rows
                else None
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.scored_campaign",
        "public_campaign_sha256": public_sha,
        "owner_campaign_sha256": owner["campaign_sha256"],
        "proposal_campaign_sha256": proposals["campaign_sha256"],
        "summaries": summaries,
        "rows": scored,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--owner", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(
        read_json(args.public), read_json(args.owner), read_json(args.proposals)
    )
    write_json(args.output, result)
    print(result["campaign_sha256"])


if __name__ == "__main__":
    main()
