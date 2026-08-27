"""Construct public-only consensus proposals and score after serialization."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

from ..contract import PUBLIC_ABSTAIN_OBJECTIVE, digest
from ..io_utils import read_json, write_json
from ..score import score


LANE = Path(__file__).resolve().parent
FREEZE = LANE / "DESIGN_FREEZE_PREEXECUTION.md"
POLICIES = (
    "adjacent_node_consensus",
    "all_node_consensus",
    "all_mode_consensus",
    "two_axis_consensus",
)


def signature(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(row["terminal_status"]), tuple(row.get("terminal_support") or ())


def terminal(row: Mapping[str, Any]) -> bool:
    return str(row["terminal_status"]) in {"IDENTIFIED", "OUT_OF_GRAMMAR"}


def choose(episode: Mapping[str, Any], policy: str) -> tuple[str | None, int, list[str]]:
    rows = list(episode["actions"])
    by = {(int(r["nodes"]), int(r["temporal_modes"])): r for r in rows}
    candidates: list[tuple[float, int, str, list[str]]] = []
    if policy == "adjacent_node_consensus":
        for modes in (2, 3, 4, 5):
            for low, high in ((7, 13), (13, 25)):
                pair = [by[(low, modes)], by[(high, modes)]]
                if terminal(pair[0]) and signature(pair[0]) == signature(pair[1]):
                    candidates.append(
                        (min(float(r["public_objective"]) for r in pair), high,
                         str(pair[1]["action_token"]), [str(r["action_token"]) for r in pair])
                    )
    elif policy == "all_node_consensus":
        for modes in (2, 3, 4, 5):
            group = [by[(nodes, modes)] for nodes in (7, 13, 25)]
            if terminal(group[0]) and len({signature(r) for r in group}) == 1:
                candidates.append(
                    (min(float(r["public_objective"]) for r in group), 25,
                     str(group[-1]["action_token"]), [str(r["action_token"]) for r in group])
                )
    elif policy == "all_mode_consensus":
        for nodes in (7, 13, 25):
            group = [by[(nodes, modes)] for modes in (2, 3, 4, 5)]
            if terminal(group[0]) and len({signature(r) for r in group}) == 1:
                candidates.append(
                    (min(float(r["public_objective"]) for r in group), nodes,
                     str(group[-1]["action_token"]), [str(r["action_token"]) for r in group])
                )
    elif policy == "two_axis_consensus":
        groups: dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = {}
        for row in rows:
            if terminal(row):
                groups.setdefault(signature(row), []).append(row)
        for group in groups.values():
            nodes = {int(r["nodes"]) for r in group}
            modes = {int(r["temporal_modes"]) for r in group}
            if len(nodes) >= 2 and len(modes) >= 2:
                selected = min(
                    group,
                    key=lambda r: (
                        int(r["nodes"]), -float(r["public_objective"]), str(r["action_token"])
                    ),
                )
                candidates.append(
                    (float(selected["public_objective"]), max(nodes),
                     str(selected["action_token"]), [str(r["action_token"]) for r in group])
                )
    else:
        raise ValueError(f"unknown policy {policy}")
    if not candidates:
        # Every consensus policy receives and inspects the complete public
        # 25-node bank.  An abstention therefore cannot be misreported as
        # zero evidence use merely because it emits no action token.
        return None, 25, []
    objective, consulted, token, evidence = min(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    )
    return token, consulted, sorted(evidence)


def proposal_campaign(public: Mapping[str, Any]) -> dict:
    proposals = []
    producer = str(public["config"]["producer"])
    for episode in public["episodes"]:
        for policy in POLICIES:
            token, consulted, evidence = choose(episode, policy)
            body = {
                "schema_version": "verisigs.adversarial_verifier_gaming.consensus_v1.proposal",
                "episode_id": episode["episode_id"],
                "producer": producer,
                "public_episode_sha256": episode["episode_sha256"],
                "policy": policy,
                "proposal_valid": token is not None,
                "selected_action_token": token,
                "scientific_authority": False,
                "hidden_reference_accessed": False,
                "metadata": {
                    "consulted_nodes": consulted,
                    "consensus_evidence_tokens": evidence,
                },
            }
            body["proposal_sha256"] = digest(body)
            proposals.append(body)
    payload = {
        "schema_version": "verisigs.adversarial_verifier_gaming.consensus_v1.campaign",
        "public_campaign_sha256": public["campaign_sha256"],
        "config": {"producer": producer, "policies": list(POLICIES)},
        "proposals": proposals,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


def enrich(scored: dict, proposals: dict) -> dict:
    metadata = {p["proposal_sha256"]: p["metadata"] for p in proposals["proposals"]}
    for row in scored["rows"]:
        row["consulted_nodes"] = metadata[row["proposal_sha256"]]["consulted_nodes"]
        if not row["proposal_valid"]:
            row["public_objective_unconditional"] = PUBLIC_ABSTAIN_OBJECTIVE
    for key, summary in scored["summaries"].items():
        producer, policy = key.split("::", 1)
        rows = [r for r in scored["rows"] if r["producer"] == producer and r["policy"] == policy]
        summary["mean_consensus_witness_nodes"] = (
            sum(r["consulted_nodes"] for r in rows) / len(rows)
        )
    body = dict(scored)
    body.pop("campaign_sha256", None)
    body["campaign_sha256"] = digest(body)
    return body


def run(public_path: Path, owner_path: Path) -> dict:
    public, owner = read_json(public_path), read_json(owner_path)
    proposals = proposal_campaign(public)
    scored = enrich(score(public, owner, proposals), proposals)
    producer = str(public["config"]["producer"])
    useful = {}
    for policy in POLICIES:
        summary = scored["summaries"][f"{producer}::{policy}"]
        useful[policy] = {
            # Literal mismatch to the locked reference.  This deliberately
            # counts an abstention against a terminal reference as a changed
            # terminal rather than hiding it behind the narrower Goodhart
            # predicate (public terminal AND changed reference).
            "reference_change_rate": 1.0 - summary["reference_signature_agreement_rate"],
            "goodhart_event_rate": summary["goodhart_event_rate"],
            "reference_agreement_rate": summary["reference_signature_agreement_rate"],
            "public_terminal_rate": summary["public_terminal_rate"],
            "valid_rate": summary["valid_rate"],
            "mean_public_objective_unconditional": summary["mean_public_objective_unconditional"],
            "mean_consensus_witness_nodes": summary["mean_consensus_witness_nodes"],
        }
    return {
        "schema": "verisigs.adversarial_verifier_gaming.consensus_screen.v1",
        "status": "EXPLORATORY_LOCKED_REPLAY_ONLY",
        "freeze_sha256": hashlib.sha256(FREEZE.read_bytes()).hexdigest(),
        "producer": producer,
        "public_campaign_sha256": public["campaign_sha256"],
        "owner_campaign_sha256": owner["campaign_sha256"],
        "proposal_campaign_sha256": proposals["campaign_sha256"],
        "summary": useful,
        "scored_campaign": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.public, args.owner)
    write_json(args.output, payload)
    print(payload["summary"])


if __name__ == "__main__":
    main()
