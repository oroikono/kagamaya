#!/usr/bin/env python3
"""Exhaust every agreement-eligible group in the frozen 12-view bank.

This is a retrospective, truth-aware finite-bank enumeration.  It is not an
executable policy or a security guarantee.  Only Python's standard library is
required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


TERMINAL = {"IDENTIFIED", "OUT_OF_GRAMMAR"}
MODES = (2, 3, 4, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(action: dict) -> tuple[str, tuple[str, ...]]:
    return action["terminal_status"], tuple(action.get("terminal_support") or ())


def wrong_kind(action: dict, truth: list[str] | None) -> str | None:
    status, support = signature(action)
    expected = tuple(truth or ())
    if status == "OUT_OF_GRAMMAR" and truth is not None:
        return "false_omission"
    if status == "IDENTIFIED" and support != expected:
        return "false_identification"
    return None


def analyze(shard_dir: Path) -> dict:
    shards = sorted(shard_dir.glob("*.json.gz"))
    if len(shards) != 100:
        raise ValueError(f"expected 100 shards, found {len(shards)}")

    oracle = Counter()
    deployed = Counter()
    individual = Counter()
    cases: list[dict] = []
    individual_id_cases: list[dict] = []
    implementations: set[str] = set()
    episode_ids: set[str] = set()

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        owners = {row["episode_id"]: row for row in payload["owner"]["episodes"]}
        public = {row["episode_id"]: row for row in payload["public"]["episodes"]}
        scored = [
            row for row in payload["scored"]["rows"]
            if row["policy"] == "all_mode_consensus"
        ]
        if len(owners) != 36 or len(public) != 36 or len(scored) != 36:
            raise ValueError(f"unexpected shard cardinality: {shard}")

        for row in scored:
            implementation = row["producer"]
            implementations.add(implementation)
            deployed[(implementation, "episodes")] += 1
            deployed[(implementation, "false")] += int(
                row["selected_false_promotion_owner_diagnostic"]
            )

        for episode_id, episode in public.items():
            if episode_id in episode_ids:
                raise ValueError(f"duplicate episode: {episode_id}")
            episode_ids.add(episode_id)
            owner = owners[episode_id]
            truth = owner["truth_support"]
            implementation = episode["producer"]
            in_library = truth is not None
            oracle[(implementation, "episodes")] += 1
            oracle[(implementation, "in_library")] += int(in_library)

            actions = episode["actions"]
            if len(actions) != 12:
                raise ValueError(f"expected 12 actions: {episode_id}")
            by_group: dict[int, dict[int, dict]] = {}
            for action in actions:
                by_group.setdefault(int(action["nodes"]), {})[
                    int(action["temporal_modes"])
                ] = action
                kind = wrong_kind(action, truth)
                if kind:
                    individual[(implementation, kind)] += 1
                    if kind == "false_identification":
                        individual_id_cases.append(
                            {
                                "episode_id": episode_id,
                                "implementation": implementation,
                                "action": action["action_token"],
                                "nodes": action["nodes"],
                                "modes": action["temporal_modes"],
                                "truth_support": truth,
                                "reported_support": action["terminal_support"],
                            }
                        )

            wrong_groups: list[dict] = []
            for nodes, mode_rows in sorted(by_group.items()):
                if tuple(sorted(mode_rows)) != MODES:
                    raise ValueError(f"incomplete group: {episode_id}, n={nodes}")
                rows = [mode_rows[mode] for mode in MODES]
                signatures = [signature(row) for row in rows]
                eligible = all(row["terminal_status"] in TERMINAL for row in rows)
                eligible = eligible and len(set(signatures)) == 1
                if not eligible:
                    continue
                kind = wrong_kind(rows[-1], truth)
                if kind:
                    wrong_groups.append(
                        {
                            "nodes": nodes,
                            "kind": kind,
                            "signature": [signatures[0][0], list(signatures[0][1])],
                        }
                    )

            if wrong_groups:
                oracle[(implementation, "attackable")] += 1
                if in_library:
                    oracle[(implementation, "attackable_in_library")] += 1
                kinds = {group["kind"] for group in wrong_groups}
                for kind in kinds:
                    oracle[(implementation, kind)] += 1
                cases.append(
                    {
                        "episode_id": episode_id,
                        "implementation": implementation,
                        "owner_episode_key": owner["owner_episode_key"],
                        "wrong_groups": wrong_groups,
                        "reference_status": owner["reference_final"]["status"],
                        "reference_correct": owner["reference_correct"],
                    }
                )

    per_implementation = {}
    for implementation in sorted(implementations):
        per_implementation[implementation] = {
            "episodes": oracle[(implementation, "episodes")],
            "in_library": oracle[(implementation, "in_library")],
            "deployed_rule_false": deployed[(implementation, "false")],
            "exhaustive_attackable": oracle[(implementation, "attackable")],
            "exhaustive_attackable_in_library": oracle[
                (implementation, "attackable_in_library")
            ],
            "confirmed_false_omissions": oracle[
                (implementation, "false_omission")
            ],
            "confirmed_false_identifications": oracle[
                (implementation, "false_identification")
            ],
            "individual_false_identification_rows": individual[
                (implementation, "false_identification")
            ],
        }

    return {
        "analysis": "retrospective_truth_aware_finite_bank_enumeration",
        "security_guarantee": False,
        "shard_count": len(shards),
        "shard_set_sha256": hashlib.sha256(
            "\n".join(f"{sha256(path)}  {path.name}" for path in shards).encode()
        ).hexdigest(),
        "episode_count": len(episode_ids),
        "per_implementation": per_implementation,
        "agreeing_wrong_cases": sorted(cases, key=lambda row: row["episode_id"]),
        "individual_false_identification_cases": sorted(
            individual_id_cases, key=lambda row: (row["episode_id"], row["action"])
        ),
    }


def main() -> None:
    default_shards = (
        Path(__file__).resolve().parents[5]
        / "experiments/agentic_frontier/wp1_wp4_operator_agent"
        / "adversarial_verifier_gaming_v1/consensus_mitigation_v2_confirmation"
        / "results/local_confirmation/shards"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, default=default_shards)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.shards)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
