"""Run the frozen matched high-noise attack-margin comparison."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .contract import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    NOISES,
    POLICIES,
    PRODUCERS,
    SEEDS,
    digest,
    finite,
    load_freeze,
    percentile,
    publish_exclusive,
)
from ..adversarial_verifier_gaming_v1.campaign import _cell, _public_row
from ..adversarial_verifier_gaming_v1.consensus_mitigation_v1.screen import choose
from ..adversarial_verifier_gaming_v1.contract import ACTIONS
from ..adversarial_verifier_gaming_v1.proposers import propose_non_llm
from ..adversarial_verifier_gaming_v1.public_interface import (
    build_public_episode,
    validate_public_episode,
)
from ..evidence_view_controls_v1.campaign import FAMILIES, _standard_episodes


def signature(final: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(final["status"]), tuple(final.get("support") or ())


def owner_outcome(cell: dict[str, Any]) -> dict[str, Any]:
    final = cell["final"]
    return {
        "status": str(final["status"]),
        "support": None if final.get("support") is None else list(final["support"]),
        "correct": bool(cell["correct"]),
        "false_promotion": bool(cell["false_promotion"]),
        "abstained": str(final["status"]) == "UNIDENTIFIED",
    }


def _episode_id(producer: str, family_index: int, seed_index: int,
                noise_model: str, noise_index: int) -> str:
    # Public IDs are opaque ordinals and contain no family or seed label.
    producer_index = PRODUCERS.index(producer)
    model_index = ("iid", "correlated").index(noise_model)
    ordinal = (((producer_index * len(FAMILIES) + family_index) * len(SEEDS)
                + seed_index) * 2 + model_index) * len(NOISES) + noise_index
    return f"hnam_ep_{ordinal:05d}"


def _policy_locks(public: dict[str, Any]) -> dict[str, dict[str, Any]]:
    validate_public_episode(public)
    confidence = propose_non_llm(public, "confidence_max")
    fixed = propose_non_llm(public, "fixed_25_m5")
    consensus_token, consulted_nodes, evidence = choose(public, "all_mode_consensus")
    locks = {
        "reference_49_m5": {
            "selected_action_token": "REFERENCE_49_M5",
            "selected_nodes": 49,
            "selected_modes": 5,
            "proposal_valid": True,
            "public_episode_sha256": public["episode_sha256"],
        },
        "fixed_25_m5": {
            "selected_action_token": fixed["selected_action_token"],
            "selected_nodes": 25,
            "selected_modes": 5,
            "proposal_valid": True,
            "public_episode_sha256": public["episode_sha256"],
        },
        "confidence_max": {
            "selected_action_token": confidence["selected_action_token"],
            "proposal_valid": True,
            "public_episode_sha256": public["episode_sha256"],
        },
        "all_mode_consensus": {
            "selected_action_token": consensus_token,
            "proposal_valid": consensus_token is not None,
            "consulted_nodes": consulted_nodes,
            "consensus_evidence_tokens": evidence,
            "public_episode_sha256": public["episode_sha256"],
        },
    }
    by_token = {row["action_token"]: row for row in public["actions"]}
    for policy in ("confidence_max", "all_mode_consensus"):
        token = locks[policy]["selected_action_token"]
        if token is not None:
            locks[policy]["selected_nodes"] = int(by_token[token]["nodes"])
            locks[policy]["selected_modes"] = int(by_token[token]["temporal_modes"])
    locks["all_mode_consensus"].setdefault("selected_nodes", None)
    locks["all_mode_consensus"].setdefault("selected_modes", None)
    for policy, body in locks.items():
        body["policy"] = policy
        body["lock_sha256"] = digest(body)
    return locks


def _score_episode(*, producer: str, family, family_index: int, seed: int,
                   seed_index: int, noise_model: str, noise: float,
                   baseline, channels, truth) -> list[dict[str, Any]]:
    action_cells: dict[str, dict[str, Any]] = {}
    public_rows = []
    for action in ACTIONS:
        cell = _cell(
            producer, baseline, channels[action.nodes], channels[49],
            truth.support, action.temporal_modes,
        )
        action_cells[action.token] = cell
        public_rows.append(_public_row(action, cell, baseline, channels[action.nodes]))
    episode_id = _episode_id(
        producer, family_index, seed_index, noise_model, NOISES.index(noise)
    )
    public = build_public_episode(
        episode_id=episode_id,
        producer=producer,
        episode_metadata={"noise_model": noise_model, "relative_noise": noise},
        rows=public_rows,
    )
    locks = _policy_locks(public)  # policy decisions occur before owner join
    reference_cell = _cell(
        producer, baseline, channels[49], channels[49], truth.support, 5
    )
    reference = owner_outcome(reference_cell)
    rows = []
    public_by_token = {row["action_token"]: row for row in public["actions"]}
    for policy in POLICIES:
        lock = locks[policy]
        token = lock["selected_action_token"]
        if policy == "reference_49_m5":
            outcome = reference
            objective = None
        elif token is None:
            outcome = {
                "status": "UNIDENTIFIED", "support": None, "correct": False,
                "false_promotion": False, "abstained": True,
            }
            objective = None
        else:
            outcome = owner_outcome(action_cells[str(token)])
            objective = finite(public_by_token[str(token)]["public_objective"])
        row = {
            "episode_id": episode_id,
            "producer": producer,
            "family": family.value,
            "seed": seed,
            "noise_model": noise_model,
            "relative_noise": noise,
            "policy": policy,
            "lock_sha256": lock["lock_sha256"],
            "public_episode_sha256": public["episode_sha256"],
            "selected_action_token": token,
            "selected_nodes": lock.get("selected_nodes"),
            "selected_modes": lock.get("selected_modes"),
            "proposal_valid": bool(lock["proposal_valid"]),
            "selected_public_objective": objective,
            **outcome,
            "reference_status": reference["status"],
            "reference_support": reference["support"],
            "reference_correct": reference["correct"],
            "reference_false_promotion": reference["false_promotion"],
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    return rows


def _bootstrap_difference(rows: list[dict[str, Any]], producer: str,
                          noise: float) -> dict[str, Any]:
    relevant = [row for row in rows if row["producer"] == producer
                and row["relative_noise"] == noise
                and row["policy"] in {"reference_49_m5", "confidence_max"}]
    by = {(row["seed"], row["episode_id"], row["policy"]): row for row in relevant}
    per_seed = {}
    for seed in SEEDS:
        episode_ids = sorted({row["episode_id"] for row in relevant if row["seed"] == seed})
        differences = [
            int(by[(seed, episode_id, "confidence_max")]["false_promotion"])
            - int(by[(seed, episode_id, "reference_49_m5")]["false_promotion"])
            for episode_id in episode_ids
        ]
        per_seed[seed] = sum(differences) / len(differences)
    rng = np.random.default_rng(BOOTSTRAP_SEED + PRODUCERS.index(producer) * 100 + NOISES.index(noise))
    seed_values = np.asarray([per_seed[seed] for seed in SEEDS], dtype=float)
    replicates = [float(np.mean(rng.choice(seed_values, size=len(SEEDS), replace=True)))
                  for _ in range(BOOTSTRAP_REPLICATES)]
    return {
        "method": "descriptive_seed_cluster_bootstrap",
        "replicates": BOOTSTRAP_REPLICATES,
        "estimate": float(seed_values.mean()),
        "interval_95": [percentile(replicates, 0.025), percentile(replicates, 0.975)],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    observed_producers = tuple(p for p in PRODUCERS if any(r["producer"] == p for r in rows))
    if not observed_producers:
        raise ValueError("cannot summarize an empty campaign")
    for producer in observed_producers:
        for noise in NOISES:
            group = [row for row in rows if row["producer"] == producer
                     and row["relative_noise"] == noise]
            for policy in POLICIES:
                subset = [row for row in group if row["policy"] == policy]
                key = f"{producer}::{noise:.2f}::{policy}"
                n = len(subset)
                summaries[key] = {
                    "episodes": n,
                    "correct": sum(bool(row["correct"]) for row in subset),
                    "false_promotions": sum(bool(row["false_promotion"]) for row in subset),
                    "abstentions": sum(bool(row["abstained"]) for row in subset),
                }
                for metric in ("correct", "false_promotions", "abstentions"):
                    summaries[key][f"{metric}_rate"] = summaries[key][metric] / n
                selections[key] = dict(sorted(Counter(
                    str(row["selected_action_token"]) for row in subset
                ).items()))
            by = {(row["episode_id"], row["policy"]): row for row in group}
            episode_ids = sorted({row["episode_id"] for row in group})
            attack_only = reference_only = both_false = neither_false = 0
            for episode_id in episode_ids:
                ref = bool(by[(episode_id, "reference_49_m5")]["false_promotion"])
                attack = bool(by[(episode_id, "confidence_max")]["false_promotion"])
                attack_only += attack and not ref
                reference_only += ref and not attack
                both_false += ref and attack
                neither_false += not ref and not attack
            ref_rate = summaries[f"{producer}::{noise:.2f}::reference_49_m5"]["false_promotions_rate"]
            attack_rate = summaries[f"{producer}::{noise:.2f}::confidence_max"]["false_promotions_rate"]
            paired[f"{producer}::{noise:.2f}"] = {
                "episodes": len(episode_ids),
                "reference_false_rate": ref_rate,
                "attack_false_rate": attack_rate,
                "attack_minus_reference_false_rate": attack_rate - ref_rate,
                "attack_only_false": attack_only,
                "reference_only_false": reference_only,
                "both_false": both_false,
                "neither_false": neither_false,
                "cluster_bootstrap": _bootstrap_difference(rows, producer, noise),
            }
    family_strata = {}
    for producer in observed_producers:
        for noise in NOISES:
            for family in FAMILIES:
                for policy in ("reference_49_m5", "confidence_max"):
                    subset = [row for row in rows if row["producer"] == producer
                              and row["relative_noise"] == noise
                              and row["family"] == family.value and row["policy"] == policy]
                    key = f"{producer}::{noise:.2f}::{family.value}::{policy}"
                    family_strata[key] = {
                        "episodes": len(subset),
                        "false_promotions": sum(bool(r["false_promotion"]) for r in subset),
                        "correct": sum(bool(r["correct"]) for r in subset),
                        "abstentions": sum(bool(r["abstained"]) for r in subset),
                    }
    return {
        "dependence_unit": "seed_cluster",
        "producers_are_paired_not_independent": True,
        "policy_summaries": summaries,
        "paired_attack_margins": paired,
        "selection_distributions": selections,
        "family_strata": family_strata,
    }


def build(producer: str) -> dict[str, Any]:
    freeze = load_freeze()
    if producer not in PRODUCERS:
        raise ValueError("unsupported producer")
    rows: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for seed_index, seed in enumerate(SEEDS):
            episodes = _standard_episodes(producer, family, seed, NOISES)
            for noise_model, noise, baseline, channels, truth in episodes:
                rows.extend(_score_episode(
                    producer=producer, family=family, family_index=family_index,
                    seed=seed, seed_index=seed_index, noise_model=noise_model,
                    noise=noise, baseline=baseline, channels=channels, truth=truth,
                ))
    body = {
        "schema_version": "verisigs.high_noise_attack_margin.result.v1",
        "freeze_sha256": freeze["freeze_sha256"],
        "producer": producer,
        "seeds": list(SEEDS),
        "rows": rows,
        "summary": summarize(rows),
    }
    body["content_sha256"] = digest(body)
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True, choices=PRODUCERS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish_exclusive(args.output, build(args.producer))


if __name__ == "__main__":
    main()
