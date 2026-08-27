"""Exhaustive ordered fixed-pair analysis before any learner is admitted."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from .contract import DIAGNOSE_VIEW_ID, GATES, NEVER_SEEN_FAMILIES, VIEW_BANK
from .control import Terminal, score


VIEW_MAP = {view.view_id: view for view in VIEW_BANK}


def load_shards(root: Path) -> tuple[list[dict], dict[str, dict]]:
    episodes: list[dict] = []
    labels: dict[str, dict] = {}
    for path in sorted(root.glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        seed = int(payload["metadata"]["seed"])
        for episode in payload["episodes"]:
            episode["_seed"] = seed
            episodes.append(episode)
        for owner in payload["owner_labels"]:
            if owner["episode_id"] in labels:
                raise ValueError("duplicate owner label across shards")
            labels[owner["episode_id"]] = owner
    if {episode["episode_id"] for episode in episodes} != set(labels):
        raise ValueError("public/owner join mismatch")
    return episodes, labels


def cp_upper(false_count: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    if false_count == total:
        return 1.0
    return float(beta.ppf(confidence, false_count + 1, total - false_count))


def _outcome(episode: dict, owner: dict, diagnose: str, submit: str):
    record = next(row for row in episode["views"] if row["view_id"] == submit)
    cost = VIEW_MAP[diagnose].registered_cost
    if diagnose != submit:
        cost += VIEW_MAP[submit].registered_cost
    terminal = Terminal(record["status"], tuple(record["support"]), "fixed_submit",
                        diagnose, submit, cost)
    return score(terminal, owner), cost


def analyze(episodes: list[dict], labels: dict[str, dict]) -> dict:
    eligible = [episode for episode in episodes
                if labels[episode["episode_id"]]["family"] not in NEVER_SEEN_FAMILIES]
    fixed: list[dict] = []
    for diagnose in VIEW_MAP:
        for submit in VIEW_MAP:
            correct = false = inconclusive = 0
            costs = []
            seed_false = defaultdict(list)
            family_counts = defaultdict(lambda: [0, 0, 0])
            for episode in eligible:
                owner = labels[episode["episode_id"]]
                outcome, cost = _outcome(episode, owner, diagnose, submit)
                correct += int(outcome.correct)
                false += int(outcome.false_promotion)
                inconclusive += int(outcome.inconclusive)
                costs.append(cost)
                seed_false[int(episode["_seed"])].append(int(outcome.false_promotion))
                slot = family_counts[owner["family"]]
                slot[0] += int(outcome.correct)
                slot[1] += int(outcome.false_promotion)
                slot[2] += 1
            fixed.append({
                "diagnose": diagnose, "submit": submit,
                "correct": correct, "false": false, "inconclusive": inconclusive,
                "n": len(eligible), "correct_rate": correct / len(eligible),
                "false_rate": false / len(eligible),
                # Sensitivity only: episodes within a seed are not IID trials.
                "false_iid_cp_upper_95_sensitivity": cp_upper(false, len(eligible)),
                "max_seed_false_rate": max(float(np.mean(values))
                                            for values in seed_false.values()),
                "registered_cost": costs[0],
                "families": {key: {"correct": val[0], "false": val[1], "n": val[2]}
                             for key, val in sorted(family_counts.items())},
            })
    # The gate is a finite-factorial empirical ceiling. Dependence-aware
    # uncertainty is applied to the selected comparator and final paired claim,
    # not by pretending the episode rows are IID Bernoulli trials.
    risk_safe = [row for row in fixed if row["false_rate"] <= GATES.risk_ceiling]
    selected_pool = risk_safe
    best_fixed = max(selected_pool, key=lambda row: (
        row["correct"], -row["false"], -row["registered_cost"],
        row["diagnose"], row["submit"])) if selected_pool else None

    oracle_correct = 0
    submit_choices = Counter()
    for episode in eligible:
        owner = labels[episode["episode_id"]]
        successful = []
        for submit in VIEW_MAP:
            outcome, cost = _outcome(episode, owner, submit, submit)
            if outcome.correct:
                successful.append((cost, submit))
        if successful:
            oracle_correct += 1
            submit_choices[min(successful)[1]] += 1
    best_rate = 0.0 if best_fixed is None else best_fixed["correct_rate"]
    oracle_rate = oracle_correct / len(eligible)
    matched_interface = [row for row in risk_safe
                         if row["diagnose"] == DIAGNOSE_VIEW_ID]
    best_matched_interface = (max(matched_interface, key=lambda row: (
        row["correct"], -row["false"], -row["registered_cost"], row["submit"]))
        if matched_interface else None)
    gate = {
        "finite_risk_safe_fixed_exists": bool(risk_safe),
        "oracle_headroom": oracle_rate - best_rate,
        "oracle_headroom_pass": oracle_rate - best_rate >= GATES.calibration_oracle_headroom_min,
        "unique_oracle_submit_views": len(submit_choices),
        "unique_submit_views_pass": len(submit_choices) >= GATES.calibration_unique_submit_views_min,
    }
    gate["learning_admitted"] = all((gate["finite_risk_safe_fixed_exists"],
                                      gate["oracle_headroom_pass"],
                                      gate["unique_submit_views_pass"]))
    return {
        "schema_version": "maintrack.fixed-dominance.v3r2",
        "eligible_episode_count": len(eligible),
        "excluded_never_seen_episode_count": len(episodes) - len(eligible),
        "ordered_fixed_pair_count": len(fixed),
        "risk_safe_pair_count": len(risk_safe),
        "best_fixed": best_fixed,
        "best_same_diagnosis_fixed": best_matched_interface,
        "oracle": {"correct": oracle_correct, "n": len(eligible),
                   "correct_rate": oracle_rate,
                   "submit_choice_counts": dict(sorted(submit_choices.items()))},
        "gate": gate,
        "fixed_pairs": fixed,
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if path.exists():
        raise FileExistsError(path)
    path.write_text(raw + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes, labels = load_shards(args.shards)
    result = analyze(episodes, labels)
    result["content_sha256"] = hashlib.sha256(json.dumps(
        result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "gate": result["gate"],
                      "best_fixed": result["best_fixed"],
                      "oracle": result["oracle"]}, sort_keys=True))


if __name__ == "__main__":
    main()
