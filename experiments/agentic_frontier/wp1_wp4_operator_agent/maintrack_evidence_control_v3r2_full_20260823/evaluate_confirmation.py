"""One-pass confirmation scoring against the calibration-selected fixed pair."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from .contract import DIAGNOSE_VIEW_ID, GATES, NEVER_SEEN_FAMILIES, VIEW_BANK
from .control import Terminal, score
from .dominance import load_shards
from .learning import (_attach_seed, action_outcome, evaluate_policy)
from .freeze_confirmation import semantic_digest


VIEW_MAP = {view.view_id: view for view in VIEW_BANK}


def fixed_rows(episodes, labels, diagnose, submit):
    rows = []
    for episode in episodes:
        owner = labels[episode["episode_id"]]
        record = next(row for row in episode["views"] if row["view_id"] == submit)
        cost = VIEW_MAP[diagnose].registered_cost
        if diagnose != submit:
            cost += VIEW_MAP[submit].registered_cost
        terminal = Terminal(record["status"], tuple(record["support"]), "fixed_submit",
                            diagnose, submit, cost)
        outcome = score(terminal, owner)
        rows.append({
            "episode_id": episode["episode_id"], "seed": episode["_seed"],
            "family": owner["family"], "origin": episode["origin"],
            "noise_model": episode["noise_model"], "noise_level": episode["noise_level"],
            "correct": outcome.correct, "false_promotion": outcome.false_promotion,
            "inconclusive": outcome.inconclusive, "registered_cost": cost,
        })
    return rows


def _summary(rows):
    n = len(rows)
    return {"n": n, "correct": sum(row["correct"] for row in rows),
            "false": sum(row["false_promotion"] for row in rows),
            "inconclusive": sum(row["inconclusive"] for row in rows),
            "correct_rate": sum(row["correct"] for row in rows) / n,
            "false_rate": sum(row["false_promotion"] for row in rows) / n,
            "mean_cost": float(np.mean([row["registered_cost"] for row in rows]))}


def cluster_bootstrap(agent, fixed, repetitions=10000):
    fixed_by_id = {row["episode_id"]: row for row in fixed}
    paired = [(row, fixed_by_id[row["episode_id"]]) for row in agent]
    by_seed = defaultdict(list)
    for pair in paired:
        by_seed[pair[0]["seed"]].append(pair)
    seeds = np.asarray(sorted(by_seed))
    rng = np.random.default_rng(20260823)
    correct_diffs = np.empty(repetitions)
    risk_diffs = np.empty(repetitions)
    agent_risks = np.empty(repetitions)
    cost_diffs = np.empty(repetitions)
    for index in range(repetitions):
        draw = rng.choice(seeds, size=len(seeds), replace=True)
        cells = [pair for seed in draw for pair in by_seed[int(seed)]]
        correct_diffs[index] = np.mean([a["correct"] - b["correct"] for a, b in cells])
        risk_diffs[index] = np.mean([a["false_promotion"] - b["false_promotion"]
                                     for a, b in cells])
        agent_risks[index] = np.mean([a["false_promotion"] for a, _ in cells])
        cost_diffs[index] = np.mean([a["registered_cost"] - b["registered_cost"]
                                     for a, b in cells])
    return {
        "repetitions": repetitions,
        "correct_advantage_ci95": np.quantile(correct_diffs, [0.025, 0.975]).tolist(),
        "risk_difference_ci95": np.quantile(risk_diffs, [0.025, 0.975]).tolist(),
        "agent_risk_ci95": np.quantile(agent_risks, [0.025, 0.975]).tolist(),
        "cost_difference_ci95": np.quantile(cost_diffs, [0.025, 0.975]).tolist(),
    }


def family_table(agent_rows, fixed_rows_):
    result = {}
    for family in sorted({row["family"] for row in agent_rows}):
        agent = [row for row in agent_rows if row["family"] == family]
        fixed = [row for row in fixed_rows_ if row["family"] == family]
        sa, sf = _summary(agent), _summary(fixed)
        result[family] = {
            "never_seen": family in NEVER_SEEN_FAMILIES,
            "agent": sa, "fixed": sf,
            "correct_advantage": sa["correct_rate"] - sf["correct_rate"],
            "risk_difference": sa["false_rate"] - sf["false_rate"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--dominance", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dominance = json.loads(args.dominance.read_text())
    receipt = json.loads(args.receipt.read_text())
    authorization = json.loads(args.authorization.read_text())
    if not authorization["calibration_gates_passed"]:
        raise SystemExit("confirmation was not authorized")
    if receipt.get("receipt_sha256") != semantic_digest(receipt, "receipt_sha256"):
        raise SystemExit("model receipt semantic digest mismatch")
    if dominance.get("content_sha256") != semantic_digest(dominance, "content_sha256"):
        raise SystemExit("dominance semantic digest mismatch")
    if authorization.get("authorization_sha256") != semantic_digest(
            authorization, "authorization_sha256"):
        raise SystemExit("authorization semantic digest mismatch")
    if (authorization.get("model_receipt_sha256") != receipt["receipt_sha256"]
            or authorization.get("model_sha256") != receipt["model_sha256"]
            or authorization.get("dominance_content_sha256")
            != dominance["content_sha256"]):
        raise SystemExit("confirmation dependency binding mismatch")
    if (authorization.get("selected_risk_threshold")
            != receipt["selected_risk_threshold"]
            or authorization.get("matched_fixed") != receipt["matched_fixed"]
            or authorization.get("lofo_gate") != receipt["lofo_gate"]):
        raise SystemExit("confirmation decision binding mismatch")
    models_path = args.receipt.parent / "models.pkl"
    if hashlib.sha256(models_path.read_bytes()).hexdigest() != receipt["model_sha256"]:
        raise SystemExit("model bytes do not match receipt")
    model_correct, model_false = pickle.loads(models_path.read_bytes())
    episodes, labels = load_shards(args.shards)
    _attach_seed(args.shards, episodes)
    agent = evaluate_policy(model_correct, model_false, episodes, labels,
                            receipt["selected_risk_threshold"])
    best = receipt["matched_fixed"]
    fixed = fixed_rows(episodes, labels, DIAGNOSE_VIEW_ID, best["submit"])
    agent_rows = agent.pop("rows")
    overall_fixed = _summary(fixed)
    families = family_table(agent_rows, fixed)
    unseen = [family for family in families.values() if family["never_seen"]]
    correct_advantage = agent["correct_rate"] - overall_fixed["correct_rate"]
    risk_difference = agent["false_rate"] - overall_fixed["false_rate"]
    cost_difference = agent["mean_cost"] - overall_fixed["mean_cost"]
    bootstrap = cluster_bootstrap(agent_rows, fixed, GATES.bootstrap_replicates)
    heldout_advantage = (sum(row["agent"]["correct"] for row in unseen)
                         / sum(row["agent"]["n"] for row in unseen)
                         - sum(row["fixed"]["correct"] for row in unseen)
                         / sum(row["fixed"]["n"] for row in unseen))
    gates = {
        "confirmation_seed_clusters": len({row["seed"] for row in agent_rows}),
        "seed_count_pass": len({row["seed"] for row in agent_rows}) >= GATES.confirmation_seed_clusters_min,
        "absolute_risk_pass": (agent["false_rate"] <= GATES.risk_ceiling
                               and bootstrap["agent_risk_ci95"][1]
                               <= GATES.risk_ceiling),
        "matched_risk_pass": (risk_difference <= GATES.risk_difference_margin
                              and bootstrap["risk_difference_ci95"][1]
                              <= GATES.risk_difference_margin),
        "correct_advantage_pass": (correct_advantage >= GATES.correct_advantage_min
                                   and bootstrap["correct_advantage_ci95"][0] > 0),
        "heldout_advantage_pass": heldout_advantage >= GATES.heldout_correct_advantage_min,
        "heldout_family_count_pass": sum(row["correct_advantage"] > 0 for row in unseen)
                                     >= GATES.heldout_families_positive_min,
        "matched_cost_pass": (cost_difference <= GATES.cost_difference_margin
                              and bootstrap["cost_difference_ci95"][1]
                              <= GATES.cost_difference_margin),
    }
    gates["primary_claim_pass"] = all(gates.values())
    result = {
        "schema_version": "maintrack.evidence-control.confirmation.v3r2",
        "agent": agent, "fixed": overall_fixed,
        "fixed_policy": {"diagnose": DIAGNOSE_VIEW_ID, "submit": best["submit"]},
        "correct_advantage": correct_advantage, "risk_difference": risk_difference,
        "cost_difference": cost_difference,
        "heldout_correct_advantage": heldout_advantage,
        "families": families, "bootstrap": bootstrap, "gates": gates,
        "agent_rows": agent_rows, "fixed_rows": fixed,
    }
    raw = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["content_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"gates": gates, "agent": agent, "fixed": overall_fixed,
                      "bootstrap": bootstrap}, sort_keys=True))


if __name__ == "__main__":
    main()
