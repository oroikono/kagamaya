"""Family-general risk-aware learner with content-addressed receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform

import numpy as np
from scipy.stats import beta

from .contract import (COST_PENALTY, DIAGNOSE_VIEW_ID, GATES, MODEL_CONFIG,
                       LOFO_CALIBRATION_SEEDS, LOFO_FIT_SEEDS,
                       NEVER_SEEN_FAMILIES, POLICY_FEATURES, RISK_THRESHOLD_GRID,
                       UTILITY_FALSE_PENALTY, VIEW_BANK)
from .control import Terminal, score
from .dominance import load_shards


VIEW_MAP = {view.view_id: view for view in VIEW_BANK}


def _record(episode: dict, view_id: str) -> dict:
    return next(row for row in episode["views"] if row["view_id"] == view_id)


def feature_vector(diagnosis: dict, candidate_view_id: str) -> np.ndarray:
    view = VIEW_MAP[candidate_view_id]
    status = diagnosis["status"]
    return np.asarray([
        diagnosis["residual"],
        diagnosis["score_gap"] if math.isfinite(diagnosis["score_gap"]) else 1e6,
        math.log1p(min(diagnosis["condition"], 1e30)),
        diagnosis["minimum_coefficient"],
        diagnosis["ambiguity_size"],
        float(status == "IDENTIFIED"),
        float(status == "OUTSIDE_LIBRARY"),
        view.spatial_nodes,
        view.temporal_samples,
        view.test_mode_count,
        view.window_start,
        view.window_end,
        view.registered_cost,
        float(candidate_view_id == DIAGNOSE_VIEW_ID),
    ], dtype=np.float64)


def action_outcome(episode: dict, owner: dict, candidate_view_id: str):
    row = _record(episode, candidate_view_id)
    cost = VIEW_MAP[DIAGNOSE_VIEW_ID].registered_cost
    if candidate_view_id != DIAGNOSE_VIEW_ID:
        cost += VIEW_MAP[candidate_view_id].registered_cost
    terminal = Terminal(row["status"], tuple(row["support"]), "learned_submit",
                        DIAGNOSE_VIEW_ID, candidate_view_id, cost)
    return score(terminal, owner), terminal


def matrix(episodes: list[dict], labels: dict[str, dict], *,
           include_families: set[str] | None = None,
           exclude_families: set[str] | None = None,
           seeds: set[int] | None = None):
    features = []
    correct = []
    false = []
    row_keys = []
    for episode in episodes:
        owner = labels[episode["episode_id"]]
        family = owner["family"]
        if include_families is not None and family not in include_families:
            continue
        if exclude_families is not None and family in exclude_families:
            continue
        # Public nonces intentionally contain no seed. The owner-side shard
        # metadata is the only source for seed filtering, supplied by caller.
        if seeds is not None and episode.get("_seed") not in seeds:
            continue
        diagnosis = _record(episode, DIAGNOSE_VIEW_ID)
        for view in VIEW_BANK:
            outcome, _ = action_outcome(episode, owner, view.view_id)
            features.append(feature_vector(diagnosis, view.view_id))
            correct.append(int(outcome.correct))
            false.append(int(outcome.false_promotion))
            row_keys.append((episode["episode_id"], view.view_id))
    return (np.asarray(features), np.asarray(correct, dtype=np.int8),
            np.asarray(false, dtype=np.int8), row_keys)


def _attach_seed(root: Path, episodes: list[dict]) -> None:
    seed_by_id = {}
    import gzip
    for path in sorted(root.glob("*.json.gz")):
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
        seed = int(payload["metadata"]["seed"])
        for episode in payload["episodes"]:
            seed_by_id[episode["episode_id"]] = seed
    for episode in episodes:
        episode["_seed"] = seed_by_id[episode["episode_id"]]


def _fit_binary(x: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier
    config = dict(MODEL_CONFIG)
    config.pop("class")
    model = HistGradientBoostingClassifier(**config)
    positives = max(int(y.sum()), 1)
    negatives = max(int(len(y) - y.sum()), 1)
    weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
    return model.fit(x, y, sample_weight=weights)


def fit_models(x: np.ndarray, y_correct: np.ndarray, y_false: np.ndarray):
    return _fit_binary(x, y_correct), _fit_binary(x, y_false)


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, classes.index(1)]


def choose(model_correct, model_false, episode: dict, threshold: float) -> str | None:
    diagnosis = _record(episode, DIAGNOSE_VIEW_ID)
    x = np.vstack([feature_vector(diagnosis, view.view_id) for view in VIEW_BANK])
    pc = _positive_probability(model_correct, x)
    pf = _positive_probability(model_false, x)
    allowed = np.flatnonzero(pf <= threshold)
    if not len(allowed):
        return None
    costs = np.asarray([
        VIEW_MAP[DIAGNOSE_VIEW_ID].registered_cost
        + (0 if view.view_id == DIAGNOSE_VIEW_ID else view.registered_cost)
        for view in VIEW_BANK])
    utility = pc - UTILITY_FALSE_PENALTY * pf - COST_PENALTY * costs
    # Stable view-id tie break.
    return VIEW_BANK[int(allowed[np.argmax(utility[allowed])])].view_id


def cp_upper(false_count: int, total: int) -> float:
    if false_count >= total:
        return 1.0
    return float(beta.ppf(GATES.confidence_level, false_count + 1,
                          total - false_count))


def evaluate_policy(model_correct, model_false, episodes: list[dict],
                    labels: dict[str, dict], threshold: float,
                    allowed_families: set[str] | None = None,
                    allowed_seeds: set[int] | None = None) -> dict:
    rows = []
    for episode in episodes:
        owner = labels[episode["episode_id"]]
        if allowed_families is not None and owner["family"] not in allowed_families:
            continue
        if allowed_seeds is not None and episode["_seed"] not in allowed_seeds:
            continue
        selected = choose(model_correct, model_false, episode, threshold)
        if selected is None:
            outcome = {"correct": False, "false_promotion": False,
                       "inconclusive": True}
            cost = VIEW_MAP[DIAGNOSE_VIEW_ID].registered_cost
            status = "INCONCLUSIVE"
            support = []
        else:
            scored, terminal = action_outcome(episode, owner, selected)
            outcome = scored.__dict__
            cost = terminal.registered_cost
            status = terminal.status
            support = list(terminal.support)
        rows.append({
            "episode_id": episode["episode_id"], "seed": episode["_seed"],
            "origin": episode["origin"], "noise_model": episode["noise_model"],
            "noise_level": episode["noise_level"], "family": owner["family"],
            "never_seen_family": owner["family"] in NEVER_SEEN_FAMILIES,
            "selected_view_id": selected, "status": status, "support": support,
            "registered_cost": cost, **outcome,
        })
    n = len(rows)
    false_count = sum(row["false_promotion"] for row in rows)
    cluster_risk = cluster_bootstrap_risk(rows, GATES.calibration_bootstrap_replicates)
    return {
        "n": n, "correct": sum(row["correct"] for row in rows),
        "false": false_count, "inconclusive": sum(row["inconclusive"] for row in rows),
        "correct_rate": sum(row["correct"] for row in rows) / n,
        "false_rate": false_count / n,
        "false_iid_cp_upper_95_sensitivity": cp_upper(false_count, n),
        "cluster_risk": cluster_risk,
        "mean_cost": float(np.mean([row["registered_cost"] for row in rows])),
        "rows": rows,
    }


def cluster_bootstrap_risk(rows: list[dict], repetitions: int) -> dict:
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seeds = np.asarray(sorted(by_seed))
    if not len(seeds):
        raise ValueError("cluster risk requires at least one seed")
    seed_rates = np.asarray([
        np.mean([item["false_promotion"] for item in by_seed[int(seed)]])
        for seed in seeds], dtype=float)
    rng = np.random.default_rng(20260823)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        means[index] = float(np.mean(rng.choice(seed_rates, size=len(seeds), replace=True)))
    return {
        "seed_count": int(len(seeds)),
        "mean_seed_false_rate": float(np.mean(seed_rates)),
        "max_seed_false_rate": float(np.max(seed_rates)),
        "mean_seed_false_rate_ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "repetitions": repetitions,
    }


def select_threshold(model_correct, model_false, episodes, labels, families,
                     seeds: set[int] | None = None):
    candidates = []
    for threshold in RISK_THRESHOLD_GRID:
        result = evaluate_policy(model_correct, model_false, episodes, labels,
                                 threshold, families, seeds)
        candidates.append({key: value for key, value in result.items() if key != "rows"}
                          | {"threshold": threshold})
    safe = [row for row in candidates
            if row["false_rate"] <= GATES.risk_ceiling
            and row["cluster_risk"]["mean_seed_false_rate_ci95"][1]
            <= GATES.risk_ceiling]
    if not safe:
        return None, candidates
    chosen = max(safe, key=lambda row: (row["correct_rate"], -row["false_rate"],
                                       -row["mean_cost"], -row["threshold"]))
    return chosen["threshold"], candidates


def evaluate_fixed(episodes, labels, submit_view_id: str,
                   allowed_families: set[str] | None = None,
                   allowed_seeds: set[int] | None = None) -> dict:
    rows = []
    diagnose_cost = VIEW_MAP[DIAGNOSE_VIEW_ID].registered_cost
    submit_cost = VIEW_MAP[submit_view_id].registered_cost
    cost = diagnose_cost + (0.0 if submit_view_id == DIAGNOSE_VIEW_ID else submit_cost)
    for episode in episodes:
        owner = labels[episode["episode_id"]]
        if allowed_families is not None and owner["family"] not in allowed_families:
            continue
        if allowed_seeds is not None and episode["_seed"] not in allowed_seeds:
            continue
        record = _record(episode, submit_view_id)
        terminal = Terminal(record["status"], tuple(record["support"]),
                            "fixed_submit", DIAGNOSE_VIEW_ID, submit_view_id, cost)
        outcome = score(terminal, owner)
        rows.append({
            "episode_id": episode["episode_id"], "seed": episode["_seed"],
            "correct": outcome.correct, "false_promotion": outcome.false_promotion,
            "inconclusive": outcome.inconclusive, "registered_cost": cost,
        })
    n = len(rows)
    false_count = sum(row["false_promotion"] for row in rows)
    return {
        "submit": submit_view_id, "n": n,
        "correct": sum(row["correct"] for row in rows),
        "false": false_count,
        "inconclusive": sum(row["inconclusive"] for row in rows),
        "correct_rate": sum(row["correct"] for row in rows) / n,
        "false_rate": false_count / n,
        "registered_cost": cost,
        "cluster_risk": cluster_bootstrap_risk(
            rows, GATES.calibration_bootstrap_replicates),
        "rows": rows,
    }


def select_matched_fixed(episodes, labels, families: set[str],
                         cost_ceiling: float,
                         seeds: set[int] | None = None) -> tuple[dict | None, list[dict]]:
    candidates = []
    for view in VIEW_BANK:
        result = evaluate_fixed(episodes, labels, view.view_id, families, seeds)
        candidates.append({key: value for key, value in result.items() if key != "rows"})
    safe = [row for row in candidates
            if row["registered_cost"] <= cost_ceiling + GATES.cost_difference_margin
            and row["false_rate"] <= GATES.risk_ceiling
            and row["cluster_risk"]["mean_seed_false_rate_ci95"][1]
            <= GATES.risk_ceiling]
    if not safe:
        return None, candidates
    return max(safe, key=lambda row: (row["correct_rate"], -row["false_rate"],
                                      -row["registered_cost"], row["submit"])), candidates


def semantic_matrix_digest(x, yc, yf, keys) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).view(np.uint8))
    digest.update(np.ascontiguousarray(yc).view(np.uint8))
    digest.update(np.ascontiguousarray(yf).view(np.uint8))
    digest.update(json.dumps(keys, separators=(",", ":")).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", type=Path, required=True)
    parser.add_argument("--calibration-shards", type=Path, required=True)
    parser.add_argument("--dominance", type=Path, required=True)
    parser.add_argument("--source-closure", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dominance = json.loads(args.dominance.read_text())
    if not dominance["gate"]["learning_admitted"]:
        raise SystemExit("fixed-policy dominance gate blocks learning")
    train_episodes, train_labels = load_shards(args.train_shards)
    cal_episodes, cal_labels = load_shards(args.calibration_shards)
    _attach_seed(args.train_shards, train_episodes)
    _attach_seed(args.calibration_shards, cal_episodes)
    eligible_families = sorted({owner["family"] for owner in train_labels.values()}
                               - set(NEVER_SEEN_FAMILIES))
    x, yc, yf, keys = matrix(train_episodes, train_labels,
                             exclude_families=set(NEVER_SEEN_FAMILIES))
    model_correct, model_false = fit_models(x, yc, yf)
    threshold, threshold_table = select_threshold(
        model_correct, model_false, cal_episodes, cal_labels, set(eligible_families))
    if threshold is None:
        raise SystemExit("no calibration threshold satisfies the frozen risk gate")
    final_calibration = evaluate_policy(
        model_correct, model_false, cal_episodes, cal_labels, threshold,
        set(eligible_families))
    matched_fixed, matched_fixed_table = select_matched_fixed(
        cal_episodes, cal_labels, set(eligible_families),
        final_calibration["mean_cost"])
    if matched_fixed is None:
        raise SystemExit("no fixed comparator satisfies matched cost/risk gates")

    # Nested family-held-out audit. Models use only the reserved inner-fit
    # seeds; thresholds and fixed comparators use other inner seeds. The held
    # family is absent from fitting, threshold choice, and comparator choice.
    lofo = []
    for heldout in eligible_families:
        lx, lyc, lyf, _ = matrix(train_episodes, train_labels,
                                 exclude_families=set(NEVER_SEEN_FAMILIES) | {heldout},
                                 seeds=set(LOFO_FIT_SEEDS))
        lmc, lmf = fit_models(lx, lyc, lyf)
        remaining = set(eligible_families) - {heldout}
        lt, _ = select_threshold(lmc, lmf, train_episodes, train_labels,
                                 remaining, set(LOFO_CALIBRATION_SEEDS))
        if lt is None:
            lofo.append({"family": heldout, "status": "NO_SAFE_THRESHOLD"})
            continue
        evaluation = evaluate_policy(
            lmc, lmf, train_episodes, train_labels, lt, {heldout},
            set(LOFO_CALIBRATION_SEEDS))
        fold_fixed, _ = select_matched_fixed(
            train_episodes, train_labels, remaining, evaluation["mean_cost"],
            set(LOFO_CALIBRATION_SEEDS))
        if fold_fixed is None:
            lofo.append({"family": heldout, "status": "NO_MATCHED_FIXED"})
            continue
        heldout_fixed = evaluate_fixed(
            train_episodes, train_labels, fold_fixed["submit"], {heldout},
            set(LOFO_CALIBRATION_SEEDS))
        lofo.append({"family": heldout, "status": "EVALUATED", "threshold": lt,
                     "agent": {key: val for key, val in evaluation.items()
                               if key != "rows"},
                     "fixed": {key: val for key, val in heldout_fixed.items()
                               if key != "rows"},
                     "fixed_selected_without_heldout": fold_fixed,
                     "correct_advantage": (evaluation["correct_rate"]
                                           - heldout_fixed["correct_rate"]),
                     "risk_difference": (evaluation["false_rate"]
                                          - heldout_fixed["false_rate"]),
                     "risk_pass": (evaluation["false_rate"] <= GATES.lofo_risk_ceiling
                                   and evaluation["cluster_risk"]
                                   ["mean_seed_false_rate_ci95"][1]
                                   <= GATES.lofo_risk_ceiling)})

    evaluated_lofo = [row for row in lofo if row.get("status") == "EVALUATED"]
    lofo_gate = {
        "all_families_evaluated": len(evaluated_lofo) == len(eligible_families),
        "all_family_risk_pass": bool(evaluated_lofo)
        and all(row["risk_pass"] for row in evaluated_lofo),
        "positive_advantage_families": sum(
            row["correct_advantage"] > 0 for row in evaluated_lofo),
        "positive_advantage_required": math.ceil(
            GATES.lofo_positive_fraction_min * len(eligible_families)),
    }
    lofo_gate["positive_fraction_pass"] = (
        lofo_gate["positive_advantage_families"]
        >= lofo_gate["positive_advantage_required"])
    lofo_gate["all_passed"] = all((
        lofo_gate["all_families_evaluated"],
        lofo_gate["all_family_risk_pass"],
        lofo_gate["positive_fraction_pass"],
    ))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / "models.pkl"
    model_path.write_bytes(pickle.dumps((model_correct, model_false), protocol=5))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    import scipy
    import sklearn
    receipt = {
        "schema_version": "maintrack.evidence-control.model-receipt.v3r2",
        "source_closure": args.source_closure,
        "features": list(POLICY_FEATURES), "diagnose_view_id": DIAGNOSE_VIEW_ID,
        "model_config": MODEL_CONFIG,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__},
        "training_matrix_sha256": semantic_matrix_digest(x, yc, yf, keys),
        "training_rows": len(x), "training_episodes": len(x) // len(VIEW_BANK),
        "eligible_families": eligible_families,
        "never_seen_families": list(NEVER_SEEN_FAMILIES),
        "selected_risk_threshold": threshold,
        "threshold_table": threshold_table,
        "calibration_agent": {key: value for key, value in final_calibration.items()
                              if key != "rows"},
        "matched_fixed": matched_fixed,
        "matched_fixed_candidate_table": matched_fixed_table,
        "lofo": lofo,
        "lofo_gate": lofo_gate,
        "model_sha256": model_sha,
    }
    receipt_raw = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    receipt["receipt_sha256"] = hashlib.sha256(receipt_raw.encode()).hexdigest()
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "receipt_sha256": receipt["receipt_sha256"],
                      "threshold": threshold, "lofo": lofo}, sort_keys=True))


if __name__ == "__main__":
    main()
