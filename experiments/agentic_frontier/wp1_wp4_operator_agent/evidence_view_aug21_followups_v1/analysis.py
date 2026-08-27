"""Deterministic replay of the two Aug-21 evidence-view controls."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..adversarial_verifier_gaming_v1.campaign import _cell
from ..adversarial_verifier_gaming_v1.contract import ACTIONS, public_objective
from ..evidence_view_controls_v1.campaign import (
    A10_SEEDS,
    A11_SEEDS,
    FAMILIES,
    HIGH_NOISES,
    PRODUCERS,
    STANDARD_NOISES,
    TEMPORAL_MODES,
    _standard_episodes,
    _summary_final,
)

LANE = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_REPLICATES = 2000


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def auc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = sorted(score for label, score in zip(labels, scores) if not label)
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    wins = 0.0
    for score in positives:
        lo = bisect.bisect_left(negatives, score)
        hi = bisect.bisect_right(negatives, score)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(positives) * len(negatives))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    weight = position - lo
    return float((1.0 - weight) * ordered[lo] + weight * ordered[hi])


def _seed_bootstrap_auc(rows: list[dict[str, Any]]) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = sorted({int(row["seed"]) for row in rows})
    values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(seeds, size=len(seeds), replace=True)
        replay: list[dict[str, Any]] = []
        for seed in sampled:
            replay.extend(row for row in rows if int(row["seed"]) == int(seed))
        values.append(auc(
            [bool(row["genuine_oog"]) for row in replay],
            [float(row["coarse_residual"]) for row in replay],
        ))
    return values


def run_coarse_residual() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for producer in PRODUCERS:
        for family in FAMILIES:
            for seed in A10_SEEDS:
                for model, noise, baseline, channels, truth in _standard_episodes(
                    producer, family, seed, STANDARD_NOISES
                ):
                    cell = _cell(
                        producer, baseline, channels[7], channels[49],
                        truth.support, TEMPORAL_MODES,
                    )
                    final = _summary_final(cell)
                    genuine = truth.support is None
                    false_oog = (not genuine) and final["status"] == "OUT_OF_GRAMMAR"
                    if not (genuine or false_oog):
                        continue
                    rows.append({
                        "producer": producer,
                        "family": family.value,
                        "seed": seed,
                        "noise_model": model,
                        "relative_noise": noise,
                        "genuine_oog": genuine,
                        "false_coarse_oog": false_oog,
                        "coarse_residual": float(final["best_residual"]),
                        "coarse_status": final["status"],
                    })
    results: dict[str, Any] = {}
    for producer in PRODUCERS:
        group = [row for row in rows if row["producer"] == producer]
        boot = _seed_bootstrap_auc(group)
        genuine = [row["coarse_residual"] for row in group if row["genuine_oog"]]
        false = [row["coarse_residual"] for row in group if row["false_coarse_oog"]]
        results[producer] = {
            "rows": len(group),
            "genuine_oog": len(genuine),
            "false_coarse_oog": len(false),
            "residual_auroc_genuine_vs_false": auc(
                [bool(row["genuine_oog"]) for row in group],
                [float(row["coarse_residual"]) for row in group],
            ),
            "seed_cluster_bootstrap_95pct": [percentile(boot, 0.025), percentile(boot, 0.975)],
            "genuine_residual_range": [min(genuine), max(genuine)],
            "false_residual_range": [min(false), max(false)],
            "ranges_overlap": max(min(genuine), min(false)) <= min(max(genuine), max(false)),
        }
    return {"rows": rows, "summary": results}


def _public_final(final: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": final["status"],
        "best_residual": final["best_residual"],
        "score_gap": final["score_gap"],
        "best_condition": final["best_condition"],
    }


def run_noise_attack() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for producer in PRODUCERS:
        for family in FAMILIES:
            for seed in A11_SEEDS:
                for model, noise, baseline, channels, truth in _standard_episodes(
                    producer, family, seed, HIGH_NOISES
                ):
                    if noise != 0.01:
                        continue
                    candidates: list[tuple[float, int, str, dict[str, Any]]] = []
                    for order, action in enumerate(ACTIONS):
                        cell = _cell(
                            producer, baseline, channels[action.nodes], channels[49],
                            truth.support, action.temporal_modes,
                        )
                        final = _summary_final(cell)
                        objective = public_objective(_public_final(final), action.nodes)
                        candidates.append((objective, -order, action.token, final))
                    objective, _, action_token, attack = max(candidates)
                    baseline_cell = _cell(
                        producer, baseline, channels[49], channels[49],
                        truth.support, TEMPORAL_MODES,
                    )
                    authority = _summary_final(baseline_cell)
                    rows.append({
                        "producer": producer,
                        "family": family.value,
                        "seed": seed,
                        "noise_model": model,
                        "relative_noise": noise,
                        "truth_in_library": truth.support is not None,
                        "chosen_action": action_token,
                        "chosen_objective": objective,
                        "baseline_status": authority["status"],
                        "baseline_false": authority["false_promotion"],
                        "attack_status": attack["status"],
                        "attack_false": attack["false_promotion"],
                    })
    results: dict[str, Any] = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    for producer in PRODUCERS:
        group = [row for row in rows if row["producer"] == producer]
        baseline_false = sum(row["baseline_false"] for row in group)
        attack_false = sum(row["attack_false"] for row in group)
        attack_only = sum(row["attack_false"] and not row["baseline_false"] for row in group)
        baseline_only = sum(row["baseline_false"] and not row["attack_false"] for row in group)
        seeds = sorted({int(row["seed"]) for row in group})
        differences: list[float] = []
        for _ in range(BOOTSTRAP_REPLICATES):
            sampled = rng.choice(seeds, size=len(seeds), replace=True)
            replay: list[dict[str, Any]] = []
            for seed in sampled:
                replay.extend(row for row in group if int(row["seed"]) == int(seed))
            differences.append(
                sum(row["attack_false"] for row in replay) / len(replay)
                - sum(row["baseline_false"] for row in replay) / len(replay)
            )
        in_library = [row for row in group if row["truth_in_library"]]
        results[producer] = {
            "episodes": len(group),
            "in_library_episodes": len(in_library),
            "baseline_false": baseline_false,
            "baseline_false_rate": baseline_false / len(group),
            "attack_false": attack_false,
            "attack_false_rate": attack_false / len(group),
            "paired_attack_minus_baseline_pp": 100.0 * (attack_false - baseline_false) / len(group),
            "paired_seed_cluster_bootstrap_95pct_pp": [
                100.0 * percentile(differences, 0.025),
                100.0 * percentile(differences, 0.975),
            ],
            "attack_only_false": attack_only,
            "baseline_only_false": baseline_only,
            "in_library_baseline_false_rate": sum(row["baseline_false"] for row in in_library) / len(in_library),
            "in_library_attack_false_rate": sum(row["attack_false"] for row in in_library) / len(in_library),
            "chosen_action_counts": dict(sorted(Counter(row["chosen_action"] for row in group).items())),
        }
    return {"rows": rows, "summary": results}


def build_payload() -> dict[str, Any]:
    coarse = run_coarse_residual()
    noise = run_noise_attack()
    body = {
        "schema_version": "verisigs.evidence_view_aug21_followups.v1",
        "status": "COMPLETED_DETERMINISTIC_POSTPRIMARY_CONTROLS",
        "scope": "finite_synthetic_development_replay_not_independent_confirmation",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "coarse_residual": coarse,
        "noise_attack": noise,
    }
    return body | {"content_sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(encoded)
    print(json.dumps({
        "coarse_residual": payload["coarse_residual"]["summary"],
        "noise_attack": payload["noise_attack"]["summary"],
        "content_sha256": payload["content_sha256"],
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
