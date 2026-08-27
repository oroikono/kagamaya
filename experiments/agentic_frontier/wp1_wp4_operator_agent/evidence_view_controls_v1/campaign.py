"""Execute frozen A10--A12 CPU controls without modifying authorities."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contract import (
    A10_SEEDS, A11_SEEDS, A12_SEEDS, BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED, HIGH_NOISES, NODES, NOISE_MODELS, PRODUCERS,
    PUBLIC_SEED, REFERENCE_NODES, STANDARD_NOISES, TEMPORAL_MODES,
    RunContract, digest, finite, load_and_validate_freeze, percentile,
    publish_exclusive,
)
from ..adversarial_verifier_gaming_v1.campaign import _cell
from ..independent_fd_producer_v1.simulator import (
    BASELINE as FD_BASELINE,
    GRADIENT as FD_GRADIENT,
    Family as FDFamily,
    ProducerConfig as FDProducerConfig,
    clean_pair as fd_clean_pair,
    observed_trajectory as fd_observed,
    solve_clean as fd_solve_clean,
)
from ..maintrack_suite_v41.numerics import integrate_etdrk4
from ..maintrack_suite_v41.simulator import Family, HiddenTruth, Trajectory, TRUE_SUPPORTS
from ..maintrack_suite_v41.weakform import DiscoveryConfig, build_evidence
from ..unified_agent_v2_1_scaled.simulator import (
    BASELINE as SPECTRAL_BASELINE,
    Channel,
    _initial,
    _noise,
    _problem,
    simulate,
)
from ..unified_quadrature_bias_v1.simulator import finest_noisy_trajectory


LANE_ROOT = Path(__file__).resolve().parent
RESULTS = LANE_ROOT / "results"
FAMILIES = tuple(Family)
REFERENCE97_CHANNEL = Channel("gradient_space", 96, 0.12, 0.00125, 1.0)
FD_REFERENCE97 = FDProducerConfig(96, 0.12, 97, "gradient_space")


def _signature(final: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return str(final["status"]), tuple(final.get("support") or ())


def _summary_final(cell: dict[str, Any]) -> dict[str, Any]:
    final = cell["final"]
    return {
        "status": str(final["status"]),
        "support": None if final.get("support") is None else list(final["support"]),
        "best_residual": finite(final["best_residual"]),
        "score_gap": finite(final["score_gap"]),
        "best_condition": None if final.get("best_condition") is None else finite(final["best_condition"]),
        "top_fit_support": None if final.get("top_fit_support") is None else list(final["top_fit_support"]),
        "correct": bool(cell["correct"]),
        "false_promotion": bool(cell["false_promotion"]),
    }


def _nested(fine: Any, nodes: int) -> Any:
    if nodes < 3 or nodes % 2 == 0 or (fine.t.size - 1) % (nodes - 1):
        raise ValueError("requested temporal grid is not an odd exact nested subsample")
    stride = (fine.t.size - 1) // (nodes - 1)
    index = np.arange(0, fine.t.size, stride)
    if index.size != nodes or index[-1] != fine.t.size - 1:
        raise ValueError("nested subsample lost endpoint or cardinality")
    return type(fine)(
        f"{fine.sample_id}-n{nodes}", fine.action, fine.x.copy(),
        fine.t[index].copy(), fine.u[index].copy(), fine.noise_model,
        fine.relative_noise,
    )


def _spectral_observed(clean: Trajectory, seed: int, model: str, noise: float) -> Trajectory:
    noise_seed = int(np.random.SeedSequence([
        seed, PUBLIC_SEED, 991, sum(map(ord, clean.action))
    ]).generate_state(1)[0])
    observed = _noise(clean.u, noise, model, noise_seed)
    return Trajectory(
        clean.sample_id, clean.action, clean.x.copy(), clean.t.copy(), observed,
        model, noise,
    )


def _standard_episodes(producer: str, family: Family, seed: int,
                       noises: tuple[float, ...]):
    """Generate clean trajectories once, then apply the frozen matched noise draws."""

    if producer == "spectral_etdrk4":
        baseline_clean, baseline_truth = simulate(
            family, SPECTRAL_BASELINE, seed, PUBLIC_SEED, 0.0, "iid"
        )
        fine_clean, truth = finest_noisy_trajectory(
            family, seed, PUBLIC_SEED, 0.0, "iid"
        )
        if baseline_truth != truth:
            raise AssertionError("spectral standard baseline/channel truth mismatch")
        for model in NOISE_MODELS:
            for noise in noises:
                baseline = _spectral_observed(baseline_clean, seed, model, noise)
                fine = _spectral_observed(fine_clean, seed, model, noise)
                yield model, noise, baseline, {
                    nodes: _nested(fine, nodes) for nodes in NODES
                }, truth
        return
    fd_family = FDFamily(family.value)
    baseline_clean, gradient_clean, truth = fd_clean_pair(fd_family, seed, PUBLIC_SEED)
    for model in NOISE_MODELS:
        for noise in noises:
            baseline = fd_observed(
                FD_BASELINE, seed, PUBLIC_SEED, baseline_clean[2], baseline_clean[0],
                baseline_clean[1], noise, model,
            )
            fine = fd_observed(
                FD_GRADIENT, seed, PUBLIC_SEED, gradient_clean[2], gradient_clean[0],
                gradient_clean[1], noise, model,
            )
            yield model, noise, baseline, {
                nodes: _nested(fine, nodes) for nodes in NODES
            }, truth


def _spectral_clean97(family: Family, seed: int) -> tuple[np.ndarray, HiddenTruth]:
    internal_dt = 0.00025
    steps = int(round(REFERENCE97_CHANNEL.t_final / internal_dt))
    stride = int(round(REFERENCE97_CHANNEL.save_dt / internal_dt))
    if steps != 480 or stride != 5:
        raise AssertionError("frozen spectral 97-node integration schedule drift")
    x = np.arange(REFERENCE97_CHANNEL.nx) * (2.0 * np.pi / REFERENCE97_CHANNEL.nx)
    k = 2.0 * np.pi * np.fft.fftfreq(
        REFERENCE97_CHANNEL.nx, d=2.0 * np.pi / REFERENCE97_CHANNEL.nx
    )
    linear, nonlinear, coefficients = _problem(family, k, np.random.default_rng(seed))
    clean = integrate_etdrk4(
        _initial(REFERENCE97_CHANNEL, x, PUBLIC_SEED), linear, nonlinear,
        internal_dt, steps, stride,
    )
    if clean.shape != (97, REFERENCE97_CHANNEL.nx):
        raise RuntimeError(f"unexpected spectral reference shape {clean.shape}")
    return clean, HiddenTruth(family, TRUE_SUPPORTS[family], coefficients, seed)


def _spectral_observed97(seed: int, model: str, noise: float,
                         clean: np.ndarray) -> Trajectory:
    noise_seed = int(np.random.SeedSequence([
        seed, PUBLIC_SEED, 991, sum(map(ord, REFERENCE97_CHANNEL.name))
    ]).generate_state(1)[0])
    observed = _noise(clean, noise, model, noise_seed)
    x = np.arange(REFERENCE97_CHANNEL.nx) * (2.0 * np.pi / REFERENCE97_CHANNEL.nx)
    t = np.linspace(0.0, REFERENCE97_CHANNEL.t_final, 97)
    fine = Trajectory(
        f"e{seed}-gradient-space-97", REFERENCE97_CHANNEL.name, x, t,
        observed, model, noise,
    )
    return fine


def _reference97_episodes(producer: str, family: Family, seed: int) -> Iterable[tuple[str, float, Any, Any, Any]]:
    if producer == "spectral_etdrk4":
        clean, truth = _spectral_clean97(family, seed)
        baseline_clean, baseline_truth = simulate(
            family, SPECTRAL_BASELINE, seed, PUBLIC_SEED, 0.0, "iid"
        )
        if baseline_truth != truth:
            raise AssertionError("spectral baseline/reference truth mismatch")
        for model in NOISE_MODELS:
            for noise in STANDARD_NOISES:
                fine = _spectral_observed97(seed, model, noise, clean)
                baseline = _spectral_observed(baseline_clean, seed, model, noise)
                yield model, noise, baseline, fine, truth
        return
    fd_family = FDFamily(family.value)
    bx, bt, bu, truth = fd_solve_clean(fd_family, FD_BASELINE, seed, PUBLIC_SEED)
    gx, gt, gu, truth_reference = fd_solve_clean(
        fd_family, FD_REFERENCE97, seed, PUBLIC_SEED
    )
    if truth != truth_reference:
        raise AssertionError("FD baseline/reference truth mismatch")
    for model in NOISE_MODELS:
        for noise in STANDARD_NOISES:
            baseline = fd_observed(FD_BASELINE, seed, PUBLIC_SEED, bu, bx, bt, noise, model)
            fine = fd_observed(FD_REFERENCE97, seed, PUBLIC_SEED, gu, gx, gt, noise, model)
            yield model, noise, baseline, fine, truth


def _ols_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(tuple(xs), dtype=float)
    y = np.asarray(tuple(ys), dtype=float)
    centered = x - x.mean()
    return finite(np.dot(centered, y - y.mean()) / np.dot(centered, centered))


def _auc(labels: list[bool], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label]
    negative = sorted(score for label, score in zip(labels, scores) if not label)
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    wins = 0.0
    for pos in positive:
        lower = bisect.bisect_left(negative, pos)
        upper = bisect.bisect_right(negative, pos)
        wins += lower + 0.5 * (upper - lower)
    return wins / (len(positive) * len(negative))


def _roc(labels: list[bool], scores: list[float]) -> list[dict[str, float]]:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    maximum, minimum = max(scores), min(scores)
    thresholds = [
        math.nextafter(maximum, math.inf), *sorted(set(scores), reverse=True),
        math.nextafter(minimum, -math.inf),
    ]
    rows = []
    for threshold in thresholds:
        predicted = [score >= threshold for score in scores]
        tp = sum(label and guess for label, guess in zip(labels, predicted))
        fp = sum((not label) and guess for label, guess in zip(labels, predicted))
        rows.append({
            "threshold": threshold,
            "sensitivity": tp / positives,
            "specificity": 1.0 - fp / negatives,
            "balanced_accuracy": 0.5 * (tp / positives + 1.0 - fp / negatives),
        })
    return rows


def _best_roc_point(points: list[dict[str, float]]) -> dict[str, float]:
    return max(points, key=lambda row: (
        row["balanced_accuracy"], row["specificity"], row["sensitivity"],
        -abs(row["threshold"]) if math.isfinite(row["threshold"]) else float("-inf"),
    ))


def _a10_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_names = ("residual_score", "slope_score", "composite_score")
    producer_results: dict[str, Any] = {}
    for producer in PRODUCERS:
        group = [row for row in rows if row["producer"] == producer]
        labels = [bool(row["label_oog"]) for row in group]
        score_results = {}
        for name in score_names:
            scores = [float(row[name]) for row in group]
            points = _roc(labels, scores)
            score_results[name] = {
                "auroc": _auc(labels, scores),
                "best_balanced_accuracy_point": _best_roc_point(points),
                "roc": points,
            }
        threshold_guesses = [row["residual49"] >= 0.20 for row in group]
        positives, negatives = sum(labels), len(labels) - sum(labels)
        tp = sum(label and guess for label, guess in zip(labels, threshold_guesses))
        fp = sum((not label) and guess for label, guess in zip(labels, threshold_guesses))
        residual_point = score_results["residual_score"]["best_balanced_accuracy_point"]
        producer_results[producer] = {
            "rows": len(group),
            "positive_oog": positives,
            "negative_in_library": negatives,
            "scores": score_results,
            "frozen_residual_0_20": {
                "sensitivity": tp / positives,
                "specificity": 1.0 - fp / negatives,
            },
            "some_residual_threshold_has_sensitivity_and_specificity_ge_0_95": bool(
                residual_point["sensitivity"] >= 0.95 and residual_point["specificity"] >= 0.95
            ),
        }
    macro = {
        name: sum(producer_results[p]["scores"][name]["auroc"] for p in PRODUCERS) / len(PRODUCERS)
        for name in score_names
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seeds = list(A10_SEEDS)
    replicates = {name: [] for name in score_names}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = list(rng.choice(seeds, size=len(seeds), replace=True))
        sampled_rows = []
        for sampled_seed in sampled:
            sampled_rows.extend(row for row in rows if row["seed"] == int(sampled_seed))
        for name in score_names:
            values = []
            for producer in PRODUCERS:
                group = [row for row in sampled_rows if row["producer"] == producer]
                values.append(_auc(
                    [bool(row["label_oog"]) for row in group],
                    [float(row[name]) for row in group],
                ))
            replicates[name].append(sum(values) / len(values))
    intervals = {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in replicates.items()
    }
    residual_triviality = all(
        producer_results[p]["scores"]["residual_score"]["auroc"] >= 0.99
        and producer_results[p]["some_residual_threshold_has_sensitivity_and_specificity_ge_0_95"]
        for p in PRODUCERS
    )
    return {
        "dependence_unit": "episode_seed_cluster",
        "producer_results": producer_results,
        "macro_auroc": macro,
        "macro_auroc_seed_cluster_bootstrap_95pct": intervals,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "residual_triviality_criterion_passed": residual_triviality,
        "interpretation": (
            "FALSIFIES_NONTRIVIALITY_ARGUMENT"
            if residual_triviality else "STRENGTHENS_WHY_NOT_RESIDUAL_THRESHOLD_ANSWER"
        ),
    }


def run_a10() -> dict[str, Any]:
    RunContract("a10", A10_SEEDS).validate()
    rows = []
    for producer in PRODUCERS:
        for family in FAMILIES:
            for seed in A10_SEEDS:
                for model, noise, baseline, channels, truth in _standard_episodes(
                    producer, family, seed, STANDARD_NOISES
                ):
                        residuals = {}
                        final49 = None
                        for nodes in NODES:
                            cell = _cell(
                                producer, baseline, channels[nodes], channels[49],
                                truth.support, TEMPORAL_MODES,
                            )
                            residuals[str(nodes)] = finite(cell["final"]["best_residual"])
                            if nodes == 49:
                                final49 = _summary_final(cell)
                        assert final49 is not None
                        residual_score = math.log10(max(residuals["49"], 1e-12))
                        slope_score = _ols_slope(
                            (math.log10(node) for node in NODES),
                            (math.log10(max(residuals[str(node)], 1e-12)) for node in NODES),
                        )
                        rows.append({
                            "producer": producer, "family": family.value,
                            "seed": seed, "noise_model": model,
                            "relative_noise": noise,
                            "label_oog": truth.support is None,
                            "residuals": residuals,
                            "residual49": residuals["49"],
                            "residual_score": residual_score,
                            "slope_score": slope_score,
                            "composite_score": residual_score + slope_score,
                            "final49": final49,
                        })
    return _payload("a10", rows, _a10_summary(rows))


def _rate(group: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in group) / len(group)


def _a11_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_producer: dict[str, Any] = {}
    valid_by_level = {str(noise): True for noise in HIGH_NOISES}
    for producer in PRODUCERS:
        levels = {}
        correct_rates, failure_rates = [], []
        for noise in HIGH_NOISES:
            group = [row for row in rows if row["producer"] == producer and row["relative_noise"] == noise]
            correct = _rate(group, "correct")
            false = _rate(group, "false_promotion")
            abstain = sum(row["status"] == "UNIDENTIFIED" for row in group) / len(group)
            valid = false <= 0.01 and correct >= 0.50
            valid_by_level[str(noise)] = valid_by_level[str(noise)] and valid
            correct_rates.append(correct)
            failure_rates.append(false + abstain)
            levels[str(noise)] = {
                "rows": len(group), "correct_terminal_rate": correct,
                "false_promotion_rate": false, "abstention_rate": abstain,
                "status_counts": dict(sorted(Counter(row["status"] for row in group).items())),
                "controlled_valid_regime": valid,
            }
        by_producer[producer] = {
            "levels": levels,
            "correct_rate_nonincreasing": all(
                left >= right for left, right in zip(correct_rates, correct_rates[1:])
            ),
            "false_plus_abstention_non_decreasing": all(
                left <= right for left, right in zip(failure_rates, failure_rates[1:])
            ),
        }
        by_producer[producer]["aggregate_difficulty_gradient"] = bool(
            by_producer[producer]["correct_rate_nonincreasing"]
            and by_producer[producer]["false_plus_abstention_non_decreasing"]
        )
    return {
        "dependence_unit": "episode_seed_cluster",
        "producer_results": by_producer,
        "joint_controlled_valid_regime_by_noise": valid_by_level,
        "difficulty_gradient_established_both_producers": all(
            by_producer[p]["aggregate_difficulty_gradient"] for p in PRODUCERS
        ),
        "one_percent_level_valid_both_producers": valid_by_level[str(0.01)],
    }


def run_a11() -> dict[str, Any]:
    RunContract("a11", A11_SEEDS).validate()
    rows = []
    for producer in PRODUCERS:
        for family in FAMILIES:
            for seed in A11_SEEDS:
                for model, noise, baseline, channels, truth in _standard_episodes(
                    producer, family, seed, HIGH_NOISES
                ):
                        cell = _cell(
                            producer, baseline, channels[49], channels[49],
                            truth.support, TEMPORAL_MODES,
                        )
                        final = _summary_final(cell)
                        rows.append({
                            "producer": producer, "family": family.value,
                            "seed": seed, "noise_model": model,
                            "relative_noise": noise,
                            "truth_support": None if truth.support is None else list(truth.support),
                            "status": final["status"], "support": final["support"],
                            "best_residual": final["best_residual"],
                            "score_gap": final["score_gap"],
                            "best_condition": final["best_condition"],
                            "correct": final["correct"],
                            "false_promotion": final["false_promotion"],
                        })
    return _payload("a11", rows, _a11_summary(rows))


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    return finite(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-15))


def _a12_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    producer_results = {}
    for producer in PRODUCERS:
        group = [row for row in rows if row["producer"] == producer]
        count = len(group)
        correct49 = sum(row["final49"]["correct"] for row in group)
        correct97 = sum(row["final97"]["correct"] for row in group)
        false49 = sum(row["final49"]["false_promotion"] for row in group)
        false97 = sum(row["final97"]["false_promotion"] for row in group)
        agreement = sum(row["signature_equal"] for row in group) / count
        gates = {
            "signature_agreement_ge_0_99": agreement >= 0.99,
            "false97_no_more_than_0_5pp_above_49": (false97 - false49) / count <= 0.005,
            "correct97_no_more_than_1pp_below_49": (correct49 - correct97) / count <= 0.01,
        }
        producer_results[producer] = {
            "rows": count, "signature_agreement": agreement,
            "status_agreement": sum(row["status_equal"] for row in group) / count,
            "support_agreement": sum(row["support_equal"] for row in group) / count,
            "correct49": correct49, "correct97": correct97,
            "false_promotions49": false49, "false_promotions97": false97,
            "correct_rate49": correct49 / count, "correct_rate97": correct97 / count,
            "false_promotion_rate49": false49 / count,
            "false_promotion_rate97": false97 / count,
            "median_design_error_49_to_97": float(np.median([row["design_error_49_to_97"] for row in group])),
            "median_target_error_49_to_97": float(np.median([row["target_error_49_to_97"] for row in group])),
            "gates": gates, "positive_gate": all(gates.values()),
        }
    return {
        "dependence_unit": "episode_seed_cluster",
        "producer_results": producer_results,
        "positive_gate": all(producer_results[p]["positive_gate"] for p in PRODUCERS),
        "interpretation": (
            "SUPPORTS_49_NODE_NUMERICAL_STABILITY"
            if all(producer_results[p]["positive_gate"] for p in PRODUCERS)
            else "FALSIFIES_49_NODE_REFERENCE_STABILITY"
        ),
    }


def run_a12() -> dict[str, Any]:
    RunContract("a12", A12_SEEDS).validate()
    rows = []
    cfg = DiscoveryConfig(temporal_modes=TEMPORAL_MODES)
    for producer in PRODUCERS:
        for family in FAMILIES:
            for seed in A12_SEEDS:
                for model, noise, baseline, fine97, truth in _reference97_episodes(
                    producer, family, seed
                ):
                    channel49 = _nested(fine97, 49)
                    if not np.array_equal(channel49.t, fine97.t[::2]) or not np.array_equal(channel49.u, fine97.u[::2]):
                        raise AssertionError("49-node observation is not exact nested 97-node data")
                    cell49 = _cell(
                        producer, baseline, channel49, fine97, truth.support,
                        TEMPORAL_MODES,
                    )
                    cell97 = _cell(
                        producer, baseline, fine97, fine97, truth.support,
                        TEMPORAL_MODES,
                    )
                    final49, final97 = _summary_final(cell49), _summary_final(cell97)
                    design49, target49, _ = build_evidence((channel49,), cfg)
                    design97, target97, _ = build_evidence((fine97,), cfg)
                    rows.append({
                        "producer": producer, "family": family.value,
                        "seed": seed, "noise_model": model,
                        "relative_noise": noise,
                        "truth_support": None if truth.support is None else list(truth.support),
                        "final49": final49, "final97": final97,
                        "signature_equal": _signature(final49) == _signature(final97),
                        "status_equal": final49["status"] == final97["status"],
                        "support_equal": final49["support"] == final97["support"],
                        "design_error_49_to_97": _relative_error(design49, design97),
                        "target_error_49_to_97": _relative_error(target49, target97),
                    })
    return _payload("a12", rows, _a12_summary(rows))


def _payload(lane: str, rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    freeze = load_and_validate_freeze()
    body = {
        "schema_version": f"verisigs.evidence_view_controls.{lane}.v1",
        "status": "COMPLETED_CONTROL_NOT_PHYSICAL_REPLICATION",
        "lane": lane, "run_contract": RunContract(lane, tuple({"a10": A10_SEEDS, "a11": A11_SEEDS, "a12": A12_SEEDS}[lane])).to_dict(),
        "preoutcome_freeze_sha256": freeze["freeze_sha256"],
        "design_sha256": freeze["design_sha256"],
        "row_count": len(rows), "summary": summary, "rows": rows,
    }
    return body | {"content_sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True, choices=("a10", "a11", "a12"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    load_and_validate_freeze()
    payload = {"a10": run_a10, "a11": run_a11, "a12": run_a12}[args.lane]()
    publish_exclusive(args.output, payload)
    print(json.dumps(payload["summary"], sort_keys=True, indent=2))
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()
