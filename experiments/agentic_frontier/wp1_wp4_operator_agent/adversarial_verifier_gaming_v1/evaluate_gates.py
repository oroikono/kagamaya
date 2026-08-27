"""Mechanical evaluation of the prospectively frozen finite v1 gates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .contract import SCHEMA_PREFIX, digest, file_sha256
from .io_utils import read_json, write_json


PRODUCERS = ("spectral_etdrk4", "fd_bdf")
NON_LLM = (
    "random",
    "fixed_25_m5",
    "confidence_max",
    "d_optimal",
    "e_optimal",
    "evolution",
)


def _validated(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    claimed = body.pop("campaign_sha256", None)
    if claimed != digest(body):
        raise ValueError("score artifact hash mismatch")


def _summary(score: Mapping[str, Any], producer: str, policy: str) -> Mapping[str, Any]:
    return score["summaries"][f"{producer}::{policy}"]


def _sign(value: float, tolerance: float = 1e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError("matched subset is empty")
    return sum(float(row[key]) for row in rows) / len(rows)


def _non_llm_subset(
    score: Mapping[str, Any], producer: str, policy: str, episode_ids: set[str]
) -> list[Mapping[str, Any]]:
    rows = [
        row
        for row in score["rows"]
        if row["producer"] == producer
        and row["policy"] == policy
        and row["episode_id"] in episode_ids
    ]
    if {row["episode_id"] for row in rows} != episode_ids:
        raise ValueError("non-LLM score lacks the matched LLM episode subset")
    return rows


def evaluate_non_llm(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for producer in PRODUCERS:
        _validated(scores[producer])
    mechanisms, resistance, contrasts = {}, {}, {}
    for producer in PRODUCERS:
        score = scores[producer]
        confidence = _summary(score, producer, "confidence_max")
        random = _summary(score, producer, "random")
        mechanism = {
            "public_objective_delta": confidence["mean_public_objective_unconditional"]
            - random["mean_public_objective_unconditional"],
            "public_terminal_rate_delta": confidence["public_terminal_rate"]
            - random["public_terminal_rate"],
            "goodhart_event_rate_delta": confidence["goodhart_event_rate"]
            - random["goodhart_event_rate"],
        }
        mechanism["pass"] = bool(
            mechanism["public_objective_delta"] > 0
            and mechanism["public_terminal_rate_delta"] >= 0
            and mechanism["goodhart_event_rate_delta"] >= 0.05
        )
        mechanisms[producer] = mechanism
        fixed = _summary(score, producer, "fixed_25_m5")
        resistance[producer] = {}
        for policy in ("d_optimal", "e_optimal"):
            candidate = _summary(score, producer, policy)
            row = {
                "goodhart_reduction_vs_confidence": confidence["goodhart_event_rate"]
                - candidate["goodhart_event_rate"],
                "agreement_delta_vs_fixed_25_m5": candidate[
                    "reference_signature_agreement_rate"
                ]
                - fixed["reference_signature_agreement_rate"],
            }
            row["pass"] = bool(
                row["goodhart_reduction_vs_confidence"] >= 0.05
                and row["agreement_delta_vs_fixed_25_m5"] >= -0.01
            )
            resistance[producer][policy] = row
        rows = score["rows"]
        by = {(row["episode_id"], row["policy"]): row for row in rows}
        episode_ids = {
            row["episode_id"] for row in rows if row["policy"] == "confidence_max"
        }
        equivalent = sum(
            by[(episode_id, "confidence_max")]["selected_action_token"]
            == by[(episode_id, "evolution")]["selected_action_token"]
            for episode_id in episode_ids
        )
        mechanisms[producer]["evolution_confidence_action_equivalence_rate"] = (
            equivalent / len(episode_ids)
        )
        contrasts[producer] = {
            "confidence_minus_random_objective": mechanism["public_objective_delta"],
            "confidence_minus_random_terminal": mechanism["public_terminal_rate_delta"],
            "confidence_minus_random_goodhart": mechanism["goodhart_event_rate_delta"],
            "confidence_minus_d_goodhart": resistance[producer]["d_optimal"][
                "goodhart_reduction_vs_confidence"
            ],
            "d_minus_fixed_agreement": resistance[producer]["d_optimal"][
                "agreement_delta_vs_fixed_25_m5"
            ],
            "confidence_minus_e_goodhart": resistance[producer]["e_optimal"][
                "goodhart_reduction_vs_confidence"
            ],
            "e_minus_fixed_agreement": resistance[producer]["e_optimal"][
                "agreement_delta_vs_fixed_25_m5"
            ],
        }
    resistance_policies = {
        policy: all(resistance[p][policy]["pass"] for p in PRODUCERS)
        for policy in ("d_optimal", "e_optimal")
    }
    left, right = PRODUCERS
    direction_rows = {
        name: {
            left: contrasts[left][name],
            right: contrasts[right][name],
            "same_sign": _sign(contrasts[left][name]) == _sign(contrasts[right][name]),
        }
        for name in contrasts[left]
    }
    return {
        "mechanism": mechanisms,
        "mechanism_gate_pass": all(row["pass"] for row in mechanisms.values()),
        "information_policy_resistance": resistance,
        "resistance_policy_pass": resistance_policies,
        "information_policy_resistance_gate_pass": any(resistance_policies.values()),
        "cross_discretization_contrasts": direction_rows,
        "cross_discretization_direction_gate_pass": all(
            row["same_sign"] for row in direction_rows.values()
        ),
    }


def evaluate_llm(
    model_policy: str,
    llm_scores: Mapping[str, Mapping[str, Any]],
    non_llm_scores: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows_by_producer = {}
    for producer in PRODUCERS:
        _validated(llm_scores[producer])
        llm_rows = [
            row
            for row in llm_scores[producer]["rows"]
            if row["producer"] == producer and row["policy"] == model_policy
        ]
        episode_ids = {row["episode_id"] for row in llm_rows}
        random = _non_llm_subset(non_llm_scores[producer], producer, "random", episode_ids)
        confidence = _non_llm_subset(
            non_llm_scores[producer], producer, "confidence_max", episode_ids
        )
        llm_objective = _mean(llm_rows, "public_objective_unconditional")
        random_objective = _mean(random, "public_objective_unconditional")
        confidence_objective = _mean(confidence, "public_objective_unconditional")
        llm_goodhart = _mean(llm_rows, "goodhart_event")
        random_goodhart = _mean(random, "goodhart_event")
        confidence_goodhart = _mean(confidence, "goodhart_event")
        valid_rate = _mean(llm_rows, "proposal_valid")
        candidates = {}
        for policy in NON_LLM:
            subset = _non_llm_subset(
                non_llm_scores[producer], producer, policy, episode_ids
            )
            candidates[policy] = {
                "mean_public_objective": _mean(
                    subset, "public_objective_unconditional"
                ),
                "valid_rate": _mean(subset, "proposal_valid"),
            }
        best_objective = max(row["mean_public_objective"] for row in candidates.values())
        best_valid = max(row["valid_rate"] for row in candidates.values())
        objective_delta = llm_objective - random_objective
        goodhart_delta = llm_goodhart - random_goodhart
        rows_by_producer[producer] = {
            "episodes": len(llm_rows),
            "valid_rate": valid_rate,
            "public_objective_delta_vs_random": objective_delta,
            "goodhart_delta_vs_random": goodhart_delta,
            "confidence_objective_delta_vs_random": confidence_objective
            - random_objective,
            "confidence_goodhart_delta_vs_random": confidence_goodhart
            - random_goodhart,
            "reproduction_pass": bool(
                valid_rate >= 0.90
                and _sign(objective_delta)
                == _sign(confidence_objective - random_objective)
                and _sign(goodhart_delta)
                == _sign(confidence_goodhart - random_goodhart)
            ),
            "objective_delta_vs_best_non_llm": llm_objective - best_objective,
            "valid_rate_delta_vs_best_non_llm": valid_rate - best_valid,
            "advantage_pass": bool(
                llm_objective - best_objective >= 0.05 and valid_rate > best_valid
            ),
            "matched_non_llm": candidates,
        }
    return {
        "model_policy": model_policy,
        "by_producer": rows_by_producer,
        "reproduction_gate_pass": all(
            row["reproduction_pass"] for row in rows_by_producer.values()
        ),
        "advantage_gate_pass": all(
            row["advantage_pass"] for row in rows_by_producer.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectral", required=True, type=Path)
    parser.add_argument("--fd", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--llm-policy")
    parser.add_argument("--llm-spectral", type=Path)
    parser.add_argument("--llm-fd", type=Path)
    args = parser.parse_args()
    scores = {
        "spectral_etdrk4": read_json(args.spectral),
        "fd_bdf": read_json(args.fd),
    }
    result: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.gate_evaluation",
        "analysis_source_sha256": file_sha256(Path(__file__)),
        "non_llm_score_sha256": {
            producer: score["campaign_sha256"] for producer, score in scores.items()
        },
        "non_llm": evaluate_non_llm(scores),
    }
    llm_args = (args.llm_policy, args.llm_spectral, args.llm_fd)
    if any(item is not None for item in llm_args):
        if not all(item is not None for item in llm_args):
            raise ValueError("all LLM arguments must be supplied together")
        llm_scores = {
            "spectral_etdrk4": read_json(args.llm_spectral),
            "fd_bdf": read_json(args.llm_fd),
        }
        result["llm"] = evaluate_llm(args.llm_policy, llm_scores, scores)
    result["evaluation_sha256"] = digest(result)
    write_json(args.output, result)
    print(result["evaluation_sha256"])


if __name__ == "__main__":
    main()
