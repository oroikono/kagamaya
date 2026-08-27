"""Frozen cross-discretization campaign for the finite-difference producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .simulator import (
    BASELINE,
    GRADIENT,
    NESTED_NODES,
    Family,
    clean_pair,
    observed_trajectory,
    subsample,
)
from ..maintrack_suite_v41.weakform import (
    DiscoveryConfig,
    DiscoveryResult,
    SUPPORTS,
    build_evidence,
    discover,
)


TEMPORAL_MODES = (2, 5)
ACTION_COST = 1.0


def _manifest_complete(manifest: tuple[tuple[str, ...], ...]) -> bool:
    """Exact frozen finite-coverage predicate, localized to avoid imports."""

    return len(manifest) == len(SUPPORTS) and set(manifest) == set(SUPPORTS)


def gate(
    manifest: tuple[tuple[str, ...], ...], result: DiscoveryResult
) -> DiscoveryResult:
    """Apply the unchanged frozen residual/gap/conditioning admission gate."""

    if not _manifest_complete(manifest):
        return DiscoveryResult(
            "UNIDENTIFIED",
            None,
            result.best_residual,
            result.score_gap,
            result.ambiguity_set,
            result.fits,
        )
    if result.status == "IDENTIFIED":
        condition = result.fits[0].condition if result.fits else float("inf")
        if (
            result.best_residual <= 0.050
            and result.score_gap >= 0.006
            and condition <= 12.0
        ):
            return result
    if result.status == "OUT_OF_GRAMMAR" and result.best_residual >= 0.20:
        return result
    return DiscoveryResult(
        "UNIDENTIFIED",
        None,
        result.best_residual,
        result.score_gap,
        result.ambiguity_set,
        result.fits,
    )


def record(result: DiscoveryResult) -> dict:
    return {
        "status": result.status,
        "support": None if result.support is None else list(result.support),
        "best_residual": result.best_residual,
        "score_gap": result.score_gap,
        "best_condition": result.fits[0].condition if result.fits else None,
        "ambiguity_set": [list(item) for item in result.ambiguity_set],
    }


def evaluate(
    result: DiscoveryResult, truth: tuple[str, ...] | None
) -> tuple[bool, bool]:
    correct = (
        truth is not None
        and result.status == "IDENTIFIED"
        and result.support == truth
    ) or (truth is None and result.status == "OUT_OF_GRAMMAR")
    false_promotion = (
        result.status in {"IDENTIFIED", "OUT_OF_GRAMMAR"} and not correct
    )
    return bool(correct), bool(false_promotion)


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    families: tuple[str, ...] = tuple(item.value for item in Family)
    noise_models: tuple[str, ...] = ("iid", "correlated")
    relative_noises: tuple[float, ...] = (0.0, 0.001, 0.005)
    public_seed: int = 8675309
    split: str = "confirmation"


def design_sha() -> str:
    return hashlib.sha256(Path(__file__).with_name("DESIGN_FREEZE_PREEXECUTION.md").read_bytes()).hexdigest()


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) / max(float(np.linalg.norm(reference)), 1e-15))


def _cell_record(baseline, channel, reference, truth_support, temporal_modes: int) -> dict:
    cfg = DiscoveryConfig(temporal_modes=temporal_modes)
    raw = discover((baseline, channel), cfg)
    final = gate(tuple(SUPPORTS), raw)
    correct, false = evaluate(final, truth_support)
    channel_design, channel_target, _ = build_evidence((channel,), cfg)
    reference_design, reference_target, _ = build_evidence((reference,), cfg)
    result = record(final)
    top = raw.fits[0] if raw.fits else None
    result["top_fit_support"] = None if top is None else list(top.support)
    result["top_train_residual"] = None if top is None else top.train_residual
    return {
        "nodes": int(channel.t.size),
        "temporal_modes": temporal_modes,
        "mode_to_node_ratio": temporal_modes / int(channel.t.size),
        "action_cost": ACTION_COST,
        "final": result,
        "correct": bool(correct),
        "false_promotion": bool(false),
        "channel_design_error_to_49": _relative_error(channel_design, reference_design),
        "channel_target_error_to_49": _relative_error(channel_target, reference_target),
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _median(rows: list[dict], *keys: str) -> float:
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value[key]
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    output = {}
    for modes in TEMPORAL_MODES:
        for nodes in NESTED_NODES:
            group = [r for r in rows if r["nodes"] == nodes and r["temporal_modes"] == modes]
            output[f"nodes_{nodes}_modes_{modes}"] = {
                "episodes": len(group),
                "correct_rate": _rate(group, "correct"),
                "false_promotion_rate": _rate(group, "false_promotion"),
                "abstention_rate": sum(r["final"]["status"] == "UNIDENTIFIED" for r in group) / len(group),
                "out_of_grammar_rate": sum(r["final"]["status"] == "OUT_OF_GRAMMAR" for r in group) / len(group),
                "median_best_residual": _median(group, "final", "best_residual"),
                "median_score_gap": _median(group, "final", "score_gap"),
                "median_best_condition": _median(group, "final", "best_condition"),
                "median_design_error_to_49": _median(group, "channel_design_error_to_49"),
                "median_target_error_to_49": _median(group, "channel_target_error_to_49"),
            }
    return output


def _two_sided_sign_pvalue(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    lower = min(a_only, b_only)
    tail = sum(math.comb(n, k) for k in range(lower + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def paired_contrasts(rows: list[dict]) -> dict:
    output = {}
    for modes in TEMPORAL_MODES:
        by_episode: dict[tuple, dict[int, dict]] = {}
        for row in rows:
            if row["temporal_modes"] == modes:
                by_episode.setdefault(tuple(row["episode_key"]), {})[row["nodes"]] = row
        pairs = [(cells[7], cells[49]) for cells in by_episode.values()]
        false_7_only = sum(a["false_promotion"] and not b["false_promotion"] for a, b in pairs)
        false_49_only = sum(b["false_promotion"] and not a["false_promotion"] for a, b in pairs)
        output[f"modes_{modes}_7_vs_49"] = {
            "matched_episodes": len(pairs),
            "false_promotion_7": sum(a["false_promotion"] for a, _ in pairs),
            "false_promotion_49": sum(b["false_promotion"] for _, b in pairs),
            "paired_risk_difference_49_minus_7": (
                sum(b["false_promotion"] for _, b in pairs) - sum(a["false_promotion"] for a, _ in pairs)
            ) / len(pairs),
            "false_7_only": false_7_only,
            "false_49_only": false_49_only,
            "discordant_false_pairs": false_7_only + false_49_only,
            "exact_two_sided_sign_pvalue": _two_sided_sign_pvalue(false_7_only, false_49_only),
            "correct_7": sum(a["correct"] for a, _ in pairs),
            "correct_49": sum(b["correct"] for _, b in pairs),
            "status_changed": sum(a["final"]["status"] != b["final"]["status"] for a, b in pairs),
            "top_support_changed": sum(a["final"]["top_fit_support"] != b["final"]["top_fit_support"] for a, b in pairs),
        }
    return output


def ladder_summary(rows: list[dict]) -> dict:
    by_episode: dict[tuple, dict[int, dict]] = {}
    for row in rows:
        if row["temporal_modes"] == 5:
            by_episode.setdefault(tuple(row["episode_key"]), {})[row["nodes"]] = row
    terminals = []
    for key, cells in by_episode.items():
        selected = cells[49]
        for nodes in NESTED_NODES:
            if cells[nodes]["final"]["status"] == "IDENTIFIED":
                selected = cells[nodes]
                break
        terminals.append((key, selected, cells[49]))
    count = len(terminals)
    return {
        "episodes": count,
        "correct": sum(selected["correct"] for _, selected, _ in terminals),
        "false_promotions": sum(selected["false_promotion"] for _, selected, _ in terminals),
        "abstentions": sum(selected["final"]["status"] == "UNIDENTIFIED" for _, selected, _ in terminals),
        "mean_terminal_nodes": sum(selected["nodes"] for _, selected, _ in terminals) / count,
        "terminal_node_counts": {
            str(nodes): sum(selected["nodes"] == nodes for _, selected, _ in terminals)
            for nodes in NESTED_NODES
        },
        "fine_correct": sum(fine["correct"] for _, _, fine in terminals),
        "fine_false_promotions": sum(fine["false_promotion"] for _, _, fine in terminals),
        "status_matches_fine": sum(selected["final"]["status"] == fine["final"]["status"] for _, selected, fine in terminals),
        "support_matches_fine": sum(selected["final"]["support"] == fine["final"]["support"] for _, selected, fine in terminals),
    }


def _encode(payload: dict) -> dict:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    payload["content_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def run(config: CampaignConfig) -> dict:
    if config.split == "confirmation" and any(seed < 9600 or seed > 9619 for seed in config.seeds):
        raise ValueError("confirmation split is frozen to seeds 9600--9619")
    rows = []
    ordinal = 0
    for family_name in config.families:
        family = Family(family_name)
        for seed in config.seeds:
            baseline_clean, gradient_clean, truth = clean_pair(family, seed, config.public_seed)
            for noise_model in config.noise_models:
                for noise in config.relative_noises:
                    baseline = observed_trajectory(BASELINE, seed, config.public_seed, baseline_clean[2], baseline_clean[0], baseline_clean[1], noise, noise_model)
                    fine = observed_trajectory(GRADIENT, seed, config.public_seed, gradient_clean[2], gradient_clean[0], gradient_clean[1], noise, noise_model)
                    channels = {nodes: subsample(fine, nodes) for nodes in NESTED_NODES}
                    common = {
                        "split": config.split,
                        "episode": ordinal,
                        "episode_key": [family.value, seed, noise_model, noise],
                        "truth_support": None if truth.support is None else list(truth.support),
                    }
                    for modes in TEMPORAL_MODES:
                        for nodes in NESTED_NODES:
                            rows.append({
                                **common,
                                **_cell_record(baseline, channels[nodes], fine, truth.support, modes),
                            })
                    ordinal += 1
    payload = {
        "status": "CROSS_DISCRETIZATION_CONFIRMATION_AUDIT_ONLY",
        "design_freeze_sha256": design_sha(),
        "config": asdict(config),
        "producer": {
            "space": "periodic_centered_nine_point_finite_difference",
            "time": "scipy_solve_ivp_BDF",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_step": 0.001,
            "independence_claim": "same_repository_cross_discretization_only",
        },
        "grammar_size": len(SUPPORTS),
        "admission_gate": {"residual_max": 0.05, "gap_min": 0.006, "condition_max": 12.0, "oog_residual_min": 0.20},
        "summary": summarize(rows),
        "paired_contrasts": paired_contrasts(rows),
        "ladder": ladder_summary(rows),
        "rows": rows,
    }
    return _encode(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--split", default="confirmation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(CampaignConfig(tuple(int(x) for x in args.seeds.split(",") if x), split=args.split))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"paired": result["paired_contrasts"], "ladder": result["ladder"]}, indent=2))
    print(result["content_sha256"])


if __name__ == "__main__":
    main()
