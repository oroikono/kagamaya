"""Frozen matched factorial for channel-specific temporal quadrature bias."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .simulator import ACTION_COST, CELLS, TEMPORAL_MODES, matched_episode
from ..maintrack_suite_v41.simulator import Family
from ..maintrack_suite_v41.weakform import (
    DiscoveryConfig,
    SUPPORTS,
    build_evidence,
    discover,
)
from ..unified_agent_v2_1_scaled.campaign import evaluate, gate, record


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    families: tuple[str, ...] = tuple(item.value for item in Family)
    noise_models: tuple[str, ...] = ("iid", "correlated")
    relative_noises: tuple[float, ...] = (0.0, 0.001, 0.005)
    public_seed: int = 8675309
    split: str = "development"


def design_sha() -> str:
    return hashlib.sha256(
        Path(__file__).with_name("DESIGN_FREEZE_PREEXECUTION.md").read_bytes()
    ).hexdigest()


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(value - reference) / max(float(np.linalg.norm(reference)), 1e-15)
    )


def _cell_record(
    baseline,
    channel,
    reference_channel,
    truth_support,
    temporal_modes: int,
) -> dict:
    cfg = DiscoveryConfig(temporal_modes=temporal_modes)
    raw = discover((baseline, channel), cfg)
    final = gate(tuple(SUPPORTS), raw)
    correct, false = evaluate(final, truth_support)
    channel_design, channel_target, _ = build_evidence((channel,), cfg)
    ref_design, ref_target, _ = build_evidence((reference_channel,), cfg)
    top = raw.fits[0] if raw.fits else None
    result = record(final)
    result["top_fit_support"] = None if top is None else list(top.support)
    result["top_train_residual"] = None if top is None else top.train_residual
    return {
        "nodes": int(channel.t.size),
        "temporal_modes": temporal_modes,
        "mode_to_node_ratio": temporal_modes / int(channel.t.size),
        "action_cost": ACTION_COST,
        "final": result,
        "correct": correct,
        "false_promotion": false,
        "channel_design_error_to_49": _relative_error(channel_design, ref_design),
        "channel_target_error_to_49": _relative_error(channel_target, ref_target),
    }


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def _median(rows: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if values else float("nan")


def summarize(rows: list[dict]) -> dict:
    cells = {}
    for modes in TEMPORAL_MODES:
        for cell in CELLS:
            group = [
                row
                for row in rows
                if row["nodes"] == cell.nodes and row["temporal_modes"] == modes
            ]
            key = f"nodes_{cell.nodes}_modes_{modes}"
            cells[key] = {
                "episodes": len(group),
                "correct_rate": _rate(group, "correct"),
                "false_promotion_rate": _rate(group, "false_promotion"),
                "abstention_rate": sum(
                    row["final"]["status"] == "UNIDENTIFIED" for row in group
                )
                / len(group),
                "out_of_grammar_rate": sum(
                    row["final"]["status"] == "OUT_OF_GRAMMAR" for row in group
                )
                / len(group),
                "median_best_residual": _median(group, ("final", "best_residual")),
                "median_score_gap": _median(group, ("final", "score_gap")),
                "median_best_condition": _median(group, ("final", "best_condition")),
                "median_design_error_to_49": _median(
                    group, ("channel_design_error_to_49",)
                ),
                "median_target_error_to_49": _median(
                    group, ("channel_target_error_to_49",)
                ),
            }
    return cells


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
        group = [row for row in rows if row["temporal_modes"] == modes]
        by_episode: dict[tuple, dict[int, dict]] = {}
        for row in group:
            by_episode.setdefault(tuple(row["episode_key"]), {})[row["nodes"]] = row
        pairs = [(cells[7], cells[49]) for cells in by_episode.values()]
        false_7_only = sum(a["false_promotion"] and not b["false_promotion"] for a, b in pairs)
        false_49_only = sum(b["false_promotion"] and not a["false_promotion"] for a, b in pairs)
        false_at_7 = [pair for pair in pairs if pair[0]["false_promotion"]]
        output[f"modes_{modes}_7_vs_49"] = {
            "matched_episodes": len(pairs),
            "false_promotion_7": sum(a["false_promotion"] for a, _ in pairs),
            "false_promotion_49": sum(b["false_promotion"] for _, b in pairs),
            "paired_risk_difference_49_minus_7": (
                sum(b["false_promotion"] for _, b in pairs)
                - sum(a["false_promotion"] for a, _ in pairs)
            )
            / len(pairs),
            "false_7_only": false_7_only,
            "false_49_only": false_49_only,
            "discordant_false_pairs": false_7_only + false_49_only,
            "exact_two_sided_sign_pvalue": _two_sided_sign_pvalue(
                false_7_only, false_49_only
            ),
            "correct_7": sum(a["correct"] for a, _ in pairs),
            "correct_49": sum(b["correct"] for _, b in pairs),
            "wrong_at_7_to_correct_at_49": sum(
                b["correct"] for _, b in false_at_7
            ),
            "wrong_at_7_to_abstain_at_49": sum(
                b["final"]["status"] == "UNIDENTIFIED" for _, b in false_at_7
            ),
            "wrong_at_7_persists_wrong_at_49": sum(
                b["false_promotion"] for _, b in false_at_7
            ),
            "status_changed": sum(
                a["final"]["status"] != b["final"]["status"] for a, b in pairs
            ),
            "top_support_changed": sum(
                a["final"]["top_fit_support"] != b["final"]["top_fit_support"]
                for a, b in pairs
            ),
        }
    return output


def stratified(rows: list[dict]) -> dict:
    output = {}
    fields = (("family", 0), ("noise_model", 2), ("noise_level", 3))
    for label, index in fields:
        values = sorted({row["episode_key"][index] for row in rows}, key=str)
        for value in values:
            for nodes in (7, 49):
                group = [
                    row
                    for row in rows
                    if row["temporal_modes"] == 5
                    and row["nodes"] == nodes
                    and row["episode_key"][index] == value
                ]
                output[f"{label}={value}|nodes={nodes}|modes=5"] = {
                    "episodes": len(group),
                    "correct_rate": _rate(group, "correct"),
                    "false_promotion_rate": _rate(group, "false_promotion"),
                }
    return output


def run(config: CampaignConfig) -> dict:
    if any(seed >= 9500 for seed in config.seeds):
        raise ValueError("reserved confirmation seeds are forbidden")
    rows = []
    ordinal = 0
    for family_name in config.families:
        family = Family(family_name)
        for seed in config.seeds:
            for noise_model in config.noise_models:
                for noise in config.relative_noises:
                    baseline, channels, truth = matched_episode(
                        family,
                        seed,
                        config.public_seed,
                        noise,
                        noise_model,
                    )
                    reference = channels[49]
                    common = {
                        "split": config.split,
                        "episode": ordinal,
                        "episode_key": [family.value, seed, noise_model, noise],
                        "truth_support": (
                            None if truth.support is None else list(truth.support)
                        ),
                    }
                    for modes in TEMPORAL_MODES:
                        for cell in CELLS:
                            rows.append(
                                {
                                    **common,
                                    **_cell_record(
                                        baseline,
                                        channels[cell.nodes],
                                        reference,
                                        truth.support,
                                        modes,
                                    ),
                                }
                            )
                    ordinal += 1
    payload = {
        "status": "MATCHED_DEVELOPMENT_NOT_CONFIRMATORY",
        "design_freeze_sha256": design_sha(),
        "config": asdict(config),
        "grammar_size": len(SUPPORTS),
        "admission_gate": {
            "residual_max": 0.05,
            "gap_min": 0.006,
            "condition_max": 12.0,
            "oog_residual_min": 0.20,
        },
        "summary": summarize(rows),
        "paired_contrasts": paired_contrasts(rows),
        "stratified": stratified(rows),
        "rows": rows,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="4100,4101")
    parser.add_argument("--split", default="local_development")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = CampaignConfig(
        tuple(int(item) for item in args.seeds.split(",") if item), split=args.split
    )
    result = run(config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["paired_contrasts"], indent=2, sort_keys=True))
    print(result["content_sha256"])


if __name__ == "__main__":
    main()

