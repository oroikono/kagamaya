#!/usr/bin/env python3
"""Reanalyse verifier health diagnostics and action-level model/error alignment.

This script deliberately keeps the two empirical populations separate:

* the numerical mechanism/confirmation episodes determine per-action error;
* the language-model campaign determines per-action selection frequency.

They are joined only through the frozen action token and its (nodes, modes)
configuration.  No episode-level pairing is attempted or implied.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent


def find_repo() -> Path:
    for parent in HERE.parents:
        if (parent / "evidence_view_benchmark_v1").is_dir():
            return parent
    raise RuntimeError("could not locate the verisigs repository root")


REPO = find_repo()
CASES = REPO / "evidence_view_benchmark_v1" / "cases" / "cases.jsonl"
QUADRATURE = (
    REPO
    / "papers"
    / "neurips_2026"
    / "sim2science_verifier_fidelity_final_20260818"
    / "repro_artifacts"
    / "quadrature_bias_full_development_20seeds.json"
)
CONFIRMATION_SHARDS = (
    REPO
    / "experiments"
    / "agentic_frontier"
    / "wp1_wp4_operator_agent"
    / "adversarial_verifier_gaming_v1"
    / "consensus_mitigation_v2_confirmation"
    / "results"
    / "local_confirmation"
    / "shards"
)
AGENT_SCORED = (
    REPO
    / "evidence_view_benchmark_v1"
    / "gate5a4_claude_full_v1_20260820"
    / "results"
    / "scored_owner_join.json"
)

ACTION_CONFIG = {
    f"A{index:02d}": (nodes, modes)
    for index, (nodes, modes) in enumerate(
        (n, m) for n in (7, 13, 25) for m in (2, 3, 4, 5)
    )
}
CONFIG_ACTION = {config: action for action, config in ACTION_CONFIG.items()}
TERMINAL = {"IDENTIFIED", "OUT_OF_GRAMMAR", "OUTSIDE_LIBRARY"}
ARM_NAMES = {
    "A1": "follow_public_score",
    "A2": "accuracy",
    "A3": "decisiveness",
    "A5": "omission_rewarded",
}


def load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def is_correct(status: str, support: Iterable[str] | None, truth: Iterable[str] | None) -> bool:
    support_tuple = tuple(support or ())
    truth_tuple = None if truth is None else tuple(truth)
    return bool(
        (truth_tuple is not None and status == "IDENTIFIED" and support_tuple == truth_tuple)
        or (truth_tuple is None and status in {"OUT_OF_GRAMMAR", "OUTSIDE_LIBRARY"})
    )


def worked_example() -> dict[str, Any]:
    selected = None
    with CASES.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("legacy_case_id") == "case_00":
                selected = row
                break
    if selected is None:
        raise RuntimeError("legacy case_00 not found")
    rows = {row["action_token"]: row for row in selected["public_rows"]}
    result = {}
    for token in ("A03", "A08"):
        action = rows[token]
        if (action["nodes"], action["temporal_modes"]) != ACTION_CONFIG[token]:
            raise AssertionError(f"worked-example registry mismatch for {token}")
        public = action["spectral_etdrk4"]
        result[token] = {
            "nodes": action["nodes"],
            "temporal_modes": action["temporal_modes"],
            "best_residual": public["best_residual"],
            "score_gap": public["score_gap"],
            "best_condition": public["best_condition"],
            "gram_condition": public["gram_condition"],
            "gate_margin": public["gate_margin"],
            "public_objective": public["public_objective"],
            "terminal_status": public["terminal_status"],
            "gap_clears_identification_threshold": public["score_gap"] >= 0.006,
            "condition_clears_identification_threshold": public["best_condition"] <= 12.0,
        }
    result["comparison"] = {
        "a03_and_a08_clear_gap_threshold": all(
            result[token]["gap_clears_identification_threshold"] for token in ("A03", "A08")
        ),
        "a03_and_a08_clear_condition_threshold": all(
            result[token]["condition_clears_identification_threshold"]
            for token in ("A03", "A08")
        ),
        "a03_gram_condition_is_lower_than_a08": (
            result["A03"]["gram_condition"] < result["A08"]["gram_condition"]
        ),
        "interpretation": (
            "A03 is not rejected by the published gap or best-support condition checks; "
            "its large residual instead authorizes the outside-library result."
        ),
    }
    return result


def mechanism_by_action() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = load_json(QUADRATURE)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in payload["rows"]:
        grouped[(int(row["nodes"]), int(row["temporal_modes"]))].append(row)
    summary = {}
    for config, rows in sorted(grouped.items()):
        token = CONFIG_ACTION.get(config)
        if token is None:
            continue
        n = len(rows)
        summary[token] = {
            "nodes": config[0],
            "temporal_modes": config[1],
            "episodes": n,
            "false_promotions": sum(bool(row["false_promotion"]) for row in rows),
            "false_promotion_rate": sum(bool(row["false_promotion"]) for row in rows) / n,
            "median_best_residual": statistics.median(
                float(row["final"]["best_residual"]) for row in rows
            ),
            "median_target_error_to_49": statistics.median(
                float(row["channel_target_error_to_49"]) for row in rows
            ),
            "median_design_error_to_49": statistics.median(
                float(row["channel_design_error_to_49"]) for row in rows
            ),
        }
    expected = {"A00", "A03", "A04", "A07", "A08", "A11"}
    if set(summary) != expected:
        raise AssertionError(f"unexpected mechanism/action overlap: {sorted(summary)}")
    meta = {
        "population": "720 family-seed-noise conditions per numerical configuration",
        "available_configurations": sorted([list(config) for config in grouped]),
        "shared_agent_actions": sorted(summary),
        "join_scope": "six action configurations only; 49-sample rows have no agent action token",
    }
    return summary, meta


def confirmation_by_action() -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, str]]]:
    paths = sorted(CONFIRMATION_SHARDS.glob("*.json.gz"))
    if len(paths) != 100:
        raise AssertionError(f"expected 100 confirmation shards, found {len(paths)}")
    counts: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    registry: dict[str, tuple[int, int]] = {}
    input_hashes = []
    for path in paths:
        payload = load_json(path)
        producer = payload["config"]["producer"]
        public_by_id = {row["episode_id"]: row for row in payload["public"]["episodes"]}
        owner_by_id = {row["episode_id"]: row for row in payload["owner"]["episodes"]}
        if set(public_by_id) != set(owner_by_id):
            raise AssertionError(f"owner/public mismatch in {path.name}")
        for episode_id, episode in public_by_id.items():
            truth = owner_by_id[episode_id]["truth_support"]
            for action in episode["actions"]:
                token = action["action_token"]
                config = (int(action["nodes"]), int(action["temporal_modes"]))
                if token in registry and registry[token] != config:
                    raise AssertionError(f"confirmation registry drift for {token}")
                registry[token] = config
                if ACTION_CONFIG.get(token) != config:
                    raise AssertionError(f"confirmation/action registry mismatch for {token}")
                status = action["terminal_status"]
                support = action["terminal_support"]
                correct = is_correct(status, support, truth)
                terminal = status in TERMINAL
                counter = counts[producer][token]
                counter["episodes"] += 1
                counter["in_library"] += truth is not None
                counter["omitted"] += truth is None
                counter["correct"] += correct
                counter["false_promotion"] += terminal and not correct
                counter["false_omission"] += truth is not None and status in {
                    "OUT_OF_GRAMMAR",
                    "OUTSIDE_LIBRARY",
                }
                counter["wrong_identification"] += (
                    truth is not None and status == "IDENTIFIED" and not correct
                )
                counter["in_library_terminal"] += truth is not None and terminal
        input_hashes.append({"path": str(path.relative_to(REPO)), "sha256": sha256(path)})
    if registry != ACTION_CONFIG:
        raise AssertionError("confirmation registry does not equal the 12-action registry")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for producer, actions in sorted(counts.items()):
        result[producer] = {}
        for token in sorted(actions):
            c = actions[token]
            if c["episodes"] != 1800 or c["in_library"] != 1500 or c["omitted"] != 300:
                raise AssertionError(f"unexpected confirmation denominator for {producer}/{token}: {c}")
            result[producer][token] = {
                **dict(c),
                "nodes": ACTION_CONFIG[token][0],
                "temporal_modes": ACTION_CONFIG[token][1],
                "false_promotion_rate_all": c["false_promotion"] / c["episodes"],
                "false_omission_rate_in_library": c["false_omission"] / c["in_library"],
                "wrong_identification_rate_in_library": c["wrong_identification"] / c["in_library"],
            }
    return result, input_hashes


def agent_selections() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    payload = load_json(AGENT_SCORED)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in payload["rows"]:
        if row["in_library"]:
            grouped[(row["arm"], row["model"], row["producer"])].append(row)
    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (arm, model, producer), rows in sorted(grouped.items()):
        if len(rows) != 180:
            raise AssertionError(f"expected 180 in-library rows for {arm}/{model}/{producer}")
        counts = Counter()
        invalid = 0
        for row in rows:
            if not row["proposal_valid"] or row["selected_action"] is None:
                invalid += 1
                continue
            token = row["selected_action"]
            config = (int(row["selected_nodes"]), int(row["selected_temporal_modes"]))
            if ACTION_CONFIG.get(token) != config:
                raise AssertionError(f"agent/action registry mismatch for {token}")
            counts[token] += 1
        result.setdefault(ARM_NAMES[arm], {}).setdefault(model, {})[producer] = {
            "denominator": len(rows),
            "valid_choices": len(rows) - invalid,
            "invalid_choices": invalid,
            "action_counts": {token: counts[token] for token in sorted(ACTION_CONFIG)},
            "action_rates": {
                token: counts[token] / len(rows) for token in sorted(ACTION_CONFIG)
            },
        }
    if len(grouped) != 4 * 3 * 2:
        raise AssertionError(f"expected 24 arm/model/producer groups, found {len(grouped)}")
    return result


def join_results(
    mechanism: dict[str, dict[str, Any]],
    confirmation: dict[str, dict[str, dict[str, Any]]],
    selections: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    full_rows = []
    overlap_rows = []
    correlations: dict[str, Any] = {"full_12_action": {}, "mechanism_6_action_overlap": {}}
    for goal, models in sorted(selections.items()):
        for model, producers in sorted(models.items()):
            for producer, selection in sorted(producers.items()):
                key = f"{goal}::{model}::{producer}"
                x_full, y_full = [], []
                for token in sorted(ACTION_CONFIG):
                    sel = selection["action_rates"][token]
                    err = confirmation[producer][token]["false_omission_rate_in_library"]
                    x_full.append(sel)
                    y_full.append(err)
                    full_rows.append(
                        {
                            "goal": goal,
                            "model": model,
                            "producer": producer,
                            "action": token,
                            "nodes": ACTION_CONFIG[token][0],
                            "temporal_modes": ACTION_CONFIG[token][1],
                            "agent_selection_count": selection["action_counts"][token],
                            "agent_selection_denominator": selection["denominator"],
                            "agent_selection_rate": sel,
                            "confirmation_false_omissions": confirmation[producer][token][
                                "false_omission"
                            ],
                            "confirmation_in_library_denominator": confirmation[producer][token][
                                "in_library"
                            ],
                            "confirmation_false_omission_rate": err,
                            "confirmation_wrong_identifications": confirmation[producer][token][
                                "wrong_identification"
                            ],
                        }
                    )
                correlations["full_12_action"][key] = {
                    "actions": 12,
                    "spearman_selection_vs_false_omission": spearman(x_full, y_full),
                    "pearson_selection_vs_false_omission": pearson(x_full, y_full),
                }

                shared = sorted(mechanism)
                x_overlap, y_overlap = [], []
                shared_selection_count = 0
                for token in shared:
                    sel = selection["action_rates"][token]
                    err = mechanism[token]["false_promotion_rate"]
                    shared_selection_count += selection["action_counts"][token]
                    x_overlap.append(sel)
                    y_overlap.append(err)
                    overlap_rows.append(
                        {
                            "goal": goal,
                            "model": model,
                            "producer": producer,
                            "action": token,
                            "nodes": ACTION_CONFIG[token][0],
                            "temporal_modes": ACTION_CONFIG[token][1],
                            "agent_selection_count": selection["action_counts"][token],
                            "agent_selection_denominator": selection["denominator"],
                            "agent_selection_rate": sel,
                            "mechanism_false_promotions": mechanism[token]["false_promotions"],
                            "mechanism_denominator": mechanism[token]["episodes"],
                            "mechanism_false_promotion_rate": err,
                            "mechanism_median_target_error_to_49": mechanism[token][
                                "median_target_error_to_49"
                            ],
                            "mechanism_median_design_error_to_49": mechanism[token][
                                "median_design_error_to_49"
                            ],
                        }
                    )
                correlations["mechanism_6_action_overlap"][key] = {
                    "actions": 6,
                    "selection_count_on_shared_actions": shared_selection_count,
                    "selection_denominator": selection["denominator"],
                    "selection_coverage_on_shared_actions": (
                        shared_selection_count / selection["denominator"]
                    ),
                    "spearman_selection_vs_false_promotion": spearman(x_overlap, y_overlap),
                    "pearson_selection_vs_false_promotion": pearson(x_overlap, y_overlap),
                }
    return full_rows, overlap_rows, correlations


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    health = worked_example()
    mechanism, mechanism_meta = mechanism_by_action()
    confirmation, confirmation_hashes = confirmation_by_action()
    selections = agent_selections()
    full_rows, overlap_rows, correlations = join_results(mechanism, confirmation, selections)

    payload = {
        "schema_version": "verisigs.evidence_view_health_action_join.v1",
        "method": {
            "worked_example": (
                "Exact spectral row from legacy case_00; no score inversion or inferred diagnostic."
            ),
            "full_action_join": (
                "Twelve-action descriptive join by exact token and (nodes,modes) only. "
                "Error rates use 1,500 in-library confirmation conditions per action and "
                "implementation; selections use 180 in-library agent cases per goal/model/implementation."
            ),
            "mechanism_join": (
                "Six-action overlap by exact (nodes,modes). Mechanism rates use 720 conditions per "
                "configuration; selections remain on their separate 180-case denominator."
            ),
            "claim_ceiling": (
                "Action-level association across separate populations; not an episode-level join, "
                "causal estimate, model-ranking result, or independent replication."
            ),
        },
        "input_hashes": {
            "cases_jsonl": sha256(CASES),
            "quadrature_json": sha256(QUADRATURE),
            "agent_scored_json": sha256(AGENT_SCORED),
            "confirmation_shards": confirmation_hashes,
        },
        "worked_example_health": health,
        "mechanism_by_shared_action": mechanism,
        "mechanism_metadata": mechanism_meta,
        "confirmation_by_action": confirmation,
        "agent_selections": selections,
        "correlations": correlations,
    }
    analysis_path = HERE / "analysis.json"
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(HERE / "full_12_action_join.csv", full_rows)
    write_csv(HERE / "mechanism_6_action_overlap.csv", overlap_rows)

    selected_keys = [
        key
        for key in correlations["full_12_action"]
        if key.startswith("follow_public_score::") or key.startswith("omission_rewarded::")
    ]
    lines = [
        "# Evidence-view health and action-level join",
        "",
        "## Worked example",
        "",
        (
            f"A03: gap {health['A03']['score_gap']:.6f} (threshold 0.006), "
            f"best-support condition {health['A03']['best_condition']:.6f} "
            f"(threshold 12), Gram condition {health['A03']['gram_condition']:.3f}."
        ),
        (
            f"A08: gap {health['A08']['score_gap']:.6f}, best-support condition "
            f"{health['A08']['best_condition']:.6f}, Gram condition "
            f"{health['A08']['gram_condition']:.3f}."
        ),
        "",
        "Both rows clear the published gap and best-support condition thresholds. A03 is not "
        "flagged by those two auxiliary checks; its high residual is instead interpreted as "
        "evidence for rejecting the library.",
        "",
        "## Join validity",
        "",
        "- Full join: all 12 action tokens and (nodes, modes) pairs match exactly.",
        "- Mechanism join: only six of the 12 agent actions occur in the 720-condition mechanism "
        "sweep; the 49-sample mechanism rows have no agent action token.",
        "- Denominators remain separate. No case-level pairing is claimed.",
        "",
        "## Selected correlations",
        "",
    ]
    for key in selected_keys:
        row = correlations["full_12_action"][key]
        lines.append(
            f"- {key}: Spearman {row['spearman_selection_vs_false_omission']:.4f}; "
            f"Pearson {row['pearson_selection_vs_false_omission']:.4f}."
        )
    lines.extend(
        [
            "",
            "These correlations summarize alignment across 12 registered actions, not statistical "
            "inference over independent action samples.",
        ]
    )
    (HERE / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checksums = []
    for path in (Path(__file__), analysis_path, HERE / "full_12_action_join.csv", HERE / "mechanism_6_action_overlap.csv", HERE / "REPORT.md"):
        checksums.append(f"{sha256(path)}  {path.name}")
    (HERE / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({
        "worked_example": health,
        "full_join_selected": {key: correlations["full_12_action"][key] for key in selected_keys},
        "outputs": ["analysis.json", "full_12_action_join.csv", "mechanism_6_action_overlap.csv", "REPORT.md", "SHA256SUMS"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
