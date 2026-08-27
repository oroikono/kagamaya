"""Prospective matched confirmation for verification-fidelity control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..independent_fd_producer_v1.campaign import _cell_record as fd_cell
from ..independent_fd_producer_v1.simulator import (
    BASELINE as FD_BASELINE,
    GRADIENT as FD_GRADIENT,
    NESTED_NODES,
    Family as FDFamily,
    clean_pair,
    observed_trajectory,
    subsample,
)
from ..maintrack_suite_v41.simulator import Family
from ..unified_quadrature_bias_v1.simulator import matched_episode
from ..maintrack_suite_v41.weakform import (
    DiscoveryConfig, DiscoveryResult, SUPPORTS, build_evidence, discover,
)


CONFIRMATION_SEEDS = tuple(range(9700, 9750))
SMOKE_SEEDS = tuple(range(9650, 9700))
TEMPORAL_MODES = (2, 5)
NOISE_MODELS = ("iid", "correlated")
RELATIVE_NOISES = (0.0, 0.001, 0.005)
PUBLIC_SEED = 8675309


@dataclass(frozen=True)
class Config:
    producer: str
    seeds: tuple[int, ...]
    split: str
    public_seed: int = PUBLIC_SEED


def digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def design_sha() -> str:
    return hashlib.sha256(Path(__file__).with_name(
        "DESIGN_FREEZE_PREEXECUTION.md").read_bytes()).hexdigest()


def source_sha() -> str:
    lane = Path(__file__).resolve().parent
    root = lane.parent
    paths = (
        lane / "DESIGN_FREEZE_PREEXECUTION.md", lane / "campaign.py",
        lane / "aggregate.py", lane / "validate.py", lane / "test_protocol.py",
        root / "unified_quadrature_bias_v1" / "simulator.py",
        root / "independent_fd_producer_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "simulator.py",
        root / "maintrack_suite_v41" / "weakform.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete source closure: {missing}")
    material = b"".join(
        str(path.relative_to(root)).encode() + b"\0" + path.read_bytes()
        for path in sorted(paths)
    )
    return hashlib.sha256(material).hexdigest()


def _gate(result: DiscoveryResult) -> DiscoveryResult:
    """Local exact copy of the frozen complete-manifest admission rule."""
    if result.status == "IDENTIFIED":
        condition = result.fits[0].condition if result.fits else float("inf")
        if result.best_residual <= 0.05 and result.score_gap >= 0.006 and condition <= 12.0:
            return result
    if result.status == "OUT_OF_GRAMMAR" and result.best_residual >= 0.20:
        return result
    return DiscoveryResult("UNIDENTIFIED", None, result.best_residual,
                           result.score_gap, result.ambiguity_set, result.fits)


def _record(result: DiscoveryResult) -> dict:
    return {"status": result.status,
            "support": None if result.support is None else list(result.support),
            "best_residual": float(result.best_residual),
            "score_gap": float(result.score_gap),
            "best_condition": float(result.fits[0].condition) if result.fits else None,
            "ambiguity_set": [list(item) for item in result.ambiguity_set]}


def _evaluate(result: DiscoveryResult, truth) -> tuple[bool, bool]:
    correct = ((truth is not None and result.status == "IDENTIFIED" and result.support == truth)
               or (truth is None and result.status == "OUT_OF_GRAMMAR"))
    false = result.status in {"IDENTIFIED", "OUT_OF_GRAMMAR"} and not correct
    return bool(correct), bool(false)


def _relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(value - reference) /
                 max(float(np.linalg.norm(reference)), 1e-15))


def spectral_cell(baseline, channel, reference, truth, modes: int) -> dict:
    cfg = DiscoveryConfig(temporal_modes=modes)
    raw = discover((baseline, channel), cfg); final = _gate(raw)
    correct, false = _evaluate(final, truth)
    design, target, _ = build_evidence((channel,), cfg)
    ref_design, ref_target, _ = build_evidence((reference,), cfg)
    result = _record(final); top = raw.fits[0] if raw.fits else None
    result["top_fit_support"] = None if top is None else list(top.support)
    result["top_train_residual"] = None if top is None else float(top.train_residual)
    return {"nodes": int(channel.t.size), "temporal_modes": modes,
            "mode_to_node_ratio": modes / int(channel.t.size), "action_cost": 1.0,
            "final": result, "correct": correct, "false_promotion": false,
            "channel_design_error_to_49": _relative_error(design, ref_design),
            "channel_target_error_to_49": _relative_error(target, ref_target)}


def signature(row: dict) -> tuple[str, tuple[str, ...]]:
    return row["final"]["status"], tuple(row["final"]["support"] or ())


def select(cells: dict[int, dict], policy: str) -> dict:
    if policy.startswith("fixed_"):
        return cells[int(policy.split("_")[1])]
    if policy == "adaptive_identified":
        for nodes in NESTED_NODES:
            if cells[nodes]["final"]["status"] == "IDENTIFIED":
                return cells[nodes]
        return cells[49]
    if policy == "unsafe_first_terminal":
        for nodes in NESTED_NODES:
            if cells[nodes]["final"]["status"] in {"IDENTIFIED", "OUT_OF_GRAMMAR"}:
                return cells[nodes]
        return cells[49]
    if policy == "two_resolution_stable":
        for left, right in zip(NESTED_NODES[:-1], NESTED_NODES[1:]):
            if signature(cells[left]) == signature(cells[right]) and cells[right]["final"]["status"] != "UNIDENTIFIED":
                return cells[right]
        return cells[49]
    if policy == "oracle_match":
        target = signature(cells[49])
        return next(cells[nodes] for nodes in NESTED_NODES if signature(cells[nodes]) == target)
    raise ValueError(f"unknown policy {policy}")


POLICIES = tuple(f"fixed_{n}" for n in NESTED_NODES) + (
    "adaptive_identified", "unsafe_first_terminal", "two_resolution_stable",
    "oracle_match",
)


def _producer_episode(producer: str, family: Family, seed: int, noise_model: str,
                      noise: float, public_seed: int) -> tuple[dict[int, object], object, object]:
    if producer == "spectral_etdrk4":
        baseline, channels, truth = matched_episode(
            family, seed, public_seed, noise, noise_model
        )
        return channels, baseline, truth
    if producer == "fd_bdf":
        # The FD producer owns a distinct Enum class with the same serialized
        # values.  Bridge explicitly: identity-based family dispatch otherwise
        # falls through to the producer's final branch.
        fd_family = FDFamily(family.value)
        baseline_clean, gradient_clean, truth = clean_pair(fd_family, seed, public_seed)
        baseline = observed_trajectory(
            FD_BASELINE, seed, public_seed, baseline_clean[2], baseline_clean[0],
            baseline_clean[1], noise, noise_model,
        )
        fine = observed_trajectory(
            FD_GRADIENT, seed, public_seed, gradient_clean[2], gradient_clean[0],
            gradient_clean[1], noise, noise_model,
        )
        return {nodes: subsample(fine, nodes) for nodes in NESTED_NODES}, baseline, truth
    raise ValueError("producer must be spectral_etdrk4 or fd_bdf")


def _summarize(rows: list[dict]) -> dict:
    output = {}
    for modes in TEMPORAL_MODES:
        mode_rows = [row for row in rows if row["temporal_modes"] == modes]
        by_episode: dict[tuple, dict[int, dict]] = {}
        for row in mode_rows:
            by_episode.setdefault(tuple(row["episode_key"]), {})[row["nodes"]] = row
        policies = {}
        for policy in POLICIES:
            chosen = [select(cells, policy) for cells in by_episode.values()]
            fine = [cells[49] for cells in by_episode.values()]
            policies[policy] = {
                "episodes": len(chosen),
                "signature_match_fine": sum(signature(a) == signature(b) for a, b in zip(chosen, fine)),
                "correct": sum(bool(row["correct"]) for row in chosen),
                "false_promotions": sum(bool(row["false_promotion"]) for row in chosen),
                "mean_nodes": sum(row["nodes"] for row in chosen) / len(chosen),
                "terminal_node_counts": {
                    str(n): sum(row["nodes"] == n for row in chosen) for n in NESTED_NODES
                },
            }
        adaptive, fixed25 = policies["adaptive_identified"], policies["fixed_25"]
        n = adaptive["episodes"]
        gates = {
            "match_ge_99_5pct": adaptive["signature_match_fine"] / n >= 0.995,
            "false_le_fine_plus_0_2pp": adaptive["false_promotions"] / n <= policies["fixed_49"]["false_promotions"] / n + 0.002,
            "mean_nodes_le_30": adaptive["mean_nodes"] <= 30.0,
            "match_advantage_over_fixed25_ge_2pp": (adaptive["signature_match_fine"] - fixed25["signature_match_fine"]) / n >= 0.02,
            "mean_nodes_le_fixed25_plus_3": adaptive["mean_nodes"] <= fixed25["mean_nodes"] + 3.0,
        }
        output[f"modes_{modes}"] = {
            "policies": policies, "gates": gates,
            "positive_gate": all(gates.values()),
        }
    return output


def run(config: Config) -> dict:
    allowed = {"confirmation": set(CONFIRMATION_SEEDS), "smoke": set(SMOKE_SEEDS)}
    if config.split not in allowed or not set(config.seeds) <= allowed[config.split]:
        raise ValueError("seed range does not match frozen split")
    rows, ordinal = [], 0
    for family in Family:
        for seed in config.seeds:
            for noise_model in NOISE_MODELS:
                for noise in RELATIVE_NOISES:
                    channels, baseline, truth = _producer_episode(
                        config.producer, family, seed, noise_model, noise,
                        config.public_seed,
                    )
                    reference = channels[49]
                    truth_support = truth.support
                    for modes in TEMPORAL_MODES:
                        for nodes in NESTED_NODES:
                            if config.producer == "spectral_etdrk4":
                                cell = spectral_cell(baseline, channels[nodes], reference, truth_support, modes)
                            else:
                                cell = fd_cell(baseline, channels[nodes], reference, truth_support, modes)
                            rows.append({
                                "producer": config.producer, "split": config.split,
                                "episode": ordinal,
                                "episode_key": [family.value, seed, noise_model, noise],
                                "truth_support": None if truth_support is None else list(truth_support),
                                **cell,
                            })
                    ordinal += 1
    summary = _summarize(rows)
    payload = {
        "status": "CONFIRMATION_CANDIDATE_NOT_INDEPENDENT_REPLICATION",
        "design_freeze_sha256": design_sha(), "source_closure_sha256": source_sha(),
        "config": asdict(config), "policies": list(POLICIES),
        "summary": summary, "rows": rows,
    }
    payload["positive_gate"] = bool(summary["modes_5"]["positive_gate"])
    payload["content_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True, choices=("spectral_etdrk4", "fd_bdf"))
    parser.add_argument("--split", required=True, choices=("smoke", "confirmation"))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(Config(args.producer, tuple(int(x) for x in args.seeds.split(",") if x), args.split))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"]["modes_5"], indent=2, sort_keys=True))
    print(result["content_sha256"])


if __name__ == "__main__":
    main()
