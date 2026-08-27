"""Reserved-seed sensitivity test for nested versus disjoint weak-test banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..maintrack_suite_v41.simulator import Family, Trajectory
from ..maintrack_suite_v41.weakform import (
    DiscoveryConfig,
    DiscoveryResult,
    SUPPORTS,
    _space_tests,
    build_evidence,
    discover,
    enumerate_fits,
    trajectory_evidence,
)
from ..unified_agent_v2_1_scaled.campaign import evaluate, gate, record
from ..unified_quadrature_bias_v1.simulator import matched_episode


NODES = (7, 13, 25, 49)
LOW_MODES = (1, 2)
CUMULATIVE_MODES = (1, 2, 3, 4, 5)
HIGH_MODES = (3, 4, 5)
CONFIRMATION_SEEDS = tuple(range(62000, 62020))
NOISE_MODELS = ("iid", "correlated")
RELATIVE_NOISES = (0.0, 0.001, 0.005)
PUBLIC_SEED = 8675309


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    split: str = "confirmation"
    families: tuple[str, ...] = tuple(item.value for item in Family)
    noise_models: tuple[str, ...] = NOISE_MODELS
    relative_noises: tuple[float, ...] = RELATIVE_NOISES
    public_seed: int = PUBLIC_SEED


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def design_sha() -> str:
    return hashlib.sha256(
        Path(__file__).with_name("DESIGN_FREEZE_PREEXECUTION.md").read_bytes()
    ).hexdigest()


def source_sha() -> str:
    lane = Path(__file__).resolve().parent
    operator_root = lane.parent
    paths = (
        lane / "denested_bank.py",
        lane / "validate.py",
        operator_root / "unified_quadrature_bias_v1" / "simulator.py",
        operator_root / "unified_agent_v2_1_scaled" / "campaign.py",
        operator_root / "maintrack_suite_v41" / "weakform.py",
        operator_root / "maintrack_suite_v41" / "simulator.py",
        operator_root / "maintrack_suite_v41" / "numerics.py",
    )
    payload = b"".join(
        str(path.relative_to(operator_root)).encode() + b"\0" + path.read_bytes()
        for path in sorted(paths)
    )
    return hashlib.sha256(payload).hexdigest()


def trajectory_evidence_modes(
    trajectory: Trajectory,
    temporal_modes: tuple[int, ...],
    spatial_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The frozen authority evidence with an explicit temporal-mode index set."""

    x, t, u = trajectory.x, trajectory.t, trajectory.u
    if u.shape != (t.size, x.size) or t.size % 2 == 0:
        raise ValueError("trajectory requires matching shapes and odd saved-time count")
    if not temporal_modes or len(set(temporal_modes)) != len(temporal_modes):
        raise ValueError("temporal mode indices must be nonempty and unique")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("uniform time grid required")
    dx = float(x[1] - x[0])
    duration = float(t[-1] - t[0])
    scaled = (t - t[0]) / duration
    q = np.ones(t.size)
    q[1:-1:2] = 4.0
    q[2:-1:2] = 2.0
    q *= dt / 3.0
    fields = (np.ones_like(u), u, u**2, u**3)
    rows: list[list[float]] = []
    targets: list[float] = []
    train: list[bool] = []
    for tm in temporal_modes:
        eta = np.sin(tm * np.pi * scaled)
        deta = tm * np.pi / duration * np.cos(tm * np.pi * scaled)
        wet = q * eta
        wdt = q * deta
        for sm, (psi, psi1, psi2, psi3) in enumerate(_space_tests(x, spatial_modes)):
            def integ(wt: np.ndarray, field: np.ndarray, wx: np.ndarray) -> float:
                return float(dx * np.einsum("t,tx,x->", wt, field, wx))

            targets.append(-integ(wdt, u, psi))
            rows.append(
                [
                    integ(wet, fields[0], psi),
                    integ(wet, fields[1], psi),
                    integ(wet, fields[2], psi),
                    integ(wet, fields[3], psi),
                    -integ(wet, u, psi1),
                    integ(wet, u, psi2),
                    -integ(wet, u, psi3),
                    -0.5 * integ(wet, fields[2], psi1),
                ]
            )
            train.append((tm + sm) % 2 == 0)
    return np.asarray(rows), np.asarray(targets), np.asarray(train, dtype=bool)


def _discover_modes(
    trajectories: tuple[Trajectory, ...], temporal_modes: tuple[int, ...]
) -> DiscoveryResult:
    cfg = DiscoveryConfig()
    pieces = tuple(
        trajectory_evidence_modes(item, temporal_modes, cfg.spatial_modes)
        for item in trajectories
    )
    design = np.vstack([piece[0] for piece in pieces])
    target = np.concatenate([piece[1] for piece in pieces])
    train = np.concatenate([piece[2] for piece in pieces])
    fits = enumerate_fits(design, target, train, cfg)
    if len(fits) < 2:
        return DiscoveryResult("FAILED", None, float("inf"), 0.0, (), fits)
    best = fits[0]
    coefficient_ok = all(abs(value) >= cfg.coefficient_floor for value in best.coefficients)
    best_terms = set(best.support)
    competitors = tuple(item for item in fits[1:] if not best_terms < set(item.support))
    if not competitors:
        return DiscoveryResult("FAILED", None, best.validation_residual, 0.0, (), fits)
    gap = float(competitors[0].score - best.score)
    ambiguity = tuple(
        item.support
        for item in (best, *competitors)
        if item.score <= best.score + cfg.score_gap_min
    )
    if best.validation_residual > cfg.fit_residual_max:
        status, support = "OUT_OF_GRAMMAR", None
    elif not coefficient_ok or gap < cfg.score_gap_min or best.condition > cfg.condition_max:
        status, support = "UNIDENTIFIED", None
    else:
        status, support = "IDENTIFIED", best.support
    return DiscoveryResult(status, support, best.validation_residual, gap, ambiguity, fits)


def _public(result: DiscoveryResult, bank: str, nodes: int) -> dict:
    endpoint = record(result)
    return {"bank": bank, "nodes": nodes, **endpoint}


def _key(endpoint: dict) -> tuple[str, tuple[str, ...]]:
    return endpoint["status"], tuple(endpoint.get("support") or ())


def _terminal(endpoint: dict) -> bool:
    return endpoint["status"] in {"IDENTIFIED", "OUT_OF_GRAMMAR"}


def _abstention(trace: list[dict]) -> dict:
    return {
        "bank": "controller",
        "nodes": 49,
        "status": "UNIDENTIFIED",
        "support": None,
        "best_residual": None,
        "score_gap": None,
        "best_condition": None,
        "ambiguity_set": [],
        "trace": trace,
    }


def agreement_gate(
    cells: dict[tuple[int, str], dict], left_bank: str, right_bank: str
) -> dict:
    """Truth-blind demotion/refinement gate over two named public banks."""

    trace: list[dict] = []
    for nodes in NODES:
        left = cells[(nodes, left_bank)]
        right = cells[(nodes, right_bank)]
        trace.append({"nodes": nodes, "left": left, "right": right})
        if _terminal(left) and _key(left) == _key(right):
            return {**right, "trace": trace}
    return _abstention(trace)


def _score(endpoint: dict, truth_support: tuple[str, ...] | None) -> dict:
    status = endpoint["status"]
    support = tuple(endpoint.get("support") or ())
    correct = (
        truth_support is not None and status == "IDENTIFIED" and support == truth_support
    ) or (truth_support is None and status == "OUT_OF_GRAMMAR")
    false = status in {"IDENTIFIED", "OUT_OF_GRAMMAR"} and not correct
    return {
        "status": status,
        "support": endpoint.get("support"),
        "nodes": int(endpoint["nodes"]),
        "correct": bool(correct),
        "false_terminal": bool(false),
        "unidentified": status == "UNIDENTIFIED",
    }


def _episode(
    config: CampaignConfig,
    family: Family,
    seed: int,
    noise_model: str,
    noise: float,
) -> dict:
    baseline, channels, truth = matched_episode(
        family, seed, config.public_seed, noise, noise_model
    )
    cells: dict[tuple[int, str], dict] = {}
    mode_sets = {
        "low": LOW_MODES,
        "cumulative": CUMULATIVE_MODES,
        "high": HIGH_MODES,
    }
    for nodes in NODES:
        for bank, indices in mode_sets.items():
            result = gate(tuple(SUPPORTS), _discover_modes((baseline, channels[nodes]), indices))
            cells[(nodes, bank)] = _public(result, bank, nodes)

    nested = agreement_gate(cells, "low", "cumulative")
    disjoint = agreement_gate(cells, "low", "high")
    fixed_fine = cells[(49, "cumulative")]
    return {
        "episode_key": [family.value, seed, noise_model, noise],
        "truth_support": None if truth.support is None else list(truth.support),
        "policies": {
            "nested_gate": _score(nested, truth.support),
            "disjoint_gate": _score(disjoint, truth.support),
            "fixed_fine_cumulative": _score(fixed_fine, truth.support),
        },
    }


def _summary(episodes: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for policy in ("nested_gate", "disjoint_gate", "fixed_fine_cumulative"):
        rows = [episode["policies"][policy] for episode in episodes]
        output[policy] = {
            "factorial_conditions": len(rows),
            "hidden_trajectory_realizations": len(
                {(episode["episode_key"][0], episode["episode_key"][1]) for episode in episodes}
            ),
            "correct": sum(row["correct"] for row in rows),
            "false_terminals": sum(row["false_terminal"] for row in rows),
            "unidentified": sum(row["unidentified"] for row in rows),
            "mean_terminal_nodes": sum(row["nodes"] for row in rows) / len(rows),
            "stop_nodes": {
                str(nodes): sum(row["nodes"] == nodes for row in rows) for nodes in NODES
            },
        }
    return output


def _transitions(episodes: list[dict]) -> dict:
    labels: dict[str, int] = {}
    for episode in episodes:
        nested = episode["policies"]["nested_gate"]
        disjoint = episode["policies"]["disjoint_gate"]
        left = "correct" if nested["correct"] else "false" if nested["false_terminal"] else "unidentified"
        right = "correct" if disjoint["correct"] else "false" if disjoint["false_terminal"] else "unidentified"
        labels[f"{left}_to_{right}"] = labels.get(f"{left}_to_{right}", 0) + 1
    return dict(sorted(labels.items()))


def _family_summary(episodes: list[dict]) -> dict:
    output: dict[str, dict] = {}
    families = sorted({episode["episode_key"][0] for episode in episodes})
    for family in families:
        group = [episode for episode in episodes if episode["episode_key"][0] == family]
        for policy in ("nested_gate", "disjoint_gate"):
            rows = [episode["policies"][policy] for episode in group]
            output[f"{family}|{policy}"] = {
                "factorial_conditions": len(rows),
                "correct": sum(row["correct"] for row in rows),
                "false_terminals": sum(row["false_terminal"] for row in rows),
            }
    return output


def _gates(summary: dict, family_summary: dict, episodes: list[dict]) -> dict:
    nested = summary["nested_gate"]
    disjoint = summary["disjoint_gate"]
    count = disjoint["factorial_conditions"]
    new_false = sum(
        episode["policies"]["disjoint_gate"]["false_terminal"]
        and not episode["policies"]["nested_gate"]["false_terminal"]
        for episode in episodes
    )
    family_losses = {}
    for family in sorted({key.split("|")[0] for key in family_summary}):
        base = family_summary[f"{family}|nested_gate"]
        test = family_summary[f"{family}|disjoint_gate"]
        family_losses[family] = test["correct"] / test["factorial_conditions"] - base["correct"] / base["factorial_conditions"]
    checks = {
        "false_rate_le_1pct": disjoint["false_terminals"] / count <= 0.01,
        "correct_within_5pp_of_nested": disjoint["correct"] / count >= nested["correct"] / count - 0.05,
        "mean_nodes_below_30": disjoint["mean_terminal_nodes"] < 30.0,
        "new_false_terminals_le_5": new_false <= 5,
        "all_family_correct_losses_ge_minus_10pp": min(family_losses.values()) >= -0.10,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "new_false_terminals": new_false,
        "family_correct_rate_deltas": family_losses,
    }


def run(config: CampaignConfig) -> dict:
    if config.split != "confirmation" or tuple(config.seeds) != CONFIRMATION_SEEDS:
        raise ValueError("reserved seed/split contract violation")
    episodes: list[dict] = []
    for family_name in config.families:
        family = Family(family_name)
        for seed in config.seeds:
            for noise_model in config.noise_models:
                for noise in config.relative_noises:
                    episodes.append(_episode(config, family, seed, noise_model, noise))
    summary = _summary(episodes)
    family_summary = _family_summary(episodes)
    payload = {
        "status": "FROZEN_RESERVED_SEED_CONFIRMATION",
        "design_freeze_sha256": design_sha(),
        "source_closure_sha256": source_sha(),
        "config": asdict(config),
        "denominator_contract": {
            "hidden_trajectory_realizations": 120,
            "factorial_observation_conditions": 720,
            "factorial_conditions_are_not_claimed_iid": True,
        },
        "bank_definitions": {
            "low": list(LOW_MODES),
            "cumulative": list(CUMULATIVE_MODES),
            "high": list(HIGH_MODES),
        },
        "summary": summary,
        "transitions": _transitions(episodes),
        "family_summary": family_summary,
        "gate_verdict": _gates(summary, family_summary, episodes),
        "episodes": episodes,
    }
    payload["content_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = run(CampaignConfig())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(json.dumps(payload["gate_verdict"], indent=2, sort_keys=True))
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()

