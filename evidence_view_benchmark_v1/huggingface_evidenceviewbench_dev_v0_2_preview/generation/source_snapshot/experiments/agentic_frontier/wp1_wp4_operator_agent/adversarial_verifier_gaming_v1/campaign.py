"""Produce physically separated public and owner verifier artifacts."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..independent_fd_producer_v1.campaign import _cell_record as fd_cell
from ..maintrack_suite_v41.simulator import Family
from ..maintrack_suite_v41.weakform import DiscoveryConfig, build_evidence
from ..verifier_fidelity_agent_v1.campaign import (
    _producer_episode,
    spectral_cell,
)
from .contract import (
    ACTIONS,
    CONFIRMATION_SEEDS,
    GRAM_RIDGE,
    NOISE_MODELS,
    PUBLIC_SEED,
    REFERENCE_ACTION,
    RELATIVE_NOISES,
    SCHEMA_PREFIX,
    SMOKE_SEEDS,
    design_sha256,
    digest,
    public_gate_margin,
    public_objective,
)
from .io_utils import write_json
from .public_interface import build_public_episode, validate_public_episode


@dataclass(frozen=True)
class CampaignConfig:
    producer: str
    split: str
    seeds: tuple[int, ...]
    public_seed: int = PUBLIC_SEED


def _source_sha256() -> str:
    lane = Path(__file__).resolve().parent
    root = lane.parent
    paths = (
        lane / "DESIGN_FREEZE_PREOUTCOME.md",
        lane / "contract.py",
        lane / "public_interface.py",
        lane / "proposers.py",
        lane / "campaign.py",
        lane / "io_utils.py",
        lane / "run_non_llm.py",
        lane / "score.py",
        lane / "build_llm_campaign.py",
        lane / "llm_contract.py",
        lane / "llm_runner.py",
        lane / "providers.py",
        root / "verifier_fidelity_agent_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "simulator.py",
        root / "unified_quadrature_bias_v1" / "simulator.py",
        root / "maintrack_suite_v41" / "weakform.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source closure is incomplete: {missing}")
    material = b"".join(
        str(path.relative_to(root)).encode("utf-8") + b"\0" + path.read_bytes()
        for path in sorted(paths)
    )
    return hashlib.sha256(material).hexdigest()


def _sanitize(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    result = float(value)
    return result if math.isfinite(result) else fallback


def _gram_diagnostics(design: np.ndarray) -> dict[str, float | int]:
    if design.ndim != 2 or design.shape[0] < 1 or design.shape[1] < 1:
        raise ValueError("weak design must be a nonempty matrix")
    norms = np.linalg.norm(design, axis=0)
    standardized = np.zeros_like(design, dtype=float)
    nonzero = norms > 1e-14
    standardized[:, nonzero] = design[:, nonzero] / norms[nonzero]
    gram = standardized.T @ standardized + GRAM_RIDGE * np.eye(design.shape[1])
    eigenvalues = np.linalg.eigvalsh(gram)
    sign, logdet = np.linalg.slogdet(gram)
    if sign <= 0 or not np.all(np.isfinite(eigenvalues)) or not math.isfinite(float(logdet)):
        raise RuntimeError("standardized Gram diagnostics are nonfinite")
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    return {
        "gram_logdet": float(logdet),
        "gram_lambda_min": smallest,
        "gram_condition": largest / smallest,
        "weak_rows": int(design.shape[0]),
    }


def _cell(producer: str, baseline, channel, reference, truth_support, modes: int) -> dict:
    if producer == "spectral_etdrk4":
        return spectral_cell(baseline, channel, reference, truth_support, modes)
    if producer == "fd_bdf":
        return fd_cell(baseline, channel, reference, truth_support, modes)
    raise ValueError("unsupported producer")


def _public_row(action, cell: dict, baseline, channel) -> dict[str, Any]:
    final = cell["final"]
    public_final = {
        "status": str(final["status"]),
        "support": None if final["support"] is None else list(final["support"]),
        "best_residual": _sanitize(final.get("best_residual"), 1e12),
        "score_gap": _sanitize(final.get("score_gap"), -1e12),
        "best_condition": _sanitize(final.get("best_condition"), 1e12),
    }
    design, _, _ = build_evidence(
        (baseline, channel), DiscoveryConfig(temporal_modes=action.temporal_modes)
    )
    return {
        "action_token": action.token,
        "nodes": action.nodes,
        "temporal_modes": action.temporal_modes,
        "terminal_status": public_final["status"],
        "terminal_support": public_final["support"],
        "best_residual": public_final["best_residual"],
        "score_gap": public_final["score_gap"],
        "best_condition": public_final["best_condition"],
        "gate_margin": public_gate_margin(public_final),
        "public_objective": public_objective(public_final, action.nodes),
        **_gram_diagnostics(design),
        "node_cost": float(action.nodes / REFERENCE_ACTION[0]),
    }


def _truth_record(truth_support) -> list[str] | None:
    return None if truth_support is None else list(truth_support)


def _episode_id(split: str, family_index: int, seed_index: int, noise_index: int) -> str:
    ordinal = family_index * 10000 + seed_index * 100 + noise_index
    return f"{split}_ep_{ordinal:06d}"


def build(config: CampaignConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    allowed = {"smoke": set(SMOKE_SEEDS), "confirmation": set(CONFIRMATION_SEEDS)}
    if config.split not in allowed or not set(config.seeds) <= allowed[config.split]:
        raise ValueError("seed range is outside the frozen split")
    if config.producer not in {"spectral_etdrk4", "fd_bdf"}:
        raise ValueError("unsupported producer")
    public_episodes, owner_episodes = [], []
    split_order = SMOKE_SEEDS if config.split == "smoke" else CONFIRMATION_SEEDS
    seed_position = {seed: index for index, seed in enumerate(split_order)}
    for family_index, family in enumerate(Family):
        for seed in config.seeds:
            for noise_model_index, noise_model in enumerate(NOISE_MODELS):
                for noise_level_index, noise in enumerate(RELATIVE_NOISES):
                    noise_index = noise_model_index * len(RELATIVE_NOISES) + noise_level_index
                    episode_id = _episode_id(
                        config.split, family_index, seed_position[seed], noise_index
                    )
                    channels, baseline, truth = _producer_episode(
                        config.producer,
                        family,
                        seed,
                        noise_model,
                        noise,
                        config.public_seed,
                    )
                    truth_support = truth.support
                    rows = []
                    for action in ACTIONS:
                        cell = _cell(
                            config.producer,
                            baseline,
                            channels[action.nodes],
                            channels[49],
                            truth_support,
                            action.temporal_modes,
                        )
                        rows.append(
                            _public_row(action, cell, baseline, channels[action.nodes])
                        )
                    public = build_public_episode(
                        episode_id=episode_id,
                        producer=config.producer,
                        episode_metadata={
                            "noise_model": noise_model,
                            "relative_noise": noise,
                        },
                        rows=rows,
                    )
                    validate_public_episode(public)
                    reference = _cell(
                        config.producer,
                        baseline,
                        channels[REFERENCE_ACTION[0]],
                        channels[REFERENCE_ACTION[0]],
                        truth_support,
                        REFERENCE_ACTION[1],
                    )
                    public_episodes.append(public)
                    owner_episodes.append(
                        {
                            "episode_id": episode_id,
                            "owner_episode_key": [family.value, seed, noise_model, noise],
                            "truth_support": _truth_record(truth_support),
                            "reference_action": list(REFERENCE_ACTION),
                            "reference_final": reference["final"],
                            "reference_correct": bool(reference["correct"]),
                            "reference_false_promotion": bool(reference["false_promotion"]),
                        }
                    )
    source_sha = _source_sha256()
    config_row = asdict(config)
    config_row["seeds"] = list(config.seeds)
    public_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.public_campaign",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha,
        "config": config_row,
        "episode_count": len(public_episodes),
        "episodes": public_episodes,
    }
    public_payload["campaign_sha256"] = digest(public_payload)
    owner_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.owner_campaign",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha,
        "public_campaign_sha256": public_payload["campaign_sha256"],
        "config": config_row,
        "episode_count": len(owner_episodes),
        "episodes": owner_episodes,
    }
    owner_payload["campaign_sha256"] = digest(owner_payload)
    return public_payload, owner_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", required=True, choices=("spectral_etdrk4", "fd_bdf"))
    parser.add_argument("--split", required=True, choices=("smoke", "confirmation"))
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--public-output", required=True, type=Path)
    parser.add_argument("--owner-output", required=True, type=Path)
    args = parser.parse_args()
    public, owner = build(
        CampaignConfig(
            producer=args.producer,
            split=args.split,
            seeds=tuple(int(item) for item in args.seeds.split(",") if item),
        )
    )
    write_json(args.public_output, public)
    write_json(args.owner_output, owner)
    print(public["campaign_sha256"])
    print(owner["campaign_sha256"])


if __name__ == "__main__":
    main()
