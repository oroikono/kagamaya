"""Build fresh public/owner episodes, serialize proposals, and score them."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..campaign import _cell, _producer_episode, _public_row
from ..contract import (
    ACTIONS,
    NOISE_MODELS,
    PUBLIC_SEED,
    REFERENCE_ACTION,
    RELATIVE_NOISES,
    SCHEMA_PREFIX,
    digest,
)
from ..io_utils import write_json
from ..proposers import propose_non_llm
from ..public_interface import build_public_episode, validate_public_episode
from ..score import score
from ..consensus_mitigation_v1.screen import choose
from ...maintrack_suite_v41.simulator import Family


LANE = Path(__file__).resolve().parent
CONFIRMATION_SEEDS = tuple(range(10200, 10250))
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
POLICIES = ("all_mode_consensus", "confidence_max", "fixed_25_m5")
SOURCE_FILES = (
    LANE / "DESIGN_FREEZE_PREEXECUTION.md",
    LANE / "confirmation.py",
    LANE / "aggregate.py",
    LANE / "validate.py",
    LANE / "test_protocol.py",
    LANE / "euler_array.sbatch",
    LANE / "euler_aggregate.sbatch",
    LANE.parent / "campaign.py",
    LANE.parent / "contract.py",
    LANE.parent / "public_interface.py",
    LANE.parent / "score.py",
    LANE.parent / "proposers.py",
    LANE.parent / "consensus_mitigation_v1" / "screen.py",
    LANE.parent.parent / "verifier_fidelity_agent_v1" / "campaign.py",
    LANE.parent.parent / "independent_fd_producer_v1" / "campaign.py",
    LANE.parent.parent / "independent_fd_producer_v1" / "simulator.py",
    LANE.parent.parent / "unified_quadrature_bias_v1" / "simulator.py",
    LANE.parent.parent / "maintrack_suite_v41" / "weakform.py",
)


@dataclass(frozen=True)
class Config:
    producer: str
    seed: int
    public_seed: int = PUBLIC_SEED


def design_sha256() -> str:
    return hashlib.sha256((LANE / "DESIGN_FREEZE_PREEXECUTION.md").read_bytes()).hexdigest()


def source_sha256() -> str:
    missing = [str(path) for path in SOURCE_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source closure incomplete: {missing}")
    common = LANE.parent.parent.parent.parent.parent
    material = b"".join(
        str(path.relative_to(common)).encode() + b"\0" + path.read_bytes()
        for path in sorted(SOURCE_FILES)
    )
    return hashlib.sha256(material).hexdigest()


def _episode_id(producer: str, seed: int, family_index: int, noise_index: int) -> str:
    return f"confirm_v2_{producer}_{seed}_{family_index}_{noise_index}"


def _truth_record(truth_support) -> list[str] | None:
    return None if truth_support is None else list(truth_support)


def _proposal(episode: dict[str, Any], producer: str, policy: str) -> dict[str, Any]:
    if policy == "all_mode_consensus":
        token, consulted, evidence = choose(episode, policy)
        body: dict[str, Any] = {
            "schema_version": "verisigs.adversarial_verifier_gaming.consensus_v2.proposal",
            "episode_id": episode["episode_id"],
            "producer": producer,
            "public_episode_sha256": episode["episode_sha256"],
            "policy": policy,
            "proposal_valid": token is not None,
            "selected_action_token": token,
            "scientific_authority": False,
            "hidden_reference_accessed": False,
            "metadata": {
                "consensus_witness_nodes": consulted,
                "consensus_evidence_tokens": evidence,
            },
        }
    else:
        body = propose_non_llm(episode, policy)
        body.pop("proposal_sha256")
        body["producer"] = producer
    body["proposal_sha256"] = digest(body)
    return body


def build(config: Config) -> dict[str, Any]:
    if config.seed not in CONFIRMATION_SEEDS:
        raise ValueError("seed is outside the frozen fresh-confirmation split")
    if config.producer not in PRODUCERS:
        raise ValueError("unsupported producer")
    public_episodes: list[dict[str, Any]] = []
    owner_episodes: list[dict[str, Any]] = []
    for family_index, family in enumerate(Family):
        for noise_model_index, noise_model in enumerate(NOISE_MODELS):
            for noise_level_index, noise in enumerate(RELATIVE_NOISES):
                noise_index = noise_model_index * len(RELATIVE_NOISES) + noise_level_index
                episode_id = _episode_id(config.producer, config.seed, family_index, noise_index)
                channels, baseline, truth = _producer_episode(
                    config.producer, family, config.seed, noise_model, noise, config.public_seed
                )
                rows = []
                for action in ACTIONS:
                    cell = _cell(
                        config.producer,
                        baseline,
                        channels[action.nodes],
                        channels[49],
                        truth.support,
                        action.temporal_modes,
                    )
                    rows.append(_public_row(action, cell, baseline, channels[action.nodes]))
                public = build_public_episode(
                    episode_id=episode_id,
                    producer=config.producer,
                    episode_metadata={"noise_model": noise_model, "relative_noise": noise},
                    rows=rows,
                )
                validate_public_episode(public)
                reference = _cell(
                    config.producer,
                    baseline,
                    channels[REFERENCE_ACTION[0]],
                    channels[REFERENCE_ACTION[0]],
                    truth.support,
                    REFERENCE_ACTION[1],
                )
                public_episodes.append(public)
                owner_episodes.append(
                    {
                        "episode_id": episode_id,
                        "owner_episode_key": [family.value, config.seed, noise_model, noise],
                        "truth_support": _truth_record(truth.support),
                        "reference_action": list(REFERENCE_ACTION),
                        "reference_final": reference["final"],
                        "reference_correct": bool(reference["correct"]),
                        "reference_false_promotion": bool(reference["false_promotion"]),
                    }
                )
    config_row = asdict(config)
    source_sha = source_sha256()
    public_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.consensus_v2.public_campaign",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha,
        "config": config_row,
        "episode_count": len(public_episodes),
        "episodes": public_episodes,
    }
    public_payload["campaign_sha256"] = digest(public_payload)
    owner_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.consensus_v2.owner_campaign",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha,
        "public_campaign_sha256": public_payload["campaign_sha256"],
        "config": config_row,
        "episode_count": len(owner_episodes),
        "episodes": owner_episodes,
    }
    owner_payload["campaign_sha256"] = digest(owner_payload)
    proposals = [
        _proposal(episode, config.producer, policy)
        for episode in public_episodes
        for policy in POLICIES
    ]
    proposal_payload: dict[str, Any] = {
        "schema_version": f"{SCHEMA_PREFIX}.consensus_v2.proposal_campaign",
        "public_campaign_sha256": public_payload["campaign_sha256"],
        "config": {"producer": config.producer, "seed": config.seed, "policies": list(POLICIES)},
        "proposals": proposals,
    }
    proposal_payload["campaign_sha256"] = digest(proposal_payload)
    scored = score(public_payload, owner_payload, proposal_payload)
    payload: dict[str, Any] = {
        "schema_version": "verisigs.adversarial_verifier_gaming.consensus_v2.shard",
        "design_sha256": design_sha256(),
        "source_closure_sha256": source_sha,
        "config": config_row,
        "public": public_payload,
        "owner": owner_payload,
        "proposals": proposal_payload,
        "scored": scored,
    }
    payload["content_sha256"] = digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", choices=PRODUCERS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(Config(args.producer, args.seed))
    write_json(args.output, payload)
    print(payload["content_sha256"])


if __name__ == "__main__":
    main()
