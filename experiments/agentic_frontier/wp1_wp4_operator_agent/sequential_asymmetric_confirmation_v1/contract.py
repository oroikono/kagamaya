"""Immutable split, source, and artifact contract for confirmation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..sequential_consensus_controller_v1.contract import (
    NOISE_MODELS,
    POLICIES,
    PRODUCERS,
    RELATIVE_NOISES,
    digest,
    file_sha256,
)


LANE = Path(__file__).resolve().parent
REPO = LANE.parents[3]
CONFIRMATION_SEEDS = tuple(range(10500, 10550))
CALIBRATION_VERDICT_PATH = (
    LANE.parent
    / "sequential_asymmetric_calibration_v1r2"
    / "results"
    / "FINAL_GATE_VERDICT.json"
)
FAMILIES = (
    "advection_diffusion",
    "viscous_burgers",
    "fisher_kpp",
    "allen_cahn",
    "kdv",
    "hyperdiffusion_oog",
)
EPISODES_PER_SHARD = len(FAMILIES) * len(NOISE_MODELS) * len(RELATIVE_NOISES)
FREEZE_PATH = LANE / "PREOUTCOME_FREEZE.json"


def design_sha256() -> str:
    return file_sha256(LANE / "DESIGN_FREEZE_PREEXECUTION.md")


def source_files() -> tuple[Path, ...]:
    root = LANE.parent
    local = tuple(
        LANE / name
        for name in (
            "__init__.py",
            "DESIGN_FREEZE_PREEXECUTION.md",
            "contract.py",
            "freeze.py",
            "campaign.py",
            "validate.py",
            "aggregate.py",
            "validate_numeric.py",
            "test_protocol.py",
        )
    )
    v2 = root / "sequential_asymmetric_controller_v2"
    v1 = root / "sequential_consensus_controller_v1"
    upstream = (
        v2 / "contract.py",
        v2 / "policy.py",
        v1 / "contract.py",
        v1 / "policy.py",
        v1 / "campaign.py",
        root / "adversarial_verifier_gaming_v1" / "campaign.py",
        root / "adversarial_verifier_gaming_v1" / "contract.py",
        root / "adversarial_verifier_gaming_v1" / "io_utils.py",
        root / "verifier_fidelity_agent_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "simulator.py",
        root / "unified_quadrature_bias_v1" / "simulator.py",
        root / "maintrack_suite_v41" / "simulator.py",
        root / "maintrack_suite_v41" / "weakform.py",
    )
    return local + upstream


def source_manifest() -> dict[str, Any]:
    missing = [str(path) for path in source_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source closure is incomplete: {missing}")
    records = [
        {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)}
        for path in sorted(source_files())
    ]
    body = {"files": records}
    return {**body, "source_closure_sha256": digest(body)}


def expected_freeze_body() -> dict[str, Any]:
    return {
        "schema_version": "verisigs.sequential_asymmetric.confirmation_freeze.v1",
        "status": "FROZEN_PREOUTCOME",
        "design_sha256": design_sha256(),
        "source_manifest": source_manifest(),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "calibration_verdict_path": str(CALIBRATION_VERDICT_PATH.relative_to(REPO)),
        "calibration_verdict_sha256": file_sha256(CALIBRATION_VERDICT_PATH),
        "producers": list(PRODUCERS),
        "confirmation_access_count": 0,
    }


def validate_freeze() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("pre-outcome freeze is absent")
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    body = dict(payload)
    receipt = body.pop("freeze_sha256", None)
    if body != expected_freeze_body() or receipt != digest(body):
        raise ValueError("pre-outcome freeze does not match current source/design")
    return payload


def validate_calibration_eligibility() -> dict[str, Any]:
    payload = json.loads(CALIBRATION_VERDICT_PATH.read_text(encoding="utf-8"))
    if payload.get("valid") is not True:
        raise ValueError("calibration verdict is not valid")
    if payload.get("status") != "CALIBRATION_PASS_FREEZE_CONFIRMATION_NEXT":
        raise ValueError("calibration did not authorize a confirmation freeze")
    if not all(row.get("all_passed") is True for row in payload.get("final_gates", {}).values()):
        raise ValueError("not every calibration gate passed")
    return payload


def shard_ordinal(producer: str, seed: int, within_shard: int) -> int:
    return PRODUCERS.index(producer) * 100_000_000 + seed * 1_000 + within_shard
