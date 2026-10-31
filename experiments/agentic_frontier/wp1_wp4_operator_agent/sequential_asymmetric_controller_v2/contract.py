"""Frozen source and split contract for v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..sequential_consensus_controller_v1.contract import (
    MODES,
    NODES,
    NOISE_MODELS,
    POLICIES,
    PRODUCERS,
    PUBLIC_SEED,
    RELATIVE_NOISES,
    TERMINAL_STATUSES,
    VIEW_BY_CONFIG,
    VIEW_BY_TOKEN,
    VIEWS,
    canonical_bytes,
    digest,
    file_sha256,
    is_terminal,
    terminal_signature,
)


LANE = Path(__file__).resolve().parent
REPO = LANE.parents[3]
DEVELOPMENT_SEEDS = (10410, 10411)
RESERVED_CALIBRATION_SEEDS = tuple(range(10450, 10470))
RESERVED_CONFIRMATION_SEEDS = tuple(range(10500, 10550))


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
            "policy.py",
            "campaign.py",
            "validate.py",
            "test_protocol.py",
        )
    )
    v1 = root / "sequential_consensus_controller_v1"
    upstream = (
        v1 / "contract.py",
        v1 / "policy.py",
        v1 / "campaign.py",
        root / "adversarial_verifier_gaming_v1" / "campaign.py",
        root / "adversarial_verifier_gaming_v1" / "contract.py",
        root / "verifier_fidelity_agent_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "campaign.py",
        root / "independent_fd_producer_v1" / "simulator.py",
        root / "unified_quadrature_bias_v1" / "simulator.py",
        root / "maintrack_suite_v41" / "simulator.py",
        root / "maintrack_suite_v41" / "weakform.py",
    )
    return local + upstream


def source_manifest() -> dict[str, Any]:
    files = source_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source closure is incomplete: {missing}")
    records = [
        {"path": str(path.relative_to(REPO)), "sha256": file_sha256(path)}
        for path in sorted(files)
    ]
    body = {"files": records}
    return {**body, "source_closure_sha256": digest(body)}

