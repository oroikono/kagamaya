"""Frozen contracts and canonical helpers for A10--A12."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


LANE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = LANE_ROOT.parents[3]
DESIGN = LANE_ROOT / "DESIGN_FREEZE_PREOUTCOME.md"
FREEZE = LANE_ROOT / "PREOUTCOME_FREEZE.json"

A10_SEEDS = tuple(range(150000, 150010))
A11_SEEDS = tuple(range(151000, 151010))
A12_SEEDS = tuple(range(152000, 152010))
SEEDS = {"a10": A10_SEEDS, "a11": A11_SEEDS, "a12": A12_SEEDS}
STANDARD_NOISES = (0.0, 0.001, 0.005)
HIGH_NOISES = (0.01, 0.05, 0.10)
NOISE_MODELS = ("iid", "correlated")
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
NODES = (7, 13, 25, 49)
REFERENCE_NODES = (49, 97)
TEMPORAL_MODES = 5
PUBLIC_SEED = 8675309
BOOTSTRAP_SEED = 20260820
BOOTSTRAP_REPLICATES = 2000

SOURCE_FILES = (
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/DESIGN_FREEZE_PREOUTCOME.md",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/freeze.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/preflight.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/validate.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/test_controls.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/verifier_fidelity_agent_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/unified_quadrature_bias_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/numerics.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/weakform.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/unified_agent_v2_1_scaled/simulator.py",
)


@dataclass(frozen=True)
class RunContract:
    lane: str
    seeds: tuple[int, ...]

    def validate(self) -> None:
        if self.lane not in SEEDS:
            raise ValueError(f"unknown control lane {self.lane}")
        if tuple(self.seeds) != SEEDS[self.lane]:
            raise ValueError(f"{self.lane} must open exactly its reserved seed list")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"seeds": list(self.seeds)}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return rows


def freeze_body() -> dict[str, Any]:
    sources = source_manifest()
    return {
        "schema_version": "verisigs.evidence_view_controls.preoutcome_freeze.v1",
        "status": "PREOUTCOME_SOURCE_FROZEN_RESERVED_SEEDS_CLOSED",
        "design_sha256": file_sha256(DESIGN),
        "seed_registry": {key: list(value) for key, value in SEEDS.items()},
        "dependence_unit": "episode_seed_cluster",
        "source_files": sources,
        "source_closure_sha256": digest(sources),
    }


def load_and_validate_freeze() -> dict[str, Any]:
    if not FREEZE.is_file():
        raise FileNotFoundError("PREOUTCOME_FREEZE.json is required")
    payload = json.loads(FREEZE.read_text())
    claimed = payload.get("freeze_sha256")
    body = dict(payload)
    body.pop("freeze_sha256", None)
    if claimed != digest(body):
        raise ValueError("preoutcome freeze content digest mismatch")
    if body != freeze_body():
        raise ValueError("preoutcome source/design/seed closure drift")
    return payload


def publish_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite numerical result: {value!r}")
    return result


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]
