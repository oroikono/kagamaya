"""Immutable contract and content-addressed helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


LANE = Path(__file__).resolve().parent
REPO = LANE.parents[3]
DESIGN = LANE / "DESIGN_FREEZE_PREOUTCOME.md"
FREEZE = LANE / "PREOUTCOME_FREEZE.json"
SEEDS = tuple(range(153000, 153020))
NOISES = (0.01, 0.05, 0.10)
NOISE_MODELS = ("iid", "correlated")
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
POLICIES = ("reference_49_m5", "fixed_25_m5", "confidence_max", "all_mode_consensus")
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_REPLICATES = 2000

SOURCE_FILES = (
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/__init__.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/DESIGN_FREEZE_PREOUTCOME.md",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/freeze.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/validate.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/high_noise_attack_margin_v1/test_protocol.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/evidence_view_controls_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/proposers.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/public_interface.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/consensus_mitigation_v1/screen.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/numerics.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/weakform.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/unified_quadrature_bias_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/unified_agent_v2_1_scaled/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/verifier_fidelity_agent_v1/campaign.py",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> list[dict[str, Any]]:
    rows = []
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    return rows


def freeze_body() -> dict[str, Any]:
    sources = source_manifest()
    return {
        "schema_version": "verisigs.high_noise_attack_margin.preoutcome.v1",
        "status": "SOURCE_FROZEN_OUTCOMES_CLOSED",
        "design_sha256": file_sha(DESIGN),
        "seeds": list(SEEDS),
        "noise_levels": list(NOISES),
        "noise_models": list(NOISE_MODELS),
        "producers": list(PRODUCERS),
        "policies": list(POLICIES),
        "dependence_unit": "seed_cluster",
        "source_files": sources,
        "source_closure_sha256": digest(sources),
    }


def load_freeze() -> dict[str, Any]:
    payload = json.loads(FREEZE.read_text())
    claimed = payload.get("freeze_sha256")
    body = dict(payload)
    body.pop("freeze_sha256", None)
    if claimed != digest(body):
        raise ValueError("freeze digest mismatch")
    if body != freeze_body():
        raise ValueError("source/design drift after freeze")
    return payload


def publish_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
        raise ValueError("nonfinite result")
    return result


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(x) for x in values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return (1.0 - weight) * ordered[low] + weight * ordered[high]
