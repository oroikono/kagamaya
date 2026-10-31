"""Immutable contract for the sequential consensus pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


LANE = Path(__file__).resolve().parent
REPO = LANE.parents[3]
NODES = (7, 13, 25, 49)
MODES = (2, 3, 4, 5)
PRODUCERS = ("spectral_etdrk4", "fd_bdf")
NOISE_MODELS = ("iid", "correlated")
RELATIVE_NOISES = (0.0, 0.001, 0.005)
DEVELOPMENT_SEEDS = (10400, 10401)
RESERVED_CALIBRATION_SEEDS = tuple(range(10450, 10470))
RESERVED_CONFIRMATION_SEEDS = tuple(range(10500, 10550))
PUBLIC_SEED = 8675309
TERMINAL_STATUSES = frozenset({"IDENTIFIED", "OUT_OF_GRAMMAR"})
POLICIES = (
    "sequential_all_mode",
    "fullbank_all_mode",
    "fixed_25_m5",
    "fixed_49_m5",
    "confidence_max",
    "d_optimal",
    "e_optimal",
)


@dataclass(frozen=True)
class View:
    token: str
    nodes: int
    temporal_modes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VIEWS = tuple(
    View(f"V{index:02d}", nodes, modes)
    for index, (nodes, modes) in enumerate(
        (node, mode) for node in NODES for mode in MODES
    )
)
VIEW_BY_TOKEN = {view.token: view for view in VIEWS}
VIEW_BY_CONFIG = {(view.nodes, view.temporal_modes): view for view in VIEWS}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design_sha256() -> str:
    return file_sha256(LANE / "DESIGN_FREEZE_PREEXECUTION.md")


def terminal_signature(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    return (
        str(row["terminal_status"]),
        tuple(str(item) for item in (row.get("terminal_support") or ())),
    )


def is_terminal(row: Mapping[str, Any]) -> bool:
    return str(row["terminal_status"]) in TERMINAL_STATUSES


def source_files() -> tuple[Path, ...]:
    root = LANE.parent
    local = tuple(
        LANE / name
        for name in (
            "__init__.py",
            "MAINTRACK_BUILD_PLAN.md",
            "DESIGN_FREEZE_PREEXECUTION.md",
            "contract.py",
            "policy.py",
            "campaign.py",
            "validate.py",
            "test_protocol.py",
        )
    )
    upstream = (
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
        {
            "path": str(path.relative_to(REPO)),
            "sha256": file_sha256(path),
        }
        for path in sorted(files)
    ]
    body = {"files": records}
    return {**body, "source_closure_sha256": digest(body)}

