#!/usr/bin/env python3
"""Build the truth-free node-49 acquisition bank for Gate 3."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LANE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LANE_ROOT.parent
SRC = LANE_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC))

from evidence_view_benchmark.canonical import (  # noqa: E402
    digest,
    file_sha256,
    publish_json,
    publish_jsonl,
    read_jsonl,
)
from evidence_view_benchmark.migration import load_migration  # noqa: E402
from experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1.campaign import (  # noqa: E402
    _cell,
    _producer_episode,
    _public_row,
)
from experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1.contract import (  # noqa: E402
    Action as SourceAction,
)
from experiments.agentic_frontier.wp1_wp4_operator_agent.maintrack_suite_v41.simulator import (  # noqa: E402
    Family,
)


PRODUCERS = ("spectral_etdrk4", "fd_bdf")
PUBLIC_NODES = (7, 13, 25)
MODES = (2, 3, 4, 5)
PUBLIC_SEED = 8675309
NUMERIC_TOLERANCE = 1.0e-7
DERIVED_TOLERANCE = 2.0e-5
FORBIDDEN = (
    "truth",
    "held_out",
    "correct",
    "false",
    "owner",
    "oracle",
    "reference",
    "error_to_49",
)
SOURCE_FILES = (
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/adversarial_verifier_gaming_v1/contract.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/verifier_fidelity_agent_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/campaign.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/independent_fd_producer_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/unified_quadrature_bias_v1/simulator.py",
    "experiments/agentic_frontier/wp1_wp4_operator_agent/maintrack_suite_v41/weakform.py",
)


def _assert_public(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN):
                raise ValueError(f"forbidden public key at {location}.{key}")
            _assert_public(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_public(nested, f"{location}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN):
            raise ValueError(f"forbidden public string at {location}")


@dataclass(frozen=True)
class Comparison:
    numeric_values: int = 0
    max_absolute_difference: float = 0.0
    max_scaled_difference: float = 0.0

    def combine(self, other: "Comparison") -> "Comparison":
        return Comparison(
            self.numeric_values + other.numeric_values,
            max(self.max_absolute_difference, other.max_absolute_difference),
            max(self.max_scaled_difference, other.max_scaled_difference),
        )


def _compare(expected: Any, actual: Any, location: str) -> Comparison:
    """Compare a regenerated row to Gate 1 and summarize numeric drift."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise ValueError(f"key mismatch at {location}")
        result = Comparison()
        for key in expected:
            result = result.combine(_compare(expected[key], actual[key], f"{location}.{key}"))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"list length mismatch at {location}")
        result = Comparison()
        for index, (left, right) in enumerate(zip(expected, actual)):
            result = result.combine(_compare(left, right, f"{location}[{index}]"))
        return result
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        left, right = float(expected), float(actual)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"nonfinite value at {location}")
        scale = max(1.0, abs(left), abs(right))
        absolute = abs(left - right)
        scaled = absolute / scale
        tolerance = (
            DERIVED_TOLERANCE
            if location.endswith((".gate_margin", ".public_objective"))
            else NUMERIC_TOLERANCE
        )
        if scaled > tolerance:
            raise ValueError(f"numeric mismatch at {location}: {left} != {right}")
        return Comparison(1, absolute, scaled)
    if expected != actual:
        raise ValueError(f"discrete mismatch at {location}: {expected!r} != {actual!r}")
    return Comparison()


def _source_manifest() -> list[dict[str, Any]]:
    rows = []
    for relative in SOURCE_FILES:
        source = REPO_ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append(
            {
                "path": relative,
                "bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        )
    return rows


def _paired_row(nodes: int, modes: int, producer_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "view_token": f"N{nodes}M{modes}",
        "nodes": nodes,
        "temporal_modes": modes,
    }
    for producer in PRODUCERS:
        public = dict(producer_rows[producer])
        public.pop("action_token")
        public.pop("nodes")
        public.pop("temporal_modes")
        row[producer] = public
    return row


def build() -> dict[str, Any]:
    cases = read_jsonl(LANE_ROOT / "cases/cases.jsonl")
    migration = load_migration(LANE_ROOT / "schema/token_migration.json")
    families = tuple(Family)
    bank_rows: list[dict[str, Any]] = []
    discrete_replays = 0
    replay_comparison = Comparison()
    generated_views = 0

    for case in cases:
        family_index = int(str(case["family"])[1:])
        if family_index < 0 or family_index >= len(families):
            raise ValueError(f"invalid opaque family id for {case['case_id']}")
        family = families[family_index]
        generated_by_producer: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
        for producer in PRODUCERS:
            channels, baseline, _unused_truth = _producer_episode(
                producer,
                family,
                int(case["seed"]),
                str(case["noise_model"]),
                float(case["noise_level"]),
                PUBLIC_SEED,
            )
            regenerated: dict[tuple[int, int], dict[str, Any]] = {}
            for nodes in (*PUBLIC_NODES, 49):
                for modes in MODES:
                    action = SourceAction(f"N{nodes}M{modes}", nodes, modes)
                    # Truth is deliberately absent. It affects only owner scoring fields,
                    # which _public_row never serializes.
                    cell = _cell(
                        producer,
                        baseline,
                        channels[nodes],
                        channels[49],
                        None,
                        modes,
                    )
                    regenerated[(nodes, modes)] = migration.migrate(
                        _public_row(action, cell, baseline, channels[nodes])
                    )
            generated_by_producer[producer] = regenerated

        existing = {
            (int(row["nodes"]), int(row["temporal_modes"])): row
            for row in case["public_rows"]
        }
        if set(existing) != {(n, m) for n in PUBLIC_NODES for m in MODES}:
            raise ValueError(f"incomplete Gate-1 row grid for {case['case_id']}")
        for coordinate, paired in existing.items():
            for producer in PRODUCERS:
                expected = {
                    "action_token": paired["action_token"],
                    "nodes": paired["nodes"],
                    "temporal_modes": paired["temporal_modes"],
                    **paired[producer],
                }
                actual = dict(generated_by_producer[producer][coordinate])
                # Gate 1 uses A00--A11 while the regeneration uses coordinate tokens.
                actual["action_token"] = expected["action_token"]
                replay_comparison = replay_comparison.combine(
                    _compare(
                        expected,
                        actual,
                        f"{case['case_id']}.{producer}.{coordinate}",
                    )
                )
                discrete_replays += 1

        views = []
        for modes in MODES:
            views.append(
                _paired_row(
                    49,
                    modes,
                    {
                        producer: generated_by_producer[producer][(49, modes)]
                        for producer in PRODUCERS
                    },
                )
            )
            generated_views += 1
        body: dict[str, Any] = {
            "schema_version": "evidence_view_benchmark.acquisition_49_case.v1",
            "case_id": case["case_id"],
            "public_case_sha256": case["case_sha256"],
            "views": views,
            "source_bindings": case["source_bindings"],
        }
        _assert_public(body)
        body["acquisition_case_sha256"] = digest(body)
        bank_rows.append(body)

    bank_path = LANE_ROOT / "cases/acquisition_49_rows.jsonl"
    publish_jsonl(bank_path, bank_rows)
    source_rows = _source_manifest()
    manifest = {
        "schema_version": "evidence_view_benchmark.acquisition_49_manifest.v1",
        "case_count": len(bank_rows),
        "views_per_case": 4,
        "generated_views": generated_views,
        "producer_count": len(PRODUCERS),
        "gate1_producer_rows_replayed": discrete_replays,
        "numeric_values_replayed": replay_comparison.numeric_values,
        "max_absolute_replay_difference": replay_comparison.max_absolute_difference,
        "max_scaled_replay_difference": replay_comparison.max_scaled_difference,
        "numeric_relative_tolerance": NUMERIC_TOLERANCE,
        "derived_margin_objective_tolerance": DERIVED_TOLERANCE,
        "cases_file_sha256": file_sha256(LANE_ROOT / "cases/cases.jsonl"),
        "acquisition_bank_file_sha256": file_sha256(bank_path),
        "migration_schema_sha256": file_sha256(LANE_ROOT / "schema/token_migration.json"),
        "source_files": source_rows,
        "source_closure_sha256": digest(source_rows),
        "generation_truth_support_argument": None,
    }
    manifest["manifest_sha256"] = digest(manifest)
    publish_json(LANE_ROOT / "cases/acquisition_bank_manifest.json", manifest)
    return manifest


def main() -> None:
    result = build()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
