"""Deterministic protocol tests for the Sim2Science controls."""

from __future__ import annotations

import numpy as np

from .denested_bank import (
    CUMULATIVE_MODES,
    LOW_MODES,
    _discover_modes,
    agreement_gate,
    trajectory_evidence_modes,
)
from ..maintrack_suite_v41.simulator import Family
from ..maintrack_suite_v41.weakform import DiscoveryConfig, discover, trajectory_evidence
from ..unified_quadrature_bias_v1.simulator import matched_episode


def _endpoint(status: str, support=None, nodes: int = 7) -> dict:
    return {"status": status, "support": support, "nodes": nodes}


def test_explicit_modes_reproduce_original_low_and_cumulative_banks() -> None:
    baseline, channels, _ = matched_episode(Family.ADVECTION_DIFFUSION, 61999, 8675309, 0.001, "iid")
    for trajectory in (baseline, channels[13]):
        for indices in (LOW_MODES, CUMULATIVE_MODES):
            expected = trajectory_evidence(trajectory, len(indices), DiscoveryConfig().spatial_modes)
            actual = trajectory_evidence_modes(trajectory, indices, DiscoveryConfig().spatial_modes)
            for left, right in zip(expected, actual):
                assert np.array_equal(left, right)


def test_explicit_discover_reproduces_original_cumulative_decision() -> None:
    baseline, channels, _ = matched_episode(Family.VISCOUS_BURGERS, 61999, 8675309, 0.001, "correlated")
    expected = discover((baseline, channels[25]), DiscoveryConfig(temporal_modes=5))
    actual = _discover_modes((baseline, channels[25]), CUMULATIVE_MODES)
    assert expected.status == actual.status
    assert expected.support == actual.support
    assert expected.best_residual == actual.best_residual
    assert expected.score_gap == actual.score_gap


def test_agreement_gate_is_demotion_only() -> None:
    cells = {}
    for nodes in (7, 13, 25, 49):
        cells[(nodes, "left")] = _endpoint("IDENTIFIED", ["u_x"], nodes)
        cells[(nodes, "right")] = _endpoint("UNIDENTIFIED", None, nodes)
    endpoint = agreement_gate(cells, "left", "right")
    assert endpoint["status"] == "UNIDENTIFIED"


def test_agreement_gate_stops_on_first_exact_terminal_match() -> None:
    cells = {}
    for nodes in (7, 13, 25, 49):
        cells[(nodes, "left")] = _endpoint("UNIDENTIFIED", None, nodes)
        cells[(nodes, "right")] = _endpoint("UNIDENTIFIED", None, nodes)
    cells[(13, "left")] = _endpoint("IDENTIFIED", ["u_x"], 13)
    cells[(13, "right")] = _endpoint("IDENTIFIED", ["u_x"], 13)
    endpoint = agreement_gate(cells, "left", "right")
    assert endpoint["status"] == "IDENTIFIED"
    assert endpoint["nodes"] == 13
