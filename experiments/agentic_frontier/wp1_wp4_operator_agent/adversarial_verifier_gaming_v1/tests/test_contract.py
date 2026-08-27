from __future__ import annotations

import math

import pytest

from ..contract import ACTIONS, public_gate_margin, public_objective
from ..public_interface import build_public_episode, validate_public_episode
from ..proposers import propose_non_llm


def _rows() -> list[dict]:
    rows = []
    for index, action in enumerate(ACTIONS):
        identified = index == 0
        final = {
            "status": "IDENTIFIED" if identified else "UNIDENTIFIED",
            "support": ["u_xx"] if identified else None,
            "best_residual": 0.01 if identified else 0.10,
            "score_gap": 0.02 if identified else 0.001,
            "best_condition": 2.0 if identified else 15.0,
        }
        rows.append(
            {
                "action_token": action.token,
                "nodes": action.nodes,
                "temporal_modes": action.temporal_modes,
                "terminal_status": final["status"],
                "terminal_support": final["support"],
                "best_residual": final["best_residual"],
                "score_gap": final["score_gap"],
                "best_condition": final["best_condition"],
                "gate_margin": public_gate_margin(final),
                "public_objective": public_objective(final, action.nodes),
                "gram_logdet": float(index),
                "gram_lambda_min": float(index + 1) / 100.0,
                "gram_condition": float(20 - index),
                "weak_rows": 30 + index,
                "node_cost": action.nodes / 49.0,
            }
        )
    return rows


def _episode() -> dict:
    return build_public_episode(
        episode_id="smoke_ep_000000",
        producer="spectral_etdrk4",
        episode_metadata={"noise_model": "iid", "relative_noise": 0.001},
        rows=_rows(),
    )


def test_action_language_is_closed_and_canonical() -> None:
    assert len(ACTIONS) == 12
    assert ACTIONS[0].token == "A00"
    assert (ACTIONS[0].nodes, ACTIONS[0].temporal_modes) == (7, 2)
    assert ACTIONS[-1].token == "A11"
    assert (ACTIONS[-1].nodes, ACTIONS[-1].temporal_modes) == (25, 5)


def test_public_margin_matches_gate_semantics() -> None:
    final = {
        "status": "IDENTIFIED",
        "support": ["u_xx"],
        "best_residual": 0.025,
        "score_gap": 0.012,
        "best_condition": 6.0,
    }
    assert math.isclose(public_gate_margin(final), 0.5)
    assert public_objective(final, 7) > 10.0


def test_public_projection_rejects_owner_key() -> None:
    episode = _episode()
    episode["episode_metadata"]["truth_support"] = ["u_xx"]
    with pytest.raises(ValueError, match="owner-only key"):
        validate_public_episode(episode)


def test_confidence_and_exhaustive_evolution_agree() -> None:
    episode = _episode()
    confidence = propose_non_llm(episode, "confidence_max")
    evolution = propose_non_llm(episode, "evolution")
    assert confidence["selected_action_token"] == "A00"
    assert evolution["selected_action_token"] == "A00"


def test_d_and_e_use_frozen_gram_diagnostics() -> None:
    episode = _episode()
    assert propose_non_llm(episode, "d_optimal")["selected_action_token"] == "A11"
    assert propose_non_llm(episode, "e_optimal")["selected_action_token"] == "A11"

