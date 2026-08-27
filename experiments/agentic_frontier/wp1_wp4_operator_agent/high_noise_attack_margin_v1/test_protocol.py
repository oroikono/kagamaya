"""Adversarial protocol tests that do not open reserved outcomes."""

from __future__ import annotations

from copy import deepcopy

import pytest

from .campaign import _policy_locks
from .contract import NOISES, POLICIES, SEEDS, digest


def _episode() -> dict:
    rows = []
    token = 0
    for nodes in (7, 13, 25):
        for modes in (2, 3, 4, 5):
            rows.append({
                "action_token": f"A{token:02d}", "nodes": nodes,
                "temporal_modes": modes, "terminal_status": "IDENTIFIED",
                "terminal_support": ["u_x"], "best_residual": .01,
                "score_gap": .02, "best_condition": 2.0,
                "gate_margin": .1, "public_objective": float(token),
                "gram_logdet": 1.0, "gram_lambda_min": .1,
                "gram_condition": 10.0, "weak_rows": 20,
                "node_cost": nodes / 49,
            })
            token += 1
    body = {
        "schema_version": "verisigs.adversarial_verifier_gaming.v1.public_episode",
        "episode_id": "opaque", "producer": "spectral_etdrk4",
        "episode_metadata": {"noise_model": "iid", "relative_noise": .01},
        "candidate_budget": 12, "actions": rows,
    }
    body["episode_sha256"] = digest(body)
    return body


def test_registry_is_fresh_and_exact() -> None:
    assert SEEDS == tuple(range(153000, 153020))
    assert not set(SEEDS) & set(range(150000, 152010))
    assert NOISES == (0.01, 0.05, 0.10)
    assert len(POLICIES) == 4


def test_confidence_is_exact_argmax() -> None:
    locks = _policy_locks(_episode())
    assert locks["confidence_max"]["selected_action_token"] == "A11"
    assert locks["fixed_25_m5"]["selected_action_token"] == "A11"


def test_hidden_truth_mutation_cannot_change_policy_locks() -> None:
    public = _episode()
    before = _policy_locks(public)
    hidden = {"truth_support": ["u_x"], "reference_false": False}
    mutated = deepcopy(hidden)
    mutated["truth_support"] = ["u_xxxx"]
    mutated["reference_false"] = True
    assert hidden != mutated
    assert before == _policy_locks(public)


def test_public_mutation_requires_digest_rebinding() -> None:
    public = _episode()
    public["actions"][0]["public_objective"] = 999.0
    with pytest.raises(ValueError):
        _policy_locks(public)


def test_all_mode_consensus_abstains_on_disagreement() -> None:
    public = _episode()
    for row in public["actions"]:
        row["terminal_support"] = [f"m{row['temporal_modes']}"]
    body = dict(public)
    body.pop("episode_sha256")
    public["episode_sha256"] = digest(body)
    locks = _policy_locks(public)
    assert locks["all_mode_consensus"]["proposal_valid"] is False
    assert locks["all_mode_consensus"]["selected_action_token"] is None
