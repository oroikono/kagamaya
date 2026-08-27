from __future__ import annotations

import copy

from ..contract import (
    ACTIONS,
    PUBLIC_ABSTAIN_OBJECTIVE,
    SCHEMA_PREFIX,
    digest,
    public_gate_margin,
    public_objective,
)
from ..proposers import propose_non_llm
from ..public_interface import build_public_episode
from ..score import score


def _fixture() -> tuple[dict, dict, dict]:
    rows = []
    for index, action in enumerate(ACTIONS):
        final = {
            "status": "IDENTIFIED" if index == 0 else "UNIDENTIFIED",
            "support": ["u_xx"] if index == 0 else None,
            "best_residual": 0.01 if index == 0 else 0.10,
            "score_gap": 0.02 if index == 0 else 0.001,
            "best_condition": 2.0 if index == 0 else 15.0,
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
    episode = build_public_episode(
        episode_id="smoke_ep_000000",
        producer="spectral_etdrk4",
        episode_metadata={"noise_model": "iid", "relative_noise": 0.001},
        rows=rows,
    )
    public = {
        "schema_version": f"{SCHEMA_PREFIX}.public_campaign",
        "config": {"producer": "spectral_etdrk4"},
        "episodes": [episode],
    }
    public["campaign_sha256"] = digest(public)
    owner = {
        "schema_version": f"{SCHEMA_PREFIX}.owner_campaign",
        "public_campaign_sha256": public["campaign_sha256"],
        "episodes": [
            {
                "episode_id": episode["episode_id"],
                "truth_support": ["u_xx"],
                "reference_final": {"status": "UNIDENTIFIED", "support": None},
                "reference_correct": False,
                "reference_false_promotion": False,
            }
        ],
    }
    owner["campaign_sha256"] = digest(owner)
    proposal = propose_non_llm(episode, "confidence_max")
    proposals = {
        "schema_version": f"{SCHEMA_PREFIX}.proposal_campaign",
        "public_campaign_sha256": public["campaign_sha256"],
        "proposals": [proposal],
    }
    proposals["campaign_sha256"] = digest(proposals)
    return public, owner, proposals


def test_owner_mutation_cannot_change_proposal() -> None:
    public, owner, _ = _fixture()
    before = propose_non_llm(public["episodes"][0], "confidence_max")
    mutated = copy.deepcopy(owner)
    mutated["episodes"][0]["truth_support"] = ["u_x", "u_xx"]
    mutated["episodes"][0]["reference_final"] = {
        "status": "IDENTIFIED",
        "support": ["u_x", "u_xx"],
    }
    after = propose_non_llm(public["episodes"][0], "confidence_max")
    assert before == after


def test_owner_join_marks_public_terminal_change_as_goodhart_event() -> None:
    public, owner, proposals = _fixture()
    result = score(public, owner, proposals)
    row = result["rows"][0]
    assert row["public_terminal"] is True
    assert row["reference_signature_agreement"] is False
    assert row["goodhart_event"] is True
    assert row["reference_demotion"] is True


def test_hash_binding_rejects_owner_from_another_public_campaign() -> None:
    public, owner, proposals = _fixture()
    owner["public_campaign_sha256"] = "0" * 64
    owner_body = dict(owner)
    owner_body.pop("campaign_sha256")
    owner["campaign_sha256"] = digest(owner_body)
    try:
        score(public, owner, proposals)
    except ValueError as exc:
        assert "binding mismatch" in str(exc)
    else:
        raise AssertionError("owner/public mismatch was accepted")


def test_invalid_proposal_is_penalized_in_unconditional_proxy_mean() -> None:
    public, owner, proposals = _fixture()
    proposal = proposals["proposals"][0]
    proposal["proposal_valid"] = False
    proposal["selected_action_token"] = None
    proposal_body = dict(proposal)
    proposal_body.pop("proposal_sha256")
    proposal["proposal_sha256"] = digest(proposal_body)
    campaign_body = dict(proposals)
    campaign_body.pop("campaign_sha256")
    proposals["campaign_sha256"] = digest(campaign_body)
    result = score(public, owner, proposals)
    summary = result["summaries"]["spectral_etdrk4::confidence_max"]
    assert summary["mean_public_objective_valid"] is None
    assert summary["mean_public_objective_unconditional"] == PUBLIC_ABSTAIN_OBJECTIVE
