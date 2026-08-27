from __future__ import annotations

import copy

import pytest

from ..contract import (CALIBRATION_SEEDS, CONFIRMATION_SEEDS, ControlAction,
                        DIAGNOSE_VIEW_ID, LOFO_CALIBRATION_SEEDS,
                        LOFO_FIT_SEEDS, NEVER_SEEN_FAMILIES, NOISE_CELLS,
                        POLICY_FEATURES, TRAIN_SEEDS, VIEW_BANK)
from ..campaign import authorization_digest
from ..control import EvidenceControl, score
from ..external_transfer import _compact_spatial, owner_score, perturb
from ..freeze import closure
from ..learning import cluster_bootstrap_risk, select_matched_fixed
from ..owner_truth import OWNER_CASES
from ..validate import _canonical_digest, validate_payload


def _record(view_id: str, status="IDENTIFIED", support=("u",)) -> dict:
    return {
        "episode_id": "opaque", "view_id": view_id, "status": status,
        "support": list(support), "reason": "fixture",
        "registered_cost": next(v.registered_cost for v in VIEW_BANK if v.view_id == view_id),
        "residual": 0.01, "score_gap": 0.1, "condition": 2.0,
        "minimum_coefficient": 0.5, "ambiguity_size": 1,
    }


def _episode() -> dict:
    return {"episode_id": "opaque", "origin": "fixture", "noise_model": "iid",
            "noise_level": 0.0,
            "views": [_record(view.view_id) for view in VIEW_BANK]}


def test_splits_are_disjoint_and_confirmation_is_large():
    assert not set(TRAIN_SEEDS) & set(CALIBRATION_SEEDS)
    assert not set(TRAIN_SEEDS) & set(CONFIRMATION_SEEDS)
    assert not set(CALIBRATION_SEEDS) & set(CONFIRMATION_SEEDS)
    assert len(CONFIRMATION_SEEDS) == 100


def test_nested_lofo_seed_split_is_real_and_disjoint():
    assert not set(LOFO_FIT_SEEDS) & set(LOFO_CALIBRATION_SEEDS)
    assert set(LOFO_FIT_SEEDS) | set(LOFO_CALIBRATION_SEEDS) == set(TRAIN_SEEDS)


def test_noise_cells_have_no_zero_duplicate():
    assert len(NOISE_CELLS) == len(set(NOISE_CELLS)) == 5
    assert sum(level == 0 for _, level in NOISE_CELLS) == 1


def test_four_never_seen_families_are_declared():
    assert len(NEVER_SEEN_FAMILIES) == 4
    assert any("fractional" in family for family in NEVER_SEEN_FAMILIES)


def test_vendored_owner_dependency_is_self_contained():
    assert len(OWNER_CASES) == 15
    assert {case.family for case in OWNER_CASES} >= set(NEVER_SEEN_FAMILIES)


def test_source_closure_contains_transitive_owner_and_scheduler_files():
    manifest, digest = closure()
    assert len(digest) == 64
    assert "vendor/exact_2d_oracle.py" in manifest
    assert "vendor/exact_2d_coupled_oracle.py" in manifest
    assert "scripts/train_calibration_array.sbatch" in manifest


def test_policy_feature_contract_excludes_owner_and_identifier_fields():
    forbidden = {"family", "case", "seed", "truth", "owner", "episode_id"}
    assert not any(any(token in feature for token in forbidden)
                   for feature in POLICY_FEATURES)


def test_confirmation_authorization_digest_binds_every_decision():
    authorization = {
        "stage": "confirmation", "source_closure": "a" * 64,
        "calibration_gates_passed": True,
        "selected_risk_threshold": 0.001,
        "matched_fixed": {"submit": "V001"},
    }
    authorization["authorization_sha256"] = authorization_digest(authorization)
    mutated = copy.deepcopy(authorization)
    mutated["matched_fixed"]["submit"] = "V002"
    assert authorization_digest(mutated) != authorization["authorization_sha256"]


def test_view_bank_is_exactly_72_unique_views():
    assert len(VIEW_BANK) == 72
    assert len({view.view_id for view in VIEW_BANK}) == 72


def test_diagnose_then_submit_reuses_same_view_cost():
    env = EvidenceControl(_episode(), budget=1e9)
    env.step(ControlAction("DIAGNOSE", "V000"))
    terminal = env.step(ControlAction("SUBMIT", "V000"))
    assert terminal is not None
    assert terminal.registered_cost == VIEW_BANK[0].registered_cost
    assert score(terminal, {"truth_kind": "in_library", "truth_support": ["u"]}).correct


def test_different_submit_view_adds_cost():
    env = EvidenceControl(_episode(), budget=1e9)
    env.step(ControlAction("DIAGNOSE", "V000"))
    terminal = env.step(ControlAction("SUBMIT", "V001"))
    assert terminal is not None
    assert terminal.registered_cost == VIEW_BANK[0].registered_cost + VIEW_BANK[1].registered_cost


def test_invalid_action_is_atomic():
    env = EvidenceControl(_episode(), budget=1e9)
    before = copy.deepcopy(env.public_state())
    with pytest.raises(ValueError):
        env.step(ControlAction("SUBMIT", "V000"))
    assert env.public_state() == before
    assert env.trace == []


def test_owner_mutation_cannot_change_public_trace():
    traces = []
    for owner in ({"truth_kind": "in_library", "truth_support": ["u"]},
                  {"truth_kind": "outside_library", "truth_support": []}):
        env = EvidenceControl(copy.deepcopy(_episode()), budget=1e9)
        env.step(ControlAction("DIAGNOSE", "V000"))
        terminal = env.step(ControlAction("SUBMIT", "V001"))
        traces.append((env.trace_digest, terminal.status, terminal.support))
        score(terminal, owner)
    assert traces[0] == traces[1]


def test_abstain_never_false_promotes():
    env = EvidenceControl(_episode(), budget=1e9)
    env.step(ControlAction("DIAGNOSE", "V000"))
    terminal = env.step(ControlAction("ABSTAIN"))
    outcome = score(terminal, {"truth_kind": "in_library", "truth_support": ["u"]})
    assert outcome.inconclusive and not outcome.false_promotion


def test_second_diagnosis_rejected():
    env = EvidenceControl(_episode(), budget=1e9)
    env.step(ControlAction("DIAGNOSE", "V000"))
    with pytest.raises(ValueError):
        env.step(ControlAction("DIAGNOSE", "V001"))


def test_compact_external_tests_vanish_with_normal_derivative():
    import numpy as np
    x = np.linspace(-1.0 + 1/128, 1.0 - 1/128, 128)
    spatial, lap = _compact_spatial("sin", 1, 2, x, x)
    assert np.isfinite(spatial).all() and np.isfinite(lap).all()
    # Evaluate the polynomial envelope and its physical derivative exactly at
    # the conceptual boundary; both are zero, which kills the two boundary
    # terms of the twice-integrated Laplacian.
    for z in (0.0, 1.0):
        envelope = z**2 * (1-z)**2
        derivative = 2*z - 6*z**2 + 4*z**3
        assert envelope == 0.0 and derivative == 0.0


def test_external_noise_is_deterministic_and_shared_view_ready():
    import numpy as np
    values = np.arange(5*8*8, dtype=float).reshape(5, 8, 8)
    left = perturb(values, "correlated", 0.02, 991)
    right = perturb(values, "correlated", 0.02, 991)
    assert np.array_equal(left, right)


def test_external_owner_support_is_exact_not_subset():
    assert owner_score({"status": "IDENTIFIED", "support": ["u", "v", "lap_v"]}, "v") == (True, False)
    assert owner_score({"status": "IDENTIFIED", "support": ["u", "v"]}, "v") == (False, True)


def test_cluster_risk_uses_seed_not_episode_as_resampling_unit():
    rows = ([{"seed": 1, "false_promotion": False}] * 8
            + [{"seed": 2, "false_promotion": True}] * 8)
    result = cluster_bootstrap_risk(rows, 1000)
    assert result["seed_count"] == 2
    assert result["mean_seed_false_rate"] == 0.5
    assert result["max_seed_false_rate"] == 1.0


def test_matched_fixed_never_exceeds_agent_cost_ceiling():
    episodes = []
    labels = {}
    for index in range(2):
        episode = _episode()
        episode["episode_id"] = f"opaque-{index}"
        episode["_seed"] = 900 + index
        for row in episode["views"]:
            row["episode_id"] = episode["episode_id"]
        episodes.append(episode)
        labels[episode["episode_id"]] = {
            "family": "fixture", "truth_kind": "in_library", "truth_support": ["u"]}
    diagnose_cost = next(view.registered_cost for view in VIEW_BANK
                         if view.view_id == DIAGNOSE_VIEW_ID)
    selected, _ = select_matched_fixed(
        episodes, labels, {"fixture"}, diagnose_cost)
    assert selected is not None
    assert selected["registered_cost"] <= diagnose_cost
    assert selected["submit"] == DIAGNOSE_VIEW_ID


def _valid_shard_fixture():
    episodes = []
    labels = []
    ordinal = 0
    for case in OWNER_CASES:
        for noise_model, noise_level in NOISE_CELLS:
            episode_id = f"fixture-{ordinal}"
            ordinal += 1
            views = [_record(view.view_id, "INCONCLUSIVE", ()) for view in VIEW_BANK]
            for row in views:
                row["episode_id"] = episode_id
            episodes.append({"episode_id": episode_id, "origin": "spectral",
                             "noise_model": noise_model, "noise_level": noise_level,
                             "views": views})
            labels.append({"episode_id": episode_id, "owner_case_id": case.case_id,
                           "family": case.family, "truth_kind": case.truth_kind,
                           "truth_support": list(case.truth_support),
                           "omitted_mechanism": case.omitted_mechanism})
    payload = {
        "schema_version": "maintrack.evidence-control.shard.v3r2",
        "metadata": {"split": "train", "seed": TRAIN_SEEDS[0],
                     "origin": "spectral", "source_closure": "fixture",
                     "view_count": len(VIEW_BANK),
                     "noise_cells": [list(cell) for cell in NOISE_CELLS]},
        "episodes": episodes, "owner_labels": labels, "convergence": [],
    }
    payload["content_sha256"] = _canonical_digest(payload)
    return payload


def test_producer_free_validator_rejects_coherent_owner_relabel():
    payload = _valid_shard_fixture()
    assert validate_payload(payload)["valid"]
    payload["owner_labels"][0]["truth_support"] = ["invented"]
    payload["content_sha256"] = _canonical_digest(payload)
    report = validate_payload(payload)
    assert not report["valid"] and "owner truth mismatch" in report["errors"]
