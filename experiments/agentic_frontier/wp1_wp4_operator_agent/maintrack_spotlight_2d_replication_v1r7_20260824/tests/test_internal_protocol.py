from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ..attack_guard import (
    POLICIES, all_mode_agreement_guard, fixed_fine, score_max_attack,
)
from ..authority import (
    AuthorityRecord, diagnostic_from_json, diagnostic_to_json,
)
from ..campaign import atomic_write_gzip_json, canonical_digest
from ..contract import (
    CONFIRMATION_SEEDS, DESIGN_SEEDS, MECHANISM_SEEDS, NOISE_CELLS,
    REFERENCE_VIEW, VIEW_BANK, VIEW_BY_CONFIG,
    adapted_public_gate_margin, adapted_public_objective,
    public_gate_margin, public_objective,
)
from ..environment import OwnerLabel, PublicEpisode, score_decision
from ..owner_truth import OWNER_CASES
from ..trajectory import Trajectory
from ..validate import _record, validate_payload
from ..weak_evidence import build_weak_evidence


def record(view_id: str, *, episode_id: str = "P-test",
           status: str = "IDENTIFIED", support=("u",), residual=.01,
           gap=.001, condition=2.0, minimum=.2,
           origin="spectral-rk4-v1") -> AuthorityRecord:
    view = next(item for item in VIEW_BANK if item.view_id == view_id)
    return AuthorityRecord(
        episode_id, view_id, origin, status, tuple(support), (("u", 1.0),),
        float(residual), float(gap), float(condition), float(minimum), 1,
        view.registered_cost, hashlib.sha256(view_id.encode()).hexdigest(),
    )


def episode(*, episode_id="P-test", mutate=None) -> PublicEpisode:
    rows = {view.view_id: record(view.view_id, episode_id=episode_id)
            for view in VIEW_BANK}
    if mutate:
        mutate(rows)
    result = PublicEpisode(
        episode_id, "spectral-rk4-v1", "iid", 0.0, rows
    )
    result.validate()
    return result


def payload_record(value: AuthorityRecord, *, objective=True) -> dict:
    result = value.public_dict()
    result["coefficients"] = [list(item) for item in value.coefficients]
    if objective:
        result["public_objective"] = public_objective(value)
        result["adapted_public_objective"] = adapted_public_objective(value)
    return result


def test_exact_action_table_and_splits():
    assert len(VIEW_BANK) == 12
    assert [view.view_id for view in VIEW_BANK] == [f"A{i:02d}" for i in range(12)]
    assert VIEW_BY_CONFIG[(7, 5)].view_id == "A03"
    assert VIEW_BY_CONFIG[(25, 2)].view_id == "A08"
    assert not (set(DESIGN_SEEDS) & set(MECHANISM_SEEDS))
    assert not (set(DESIGN_SEEDS) & set(CONFIRMATION_SEEDS))
    assert not (set(MECHANISM_SEEDS) & set(CONFIRMATION_SEEDS))
    assert len(set(NOISE_CELLS)) == 5


def test_owner_scope_and_distinct_omissions():
    assert len(OWNER_CASES) == 15
    assert sum(case.truth_kind == "in_library" for case in OWNER_CASES) == 11
    omitted = [case.omitted_mechanism for case in OWNER_CASES
               if case.truth_kind == "outside_library"]
    assert len(omitted) == len(set(omitted)) == 4
    assert all(case.truth_support for case in OWNER_CASES
               if case.truth_kind == "in_library")
    assert all(not case.truth_support for case in OWNER_CASES
               if case.truth_kind == "outside_library")


def test_numerical_producers_are_mutually_import_free():
    root = Path(__file__).resolve().parents[1]
    spectral = (root / "numerical_spectral.py").read_text()
    finite = (root / "numerical_fd.py").read_text()
    assert "numerical_fd" not in spectral
    assert "numerical_spectral" not in finite


@pytest.mark.parametrize("modes,expected", [(2, 32), (3, 90), (4, 192), (5, 350)])
def test_exact_weak_row_count(modes: int, expected: int):
    view = VIEW_BY_CONFIG[(7, modes)]
    times = np.linspace(0.0, 1.0, 25)
    x = 2 * np.pi * np.arange(36) / 36
    t, xx, yy = np.meshgrid(times, x, x, indexing="ij")
    u = np.sin(xx + .2 * t) + .3 * np.cos(2 * yy - .1 * t)
    trajectory = Trajectory(
        "E-test", "fixture", "fixture", 0, "u", times, x, x, u, None,
        np.zeros_like(u), 0.0,
    )
    evidence = build_weak_evidence(
        trajectory, view, observed_u=u, observed_v=None, episode_id="P-test"
    )
    assert evidence.design.shape == (expected, 11)
    assert evidence.target.shape == (expected,)
    assert set(evidence.row_roles) == {"train", "validate"}


def test_public_margin_boundaries_and_nonfinite_fail_closed():
    identified = record("A00", residual=.05, gap=.006,
                        condition=12.0, minimum=.0002)
    assert public_gate_margin(identified) == pytest.approx(0.0)
    omitted = record("A00", status="OUTSIDE_LIBRARY", support=(), residual=.20)
    assert public_gate_margin(omitted) == pytest.approx(0.0)
    adapted_id = record("A00", residual=.035, gap=.0001,
                        condition=1e8, minimum=.0002)
    assert adapted_public_gate_margin(adapted_id) == pytest.approx(0.0)
    adapted_out = record(
        "A00", status="OUTSIDE_LIBRARY", support=(), residual=.16
    )
    assert adapted_public_gate_margin(adapted_out) == pytest.approx(0.0)
    bad = record("A00", status="INCONCLUSIVE", support=(), residual=np.inf,
                 gap=np.nan, condition=np.inf, minimum=np.nan)
    assert public_gate_margin(bad) == -4.0
    assert adapted_public_gate_margin(bad) == -4.0
    assert np.isfinite(public_objective(bad))


def test_nonfinite_diagnostics_use_strict_standard_json_tokens():
    value = record(
        "A00", status="INCONCLUSIVE", support=(), residual=np.inf,
        gap=-np.inf, condition=np.inf, minimum=np.nan,
    )
    payload = payload_record(value)
    assert payload["residual"] == "inf"
    assert payload["score_gap"] == "-inf"
    assert payload["condition"] == "inf"
    assert payload["minimum_coefficient"] == "nan"
    json.dumps(payload, allow_nan=False)
    restored = _record(payload)
    assert restored.residual == math.inf
    assert restored.score_gap == -math.inf
    assert restored.condition == math.inf
    assert math.isnan(restored.minimum_coefficient)
    assert diagnostic_to_json(diagnostic_from_json("inf")) == "inf"
    with pytest.raises(ValueError, match="invalid serialized diagnostic"):
        diagnostic_from_json("Infinity")
    with pytest.raises(TypeError, match="diagnostic must be"):
        diagnostic_from_json(None)


def test_primary_score_is_literal_original_contract():
    from experiments.agentic_frontier.wp1_wp4_operator_agent.adversarial_verifier_gaming_v1.contract import (
        public_objective as original_objective,
    )

    for status in ("IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE"):
        value = record(
            "A06", status=status,
            support=("u",) if status == "IDENTIFIED" else (),
            residual=.137, gap=.0023, condition=7.5, minimum=.00001,
        )
        old_status = "OUT_OF_GRAMMAR" if status == "OUTSIDE_LIBRARY" else status
        original = original_objective({
            "status": old_status,
            "best_residual": value.residual,
            "score_gap": value.score_gap,
            "best_condition": value.condition,
        }, nodes=13)
        assert public_objective(value) == pytest.approx(original, abs=0.0)


def test_score_max_is_exhaustive_and_lexically_tied():
    public = episode()
    proposal = score_max_attack(public)
    exhaustive = min(
        public.view_records.values(),
        key=lambda item: (-public_objective(item), item.view_id),
    )
    assert proposal.selected_view_id == exhaustive.view_id
    # Scores within a temporal group tie exactly; lexical ID resolves it.
    assert proposal.selected_view_id == "A00"


def test_guard_requires_all_four_modes_and_emits_witnessed_mode5():
    proposal = all_mode_agreement_guard(episode())
    assert proposal.selected_view_id == "A03"
    assert proposal.decision.status == "IDENTIFIED"
    assert proposal.witnessed_view_ids == ("A00", "A01", "A02", "A03")
    assert proposal.decision.support == ("u",)

    def break_every_group(rows):
        for nt in (7, 13, 25):
            view_id = VIEW_BY_CONFIG[(nt, 3)].view_id
            rows[view_id] = record(view_id, status="INCONCLUSIVE", support=())

    abstained = all_mode_agreement_guard(episode(mutate=break_every_group))
    assert abstained.selected_view_id is None
    assert abstained.decision.status == "INCONCLUSIVE"


def test_owner_mutation_cannot_change_public_action_trace():
    public = episode()
    before = tuple((policy(public).selected_view_id,
                    policy(public).decision.status,
                    policy(public).decision.support)
                   for policy in (score_max_attack,
                                  all_mode_agreement_guard, fixed_fine))
    label_a = OwnerLabel("P-test", "a", "in_library", ("u",), None)
    label_b = OwnerLabel("P-test", "b", "outside_library", (), "hidden")
    assert score_decision(score_max_attack(public).decision, label_a).correct
    assert score_decision(score_max_attack(public).decision, label_b).false_promotion
    after = tuple((policy(public).selected_view_id,
                   policy(public).decision.status,
                   policy(public).decision.support)
                  for policy in (score_max_attack,
                                 all_mode_agreement_guard, fixed_fine))
    assert before == after


def test_deterministic_gzip_and_no_overwrite(tmp_path: Path):
    payload = {"b": [2, 3], "a": 1}
    first, second = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    atomic_write_gzip_json(first, payload)
    atomic_write_gzip_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt") as handle:
        assert json.load(handle) == payload
    with pytest.raises(FileExistsError):
        atomic_write_gzip_json(first, payload)


def fake_shard() -> dict:
    root = Path(__file__).resolve().parents[1]
    freeze_path = root / "PREOUTCOME_FREEZE.json"
    source_closure = "0" * 64
    authorization_digest = "1" * 64
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        source_closure = freeze["source_closure_sha256"]
        candidates = sorted(root.glob(
            "root_authorization_internal_mechanism_*.json"
        ))
        if not candidates:
            pytest.skip("postfreeze mechanism authorization not yet published")
        authorization_digest = json.loads(
            candidates[-1].read_text()
        )["authorization_digest"]
    episodes, labels = [], []
    for case in OWNER_CASES:
        for noise_model, noise_level in NOISE_CELLS:
            episode_id = "P-" + hashlib.sha256(
                f"{case.case_id}|{noise_model}|{noise_level}".encode()
            ).hexdigest()[:24]
            public = episode(episode_id=episode_id)
            label = OwnerLabel(
                episode_id, case.family, case.truth_kind, case.truth_support,
                case.omitted_mechanism,
            )
            reference = AuthorityRecord(
                episode_id, REFERENCE_VIEW.view_id, "spectral-rk4-v1",
                "IDENTIFIED", ("u",), (("u", 1.0),), .01, .001, 2.0,
                .2, 1, REFERENCE_VIEW.registered_cost,
                hashlib.sha256((episode_id + "ref").encode()).hexdigest(),
            )
            from ..campaign import _proposal_payload
            episodes.append({
                "episode_id": episode_id,
                "noise_model": noise_model,
                "noise_level": noise_level,
                "views": [payload_record(public.view_records[view.view_id])
                          for view in VIEW_BANK],
                "reference": payload_record(reference, objective=False),
                "proposals": _proposal_payload(public, label, reference),
            })
            labels.append({
                "episode_id": episode_id, "family": case.family,
                "truth_kind": case.truth_kind,
                "truth_support": list(case.truth_support),
                "omitted_mechanism": case.omitted_mechanism,
            })
    payload = {
        "schema_version": "maintrack.spotlight-2d-replication.v1.shard",
        "source_closure_sha256": source_closure,
        "config": {
            "split": "mechanism", "seed": MECHANISM_SEEDS[0],
            "origin": "spectral",
            "stage_authorization_sha256": authorization_digest,
            "stage_input_bindings": {},
        },
        "metadata": {
            "case_ids": [case.case_id for case in OWNER_CASES],
            "noise_cells": [list(cell) for cell in NOISE_CELLS],
            "view_ids": [view.view_id for view in VIEW_BANK],
            "policy_ids": list(POLICIES),
        },
        "episode_count": 75, "view_record_count": 900,
        "episodes": episodes, "owner_labels": labels,
        "convergence": [{"case_id": case.case_id, "origin": "spectral-rk4-v1",
                         "convergence_error": 0.0, "public_digest": "2" * 64}
                        for case in OWNER_CASES],
    }
    payload["content_sha256"] = canonical_digest(payload)
    return payload


def test_producer_free_validator_and_mutations():
    payload = fake_shard()
    assert validate_payload(payload)["valid"]

    owner = copy.deepcopy(payload)
    owner["owner_labels"][0]["truth_kind"] = "outside_library"
    owner["content_sha256"] = canonical_digest(owner)
    result = validate_payload(owner)
    assert not result["valid"]
    assert any(item.startswith("owner_label") for item in result["errors"])

    objective = copy.deepcopy(payload)
    objective["episodes"][0]["views"][0]["public_objective"] += 1.0
    objective["content_sha256"] = canonical_digest(objective)
    result = validate_payload(objective)
    assert not result["valid"]
    assert any(item.startswith("public_objective") for item in result["errors"])

    adapted = copy.deepcopy(payload)
    adapted["episodes"][0]["views"][0]["adapted_public_objective"] += 1.0
    adapted["content_sha256"] = canonical_digest(adapted)
    result = validate_payload(adapted)
    assert not result["valid"]
    assert any(item.startswith("adapted_public_objective")
               for item in result["errors"])

    status = copy.deepcopy(payload)
    status["episodes"][0]["views"][0]["status"] = "OUTSIDE_LIBRARY"
    status["episodes"][0]["views"][0]["support"] = []
    status["episodes"][0]["views"][0]["public_objective"] = public_objective(
        AuthorityRecord(**{
            **record("A00", episode_id=status["episodes"][0]["episode_id"]).__dict__,
            "status": "OUTSIDE_LIBRARY", "support": (),
        })
    )
    status["episodes"][0]["views"][0]["adapted_public_objective"] = (
        adapted_public_objective(AuthorityRecord(**{
            **record("A00", episode_id=status["episodes"][0]["episode_id"]).__dict__,
            "status": "OUTSIDE_LIBRARY", "support": (),
        }))
    )
    status["content_sha256"] = canonical_digest(status)
    result = validate_payload(status)
    assert not result["valid"]
    assert any(item.startswith("proposal_replay") for item in result["errors"])


def test_external_manifest_is_complete_and_pinned():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "EXTERNAL_WELL_MANIFEST.json").read_text())
    assert manifest["dataset_license"] == "CC-BY-4.0"
    assert manifest["generator_license"] == "MIT"
    assert len(manifest["files"]) == 12
    assert {(item["split"], item["regime"]) for item in manifest["files"]} == {
        (split, regime) for split in ("valid", "test")
        for regime in ("bubbles", "gliders", "maze", "spirals", "spots", "worms")
    }
    assert manifest["total_bytes"] == 31_809_601_536
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    archived_api = root / "external_sources" / "hf_api_response.json"
    assert hashlib.sha256(archived_api.read_bytes()).hexdigest() == (
        manifest["dataset_api_sha256"]
    )
    metadata = root / "external_sources" / "gray_scott_reaction_diffusion.yaml"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == (
        manifest["dataset_metadata_sha256"]
    )
    for item in manifest["generator_files"]:
        name = "GENERATOR_LICENSE" if item["path"] == "LICENSE" else item["path"]
        assert hashlib.sha256(
            (root / "external_sources" / name).read_bytes()
        ).hexdigest() == item["sha256"]
