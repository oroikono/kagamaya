from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evidenceviewbench import (
    Action,
    EvidenceViewEnv,
    case_bundle,
    load_cases,
    load_labels,
    load_observations,
    load_view_bank,
    score_trace,
    TwoDAction,
    TwoDEvidenceEnv,
    load_two_d_owner,
    load_two_d_public,
    score_two_d_terminal,
)
from evidenceviewbench.validation import validate_release


ROOT = Path(__file__).resolve().parents[1]


def test_public_tables_have_matched_216_case_keys() -> None:
    cases = load_cases()
    observations = load_observations()
    labels = load_labels()
    banks = load_view_bank()
    assert len(cases) == len(observations) == len(labels) == len(banks) == 216
    assert {row["case_id"] for row in cases} == {row["case_id"] for row in observations}
    assert {row["case_id"] for row in cases} == {row["case_id"] for row in labels}
    assert {row["case_id"] for row in cases} == {row["case_id"] for row in banks}


def test_selection_observation_excludes_analysis_metadata() -> None:
    selection_case = load_observations()[0]
    assert not {"family", "noise_model", "noise_level", "seed", "source_bindings"} & set(selection_case)
    env = EvidenceViewEnv(selection_case, budget=60, visibility="acquire_only")
    observation = env.reset()
    assert "family" not in observation
    assert "noise_model" not in observation
    assert "noise_level" not in observation


def test_trace_replays_and_scores() -> None:
    case = load_observations()[0]
    env = EvidenceViewEnv(case, budget=60, visibility="acquire_only")
    env.reset()
    env.step(Action.acquire_view(7, 2))
    env.step(Action.acquire_view(49, 5))
    _, done = env.step(Action.commit(49, 5))
    assert done
    trace = env.export_trace()
    replay = EvidenceViewEnv.replay_trace(case, trace)
    assert replay == trace
    score = score_trace(trace, case_bundle(case["case_id"])["label"])
    assert set(score) >= {"correct", "false_promotion", "inconclusive", "utility"}


def test_complete_release_validator() -> None:
    report = validate_release()
    assert report["valid"] is True
    assert report["case_count"] == 216
    assert report["metadata_blind_smokes"] == 2


def test_selection_hash_and_parent_binding_mutations_fail_closed() -> None:
    observation = load_observations()[0]
    content_mutation = copy.deepcopy(observation)
    content_mutation["public_rows"][0]["nodes"] = 13
    with pytest.raises(ValueError, match="selection case content hash mismatch"):
        EvidenceViewEnv(content_mutation, budget=60, visibility="acquire_only")

    coherent_content_mutation = copy.deepcopy(observation)
    coherent_content_mutation["public_rows"][0]["spectral_etdrk4"]["best_residual"] += 0.01
    coherent_body = dict(coherent_content_mutation)
    coherent_body.pop("selection_case_sha256")
    from evidenceviewbench.data import digest

    coherent_content_mutation["selection_case_sha256"] = digest(coherent_body)
    with pytest.raises(ValueError, match="selector rows differ from the bound case registry"):
        EvidenceViewEnv(coherent_content_mutation, budget=60, visibility="acquire_only")

    parent_mutation = copy.deepcopy(observation)
    parent_mutation["parent_public_case_sha256"] = "0" * 64
    body = dict(parent_mutation)
    body.pop("selection_case_sha256")
    parent_mutation["selection_case_sha256"] = digest(body)
    with pytest.raises(ValueError, match="case and acquisition bank bindings differ"):
        EvidenceViewEnv(parent_mutation, budget=60, visibility="acquire_only")


def test_experiment_registry_hashes_and_semantic_checks() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_experiment_registry.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report == {
        "claim_count": 10,
        "errors": [],
        "evidence_file_count": 15,
        "semantic_check_count": 44,
        "valid": True,
    }


def test_compact_sequential_artifact_is_content_bound_and_corrected() -> None:
    path = ROOT / "experiments/sequential_asymmetric_confirmation/confirmation_50seeds_compact.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("content_sha256")
    from evidenceviewbench.data import digest

    assert digest(payload) == claimed
    transitions = payload["fixed49_to_sequential_transitions"]
    for producer in ("fd_bdf", "spectral_etdrk4"):
        assert transitions[producer]["correct_to_inconclusive"] == 4
        assert transitions[producer]["inconclusive_to_correct"] == 3
        assert transitions[producer]["false_promotion_to_inconclusive"] == 29
        assert sum(transitions[producer].values()) == 1800


def test_release_version_is_consistent() -> None:
    import evidenceviewbench

    version = json.loads((ROOT / "RELEASE_VERSION.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "EXPERIMENT_REGISTRY.json").read_text(encoding="utf-8"))
    assert evidenceviewbench.__version__ == "0.1.2"
    assert version["package_version"] == "0.1.2"
    assert registry["release_version"] == "0.1.2"


def test_public_release_remains_fail_closed() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION_PENDING.md").read_text(encoding="utf-8")
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    assert "No license" in license_text
    assert "Citation decision required" in citation_text
    assert "[ ] Project owner selects" in checklist
    assert "[ ] Author/coauthor" in checklist
    assert "[ ] Final public release" in checklist


def test_release_has_no_workstation_absolute_paths() -> None:
    workstation_root = "/" + "Users" + "/"
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == "MANIFEST.sha256":
            continue
        if path.suffix.lower() not in {".md", ".py", ".toml", ".txt", ".json", ".jsonl", ".yml", ".yaml"}:
            continue
        if workstation_root in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_two_d_preview_is_bound_and_keeps_learning_blocked() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/validate_2d_preview.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report == {
        "agent_advantage_demonstrated": False,
        "convergence_rows": 30,
        "errors": [],
        "owner_rows": 180,
        "primary_cells": 150,
        "preview_claims": 5,
        "preview_evidence_bindings": 12,
        "public_rows": 180,
        "valid": True,
    }


def test_installable_two_d_agent_harness_replays_without_owner_access() -> None:
    public_rows = load_two_d_public()
    owner_rows = load_two_d_owner()
    assert len(public_rows) == len(owner_rows) == 180
    row = public_rows[0]
    owner = next(
        label
        for label in owner_rows
        if label["origin_component"] == row["origin_component"]
        and label["episode_id"] == row["episode_id"]
    )
    chosen = ["V035", "V051"]
    required_budget = sum(
        view["registered_cost"] for view in row["views"] if view["view_id"] in chosen
    )
    env = TwoDEvidenceEnv(row, budget=required_budget)
    assert not {"family", "truth_kind", "truth_support", "omitted_mechanism"} & set(env.public_state())
    env.step(TwoDAction.acquire(chosen[0]))
    env.step(TwoDAction.challenge(chosen[1]))
    terminal = env.step(TwoDAction.commit())
    assert terminal is not None
    trace = env.export_trace()
    trace = json.loads(json.dumps(trace))
    assert TwoDEvidenceEnv.replay_trace(row, trace) == trace
    assert score_two_d_terminal(terminal, owner).inconclusive in {True, False}

    mutated_owner = copy.deepcopy(owner)
    mutated_owner["truth_support"] = ["deliberately_mutated"]
    # Owner mutation is allowed only after removing/rebinding the sidecar hash;
    # the already exported public trace itself remains byte-identical.
    mutated_owner.pop("owner_row_sha256")
    from evidenceviewbench.data import digest

    mutated_owner["owner_row_sha256"] = digest(mutated_owner)
    assert TwoDEvidenceEnv.replay_trace(row, trace) == trace
    score_two_d_terminal(terminal, mutated_owner)


def test_two_d_rejected_action_is_atomic() -> None:
    row = load_two_d_public()[0]
    env = TwoDEvidenceEnv(row, budget=0)
    before = env.export_trace()
    with pytest.raises(ValueError, match="exceeds registered budget"):
        env.step(TwoDAction.acquire("V000"))
    after = env.export_trace()
    assert after == before
    assert env.actions == []
    assert env.acquired == []
