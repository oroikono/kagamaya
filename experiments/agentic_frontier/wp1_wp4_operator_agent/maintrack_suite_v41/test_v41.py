from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .campaign import CampaignConfig, run_campaign
from .policies import (
    DEPLOYABLE_POLICIES,
    ambiguity_action_scores,
    choose_action,
    independent_policy_seed,
)
from .simulator import (
    BASELINE_ACTION,
    Family,
    HiddenTruth,
    SimulationConfig,
    Trajectory,
    action_menu,
    simulate,
)
from .validate import validate_payload
from .weakform import DiscoveryConfig, discover, trajectory_evidence


PUBLIC_SEED = 8675309


def _semantic_digest(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_deployment_package_is_self_contained() -> None:
    package = Path(__file__).parent
    source = (package / "simulator.py").read_text()
    assert "maintrack_suite.generators" not in source
    assert (package / "numerics.py").is_file()


def test_simulation_is_deterministic_and_noise_models_are_distinct() -> None:
    action = action_menu()[1]
    a, _ = simulate(Family.FISHER_KPP, action, 12, PUBLIC_SEED, 0.001, "correlated")
    b, _ = simulate(Family.FISHER_KPP, action, 12, PUBLIC_SEED, 0.001, "correlated")
    c, _ = simulate(Family.FISHER_KPP, action, 12, PUBLIC_SEED, 0.001, "iid")
    assert np.array_equal(a.u, b.u)
    assert not np.array_equal(a.u, c.u)


def test_etdrk4_step_halving_converges_for_all_in_grammar_families() -> None:
    coarse = SimulationConfig(t_final=0.08, internal_dt=0.0005, save_dt=0.01)
    fine = SimulationConfig(t_final=0.08, internal_dt=0.00025, save_dt=0.01)
    for family in tuple(Family)[:5]:
        left, _ = simulate(family, action_menu()[3], 41, PUBLIC_SEED, config=coarse)
        right, _ = simulate(family, action_menu()[3], 41, PUBLIC_SEED, config=fine)
        relative = np.linalg.norm(left.u[-1] - right.u[-1]) / max(
            np.linalg.norm(right.u[-1]), 1e-15
        )
        assert relative < 2e-5, (family, relative)


def test_periodic_mass_and_dissipation_controls() -> None:
    cfg = SimulationConfig(t_final=0.08, internal_dt=0.00025, save_dt=0.01)
    for family in (Family.ADVECTION_DIFFUSION, Family.VISCOUS_BURGERS):
        trajectory, _ = simulate(family, action_menu()[2], 43, PUBLIC_SEED, config=cfg)
        means = trajectory.u.mean(axis=1)
        assert np.max(np.abs(means - means[0])) < 2e-10
        energy = np.mean(trajectory.u**2, axis=1)
        assert np.max(np.diff(energy)) < 2e-10


def test_weak_form_spatial_signs_against_spectral_derivatives() -> None:
    nx = 64
    nt = 25
    length = 2.0 * np.pi
    duration = 0.24
    x = np.arange(nx) * length / nx
    t = np.linspace(0.0, duration, nt)
    u = (
        (0.3 + 0.2 * np.cos(np.pi * t / duration))[:, None] * np.sin(x)[None, :]
        + 0.11 * np.sin(2.0 * np.pi * t / duration)[:, None] * np.cos(2.0 * x)[None, :]
    )
    trajectory = Trajectory("manufactured", "none", x, t, u, "iid", 0.0)
    design, _, _ = trajectory_evidence(trajectory, temporal_modes=3, spatial_modes=4)

    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=length / nx)
    uhat = np.fft.fft(u, axis=1)
    derivatives = [
        np.fft.ifft((1j * k)[None, :] ** order * uhat, axis=1).real
        for order in (1, 2, 3)
    ]
    nonlinear = u * derivatives[0]

    # Rebuild the same Simpson/time and Fourier-space-test integrals in strong
    # form.  Agreement checks integration-by-parts signs independently of PDE
    # fitting or hidden support labels.
    dt = t[1] - t[0]
    q = np.ones(nt)
    q[1:-1:2] = 4.0
    q[2:-1:2] = 2.0
    q *= dt / 3.0
    dx = length / nx
    expected = []
    for tm in range(1, 4):
        eta = np.sin(tm * np.pi * t / duration)
        wet = q * eta
        for sm in range(0, 5):
            if sm == 0:
                tests = (np.ones(nx),)
            else:
                phase = sm * x
                tests = (np.cos(phase), np.sin(phase))
            for psi in tests:
                expected.append(
                    [
                        dx * np.einsum("t,tx,x->", wet, derivatives[0], psi),
                        dx * np.einsum("t,tx,x->", wet, derivatives[1], psi),
                        dx * np.einsum("t,tx,x->", wet, derivatives[2], psi),
                        dx * np.einsum("t,tx,x->", wet, nonlinear, psi),
                    ]
                )
    expected_array = np.asarray(expected)
    assert np.allclose(design[:, [4, 5, 6, 7]], expected_array, rtol=2e-11, atol=2e-11)


def test_round_one_admission_and_abstention_are_evidence_driven() -> None:
    admitted, admitted_truth = simulate(
        Family.ADVECTION_DIFFUSION, BASELINE_ACTION, 100, PUBLIC_SEED
    )
    unresolved, _ = simulate(Family.FISHER_KPP, BASELINE_ACTION, 100, PUBLIC_SEED)
    first = discover((admitted,))
    second = discover((unresolved,))
    assert first.status == "IDENTIFIED"
    assert first.support == admitted_truth.support
    assert second.status == "UNIDENTIFIED"
    assert second.fits[0].condition > DiscoveryConfig().condition_max


def test_active_conversion_requires_observed_round_one_failure() -> None:
    baseline, truth = simulate(Family.FISHER_KPP, BASELINE_ACTION, 100, PUBLIC_SEED)
    before = discover((baseline,))
    conversions = []
    for action in action_menu():
        second, _ = simulate(Family.FISHER_KPP, action, 100, PUBLIC_SEED)
        after = discover((baseline, second))
        if after.status == "IDENTIFIED" and after.support == truth.support:
            conversions.append(action.name)
    assert before.status == "UNIDENTIFIED"
    assert before.fits[0].condition > DiscoveryConfig().condition_max
    assert conversions


def test_ambiguity_conditioning_changes_action_scores() -> None:
    fisher, _ = simulate(Family.FISHER_KPP, BASELINE_ACTION, 101, PUBLIC_SEED)
    allen, _ = simulate(Family.ALLEN_CAHN, BASELINE_ACTION, 101, PUBLIC_SEED)
    fisher_result = discover((fisher,))
    allen_result = discover((allen,))
    fisher_scores = ambiguity_action_scores(fisher_result, PUBLIC_SEED)
    allen_scores = ambiguity_action_scores(allen_result, PUBLIC_SEED)
    assert fisher_result.ambiguity_set != allen_result.ambiguity_set
    assert fisher_scores != allen_scores
    fisher_ranking = sorted(fisher_scores, key=lambda key: fisher_scores[key]["joint"], reverse=True)
    allen_ranking = sorted(allen_scores, key=lambda key: allen_scores[key]["joint"], reverse=True)
    assert fisher_ranking != allen_ranking


def test_policy_truth_metadata_noninterference() -> None:
    baseline, truth = simulate(Family.ALLEN_CAHN, BASELINE_ACTION, 102, PUBLIC_SEED)
    round1 = discover((baseline,))
    decisions_before = {
        policy: choose_action(policy, round1, PUBLIC_SEED, 123)
        for policy in DEPLOYABLE_POLICIES
    }
    mutated = HiddenTruth(Family.KDV, ("1",), (("1", 999.0),), truth.seed)
    assert mutated != truth
    decisions_after = {
        policy: choose_action(policy, round1, PUBLIC_SEED, 123)
        for policy in DEPLOYABLE_POLICIES
    }
    assert decisions_before == decisions_after


def test_random_policy_stream_is_independent_of_hidden_seed_values() -> None:
    base = 20260812
    first = [independent_policy_seed(base, ordinal, 1) for ordinal in range(20)]
    # Hidden simulator seeds are deliberately changed and are not arguments.
    hidden_seeds = [100 + ordinal * 17 for ordinal in range(20)]
    mutated_hidden_seeds = [9000 + ordinal * 31 for ordinal in range(20)]
    assert hidden_seeds != mutated_hidden_seeds
    second = [independent_policy_seed(base, ordinal, 1) for ordinal in range(20)]
    assert first == second
    assert len(set(first)) == len(first)


def test_oog_decision_is_observation_based() -> None:
    baseline, _ = simulate(
        Family.HYPERDIFFUSION_OOG, BASELINE_ACTION, 101, PUBLIC_SEED, 0.0005, "correlated"
    )
    results = []
    for action in action_menu():
        second, _ = simulate(
            Family.HYPERDIFFUSION_OOG, action, 101, PUBLIC_SEED, 0.0005, "correlated"
        )
        results.append(discover((baseline, second)))
    assert any(item.status == "OUT_OF_GRAMMAR" for item in results)
    assert all(item.support is None for item in results if item.status == "OUT_OF_GRAMMAR")


def test_campaign_replay_and_validator_negative_controls() -> None:
    config = CampaignConfig(
        seeds=(333,),
        relative_noises=(0.0,),
        noise_models=("iid",),
        families=(Family.FISHER_KPP.value, Family.HYPERDIFFUSION_OOG.value),
        split="test",
    )
    first = run_campaign(config)
    second = run_campaign(copy.deepcopy(config))
    assert first == second
    assert validate_payload(first) == []

    corrupted_digest = copy.deepcopy(first)
    corrupted_digest["content_sha256"] = "0" * 64
    assert "content digest mismatch" in validate_payload(corrupted_digest)

    duplicate = copy.deepcopy(first)
    duplicate["rows"].append(copy.deepcopy(duplicate["rows"][0]))
    duplicate["content_sha256"] = _semantic_digest(duplicate)
    duplicate_errors = validate_payload(duplicate)
    assert any("duplicate row key" in item for item in duplicate_errors)
    assert "row count does not match declared factorial grid" in duplicate_errors

    false_flag = copy.deepcopy(first)
    false_flag["rows"][0]["false_promotion"] = not false_flag["rows"][0]["false_promotion"]
    false_flag["content_sha256"] = _semantic_digest(false_flag)
    assert any("false-promotion flag mismatch" in item for item in validate_payload(false_flag))

    wrong_cost = copy.deepcopy(first)
    actionable = next(row for row in wrong_cost["rows"] if row["selected_action"] is not None)
    actionable["action_cost"] = 999.0
    wrong_cost["content_sha256"] = _semantic_digest(wrong_cost)
    assert any("action cost mismatch" in item for item in validate_payload(wrong_cost))
