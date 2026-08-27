from __future__ import annotations

from ..sequential_asymmetric_controller_v2.test_protocol import _episode, _reseal
from ..sequential_asymmetric_controller_v2.policy import propose
from .aggregate import cluster_statistics, confirmation_gates
from .campaign import summarize_shard
from .contract import (
    CONFIRMATION_SEEDS,
    EPISODES_PER_SHARD,
    FAMILIES,
    POLICIES,
    PRODUCERS,
    source_manifest,
)


def test_split_is_disjoint_and_exact() -> None:
    assert len(CONFIRMATION_SEEDS) == 50
    assert CONFIRMATION_SEEDS[0] == 10500
    assert CONFIRMATION_SEEDS[-1] == 10549
    assert EPISODES_PER_SHARD == 36


def test_source_closure_is_complete() -> None:
    assert len(source_manifest()["files"]) >= 20


def test_asymmetric_policy_is_unchanged() -> None:
    identified = propose(_episode({7: "IDENTIFIED"}), "sequential_all_mode")
    oog = propose(
        _episode({7: "OUT_OF_GRAMMAR", 13: "OUT_OF_GRAMMAR"}),
        "sequential_all_mode",
    )
    assert identified["views_evaluated"] == 4
    assert oog["views_evaluated"] == 8


def test_public_mutation_cannot_add_truth() -> None:
    episode = _episode({7: "IDENTIFIED"})
    episode["truth_support"] = ["u_xx"]
    episode = _reseal(episode)
    try:
        propose(episode, "sequential_all_mode")
    except ValueError as error:
        assert "owner-only" in str(error)
    else:
        raise AssertionError("owner-only field was accepted")


def _summary(false_seq: int = 0) -> dict:
    rows = {}
    for producer in PRODUCERS:
        n = 1800
        values = {
            "sequential_all_mode": (1500, 1480, false_seq, 10.0, 25.0),
            "fullbank_all_mode": (1510, 1500, 0, 16.0, 49.0),
            "fixed_25_m5": (1450, 1420, 20, 1.0, 25.0),
            "fixed_49_m5": (1550, 1500, 15, 1.0, 49.0),
            "confidence_max": (1800, 300, 1500, 1.0, 7.0),
            "d_optimal": (900, 500, 300, 1.0, 25.0),
            "e_optimal": (900, 500, 300, 1.0, 25.0),
        }
        for policy, (terminal, correct, false, views, nodes) in values.items():
            rows[f"{producer}::{policy}"] = {
                "episodes": n,
                "terminal_count": terminal,
                "correct_count": correct,
                "false_promotion_count": false,
                "reference_agreement_count": 600,
                "terminal_rate": terminal / n,
                "correct_rate": correct / n,
                "false_promotion_rate": false / n,
                "mean_views_evaluated": views,
                "mean_max_nodes_acquired": nodes,
                "mean_stage_count": views / 4,
            }
    return rows


def _scores_owners() -> tuple[list[dict], dict[str, dict]]:
    scores = []
    owners = {}
    for producer in PRODUCERS:
        ordinal = 0
        for family in FAMILIES:
            for noise in (0.001, 0.005):
                for repeat in range(60):
                    key = f"{producer}-{ordinal}"
                    ordinal += 1
                    owners[key] = {"owner_episode_key": [family, 10500, "iid", noise]}
                    scores.append(
                        {
                            "episode_id": key,
                            "producer": producer,
                            "policy": "sequential_all_mode",
                            "correct_owner_diagnostic": True,
                            "terminal": True,
                        }
                    )
    return scores, owners


def test_gate_logic_accepts_passing_fixture() -> None:
    scores, owners = _scores_owners()
    gates = confirmation_gates(_summary(), scores, owners)
    assert all(row["all_scientific_checks_passed"] for row in gates.values())


def test_zero_false_gate_is_literal() -> None:
    scores, owners = _scores_owners()
    gates = confirmation_gates(_summary(false_seq=1), scores, owners)
    assert not any(row["all_scientific_checks_passed"] for row in gates.values())


def test_confirmation_is_a_split_not_a_policy() -> None:
    assert "confirmation" not in POLICIES


def test_one_producer_shard_summary_is_total() -> None:
    rows = []
    for policy in POLICIES:
        rows.append(
            {
                "producer": PRODUCERS[0],
                "policy": policy,
                "terminal": True,
                "correct_owner_diagnostic": True,
                "false_promotion_owner_diagnostic": False,
                "reference_signature_agreement": True,
                "views_evaluated": 4,
                "max_nodes_acquired": 7,
                "stage_count": 1,
            }
        )
    summary = summarize_shard(rows, PRODUCERS[0])
    assert len(summary) == len(POLICIES)


def test_cluster_statistics_use_seed_as_unit() -> None:
    scores = []
    owners = {}
    for seed in CONFIRMATION_SEEDS:
        episode_id = f"cluster-{seed}"
        owners[episode_id] = {"owner_episode_key": [FAMILIES[0], seed, "iid", 0.0]}
        scores.extend(
            [
                {
                    "episode_id": episode_id,
                    "producer": PRODUCERS[0],
                    "policy": "sequential_all_mode",
                    "correct_owner_diagnostic": True,
                    "false_promotion_owner_diagnostic": False,
                    "terminal": True,
                    "views_evaluated": 4,
                },
                {
                    "episode_id": episode_id,
                    "producer": PRODUCERS[0],
                    "policy": "fullbank_all_mode",
                    "correct_owner_diagnostic": True,
                    "false_promotion_owner_diagnostic": False,
                    "terminal": True,
                    "views_evaluated": 16,
                },
            ]
        )
    result = cluster_statistics(scores, owners, PRODUCERS[0])
    assert result["seed_cluster_count"] == 50
    assert result["clusters_with_any_false"] == 0
    expected = 1.0 - 0.05 ** (1.0 / 50.0)
    assert abs(
        result["zero_event_one_sided_95pct_upper_under_unverified_iid_cluster_model"]
        - expected
    ) < 1e-15
