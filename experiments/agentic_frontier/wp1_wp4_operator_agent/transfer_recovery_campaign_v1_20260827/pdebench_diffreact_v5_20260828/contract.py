"""Prospective confirmation contract for the PDEBench transfer."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
import hashlib
import json

SCHEMA = "pdebench-diffreact-transfer-v5"
SOURCE_REPOSITORY = "https://github.com/pdebench/PDEBench"
SOURCE_COMMIT = "4ff3e3a4aa1561721b5571fa3a048a0a463e0568"
DATASET_DOI = "10.18419/darus-2986"
DATAFILE_ID = 133017
FILENAME = "2D_diff-react_NA_NA.h5"
FILE_BYTES = 13_243_560_688
FILE_MD5 = "b8d0b86064193195ddc30c33be5dc949"
LICENSE = "CC-BY-4.0"

DU, DV, K = 1.0e-3, 5.0e-3, 5.0e-3
SPATIAL_NODES = 128
PUBLISHER_FINE_NODES = 512
PUBLISHER_DECIMATION_STRIDE = 4
PUBLISHER_FIRST_FINE_INDEX = 1
RAW_TIME_POINTS = 101
TEMPORAL_SAMPLES = (7, 13, 25)
TEST_MODE_COUNTS = (2, 3, 4, 5)
REFERENCE_TEMPORAL_SAMPLES = 101
REFERENCE_TEST_MODE_COUNT = 5
REFINEMENT_TEMPORAL_SAMPLES = (7, 13, 25, 49)
NOISE_LEVELS = (0.0, 0.001, 0.005)
DEVELOPMENT_IDS = tuple(range(0, 20))
AUDIT_IDS = tuple(range(20, 40))
SEALED_CONFIRMATION_IDS = tuple(range(40, 100))

UPSTREAM_V4_CONTRACT_SHA256 = "30c20d6894dc6d7af90aeefe1a8fa765fda311bdce171ced57aeb4e5b1b46867"
UPSTREAM_V4_FREEZE_SHA256 = "b952de48a9f6d1abcb7fd58cfd53604258bbbb38621058ced2e92710288f0ed9"
UPSTREAM_DEVELOPMENT_SHA256 = "00c3833387af3c164f73f7e6ee995fe7a26e7fd5e6c1fc22788ca5783484d353"
UPSTREAM_AUDIT_SHA256 = "c7384d42797d0ef251108a523217e319ae2f935008e765c7c68bc02c15a4fd32"

SELECTED_THRESHOLDS = {
    "identified_residual_max": 0.025,
    "outside_library_residual_min": 0.1,
    "condition_max": 1.0e4,
    "support_gap_min": 0.0,
    "coefficient_floor": 0.0,
}

MARGIN_CLIP = 4.0
DECISIVE_REWARD = 10.0
COST_WEIGHT = 1.0e-3
POLICIES = ("score_max", "fixed_fine", "all_mode_agreement")

ATOMS = {
    "u": ("Du_lap_u", "u", "neg_u3", "neg_k", "neg_v"),
    "v": ("Dv_lap_v", "u", "neg_v"),
}
CONTRACTS = {
    "full_u": ("u", ATOMS["u"], "in_library"),
    "full_v": ("v", ATOMS["v"], "in_library"),
    "omit_v_diffusion": ("v", tuple(a for a in ATOMS["v"] if a != "Dv_lap_v"), "outside_library"),
    "omit_u_to_v_coupling": ("v", tuple(a for a in ATOMS["v"] if a != "u"), "outside_library"),
    "omit_u_local_source_block": (
        "u", tuple(a for a in ATOMS["u"] if a not in {"u", "neg_k"}), "outside_library"
    ),
}

@dataclass(frozen=True, order=True)
class Thresholds:
    identified_residual_max: float
    outside_library_residual_min: float
    condition_max: float
    support_gap_min: float
    coefficient_floor: float

THRESHOLD_GRID = tuple(
    Thresholds(*values)
    for values in product(
        (0.025, 0.05, 0.075, 0.1, 0.15),
        (0.1, 0.15, 0.2, 0.3, 0.4),
        (1.0e4, 1.0e6, 1.0e8),
        (0.0, 1.0e-6, 1.0e-4),
        (0.0, 2.0e-4, 0.01),
    )
)

AUTHORITY_GATES = {
    "complete_correct_rate_min": 0.95,
    "complete_false_promotion_max": 0,
    "each_omission_correct_rate_min": 0.90,
}

def canonical_contract() -> dict:
    return {
        "schema": SCHEMA,
        "source": {
            "repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
            "dataset_doi": DATASET_DOI, "datafile_id": DATAFILE_ID,
            "filename": FILENAME, "bytes": FILE_BYTES, "md5": FILE_MD5,
            "license": LICENSE,
        },
        "equation": {"Du": DU, "Dv": DV, "k": K, "boundary": "homogeneous Neumann"},
        "publisher_grid": {
            "configured_domain": (-1.0, 1.0),
            "fine_nodes": PUBLISHER_FINE_NODES,
            "stored_nodes": SPATIAL_NODES,
            "first_fine_index": PUBLISHER_FIRST_FINE_INDEX,
            "decimation_stride": PUBLISHER_DECIMATION_STRIDE,
            "stored_axis_formula": "-1 + (8*j+3)/512, j=0,...,127",
            "quadrature_spacing": 2.0 / SPATIAL_NODES,
        },
        "views": {
            "temporal_samples": TEMPORAL_SAMPLES,
            "test_mode_counts": TEST_MODE_COUNTS,
            "reference": (REFERENCE_TEMPORAL_SAMPLES, REFERENCE_TEST_MODE_COUNT),
            "refinement_temporal_samples": REFINEMENT_TEMPORAL_SAMPLES,
        },
        "splits": {"development": DEVELOPMENT_IDS, "audit": AUDIT_IDS,
                   "sealed_confirmation": SEALED_CONFIRMATION_IDS},
        "noise_levels": NOISE_LEVELS,
        "contracts": CONTRACTS,
        "threshold_grid": [asdict(t) for t in THRESHOLD_GRID],
        "authority_gates": AUTHORITY_GATES,
        "upstream_authority": {
            "v4_contract_sha256": UPSTREAM_V4_CONTRACT_SHA256,
            "v4_freeze_sha256": UPSTREAM_V4_FREEZE_SHA256,
            "development_result_sha256": UPSTREAM_DEVELOPMENT_SHA256,
            "audit_result_sha256": UPSTREAM_AUDIT_SHA256,
            "selected_thresholds": SELECTED_THRESHOLDS,
        },
        "policies": {
            "names": POLICIES,
            "public_score": {
                "decisive_reward": DECISIVE_REWARD,
                "gate_margin_clip": MARGIN_CLIP,
                "normalized_cost_weight": COST_WEIGHT,
                "definition": (
                    "10*I[definitive] + threshold-normalized clipped gate margin "
                    "- 0.001*view_cost/max_selectable_view_cost"
                ),
            },
            "score_max": "canonical argmax of public score over all twelve views",
            "fixed_fine": "submit T025_M5",
            "all_mode_agreement": (
                "For each temporal count, require m=2,3,4,5 to share one exact "
                "definitive (status,support); choose the eligible group with maximal "
                "minimum public score, then lower summed cost, then lower temporal count; "
                "emit its m=5 witness, otherwise INCONCLUSIVE."
            ),
        },
        "confirmation": {
            "source_ids": SEALED_CONFIRMATION_IDS,
            "one_shard_per_source_id": True,
            "expected_shards": len(SEALED_CONFIRMATION_IDS),
            "expected_episodes": len(SEALED_CONFIRMATION_IDS) * len(NOISE_LEVELS) * len(CONTRACTS),
            "expected_view_rows": (
                len(SEALED_CONFIRMATION_IDS) * len(NOISE_LEVELS) * len(CONTRACTS)
                * len(TEMPORAL_SAMPLES) * len(TEST_MODE_COUNTS)
            ),
        },
        "threshold_selection": (
            "Among development-feasible thresholds, maximize the minimum per-omission "
            "correct rate, then complete correct rate, then total decisive coverage; "
            "break ties lexicographically by Thresholds field order. Audit is not used "
            "for selection."
        ),
    }

CONTRACT_SHA256 = hashlib.sha256(
    json.dumps(canonical_contract(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

def validate() -> None:
    assert len(TEMPORAL_SAMPLES) * len(TEST_MODE_COUNTS) == 12
    assert not (set(DEVELOPMENT_IDS) & set(AUDIT_IDS))
    assert not (set(DEVELOPMENT_IDS) & set(SEALED_CONFIRMATION_IDS))
    assert not (set(AUDIT_IDS) & set(SEALED_CONFIRMATION_IDS))
    assert len(CONTRACTS) == 5
    assert len(THRESHOLD_GRID) == 5 * 5 * 3 * 3 * 3
    assert SELECTED_THRESHOLDS == asdict(Thresholds(0.025, 0.1, 1.0e4, 0.0, 0.0))
    assert len(SEALED_CONFIRMATION_IDS) == 60
