"""Prospectively frozen interfaces for the full evidence-control campaign."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "maintrack.evidence-control.v3r2-full"
DOMAIN_LENGTH = 2.0 * 3.141592653589793

SCALAR_ATOMS = (
    "u", "u2", "u3", "ux", "uy", "uxx", "uyy", "uxy",
    "biharm", "uux", "uuy",
)
COUPLED_ATOMS = (
    "u", "v", "u2", "v2", "uv", "ux", "uy", "vx", "vy",
    "uxx", "uyy", "vxx", "vyy", "uxy", "vxy",
)

SPATIAL_NODES = (12, 18, 24, 36)
TEMPORAL_SAMPLES = (7, 13, 25)
TEST_MODE_COUNTS = (2, 4)
WINDOWS = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0))


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    spatial_nodes: int
    temporal_samples: int
    test_mode_count: int
    window_start: float
    window_end: float
    filter_id: str = "none"

    @property
    def registered_cost(self) -> float:
        return float(self.spatial_nodes**2 * self.temporal_samples
                     + 0.1 * self.test_mode_count * self.spatial_nodes**2)


def view_bank() -> tuple[ViewSpec, ...]:
    views: list[ViewSpec] = []
    ordinal = 0
    for nodes in SPATIAL_NODES:
        for times in TEMPORAL_SAMPLES:
            for modes in TEST_MODE_COUNTS:
                for start, end in WINDOWS:
                    views.append(ViewSpec(
                        f"V{ordinal:03d}", nodes, times, modes, start, end))
                    ordinal += 1
    return tuple(views)


VIEW_BANK = view_bank()


@dataclass(frozen=True)
class TerminalThresholds:
    identified_residual_max: float
    outside_library_residual_min: float
    condition_max: float
    coefficient_floor: float
    support_gap_min: float


DEFAULT_DEVELOPMENT_THRESHOLDS = TerminalThresholds(
    identified_residual_max=0.035,
    outside_library_residual_min=0.16,
    condition_max=1.0e8,
    coefficient_floor=2.0e-4,
    support_gap_min=1.0e-4,
)

# Outcome isolation: no integer occurs in more than one split.
TRAIN_SEEDS = tuple(range(61000, 61040))
LOFO_FIT_SEEDS = tuple(range(61000, 61030))
LOFO_CALIBRATION_SEEDS = tuple(range(61030, 61040))
CALIBRATION_SEEDS = tuple(range(62000, 62020))
CONFIRMATION_SEEDS = tuple(range(63000, 63100))

# These families are absent from every learner fit and calibration decision.
# They enter only the sealed confirmation join.
NEVER_SEEN_FAMILIES = (
    "fourth_order",
    "nonlinear_coupling",
    "fractional_diffusion_1p5",
    "nonlocal_convolution",
)

NOISE_CELLS = (
    ("iid", 0.0),
    ("iid", 0.005),
    ("iid", 0.02),
    ("correlated", 0.005),
    ("correlated", 0.02),
)

ORIGINS = ("spectral", "fd")
DIAGNOSE_VIEW_ID = "V051"


@dataclass(frozen=True)
class ControlAction:
    """One action in the immutable two-step public control interface."""

    kind: str
    view_id: str | None = None

    def validate(self, view_ids: frozenset[str]) -> None:
        if self.kind not in {"DIAGNOSE", "SUBMIT", "ABSTAIN"}:
            raise ValueError("unknown control action")
        if self.kind in {"DIAGNOSE", "SUBMIT"} and self.view_id not in view_ids:
            raise ValueError("evidence action must name a frozen view")
        if self.kind == "ABSTAIN" and self.view_id is not None:
            raise ValueError("ABSTAIN must set view_id to null")


@dataclass(frozen=True)
class StatisticalGates:
    risk_ceiling: float = 0.01
    risk_difference_margin: float = 0.01
    correct_advantage_min: float = 0.05
    heldout_correct_advantage_min: float = 0.02
    heldout_families_positive_min: int = 2
    calibration_oracle_headroom_min: float = 0.05
    calibration_unique_submit_views_min: int = 3
    confirmation_seed_clusters_min: int = 100
    bootstrap_replicates: int = 10000
    calibration_bootstrap_replicates: int = 4000
    confidence_level: float = 0.95
    cost_difference_margin: float = 1.0e-9
    lofo_risk_ceiling: float = 0.01
    lofo_positive_fraction_min: float = 0.5


GATES = StatisticalGates()

POLICY_FEATURES = (
    "diag_residual",
    "diag_score_gap",
    "diag_log_condition",
    "diag_minimum_coefficient",
    "diag_ambiguity_size",
    "diag_status_identified",
    "diag_status_outside",
    "candidate_spatial_nodes",
    "candidate_temporal_samples",
    "candidate_test_modes",
    "candidate_window_start",
    "candidate_window_end",
    "candidate_registered_cost",
    "same_as_diagnose",
)

MODEL_CONFIG = {
    "class": "sklearn.ensemble.HistGradientBoostingClassifier",
    "learning_rate": 0.06,
    "max_iter": 240,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 20260823,
}

RISK_THRESHOLD_GRID = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
UTILITY_FALSE_PENALTY = 4.0
COST_PENALTY = 1.0e-6


def validate_contract() -> None:
    split_sets = [set(TRAIN_SEEDS), set(CALIBRATION_SEEDS), set(CONFIRMATION_SEEDS)]
    for index, left in enumerate(split_sets):
        for right in split_sets[index + 1:]:
            if left & right:
                raise AssertionError("prospective seed splits overlap")
    if len(NOISE_CELLS) != len(set(NOISE_CELLS)):
        raise AssertionError("noise cells contain duplicates")
    if GATES.confirmation_seed_clusters_min != len(CONFIRMATION_SEEDS):
        raise AssertionError("confirmation cluster gate does not match frozen split")
    if set(LOFO_FIT_SEEDS) | set(LOFO_CALIBRATION_SEEDS) != set(TRAIN_SEEDS):
        raise AssertionError("LOFO inner split does not partition training")


validate_contract()
