"""Immutable scientific interfaces and prospective split declarations."""

from __future__ import annotations

from dataclasses import dataclass


SCHEMA_VERSION = "maintrack.2d-coupled-dominance.v2"
DOMAIN_LENGTH = 2.0 * 3.141592653589793

SCALAR_ATOMS = (
    "u", "u2", "u3", "ux", "uy", "uxx", "uyy", "uxy",
    "biharm", "uux", "uuy",
)

COUPLED_ATOMS = (
    "u", "v", "u2", "v2", "uv", "ux", "uy", "vx", "vy",
    "uxx", "uyy", "vxx", "vyy", "uxy", "vxy",
)

TERMINAL_STATUSES = ("IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE")

SPATIAL_NODES = (12, 18, 24, 36)
TEMPORAL_SAMPLES = (7, 13, 25)
TEST_MODE_COUNTS = (2, 4)
WINDOWS = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0))

DESIGN_SEEDS = tuple(range(40100, 40104))
CALIBRATION_SEEDS = tuple(range(40200, 40220))
DEVELOPMENT_SEEDS = tuple(range(40300, 40320))
CONFIRMATION_SEEDS = tuple(range(50000, 50100))


@dataclass(frozen=True)
class ViewSpec:
    """One public evidence acquisition, independent of owner truth."""

    view_id: str
    spatial_nodes: int
    temporal_samples: int
    test_mode_count: int
    window_start: float
    window_end: float
    filter_id: str = "none"

    @property
    def registered_cost(self) -> float:
        sample_cost = self.spatial_nodes**2 * self.temporal_samples
        test_cost = 0.1 * self.test_mode_count * self.spatial_nodes**2
        return float(sample_cost + test_cost)


def view_bank() -> tuple[ViewSpec, ...]:
    views: list[ViewSpec] = []
    ordinal = 0
    for n in SPATIAL_NODES:
        for nt in TEMPORAL_SAMPLES:
            for modes in TEST_MODE_COUNTS:
                for start, end in WINDOWS:
                    views.append(ViewSpec(
                        view_id=f"V{ordinal:03d}",
                        spatial_nodes=n,
                        temporal_samples=nt,
                        test_mode_count=modes,
                        window_start=start,
                        window_end=end,
                    ))
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


LEARNING_GATES = {
    "oracle_advantage_pp_min": 5.0,
    "family_headroom_pp_min": 2.0,
    "families_with_headroom_min": 4,
    "origin_headroom_pp_min": 2.0,
    "unique_optimal_sequences_min": 3,
    "max_oracle_success_share": 0.80,
    "replanning_rate_min": 0.10,
    "second_action_gain_pp_min": 5.0,
    "fixed_oracle_coverage_max": 0.95,
    "false_promotion_rate_max": 0.01,
}


def validate_split_disjointness() -> None:
    split_sets = [set(DESIGN_SEEDS), set(CALIBRATION_SEEDS),
                  set(DEVELOPMENT_SEEDS), set(CONFIRMATION_SEEDS)]
    for i, left in enumerate(split_sets):
        for right in split_sets[i + 1:]:
            if left & right:
                raise AssertionError("seed splits overlap")


validate_split_disjointness()
