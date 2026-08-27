"""Immutable scientific and statistical contract for the prospective replication."""

from __future__ import annotations

from dataclasses import dataclass
import math


SCHEMA_VERSION = "maintrack.spotlight-2d-replication.v1"
DOMAIN_LENGTH = 2.0 * math.pi

SCALAR_ATOMS = (
    "u", "u2", "u3", "ux", "uy", "uxx", "uyy", "uxy",
    "biharm", "uux", "uuy",
)
COUPLED_ATOMS = (
    "u", "v", "u2", "v2", "uv", "ux", "uy", "vx", "vy",
    "uxx", "uyy", "vxx", "vyy", "uxy", "vxy",
)
TERMINAL_STATUSES = ("IDENTIFIED", "OUTSIDE_LIBRARY", "INCONCLUSIVE")

# The public action table is deliberately isomorphic to the frozen 1-D paper:
# three temporal quadrature resolutions crossed with four temporal test counts.
# Spatial resolution and window are fixed so this experiment changes dimension
# and data origin without adding a new selector axis.
SPATIAL_NODES = 36
TEMPORAL_SAMPLES = (7, 13, 25)
TEST_MODE_COUNTS = (2, 3, 4, 5)
WINDOW = (0.0, 1.0)
REFERENCE_TEMPORAL_SAMPLES = 49

# These ranges were unused elsewhere in the repository when this contract was
# written. Design is structural only; the two scientific splits are disjoint.
DESIGN_SEEDS = tuple(range(160000, 160004))
MECHANISM_SEEDS = tuple(range(161000, 161020))
CONFIRMATION_SEEDS = tuple(range(162000, 162100))

# One zero-noise cell plus two levels under each nonzero noise model. There is
# no duplicated correlated-zero row.
NOISE_CELLS = (
    ("iid", 0.0),
    ("iid", 0.005),
    ("iid", 0.02),
    ("correlated", 0.005),
    ("correlated", 0.02),
)


@dataclass(frozen=True)
class ViewSpec:
    """One selectable 2-D weak-form representation."""

    view_id: str
    spatial_nodes: int
    temporal_samples: int
    test_mode_count: int
    window_start: float = WINDOW[0]
    window_end: float = WINDOW[1]
    filter_id: str = "none"

    @property
    def registered_cost(self) -> float:
        sample_cost = self.spatial_nodes**2 * self.temporal_samples
        test_cost = 0.1 * self.test_mode_count * self.spatial_nodes**2
        return float(sample_cost + test_cost)


def view_bank() -> tuple[ViewSpec, ...]:
    return tuple(
        ViewSpec(
            view_id=f"A{index:02d}",
            spatial_nodes=SPATIAL_NODES,
            temporal_samples=nt,
            test_mode_count=modes,
        )
        for index, (nt, modes) in enumerate(
            (nt, modes)
            for nt in TEMPORAL_SAMPLES
            for modes in TEST_MODE_COUNTS
        )
    )


VIEW_BANK = view_bank()
VIEW_BY_ID = {view.view_id: view for view in VIEW_BANK}
VIEW_BY_CONFIG = {
    (view.temporal_samples, view.test_mode_count): view for view in VIEW_BANK
}
REFERENCE_VIEW = ViewSpec(
    view_id="REFERENCE_49_M5",
    spatial_nodes=SPATIAL_NODES,
    temporal_samples=REFERENCE_TEMPORAL_SAMPLES,
    test_mode_count=5,
)


@dataclass(frozen=True)
class TerminalThresholds:
    identified_residual_max: float
    outside_library_residual_min: float
    condition_max: float
    coefficient_floor: float
    support_gap_min: float


# Byte-for-byte scientific thresholds from the source 2-D authority.
DEFAULT_DEVELOPMENT_THRESHOLDS = TerminalThresholds(
    identified_residual_max=0.035,
    outside_library_residual_min=0.16,
    condition_max=1.0e8,
    coefficient_floor=2.0e-4,
    support_gap_min=1.0e-4,
)

MARGIN_CLIP = 4.0


def _finite(value: float, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def adapted_public_gate_margin(record) -> float:
    """Gate-normalized 2-D sensitivity adapter of the public-margin score.

    The functional form is unchanged: the public score rewards a decisive
    terminal, then adds clipped slack to the evaluator's own gates. Only the
    numerical scales are replaced by the already-frozen 2-D thresholds. This
    is a prospective semantic adapter, not a byte-identical score claim.
    """

    thresholds = DEFAULT_DEVELOPMENT_THRESHOLDS
    status = str(record.status)
    residual = _finite(record.residual, math.inf)
    gap = _finite(record.score_gap, -math.inf)
    condition = _finite(record.condition, math.inf)
    minimum = _finite(record.minimum_coefficient, -math.inf)
    identified = min(
        (thresholds.identified_residual_max - residual)
        / thresholds.identified_residual_max,
        (gap - thresholds.support_gap_min) / thresholds.support_gap_min,
        (thresholds.condition_max - condition) / thresholds.condition_max,
        (minimum - thresholds.coefficient_floor)
        / thresholds.coefficient_floor,
    )
    omitted = (
        residual - thresholds.outside_library_residual_min
    ) / thresholds.outside_library_residual_min
    if status == "IDENTIFIED":
        margin = identified
    elif status == "OUTSIDE_LIBRARY":
        margin = omitted
    else:
        margin = max(identified, omitted)
    if not math.isfinite(margin):
        margin = -MARGIN_CLIP
    return float(max(-MARGIN_CLIP, min(MARGIN_CLIP, margin)))


def adapted_public_objective(record) -> float:
    """Secondary 2-D-normalized score; never the primary replication score."""

    decisive = record.status in {"IDENTIFIED", "OUTSIDE_LIBRARY"}
    view = VIEW_BY_ID[record.view_id]
    return float(
        10.0 * int(decisive)
        + adapted_public_gate_margin(record)
        - 1.0e-3 * view.temporal_samples / REFERENCE_TEMPORAL_SAMPLES
    )


def public_gate_margin(record) -> float:
    """Literal frozen 1-D margin applied to the 2-D public diagnostics.

    This primary portability test changes only the spelling of the omission
    status.  Its numerical constants and omission of the coefficient-floor
    diagnostic are byte-level values from the published 1-D score contract.
    """

    status = str(record.status)
    residual = _finite(record.residual, math.inf)
    gap = _finite(record.score_gap, -math.inf)
    condition = _finite(record.condition, math.inf)
    identified = min(
        (0.05 - residual) / 0.05,
        (gap - 0.006) / 0.006,
        (12.0 - condition) / 12.0,
    )
    omitted = (residual - 0.20) / 0.20
    if status == "IDENTIFIED":
        margin = identified
    elif status == "OUTSIDE_LIBRARY":
        margin = omitted
    else:
        margin = max(identified, omitted)
    if not math.isfinite(margin):
        margin = -MARGIN_CLIP
    return float(max(-MARGIN_CLIP, min(MARGIN_CLIP, margin)))


def public_objective(record) -> float:
    """Literal frozen score-max attack objective on the 12-action table."""

    decisive = record.status in {"IDENTIFIED", "OUTSIDE_LIBRARY"}
    view = VIEW_BY_ID[record.view_id]
    return float(
        10.0 * int(decisive)
        + public_gate_margin(record)
        - 1.0e-3 * view.temporal_samples / REFERENCE_TEMPORAL_SAMPLES
    )


PRIMARY_GATES = {
    "guard_false_promotion_rate_max": 0.01,
    "guard_coverage_min": 0.75,
    "guard_goodhart_event_rate_max": 0.01,
    "guard_false_no_greater_than_fixed_fine": True,
    "guard_coverage_delta_vs_fixed_fine_min": -0.05,
    "guard_false_reduction_vs_attack_pp_min": 50.0,
    "origins_required": 2,
    "omitted_mechanisms_required": 3,
    "confirmation_seed_clusters_required": 100,
}

# These owner-side validity gates are evaluated on the nonselectable 49-point,
# five-mode reference at zero noise.  They prevent an aliased omitted operator
# or an invalid finite authority from being marketed as attack/defence evidence.
INTERNAL_AUTHORITY_GATES = {
    "noiseless_reference_correct_rate_min": 0.95,
    "noiseless_reference_false_promotion_max": 0,
    "noiseless_omitted_reference_out_rate_min": 1.0,
}


def validate_contract() -> None:
    if len(VIEW_BANK) != 12 or len(VIEW_BY_ID) != 12:
        raise AssertionError("the replicated public action table must have 12 views")
    if VIEW_BY_CONFIG[(7, 5)].view_id != "A03":
        raise AssertionError("A03 must remain the 7-point, 5-test action")
    if VIEW_BY_CONFIG[(25, 2)].view_id != "A08":
        raise AssertionError("A08 must remain the 25-point, 2-test action")
    splits = [set(DESIGN_SEEDS), set(MECHANISM_SEEDS), set(CONFIRMATION_SEEDS)]
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            if left & right:
                raise AssertionError("prospective seed splits overlap")
    if len(set(NOISE_CELLS)) != len(NOISE_CELLS):
        raise AssertionError("noise cells must be unique")


validate_contract()
