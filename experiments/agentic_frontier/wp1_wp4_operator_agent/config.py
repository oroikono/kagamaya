"""Frozen numerical configuration for the first bounded experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FrozenConfig:
    nx: int = 128
    nt: int = 81
    fit_rows: int = 48
    selection_rows: int = 48
    final_rows: int = 48
    fit_seed: int = 1729
    selection_seed: int = 2718
    final_seed: int = 31415
    max_spatial_mode: int = 4
    max_temporal_mode: int = 4
    least_squares_rcond: float = 1.0e-10
    normalized_rank_rtol: float = 1.0e-8
    zero_column_tol: float = 1.0e-14
    relative_residual_threshold: float = 1.0e-7
    condition_number_ceiling: float = 1.0e6
    complexity_penalty: float = 0.02
    residual_floor: float = 1.0e-12
    near_best_score_tolerance: float = 0.01
    projector_distance_tolerance: float = 1.0e-3

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


FROZEN_CONFIG = FrozenConfig()
