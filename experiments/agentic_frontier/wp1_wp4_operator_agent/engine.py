"""Rank bounded operator supports and fail closed on unresolved evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .config import FROZEN_CONFIG, FrozenConfig
from .manufactured import ObservedField
from .proposals import Proposal, validate_candidate_order
from .serialization import content_digest, finite_or_none
from .susano_candidates import OperatorCandidate
from .weak_form import WeakEvidence, build_weak_evidence, candidate_matrix


STATUS_VERIFIED = "VERIFIED_WITHIN_FRAGMENT"
STATUS_UNIDENTIFIED = "UNIDENTIFIED"
STATUS_REJECTED = "REJECTED"


@dataclass(frozen=True)
class MatrixDiagnostics:
    effective_rank: int
    declared_column_count: int
    nonzero_column_count: int
    normalized_singular_values: tuple[float, ...]
    normalized_gram_eigenvalues: tuple[float, ...]
    condition_number: float
    zero_columns: tuple[int, ...]

    @property
    def rank_deficient(self) -> bool:
        return self.effective_rank < self.declared_column_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "effective_rank": self.effective_rank,
            "declared_column_count": self.declared_column_count,
            "nonzero_column_count": self.nonzero_column_count,
            "normalized_singular_values": list(self.normalized_singular_values),
            "normalized_gram_eigenvalues": list(self.normalized_gram_eigenvalues),
            "condition_number": finite_or_none(self.condition_number),
            "condition_number_is_infinite": not np.isfinite(self.condition_number),
            "zero_columns": list(self.zero_columns),
            "rank_deficient": self.rank_deficient,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    fit_relative_residual: float
    selection_relative_residual: float
    selection_score: float
    least_squares_physical_rank: int
    fit_diagnostics: MatrixDiagnostics
    selection_diagnostics: MatrixDiagnostics
    covariance: tuple[tuple[float, ...], ...] | None
    standard_errors: tuple[float, ...] | None
    covariance_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "terms": list(self.terms),
            "coefficients": list(self.coefficients),
            "fit_relative_residual": self.fit_relative_residual,
            "selection_relative_residual": self.selection_relative_residual,
            "selection_score": self.selection_score,
            "least_squares_physical_rank": self.least_squares_physical_rank,
            "fit_diagnostics": self.fit_diagnostics.to_dict(),
            "selection_diagnostics": self.selection_diagnostics.to_dict(),
            "coefficient_covariance": (
                [list(row) for row in self.covariance]
                if self.covariance is not None
                else None
            ),
            "coefficient_standard_errors": (
                list(self.standard_errors) if self.standard_errors is not None else None
            ),
            "covariance_scope": self.covariance_scope,
        }


@dataclass(frozen=True)
class DiscoveryResult:
    scientific_record: dict[str, Any]
    proposal_audit: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.scientific_record["terminal_status"])

    @property
    def scientific_digest(self) -> str:
        return str(self.scientific_record["scientific_sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.scientific_record,
            "proposal_audit": self.proposal_audit,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def matrix_diagnostics(matrix: np.ndarray, config: FrozenConfig) -> MatrixDiagnostics:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    zero_columns = tuple(
        int(index) for index, value in enumerate(norms) if value <= config.zero_column_tol
    )
    keep = [index for index, value in enumerate(norms) if value > config.zero_column_tol]
    if not keep:
        return MatrixDiagnostics(
            0, matrix.shape[1], 0, (), (), float("inf"), zero_columns
        )
    normalized = matrix[:, keep] / norms[keep]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    largest = float(singular_values[0])
    rank = int(np.sum(singular_values > config.normalized_rank_rtol * largest))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if (
            rank == matrix.shape[1]
            and not zero_columns
            and singular_values[-1] > 0.0
        )
        else float("inf")
    )
    gram_eigenvalues = np.linalg.eigvalsh(normalized.T @ normalized)
    return MatrixDiagnostics(
        effective_rank=rank,
        declared_column_count=matrix.shape[1],
        nonzero_column_count=len(keep),
        normalized_singular_values=tuple(float(value) for value in singular_values),
        normalized_gram_eigenvalues=tuple(float(value) for value in gram_eigenvalues),
        condition_number=condition,
        zero_columns=zero_columns,
    )


def _effective_basis(matrix: np.ndarray, config: FrozenConfig) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=0)
    keep = norms > config.zero_column_tol
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=float)
    normalized = matrix[:, keep] / norms[keep]
    left, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
    rank = int(
        np.sum(singular_values > config.normalized_rank_rtol * singular_values[0])
    )
    return left[:, :rank]


def subspace_diagnostics(
    first: np.ndarray, second: np.ndarray, config: FrozenConfig
) -> dict[str, Any]:
    first_basis = _effective_basis(first, config)
    second_basis = _effective_basis(second, config)
    rank_first = first_basis.shape[1]
    rank_second = second_basis.shape[1]
    if rank_first == 0 and rank_second == 0:
        projector_distance = 0.0
        angles: list[float] = []
    else:
        first_projector = first_basis @ first_basis.T
        second_projector = second_basis @ second_basis.T
        projector_distance = float(
            np.linalg.norm(first_projector - second_projector, ord=2)
        )
        if min(rank_first, rank_second) > 0:
            cosines = np.linalg.svd(
                first_basis.T @ second_basis, compute_uv=False
            )
            angles = [
                float(value)
                for value in np.arccos(np.clip(cosines, -1.0, 1.0))
            ]
        else:
            angles = []
        angles.extend([float(np.pi / 2.0)] * abs(rank_first - rank_second))
    return {
        "first_rank": rank_first,
        "second_rank": rank_second,
        "principal_angles_radians": angles,
        "projector_distance_2norm": projector_distance,
    }


def _relative_residual(matrix: np.ndarray, coefficients: np.ndarray, y: np.ndarray) -> float:
    numerator = float(np.linalg.norm(matrix @ coefficients - y))
    denominator = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
    return numerator / denominator


def _conditional_covariance(
    matrix: np.ndarray,
    residual: np.ndarray,
    *,
    physical_fit_rank: int,
) -> np.ndarray | None:
    """Use exactly the physical-design singular directions retained by the fit."""
    row_count, column_count = matrix.shape
    dof = row_count - column_count
    if physical_fit_rank != column_count or dof <= 0:
        return None
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if np.any(singular_values <= 0.0):
        return None
    sigma_squared = float(residual @ residual / dof)
    return sigma_squared * (
        (right.T * (1.0 / singular_values**2)) @ right
    )


def _fit_candidate(
    candidate: OperatorCandidate,
    fit: WeakEvidence,
    selection: WeakEvidence,
    config: FrozenConfig,
) -> CandidateEvaluation:
    fit_matrix = candidate_matrix(fit, candidate.terms)
    selection_matrix = candidate_matrix(selection, candidate.terms)
    coefficients, _, physical_fit_rank, _ = np.linalg.lstsq(
        fit_matrix, fit.y, rcond=config.least_squares_rcond
    )
    fit_residual = _relative_residual(fit_matrix, coefficients, fit.y)
    selection_residual = _relative_residual(
        selection_matrix, coefficients, selection.y
    )
    fit_diagnostics = matrix_diagnostics(fit_matrix, config)
    selection_diagnostic = matrix_diagnostics(selection_matrix, config)
    score = float(
        np.log10(max(selection_residual, config.residual_floor))
        + config.complexity_penalty * candidate.complexity
    )

    residual = fit.y - fit_matrix @ coefficients
    covariance_array = _conditional_covariance(
        fit_matrix,
        residual,
        physical_fit_rank=int(physical_fit_rank),
    )
    if (
        fit_diagnostics.effective_rank == fit_matrix.shape[1]
        and covariance_array is not None
    ):
        covariance = tuple(
            tuple(float(value) for value in row) for row in covariance_array
        )
        standard_errors = tuple(
            float(value)
            for value in np.sqrt(np.maximum(np.diag(covariance_array), 0.0))
        )
        covariance_scope = (
            "diagnostic_only_fixed_design_homoskedastic_weak_rows;"
            "ignores_design_error_row_correlation_and_post_selection"
        )
    else:
        covariance = None
        standard_errors = None
        covariance_scope = (
            "NA_normalized_or_physical_rank_deficient_or_nonpositive_residual_dof"
        )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        terms=candidate.terms,
        coefficients=tuple(float(value) for value in coefficients),
        fit_relative_residual=fit_residual,
        selection_relative_residual=selection_residual,
        selection_score=score,
        least_squares_physical_rank=int(physical_fit_rank),
        fit_diagnostics=fit_diagnostics,
        selection_diagnostics=selection_diagnostic,
        covariance=covariance,
        standard_errors=standard_errors,
        covariance_scope=covariance_scope,
    )


def _make_selection_banks(
    observed: ObservedField, config: FrozenConfig
) -> tuple[WeakEvidence, WeakEvidence]:
    common = {
        "max_spatial_mode": config.max_spatial_mode,
        "max_temporal_mode": config.max_temporal_mode,
    }
    fit = build_weak_evidence(
        observed, seed=config.fit_seed, row_count=config.fit_rows, **common
    )
    selection = build_weak_evidence(
        observed,
        seed=config.selection_seed,
        row_count=config.selection_rows,
        **common,
    )
    return fit, selection


def _make_final_bank(observed: ObservedField, config: FrozenConfig) -> WeakEvidence:
    return build_weak_evidence(
        observed,
        seed=config.final_seed,
        row_count=config.final_rows,
        max_spatial_mode=config.max_spatial_mode,
        max_temporal_mode=config.max_temporal_mode,
    )


def discover_operators(
    observed: ObservedField,
    candidates: Sequence[OperatorCandidate],
    proposal: Proposal,
    *,
    config: FrozenConfig = FROZEN_CONFIG,
    manifest_sha256: str | None = None,
) -> DiscoveryResult:
    """Evaluate every candidate canonically; use the proposal only as audit data."""
    if observed.x.size != config.nx or observed.t.size != config.nt:
        raise ValueError(
            "observed grid shape does not match the frozen configuration: "
            f"got nx={observed.x.size}, nt={observed.t.size}; "
            f"expected nx={config.nx}, nt={config.nt}"
        )
    candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    proposal_order = validate_candidate_order(proposal.candidate_order, candidates)
    fit, selection = _make_selection_banks(observed, config)
    final: WeakEvidence | None = None

    evaluations = {
        candidate.candidate_id: _fit_candidate(candidate, fit, selection, config)
        for candidate in candidates
    }
    ranking = sorted(
        evaluations,
        key=lambda candidate_id: (
            evaluations[candidate_id].selection_score,
            candidate_id,
        ),
    )
    passing = [
        candidate_id
        for candidate_id in ranking
        if evaluations[candidate_id].selection_relative_residual
        < config.relative_residual_threshold
    ]

    selection_matrices = {
        candidate.candidate_id: candidate_matrix(selection, candidate.terms)
        for candidate in candidates
    }
    pairwise: dict[str, dict[str, Any]] = {}
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            key = f"{left.candidate_id}::{right.candidate_id}"
            pairwise[key] = subspace_diagnostics(
                selection_matrices[left.candidate_id],
                selection_matrices[right.candidate_id],
                config,
            )

    selected: str | None = passing[0] if passing else None
    ambiguity: list[str] = []
    terminal_reason: str
    final_record: dict[str, Any] = {
        "bank_accessed": False,
        "candidate_id": None,
        "relative_residual": None,
        "strict_threshold": config.relative_residual_threshold,
    }
    if selected is None:
        status = STATUS_REJECTED
        terminal_reason = "no_candidate_passed_selection_residual_threshold"
    else:
        selected_evaluation = evaluations[selected]
        selected_matrix = selection_matrices[selected]
        for candidate_id in passing:
            evaluation = evaluations[candidate_id]
            score_gap = evaluation.selection_score - selected_evaluation.selection_score
            if score_gap > config.near_best_score_tolerance:
                continue
            if candidate_id == selected:
                projector_distance = 0.0
            else:
                projector_distance = subspace_diagnostics(
                    selected_matrix, selection_matrices[candidate_id], config
                )["projector_distance_2norm"]
            if projector_distance <= config.projector_distance_tolerance:
                ambiguity.append(candidate_id)
        ambiguity.sort()

        diagnostics = selected_evaluation.fit_diagnostics
        if diagnostics.rank_deficient:
            status = STATUS_UNIDENTIFIED
            terminal_reason = "selected_support_rank_deficient"
        elif diagnostics.condition_number > config.condition_number_ceiling:
            status = STATUS_UNIDENTIFIED
            terminal_reason = "selected_support_ill_conditioned"
        elif len(ambiguity) > 1:
            status = STATUS_UNIDENTIFIED
            terminal_reason = "near_best_empirically_coincident_subspaces"
        else:
            # This is the first access to final-admission evidence. Candidate
            # coefficients, ranking, ambiguity, and thresholds are already
            # locked above.
            final = _make_final_bank(observed, config)
            selected_candidate = next(
                candidate for candidate in candidates if candidate.candidate_id == selected
            )
            final_matrix = candidate_matrix(final, selected_candidate.terms)
            final_residual = _relative_residual(
                final_matrix,
                np.asarray(selected_evaluation.coefficients),
                final.y,
            )
            final_record = {
                "bank_accessed": True,
                "candidate_id": selected,
                "relative_residual": final_residual,
                "strict_threshold": config.relative_residual_threshold,
            }
            if final_residual >= config.relative_residual_threshold:
                status = STATUS_REJECTED
                terminal_reason = "selected_candidate_failed_final_admission_bank"
            else:
                status = STATUS_VERIFIED
                terminal_reason = "unique_well_conditioned_candidate_passed_frozen_banks"

    scientific_body: dict[str, Any] = {
        "schema_version": "verisigs.wp1_wp4.discovery_result.v1",
        "terminal_status": status,
        "terminal_reason": terminal_reason,
        "status_scope": (
            "bounded_candidate_fragment_and_observed_trajectory;"
            "empirical_weak_tests_not_continuum_proof"
        ),
        "selected_candidate_id": selected,
        "scientifically_identified_candidate_id": (
            selected if status == STATUS_VERIFIED else None
        ),
        "ambiguity_set": ambiguity,
        "ranking": ranking,
        "evaluations": {
            candidate_id: evaluations[candidate_id].to_dict()
            for candidate_id in sorted(evaluations)
        },
        "pairwise_selection_subspaces": pairwise,
        "final_admission": final_record,
        "bank_sha256": {
            "fit": fit.bank_sha256,
            "selection": selection.bank_sha256,
            "final_admission": final.bank_sha256 if final is not None else None,
        },
        "observed_field_sha256": observed.digest(),
        "candidate_manifest_sha256": manifest_sha256,
        "candidate_representation_sha256": {
            candidate.candidate_id: candidate.representation_digest
            for candidate in candidates
        },
        "frozen_config": config.to_dict(),
        "uncertainty_warning": (
            "coefficient covariance is conditional and diagnostic only; "
            "it is not calibrated structural uncertainty"
        ),
    }
    scientific_body["scientific_sha256"] = content_digest(scientific_body)
    proposal_audit = {
        **proposal.to_dict(),
        "candidate_order": list(proposal_order),
        "authority_boundary": (
            "proposal order excluded from scientific digest and terminal decision"
        ),
    }
    return DiscoveryResult(scientific_body, proposal_audit)
