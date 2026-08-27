"""Strict proposal-only policies for ordering the frozen candidate IDs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .serialization import content_digest
from .susano_candidates import OperatorCandidate


class ProposalError(ValueError):
    """Raised when a policy leaves the closed proposal language."""


class GenerationBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Proposal:
    policy: str
    candidate_order: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "candidate_order": list(self.candidate_order),
            "metadata": self.metadata,
        }


def validate_candidate_order(
    order: Sequence[str], candidates: Sequence[OperatorCandidate]
) -> tuple[str, ...]:
    expected = {candidate.candidate_id for candidate in candidates}
    actual = list(order)
    if len(actual) != len(expected):
        raise ProposalError("proposal must contain every frozen candidate exactly once")
    if len(set(actual)) != len(actual):
        raise ProposalError("proposal contains duplicate candidate IDs")
    if set(actual) != expected:
        raise ProposalError(
            f"proposal candidate mismatch: missing={sorted(expected - set(actual))}, "
            f"unknown={sorted(set(actual) - expected)}"
        )
    return tuple(actual)


def static_proposal(candidates: Sequence[OperatorCandidate]) -> Proposal:
    order = tuple(candidate.candidate_id for candidate in candidates)
    return Proposal("static", order, {"scientific_authority": "none"})


def random_proposal(
    candidates: Sequence[OperatorCandidate], *, seed: int
) -> Proposal:
    order = [candidate.candidate_id for candidate in candidates]
    random.Random(seed).shuffle(order)
    return Proposal(
        "random",
        validate_candidate_order(order, candidates),
        {"seed": seed, "scientific_authority": "none"},
    )


def evolutionary_proposal(
    candidates: Sequence[OperatorCandidate],
    *,
    objective: Callable[[tuple[str, ...]], float],
    seed: int,
    budget: int = 64,
) -> Proposal:
    """Run a tiny hash-chained swap-mutation archive over permutations.

    The caller owns the proposal objective. The verifier never reads its
    fitness values, and every frozen candidate is still scientifically
    evaluated, so this mechanism can prioritize but cannot admit an operator.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    parent = list(candidate.candidate_id for candidate in candidates)
    rng.shuffle(parent)
    parent_tuple = validate_candidate_order(parent, candidates)
    best_score = float(objective(parent_tuple))
    ledger: list[dict[str, Any]] = []
    previous_hash = "GENESIS"
    for generation in range(budget):
        child = list(parent_tuple)
        i, j = rng.sample(range(len(child)), 2)
        child[i], child[j] = child[j], child[i]
        child_tuple = validate_candidate_order(child, candidates)
        score = float(objective(child_tuple))
        accepted = (score, tuple(child_tuple)) > (best_score, tuple(parent_tuple))
        if accepted:
            parent_tuple, best_score = child_tuple, score
        row = {
            "generation": generation,
            "swap": [i, j],
            "candidate_order": list(child_tuple),
            "proposal_fitness": score,
            "accepted": accepted,
            "previous_sha256": previous_hash,
        }
        previous_hash = content_digest(row)
        row["entry_sha256"] = previous_hash
        ledger.append(row)
    return Proposal(
        "evolutionary_archive",
        parent_tuple,
        {
            "seed": seed,
            "budget": budget,
            "best_proposal_fitness": best_score,
            "ledger": ledger,
            "ledger_head_sha256": previous_hash,
            "scientific_authority": "none",
        },
    )


def qwen_proposal(
    candidates: Sequence[OperatorCandidate],
    *,
    backend: GenerationBackend,
    model_identity: dict[str, Any] | None = None,
) -> Proposal:
    manifest = [
        {
            "candidate_id": candidate.candidate_id,
            "canonical_string": candidate.canonical_string,
            "terms": list(candidate.terms),
            "attributes": candidate.susano_attributes,
        }
        for candidate in candidates
    ]
    prompt = (
        "Order every candidate ID once as a proposal priority. You cannot "
        "accept/reject candidates or provide coefficients. Return exactly JSON "
        'with schema {"candidate_order": ["id", "..."]}. Candidates: '
        + json.dumps(manifest, sort_keys=True)
    )
    raw = backend.generate(prompt)
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProposalError("Qwen proposal is not a JSON object") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"candidate_order"}:
        raise ProposalError("Qwen proposal must contain only candidate_order")
    if not isinstance(decoded["candidate_order"], list) or not all(
        isinstance(item, str) for item in decoded["candidate_order"]
    ):
        raise ProposalError("Qwen candidate_order must be a list of strings")
    order = validate_candidate_order(decoded["candidate_order"], candidates)
    metadata = {
        "model_identity": model_identity or {},
        "prompt_sha256": content_digest({"prompt": prompt}),
        "response_sha256": content_digest({"response": raw}),
        "scientific_authority": "none",
    }
    backend_metadata = getattr(backend, "last_generation_metadata", None)
    if isinstance(backend_metadata, dict):
        metadata["generation"] = dict(backend_metadata)
    return Proposal("qwen", order, metadata)
