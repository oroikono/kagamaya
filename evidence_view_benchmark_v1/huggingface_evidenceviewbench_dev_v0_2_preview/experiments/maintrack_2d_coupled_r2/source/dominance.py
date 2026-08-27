"""Fixed-policy exhaustion, retrospective oracle, and learning eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .contract import LEARNING_GATES, VIEW_BANK
from .environment import OwnerLabel, PublicEpisode, Score, score_decision
from .policies import FixedOpenLoopPolicy, run_policy


@dataclass(frozen=True)
class PolicyAggregate:
    policy_id: str
    correct: int
    false_promotion: int
    inconclusive: int
    mean_cost: float
    episode_count: int

    @property
    def correct_rate(self) -> float:
        return self.correct / self.episode_count

    @property
    def false_rate(self) -> float:
        return self.false_promotion / self.episode_count


def fixed_sequences(*, pair_limit: int | None = None) -> tuple[tuple[str, ...], ...]:
    singles = [(view.view_id,) for view in VIEW_BANK]
    pairs = list(combinations((view.view_id for view in VIEW_BANK), 2))
    if pair_limit is not None:
        pairs = pairs[:pair_limit]
    return tuple(singles + pairs)


def evaluate_sequence(sequence: tuple[str, ...], episodes: list[PublicEpisode],
                      labels: dict[str, OwnerLabel]) -> PolicyAggregate:
    costs: list[float] = []
    scores: list[Score] = []
    policy = FixedOpenLoopPolicy("fixed:" + "+".join(sequence), sequence)
    budget = sum(next(view.registered_cost for view in VIEW_BANK if view.view_id == view_id)
                 for view_id in sequence)
    for ordinal, episode in enumerate(episodes):
        terminal, _ = run_policy(policy, episode, budget=budget, public_seed=ordinal)
        scores.append(score_decision(terminal, labels[episode.episode_id]))
        costs.append(terminal.realized_cost)
    return PolicyAggregate(
        policy_id=policy.policy_id,
        correct=sum(score.correct for score in scores),
        false_promotion=sum(score.false_promotion for score in scores),
        inconclusive=sum(score.inconclusive for score in scores),
        mean_cost=sum(costs) / len(costs), episode_count=len(scores),
    )


def exhaustive_fixed_probe(episodes: list[PublicEpisode], labels: dict[str, OwnerLabel],
                           *, pair_limit: int | None = None) -> list[PolicyAggregate]:
    results = [evaluate_sequence(sequence, episodes, labels)
               for sequence in fixed_sequences(pair_limit=pair_limit)]
    results.sort(key=lambda item: (
        item.false_rate > LEARNING_GATES["false_promotion_rate_max"],
        -item.correct_rate, item.false_rate, item.mean_cost, item.policy_id,
    ))
    return results


def retrospective_oracle(episodes: list[PublicEpisode], labels: dict[str, OwnerLabel]
                         ) -> tuple[PolicyAggregate, dict[str, tuple[str, ...]]]:
    correct = false = inconclusive = 0
    costs: list[float] = []
    choices: dict[str, tuple[str, ...]] = {}
    for episode in episodes:
        label = labels[episode.episode_id]
        candidates = []
        for view in VIEW_BANK:
            record = episode.view_records[view.view_id]
            from .environment import TerminalDecision
            terminal = TerminalDecision(record.status, record.support, "oracle_single",
                                        (view.view_id,), record.registered_cost)
            score = score_decision(terminal, label)
            if score.correct:
                candidates.append((record.registered_cost, (view.view_id,), terminal))
        if not candidates:
            inconclusive += 1
            costs.append(0.0)
            choices[episode.episode_id] = ()
            continue
        _, sequence, terminal = min(candidates, key=lambda item: (item[0], item[1]))
        correct += 1
        costs.append(terminal.realized_cost)
        choices[episode.episode_id] = sequence
    return (PolicyAggregate("retrospective_oracle", correct, false, inconclusive,
                            sum(costs) / len(costs), len(episodes)), choices)


def learning_gate_verdict(best_fixed: PolicyAggregate, oracle: PolicyAggregate,
                          choices: dict[str, tuple[str, ...]],
                          episodes: list[PublicEpisode], labels: dict[str, OwnerLabel]) -> dict:
    advantage_pp = 100.0 * (oracle.correct_rate - best_fixed.correct_rate)
    family_totals: dict[str, list[int]] = {}
    for episode in episodes:
        family = labels[episode.episode_id].family
        family_totals.setdefault(family, [0, 0])
        family_totals[family][1] += 1
        if choices[episode.episode_id]:
            family_totals[family][0] += 1
    # A strict best-fixed family comparison is computed by the campaign report;
    # this conservative placeholder does not authorize learning by itself.
    choice_counts: dict[tuple[str, ...], int] = {}
    for sequence in choices.values():
        if sequence:
            choice_counts[sequence] = choice_counts.get(sequence, 0) + 1
    oracle_successes = max(oracle.correct, 1)
    max_share = max(choice_counts.values(), default=0) / oracle_successes
    gates = {
        "global_headroom": advantage_pp >= LEARNING_GATES["oracle_advantage_pp_min"],
        "fixed_below_oracle_coverage": best_fixed.correct_rate <= LEARNING_GATES["fixed_oracle_coverage_max"] * max(oracle.correct_rate, 1e-15),
        "sequence_diversity": len(choice_counts) >= LEARNING_GATES["unique_optimal_sequences_min"] and max_share <= LEARNING_GATES["max_oracle_success_share"],
        "absolute_false_risk": best_fixed.false_rate <= LEARNING_GATES["false_promotion_rate_max"],
        "family_headroom": False,
        "origin_headroom": False,
        "feedback_opportunity": False,
    }
    return {
        "status": "LEARNING_DESIGN_REVIEW_ELIGIBLE" if all(gates.values()) else "POLICY_LEARNING_BLOCKED_FIXED_DOMINANCE_OR_SCOPE",
        "gates": gates,
        "oracle_advantage_pp": advantage_pp,
        "unique_oracle_sequences": len(choice_counts),
        "max_oracle_sequence_share": max_share,
        "note": "family, origin, and feedback gates require campaign-level matched comparisons",
    }

