"""Matched-outcome development campaign for v4.1 active weak-form discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .policies import DEPLOYABLE_POLICIES, choose_action, independent_policy_seed
from .simulator import (
    BASELINE_ACTION,
    Family,
    SimulationConfig,
    TRUE_SUPPORTS,
    action_menu,
    simulate,
)
from .weakform import DiscoveryConfig, discover


@dataclass(frozen=True)
class CampaignConfig:
    seeds: tuple[int, ...]
    relative_noises: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002)
    noise_models: tuple[str, ...] = ("iid", "correlated")
    families: tuple[str, ...] = tuple(item.value for item in Family)
    public_action_seed: int = 8675309
    policy_seed_base: int = 20260812
    split: str = "development"


def _support_list(value: tuple[str, ...] | None) -> list[str] | None:
    return None if value is None else list(value)


def _decision_record(result: object) -> dict[str, object]:
    return {
        "status": result.status,
        "support": _support_list(result.support),
        "best_residual": result.best_residual,
        "score_gap": result.score_gap,
        "ambiguity_set": [list(item) for item in result.ambiguity_set],
        "best_condition": result.fits[0].condition if result.fits else None,
    }


def run_campaign(
    campaign: CampaignConfig,
    simulation: SimulationConfig | None = None,
    discovery: DiscoveryConfig | None = None,
) -> dict[str, object]:
    sim_cfg = simulation or SimulationConfig()
    disc_cfg = discovery or DiscoveryConfig()
    rows: list[dict[str, object]] = []
    for family_index, family_name in enumerate(campaign.families):
        family = Family(family_name)
        for seed_index, seed in enumerate(campaign.seeds):
            for noise_index, noise_model in enumerate(campaign.noise_models):
                for level_index, noise in enumerate(campaign.relative_noises):
                    episode_ordinal = (
                        (
                            (family_index * len(campaign.seeds) + seed_index)
                            * len(campaign.noise_models)
                            + noise_index
                        )
                        * len(campaign.relative_noises)
                        + level_index
                    )
                    baseline, hidden = simulate(
                        family,
                        BASELINE_ACTION,
                        episode_seed=seed,
                        public_action_seed=campaign.public_action_seed,
                        relative_noise=noise,
                        noise_model=noise_model,
                        config=sim_cfg,
                    )
                    round1 = discover((baseline,), disc_cfg)
                    outcomes = {
                        action.name: simulate(
                            family,
                            action,
                            episode_seed=seed,
                            public_action_seed=campaign.public_action_seed,
                            relative_noise=noise,
                            noise_model=noise_model,
                            config=sim_cfg,
                        )[0]
                        for action in action_menu()
                    }
                    decisions = {
                        name: discover((baseline, trajectory), disc_cfg)
                        for name, trajectory in outcomes.items()
                    }
                    for policy_idx, policy in enumerate(DEPLOYABLE_POLICIES):
                        policy_seed = independent_policy_seed(
                            campaign.policy_seed_base, episode_ordinal, policy_idx
                        )
                        selected = choose_action(
                            policy,
                            round1,
                            campaign.public_action_seed,
                            policy_seed=policy_seed,
                        )
                        final = round1 if selected is None else decisions[selected.name]
                        correct = (
                            final.status == "IDENTIFIED"
                            and hidden.support is not None
                            and final.support == hidden.support
                        )
                        false_promotion = final.status == "IDENTIFIED" and not correct
                        correct_oog = (
                            hidden.support is None and final.status == "OUT_OF_GRAMMAR"
                        )
                        rows.append(
                            {
                                "split": campaign.split,
                                "family": family.value,
                                "seed": seed,
                                "episode_ordinal": episode_ordinal,
                                "noise_model": noise_model,
                                "relative_noise": noise,
                                "policy": policy,
                                "policy_seed": policy_seed if policy == "random" else None,
                                "selected_action": None if selected is None else selected.name,
                                "action_cost": 0.0 if selected is None else selected.cost,
                                "truth_support": _support_list(hidden.support),
                                "round1": _decision_record(round1),
                                "final": _decision_record(final),
                                "correct_resolution": correct,
                                "correct_oog": correct_oog,
                                "false_promotion": false_promotion,
                                "converted": round1.status != "IDENTIFIED" and correct,
                            }
                        )
                    # Retrospective oracle is evaluation-only and is visibly labeled.
                    round1_oracle_success = (
                        hidden.support is not None
                        and round1.status == "IDENTIFIED"
                        and round1.support == hidden.support
                    ) or (
                        hidden.support is None and round1.status == "OUT_OF_GRAMMAR"
                    )
                    oracle_actions = [
                        action
                        for action in action_menu()
                        if (
                            hidden.support is not None
                            and decisions[action.name].status == "IDENTIFIED"
                            and decisions[action.name].support == hidden.support
                        )
                        or (
                            hidden.support is None
                            and decisions[action.name].status == "OUT_OF_GRAMMAR"
                        )
                    ]
                    chosen_oracle = (
                        None
                        if round1_oracle_success
                        else min(oracle_actions, key=lambda a: (a.cost, a.name))
                        if oracle_actions
                        else None
                    )
                    final_oracle = round1 if chosen_oracle is None else decisions[chosen_oracle.name]
                    rows.append(
                        {
                            "split": campaign.split,
                            "family": family.value,
                            "seed": seed,
                            "episode_ordinal": episode_ordinal,
                            "noise_model": noise_model,
                            "relative_noise": noise,
                            "policy": "oracle_retrospective",
                            "policy_seed": None,
                            "selected_action": None if chosen_oracle is None else chosen_oracle.name,
                            "action_cost": 0.0 if chosen_oracle is None else chosen_oracle.cost,
                            "truth_support": _support_list(hidden.support),
                            "round1": _decision_record(round1),
                            "final": _decision_record(final_oracle),
                            "correct_resolution": hidden.support is not None and final_oracle.status == "IDENTIFIED" and final_oracle.support == hidden.support,
                            "correct_oog": hidden.support is None and final_oracle.status == "OUT_OF_GRAMMAR",
                            "false_promotion": final_oracle.status == "IDENTIFIED" and final_oracle.support != hidden.support,
                            "converted": round1.status != "IDENTIFIED" and hidden.support is not None and final_oracle.status == "IDENTIFIED" and final_oracle.support == hidden.support,
                        }
                    )
    summary: dict[str, dict[str, float | int]] = {}
    for policy in (*DEPLOYABLE_POLICIES, "oracle_retrospective"):
        subset = [row for row in rows if row["policy"] == policy]
        n = len(subset)
        in_grammar = [row for row in subset if row["truth_support"] is not None]
        out_of_grammar = [row for row in subset if row["truth_support"] is None]
        summary[policy] = {
            "rows": n,
            "in_grammar_rows": len(in_grammar),
            "out_of_grammar_rows": len(out_of_grammar),
            "correct_in_grammar_rate": sum(bool(r["correct_resolution"]) for r in in_grammar) / max(1, len(in_grammar)),
            "correct_oog_rate": sum(bool(r["correct_oog"]) for r in out_of_grammar) / max(1, len(out_of_grammar)),
            "correct_scientific_resolution_rate": (
                sum(bool(r["correct_resolution"]) or bool(r["correct_oog"]) for r in subset)
                / n
            ),
            "false_promotion_rate": sum(bool(r["false_promotion"]) for r in subset) / n,
            "conversion_rate": sum(bool(r["converted"]) for r in subset) / n,
            "mean_action_cost": sum(float(r["action_cost"]) for r in subset) / n,
        }
    payload: dict[str, object] = {
        "status": "DEVELOPMENT_NOT_CONFIRMATORY",
        "campaign": asdict(campaign),
        "simulation": asdict(sim_cfg),
        "discovery": asdict(disc_cfg),
        "policy_scoring": "round1_ambiguity_conditioned_public_reference_bank",
        "summary": summary,
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="100,101,102")
    parser.add_argument("--relative-noises", default="0.0,0.0005,0.001,0.002")
    parser.add_argument("--noise-models", default="iid,correlated")
    parser.add_argument("--split", default="development")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    relative_noises = tuple(
        float(value) for value in args.relative_noises.split(",") if value
    )
    noise_models = tuple(value for value in args.noise_models.split(",") if value)
    result = run_campaign(
        CampaignConfig(
            seeds=seeds,
            relative_noises=relative_noises,
            noise_models=noise_models,
            split=args.split,
        )
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"content_sha256={result['content_sha256']}")


if __name__ == "__main__":
    main()
