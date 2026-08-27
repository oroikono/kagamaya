"""Adversarial semantic validator for v4.1 campaign artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .policies import DEPLOYABLE_POLICIES, independent_policy_seed
from .simulator import action_menu


TERMINAL_STATUSES = {"IDENTIFIED", "UNIDENTIFIED", "OUT_OF_GRAMMAR", "FAILED"}


def _digest(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("content_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_payload(payload: dict[str, object]) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "DEVELOPMENT_NOT_CONFIRMATORY":
        errors.append("artifact status must remain DEVELOPMENT_NOT_CONFIRMATORY")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["rows must be a nonempty list"]
    campaign = payload.get("campaign")
    if not isinstance(campaign, dict):
        return ["campaign metadata must be an object"]

    required_policies = set(DEPLOYABLE_POLICIES) | {"oracle_retrospective"}
    observed_policies = {row.get("policy") for row in rows if isinstance(row, dict)}
    if observed_policies != required_policies:
        errors.append("policy set mismatch")
    expected_episodes = (
        len(campaign.get("families", ()))
        * len(campaign.get("seeds", ()))
        * len(campaign.get("noise_models", ()))
        * len(campaign.get("relative_noises", ()))
    )
    if len(rows) != expected_episodes * len(required_policies):
        errors.append("row count does not match declared factorial grid")

    action_costs = {action.name: action.cost for action in action_menu()}
    keys: set[tuple[object, ...]] = set()
    per_episode: dict[tuple[object, ...], set[object]] = {}
    policy_seed_base = int(campaign.get("policy_seed_base", -1))
    random_index = DEPLOYABLE_POLICIES.index("random")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {idx} is not an object")
            continue
        episode_key = (
            row.get("family"),
            row.get("seed"),
            row.get("noise_model"),
            row.get("relative_noise"),
        )
        key = (*episode_key, row.get("policy"))
        if key in keys:
            errors.append(f"duplicate row key {key}")
        keys.add(key)
        per_episode.setdefault(episode_key, set()).add(row.get("policy"))

        final = row.get("final")
        round1 = row.get("round1")
        if not isinstance(final, dict) or not isinstance(round1, dict):
            errors.append(f"row {idx} lacks decisions")
            continue
        if final.get("status") not in TERMINAL_STATUSES or round1.get("status") not in TERMINAL_STATUSES:
            errors.append(f"row {idx} has unknown decision status")
        selected = row.get("selected_action")
        expected_cost = 0.0 if selected is None else action_costs.get(str(selected))
        if expected_cost is None:
            errors.append(f"row {idx} selects an unknown action")
        elif abs(float(row.get("action_cost", -1.0)) - expected_cost) > 1e-12:
            errors.append(f"row {idx} action cost mismatch")
        if selected is None and final != round1:
            errors.append(f"row {idx} changes decision without an action")

        truth = row.get("truth_support")
        correct = (
            final.get("status") == "IDENTIFIED"
            and final.get("support") == truth
            and truth is not None
        )
        false = final.get("status") == "IDENTIFIED" and not correct
        correct_oog = truth is None and final.get("status") == "OUT_OF_GRAMMAR"
        converted = round1.get("status") != "IDENTIFIED" and correct
        if bool(row.get("correct_resolution")) != correct:
            errors.append(f"row {idx} correct-resolution flag mismatch")
        if bool(row.get("false_promotion")) != false:
            errors.append(f"row {idx} false-promotion flag mismatch")
        if bool(row.get("correct_oog")) != correct_oog:
            errors.append(f"row {idx} correct-OOG flag mismatch")
        if bool(row.get("converted")) != converted:
            errors.append(f"row {idx} conversion flag mismatch")
        if final.get("status") == "OUT_OF_GRAMMAR" and final.get("support") is not None:
            errors.append(f"row {idx} OOG decision contains a support")

        if row.get("policy") == "random":
            ordinal = int(row.get("episode_ordinal", -1))
            expected_seed = independent_policy_seed(policy_seed_base, ordinal, random_index)
            if row.get("policy_seed") != expected_seed:
                errors.append(f"row {idx} random-policy seed mismatch")
        elif row.get("policy_seed") is not None:
            errors.append(f"row {idx} exposes an irrelevant policy seed")

    for episode_key, policies in per_episode.items():
        if policies != required_policies:
            errors.append(f"episode {episode_key} lacks a matched policy set")

    if payload.get("content_sha256") != _digest(payload):
        errors.append("content digest mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    args = parser.parse_args()
    payload = json.loads(Path(args.artifact).read_text())
    errors = validate_payload(payload)
    print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
