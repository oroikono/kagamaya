"""Full deterministic numerical-origin replay and final confirmation verdict."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from ..adversarial_verifier_gaming_v1.io_utils import write_json
from ..maintrack_suite_v41.simulator import Family
from ..sequential_consensus_controller_v1.campaign import _build_episode
from .aggregate import build_aggregate


FAMILY_BY_VALUE = {family.value: family for family in Family}


def _replay_one(pair: tuple[Mapping[str, Any], Mapping[str, Any]]) -> str | None:
    public, owner = pair
    try:
        family_value, seed, noise_model, relative_noise = owner["owner_episode_key"]
        ordinal = int(str(public["episode_id"]).rsplit("_", 1)[1])
        expected_public, expected_owner = _build_episode(
            producer=str(public["producer"]),
            seed=int(seed),
            family=FAMILY_BY_VALUE[str(family_value)],
            noise_model=str(noise_model),
            relative_noise=float(relative_noise),
            ordinal=ordinal,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return f"origin_replay_setup:{error}"
    if expected_public != public:
        return f"public_numeric_origin_replay:{public.get('episode_id')}"
    if expected_owner != owner:
        return f"owner_numeric_origin_replay:{public.get('episode_id')}"
    return None


def validate_numeric(
    aggregate: Mapping[str, Any], shards: list[Mapping[str, Any]], workers: int
) -> dict[str, Any]:
    errors: list[str] = []
    rebuilt = build_aggregate(shards)
    if rebuilt != aggregate:
        errors.append("aggregate_replay")
    owners = {str(row["episode_id"]): row for row in aggregate.get("owner_episodes", ())}
    pairs = [
        (row, owners[str(row["episode_id"])])
        for row in aggregate.get("public_episodes", ())
        if str(row.get("episode_id")) in owners
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        replay_errors = list(executor.map(_replay_one, pairs))
    errors.extend(item for item in replay_errors if item is not None)
    scientific = aggregate.get("confirmation_gates_before_numerical_replay", {})
    final_gates: dict[str, Any] = {}
    for producer, row in scientific.items():
        checks = dict(row["checks_before_numerical_origin_replay"])
        checks["all_structural_and_numerical_replays_pass"] = not errors
        checks["exact_confirmation_seed_count"] = aggregate.get("split", {}).get(
            "confirmation_seed_cluster_access_count"
        ) == 50
        final_gates[producer] = {
            "checks": checks,
            "all_passed": all(checks.values()),
            "family_correct_rates": row["family_correct_rates"],
            "nonzero_noise_terminal_rates": row["nonzero_noise_terminal_rates"],
        }
    all_passed = bool(final_gates) and all(row["all_passed"] for row in final_gates.values())
    return {
        "schema_version": "verisigs.sequential_asymmetric.confirmation_final_verdict.v1",
        "valid": not errors,
        "errors": sorted(set(errors)),
        "status": "FINITE_CONFIRMATION_PASS" if all_passed else "CONFIRMATION_FAIL",
        "numerical_episodes_replayed": len(pairs) if not errors else sum(item is None for item in replay_errors),
        "final_gates": final_gates,
        "confirmation_seed_cluster_access_count": aggregate.get("split", {}).get(
            "confirmation_seed_cluster_access_count"
        ),
        "claim_ceiling": (
            "finite synthetic confirmation with deterministic same-source numerical replay; "
            "not independent scientific, physical, continuum, or population validation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.shard_dir.glob("*.json"))
    ]
    report = validate_numeric(aggregate, shards, args.workers)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
