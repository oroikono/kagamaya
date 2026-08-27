"""Public execution and independent cross-view admission."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    ACQUISITION_PATH,
    BENCHMARK_ROOT,
    PRODUCERS,
    TOTAL_BUDGET,
    VIEW_GEOMETRY,
    assert_public,
    challenge_token,
    digest,
    validate_first,
    validate_second,
)

if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from envlib.environment import Action, EvidenceViewEnv  # noqa: E402


def _normalized_terminal(view: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": view["terminal"], "support": view.get("support")}


def admit_pair(first: Mapping[str, Any], challenge: Mapping[str, Any]) -> dict[str, Any]:
    one = _normalized_terminal(first)
    two = _normalized_terminal(challenge)
    if one["status"] != two["status"]:
        return {"status": "INCONCLUSIVE", "support": None, "reason": "status_disagreement"}
    if one["status"] == "IDENTIFIED" and one["support"] != two["support"]:
        return {"status": "INCONCLUSIVE", "support": None, "reason": "support_disagreement"}
    if one["status"] == "INCONCLUSIVE":
        return {"status": "INCONCLUSIVE", "support": None, "reason": "both_inconclusive"}
    return {"status": one["status"], "support": one["support"], "reason": "exact_cross_view_agreement"}


def execute_public(
    case: Mapping[str, Any],
    first_value: Mapping[str, Any],
    second_value: Mapping[str, Any],
    *,
    acquisition_path: str | Path = ACQUISITION_PATH,
) -> dict[str, Any]:
    first = validate_first(first_value)
    second = validate_second(second_value)
    public_observations: dict[str, Any] = {}
    producer_records: dict[str, Any] = {}
    for producer in PRODUCERS:
        first_token = first[f"{producer}_action"]
        nodes, modes = VIEW_GEOMETRY[first_token]
        env = EvidenceViewEnv(
            case,
            TOTAL_BUDGET,
            "acquire_only",
            producer=producer,
            acquisition_bank_path=acquisition_path,
        )
        reset = env.reset()
        scout_observation, done = env.step(Action.acquire_view(nodes, modes))
        if done or scout_observation.get("event") != "view_acquired":
            raise RuntimeError("frozen scout acquisition failed")
        scout = {
            key: scout_observation[key]
            for key in ("nodes", "modes", "terminal", "support", "residual", "score_gap", "gram_condition")
        }
        public_observations[producer] = scout
        decision = second[f"{producer}_decision"]
        challenge = None
        if decision == "COMMIT":
            admission = {**_normalized_terminal(scout), "reason": "delegated_scout_commit"}
            env.step(Action.commit(nodes, modes))
        elif decision == "ABSTAIN":
            admission = {"status": "INCONCLUSIVE", "support": None, "reason": "policy_abstention"}
            env.step(Action.abstain("policy_abstention"))
        else:
            challenge_id = challenge_token(first_token)
            cnodes, cmodes = VIEW_GEOMETRY[challenge_id]
            challenge_observation, challenge_done = env.step(Action.acquire_view(cnodes, cmodes))
            if challenge_done or challenge_observation.get("event") != "view_acquired":
                raise RuntimeError("fixed challenge acquisition failed")
            challenge = {
                key: challenge_observation[key]
                for key in ("nodes", "modes", "terminal", "support", "residual", "score_gap", "gram_condition")
            }
            admission = admit_pair(scout, challenge)
            env.step(Action.abstain("independent_admission_completed"))
        record = {
            "first_action": first_token,
            "second_decision": decision,
            "scout": scout,
            "challenge_action": challenge_token(first_token) if decision == "REQUEST_CHALLENGE" else None,
            "challenge": challenge,
            "admission": admission,
            "node_cost": env.spent,
            "trace": env.export_trace(),
            "reset_sha256": digest(reset),
        }
        assert_public(record)
        producer_records[producer] = record
    result = {
        "schema_version": "evidence_view_benchmark.multistep_public.v1",
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "first": first,
        "second": second,
        "scout_observations": public_observations,
        "producers": producer_records,
    }
    assert_public(result)
    result["content_sha256"] = digest(result)
    return json.loads(json.dumps(result, sort_keys=True))


def counterfactual_second(decision: str) -> dict[str, str]:
    if decision not in {"COMMIT", "REQUEST_CHALLENGE", "ABSTAIN"}:
        raise ValueError("unsupported counterfactual")
    return {f"{producer}_decision": decision for producer in PRODUCERS}

