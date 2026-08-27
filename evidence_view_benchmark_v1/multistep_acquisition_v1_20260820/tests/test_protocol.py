from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


LANE_ROOT = Path(__file__).resolve().parents[1]
if str(LANE_ROOT) not in sys.path:
    sys.path.insert(0, str(LANE_ROOT))

from src.contract import (  # noqa: E402
    PRODUCERS,
    VIEW_GEOMETRY,
    assert_public,
    challenge_token,
    initial_prompt,
    read_cases,
    second_prompt,
    validate_first,
    validate_second,
)
from src.protocol import admit_pair, counterfactual_second, execute_public  # noqa: E402
from src.qwen_runner import strict_json  # noqa: E402
from src.score import truth_metrics  # noqa: E402


def test_challenge_is_farthest_endpoint_same_node() -> None:
    for token, (nodes, modes) in VIEW_GEOMETRY.items():
        cnodes, cmodes = VIEW_GEOMETRY[challenge_token(token)]
        assert cnodes == nodes
        assert cmodes == (5 if modes in (2, 3) else 2)


def test_strict_parsers_fail_closed() -> None:
    with pytest.raises(ValueError):
        validate_first({"spectral_etdrk4_action": "A03"})
    with pytest.raises(ValueError):
        validate_second({f"{producer}_decision": "HACK" for producer in PRODUCERS})


def test_admission_requires_exact_agreement() -> None:
    identified = {"terminal": "IDENTIFIED", "support": ["u_x", "u_xx"]}
    assert admit_pair(identified, copy.deepcopy(identified))["status"] == "IDENTIFIED"
    changed = {"terminal": "IDENTIFIED", "support": ["u_x"]}
    assert admit_pair(identified, changed)["status"] == "INCONCLUSIVE"
    outside = {"terminal": "OUTSIDE_LIBRARY", "support": None}
    assert admit_pair(outside, outside)["status"] == "OUTSIDE_LIBRARY"


def test_public_prompt_rejects_owner_fields() -> None:
    with pytest.raises(ValueError):
        assert_public({"truth_support": ["u_x"]})


def test_real_case_trace_replays_and_prompts_are_public() -> None:
    case = read_cases()[0]
    first = {f"{producer}_action": "A03" for producer in PRODUCERS}
    second = counterfactual_second("REQUEST_CHALLENGE")
    record = execute_public(case, first, second)
    assert record["content_sha256"]
    assert "held_out" not in initial_prompt(case, "A3").lower()
    assert "truth_support" not in second_prompt(case, "A3", first, record["scout_observations"]).lower()
    for producer in PRODUCERS:
        assert record["producers"][producer]["trace"]["done"] is True


def test_owner_mutation_cannot_change_public_execution() -> None:
    case = read_cases()[0]
    first = {f"{producer}_action": "A03" for producer in PRODUCERS}
    second = counterfactual_second("COMMIT")
    baseline = execute_public(case, first, second)
    hidden = {"held_out": {"in_library": False, "truth_support": None}}
    hidden["held_out"]["in_library"] = True
    assert execute_public(case, first, second) == baseline


def test_strict_json_rejects_prose_and_duplicate_keys() -> None:
    with pytest.raises(ValueError):
        strict_json('preface {"spectral_etdrk4_action":"A03"}')
    with pytest.raises(ValueError):
        strict_json('{"a":1,"a":2}')


def test_prompt_changes_with_observation_but_not_owner_labels() -> None:
    case = read_cases()[0]
    first = {f"{producer}_action": "A03" for producer in PRODUCERS}
    record = execute_public(case, first, counterfactual_second("ABSTAIN"))
    prompt = second_prompt(case, "A5", first, record["scout_observations"])
    assert "OUTSIDE_LIBRARY" in prompt
    assert "false_promotion" not in prompt
    mutated = copy.deepcopy(record["scout_observations"])
    mutated[PRODUCERS[0]]["residual"] += 0.01
    assert second_prompt(case, "A5", first, mutated) != prompt


def test_owner_endpoint_definitions() -> None:
    assert truth_metrics("IDENTIFIED", ["u_x"], True, ["u_x"]) == (True, False)
    assert truth_metrics("OUTSIDE_LIBRARY", None, True, ["u_x"]) == (False, True)
    assert truth_metrics("OUTSIDE_LIBRARY", None, False, None) == (True, False)
    assert truth_metrics("INCONCLUSIVE", None, True, ["u_x"]) == (False, False)
