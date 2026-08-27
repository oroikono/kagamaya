from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


LANE_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = LANE_ROOT.parent
if str(LANE_ROOT) not in sys.path:
    sys.path.insert(0, str(LANE_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from multistep_acquisition_v1_20260820.src.protocol import counterfactual_second, execute_public  # noqa: E402
from multistep_acquisition_v1_20260820.src.qwen_runner import strict_json  # noqa: E402
from src.contract import (  # noqa: E402
    ARMS,
    MODEL_ID,
    PRODUCERS,
    donor_map,
    initial_prompt,
    read_cases,
    second_prompt,
    verify_generation_binding,
)


def test_initial_prompts_are_metadata_blind_and_case_independent() -> None:
    for arm in ARMS:
        prompt = initial_prompt(arm)
        lowered = prompt.lower()
        for forbidden in ("case_id", "family", "noise_model", "noise_level", "truth_support", "held_out"):
            assert forbidden not in lowered


def test_donor_permutation_is_bijective_derangement() -> None:
    mapping = donor_map()
    assert len(mapping) == 216
    assert len(set(mapping.values())) == 216
    assert all(source != donor for source, donor in mapping.items())


def test_second_prompt_changes_only_with_evidence() -> None:
    cases = read_cases()
    first = {f"{producer}_action": "A10" for producer in PRODUCERS}
    one = execute_public(cases[0], first, counterfactual_second("ABSTAIN"))["scout_observations"]
    two = execute_public(cases[1], first, counterfactual_second("ABSTAIN"))["scout_observations"]
    assert second_prompt("A3", first, one) != second_prompt("A3", first, two)
    assert "case_id" not in second_prompt("A3", first, one).lower()


def test_owner_mutation_cannot_change_public_transition() -> None:
    case = read_cases()[0]
    first = {f"{producer}_action": "A10" for producer in PRODUCERS}
    second = counterfactual_second("REQUEST_CHALLENGE")
    baseline = execute_public(case, first, second)
    hidden = {"held_out": {"in_library": False, "truth_support": None}}
    hidden["held_out"]["truth_support"] = ["u_x"]
    assert execute_public(case, first, second) == baseline


def test_generation_binding_rejects_prompt_and_model_mutations() -> None:
    prompt = initial_prompt("A2")
    good = {
        "prompt": prompt,
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "resolved_model_id": MODEL_ID,
    }
    verify_generation_binding(good, prompt)
    bad_prompt = copy.deepcopy(good)
    bad_prompt["prompt"] += "x"
    with pytest.raises(ValueError):
        verify_generation_binding(bad_prompt, prompt)
    bad_model = copy.deepcopy(good)
    bad_model["resolved_model_id"] = "different"
    with pytest.raises(ValueError):
        verify_generation_binding(bad_model, prompt)


def test_strict_json_is_total_and_fail_closed() -> None:
    with pytest.raises(ValueError):
        strict_json('preface {"spectral_etdrk4_action":"A10"}')
    with pytest.raises(ValueError):
        strict_json('{"a":1,"a":2}')

