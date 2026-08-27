from __future__ import annotations

import copy

import pytest

from src.contract import (
    ARMS,
    MODELS,
    SMOKE_CASES,
    action_geometry,
    build_m3_prompt,
    build_m5_prompt,
    prompt_inventory,
    read_authorization,
    read_cases,
    validate_tool_arguments,
)
from src.score import truth_metrics
from src.run_campaign import BudgetLedger


def test_frozen_cardinalities_and_authorization() -> None:
    cases = read_cases()
    assert len(cases) == 216
    assert len(prompt_inventory()) == 1080
    assert ARMS == ("A1", "A2", "A3", "A5")
    assert MODELS == ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5")
    assert SMOKE_CASES == ("case_00", "case_01")
    assert read_authorization()["max_paid_api_usd"] == 100.0


def test_prompt_public_only_and_digest_stable() -> None:
    case = read_cases()[0]
    prompts = [build_m3_prompt(case, arm) for arm in ARMS] + [build_m5_prompt(case)]
    forbidden = ("truth_support", "held_out", "sidecar", "false_promotion")
    for prompt in prompts:
        lowered = prompt["prompt_text"].lower()
        assert not any(token in lowered for token in forbidden)
        assert len(prompt["prompt_sha256"]) == 64


def test_tool_contract_fails_closed() -> None:
    valid = {"spectral_etdrk4_action": "A03", "fd_bdf_action": "A11"}
    assert validate_tool_arguments("M3", valid) == valid
    bad = copy.deepcopy(valid)
    bad["extra"] = "A00"
    with pytest.raises(ValueError):
        validate_tool_arguments("M3", bad)
    m5 = {**valid, "signals": ["residual", "score_gap"], "explanation": "Public residual and separation warnings."}
    assert validate_tool_arguments("M5", m5) == m5
    m5["signals"] = ["truth"]
    with pytest.raises(ValueError):
        validate_tool_arguments("M5", m5)


def test_geometry() -> None:
    assert action_geometry("A00") == (7, 2)
    assert action_geometry("A03") == (7, 5)
    assert action_geometry("A11") == (25, 5)


def test_owner_scoring_semantics() -> None:
    assert truth_metrics("IDENTIFIED", ["u_x", "u_xx"], ["u_x", "u_xx"]) == (True, False)
    assert truth_metrics("OUTSIDE_LIBRARY", None, ["u_x", "u_xx"]) == (False, True)
    assert truth_metrics("INCONCLUSIVE", None, ["u_x", "u_xx"]) == (False, False)
    assert truth_metrics("OUTSIDE_LIBRARY", None, None) == (True, False)


def test_single_grant_carries_prior_smoke_spend() -> None:
    ledger = BudgetLedger(100.0, prior_spend=0.107472)
    assert ledger.observed == 0.107472
    with pytest.raises(ValueError):
        BudgetLedger(100.0, prior_spend=100.1)
