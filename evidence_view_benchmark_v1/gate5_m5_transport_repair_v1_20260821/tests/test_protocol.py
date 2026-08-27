from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.contract import (
    HARD_CAP_USD,
    PRIOR_SPEND_USD,
    bridge_case_ids,
    build_prompt,
    digest,
    parent_contract,
    prompt_inventory,
    read_cases,
    schedule,
    validate_tool_arguments,
    verify_parent_bindings,
)
from src.run_campaign import BudgetLedger
from src.validate import _close, _content_hash


def test_parent_bindings_and_inventory() -> None:
    verify_parent_bindings()
    inventory = prompt_inventory()
    assert len(inventory) == 216
    # Zero-noise iid/correlated strata are byte-identical by construction.
    assert len({row["prompt_sha256"] for row in inventory}) == 180
    assert len(schedule()) == 456


def test_bridge_selection_is_deterministic() -> None:
    selected = bridge_case_ids()
    assert len(selected) == len(set(selected)) == 24
    assert selected == bridge_case_ids()


def test_prompt_changes_only_response_tail() -> None:
    case = read_cases()[0]
    new = build_prompt(case)["prompt_text"]
    old_tail = (Path(__file__).resolve().parents[2] / "prompts/gate5/capability_probe_v1_2.txt").read_text()
    old_rows = [
        json.loads(line)
        for line in (Path(__file__).resolve().parents[2] / "gate5a4_claude_full_v1_20260820/PROMPT_INVENTORY.jsonl").open()
    ]
    old = next(row["prompt_text"] for row in old_rows if row["module"] == "M5" and row["case_id"] == case["case_id"])
    new_tail = (Path(__file__).resolve().parents[1] / "prompts_capability_probe_v2.txt").read_text()
    assert old.removesuffix(old_tail) == new.removesuffix(new_tail)


def test_v2_schema_is_scalar_and_fail_closed() -> None:
    good = {
        "spectral_etdrk4_action": "A03",
        "fd_bdf_action": "A03",
        "primary_signal": "resolution_test_space",
        "explanation": "Coarse temporal quadrature conflicts with the finer views.",
    }
    assert validate_tool_arguments(good) == good
    bad = dict(good, primary_signal='["resolution_test_space"]')
    with pytest.raises(ValueError):
        validate_tool_arguments(bad)


def test_budget_cap_is_hard() -> None:
    ledger = BudgetLedger(HARD_CAP_USD, PRIOR_SPEND_USD)
    asyncio.run(ledger.reserve(HARD_CAP_USD - PRIOR_SPEND_USD))
    with pytest.raises(RuntimeError):
        asyncio.run(ledger.reserve(0.000001))


def test_no_owner_fields_enter_prompt() -> None:
    prompt = build_prompt(read_cases()[0])["prompt_text"].lower()
    for forbidden in ("truth_support", "held_out", "false_promotion", "owner_side"):
        assert forbidden not in prompt


def test_validator_numeric_and_content_checks() -> None:
    value = {"kind": "receipt", "count": 456}
    value["content_sha256"] = digest(value)
    assert _content_hash(value)
    assert _close(1.0, 1.0 + 1e-13)
    assert not _close(1.0, 1.001)
