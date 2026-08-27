"""Outcome-free tests for the frozen Claude comparison."""

from __future__ import annotations

from ..contract import ACTION_BY_TOKEN
from .runner import CAMPAIGN_SHA256, MODEL_SPECS, model_cost
from .runner_r2 import MODEL_SPECS_R2
from .parser_sensitivity import fence_only_decode, strip_one_json_fence


def test_registry_is_exact() -> None:
    assert tuple(MODEL_SPECS) == (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-5",
        "claude-opus-5",
    )
    assert len({row["policy"] for row in MODEL_SPECS.values()}) == 3


def test_frozen_campaign_and_optimizer_token() -> None:
    assert CAMPAIGN_SHA256 == "75e3c47142478c513cb12b605517d8c368aa35a6c11dc0d121855f59b6cb2bfd"
    assert (ACTION_BY_TOKEN["A03"].nodes, ACTION_BY_TOKEN["A03"].temporal_modes) == (7, 5)


def test_cost_caps_cover_frozen_token_counts() -> None:
    for model, spec in MODEL_SPECS.items():
        hard = model_cost(
            model,
            int(spec["expected_total_input_tokens"]),
            24 * 64,
        )
        assert hard < float(spec["maximum_authorized_cost_usd"])
    assert sum(
        model_cost(model, int(spec["expected_total_input_tokens"]), 24 * 64)
        for model, spec in MODEL_SPECS.items()
    ) < 1.08


def test_r2_registry_preserves_matched_decoding_tiers() -> None:
    assert tuple(MODEL_SPECS_R2) == ("claude-sonnet-4-6", "claude-opus-4-6")
    original = MODEL_SPECS.copy()
    try:
        MODEL_SPECS.clear()
        MODEL_SPECS.update(MODEL_SPECS_R2)
        for model, spec in MODEL_SPECS_R2.items():
            hard = model_cost(
                model,
                int(spec["expected_total_input_tokens"]),
                24 * 64,
            )
            assert hard < float(spec["maximum_authorized_cost_usd"])
    finally:
        MODEL_SPECS.clear()
        MODEL_SPECS.update(original)


def test_fence_only_sensitivity_is_narrow() -> None:
    exact = '{"spectral_etdrk4_action":"A03","fd_bdf_action":"A08"}'
    assert strip_one_json_fence(exact) == exact
    assert fence_only_decode("```json\n" + exact + "\n```") == {
        "spectral_etdrk4_action": "A03",
        "fd_bdf_action": "A08",
    }
    for invalid in (
        "Here is the answer: " + exact,
        "```\n" + exact + "\n```",
        "```json\n" + exact + "\n``` trailing",
        "```json\n```json\n" + exact + "\n```\n```",
    ):
        try:
            fence_only_decode(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - regression assertion
            raise AssertionError("sensitivity parser admitted prose or a non-exact fence")
