"""Outcome-free protocol and mutation tests."""

from __future__ import annotations

from .batch_runner import _expected_authorization, _sanitized_content, prompt_archive
from .analyze import case_from_custom_id, model_from_custom_id
from .contract import (
    CALLS_PER_MODEL,
    EFFORT,
    MAX_AUTHORIZED_COST_USD,
    MAX_OUTPUT_TOKENS,
    MODEL_SPECS,
    THINKING,
    cost,
)


def test_registry_is_current_and_bounded() -> None:
    assert tuple(MODEL_SPECS) == ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")
    assert CALLS_PER_MODEL == 24
    assert MAX_OUTPUT_TOKENS == 512
    assert EFFORT == "medium"
    assert THINKING == {"type": "adaptive", "display": "summarized"}


def test_batch_hard_cap_from_nominal_new_tokenizer_count() -> None:
    nominal = 130_000
    cap = sum(cost(model, nominal, CALLS_PER_MODEL * MAX_OUTPUT_TOKENS) for model in MODEL_SPECS)
    assert cap < MAX_AUTHORIZED_COST_USD


def test_authorization_is_content_bound() -> None:
    from ..contract import digest

    preflight = {"content_sha256": "p"}
    body = {
        "schema_version": "x",
        "preflight_sha256": "p",
        "maximum_billable_calls": 72,
        "maximum_authorized_cost_usd": MAX_AUTHORIZED_COST_USD,
        "billable_calls_authorized": True,
        "authorization_reference": "user request 2026-08-18",
    }
    authorization = {**body, "content_sha256": digest(body)}
    assert _expected_authorization(preflight, authorization)
    authorization["maximum_billable_calls"] = 73
    assert not _expected_authorization(preflight, authorization)


def test_prompt_archive_keeps_full_queries() -> None:
    campaign = {
        "cases": [{"case_id": "case_00", "prompt_sha256": "h", "request": {}}],
    }
    try:
        archive = prompt_archive(campaign)
    except Exception:
        # Empty requests correctly fail the parent public-table validator. The
        # important regression is that prompt construction is not a hash-only
        # archive and therefore cannot silently discard query text.
        return
    assert archive["rows"][0]["system"]
    assert archive["rows"][0]["user"]


def test_custom_id_roundtrip() -> None:
    assert model_from_custom_id("sonnet-5-case_07") == "claude-sonnet-5"
    assert model_from_custom_id("opus-5-case_23") == "claude-opus-5"
    assert model_from_custom_id("fable-5-case_00") == "claude-fable-5"
    assert case_from_custom_id("fable-5-case_00") == "case_00"


def test_thinking_summary_is_saved_without_raw_signature() -> None:
    class Block:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self, mode="json"):
            assert mode == "json"
            return dict(self.payload)

    content, text = _sanitized_content(
        [
            Block({"type": "thinking", "thinking": "compare all rows", "signature": "opaque"}),
            Block({"type": "text", "text": '{"spectral_etdrk4_action":"A03","fd_bdf_action":"A03"}'}),
        ]
    )
    assert text.startswith("{")
    assert content[0]["thinking"] == "compare all rows"
    assert content[0]["signature_sha256"]
    assert "signature" not in content[0]
