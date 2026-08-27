from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..contract import ACTIONS, MAX_LLM_CALLS, SCHEMA_PREFIX, digest, public_gate_margin, public_objective
from ..llm_contract import build_request, decode_response, prompt_sha256
from ..llm_runner import authorization_template, execute, preflight
from ..providers import ProviderResponse, read_anthropic_key
from ..secret_scan import scan


def _actions() -> list[dict]:
    rows = []
    for index, action in enumerate(ACTIONS):
        final = {
            "status": "IDENTIFIED" if index == 0 else "UNIDENTIFIED",
            "support": ["u_xx"] if index == 0 else None,
            "best_residual": 0.01 if index == 0 else 0.1,
            "score_gap": 0.02 if index == 0 else 0.001,
            "best_condition": 2.0 if index == 0 else 15.0,
        }
        rows.append(
            {
                "action_token": action.token,
                "nodes": action.nodes,
                "temporal_modes": action.temporal_modes,
                "terminal_status": final["status"],
                "terminal_support": final["support"],
                "best_residual": final["best_residual"],
                "score_gap": final["score_gap"],
                "best_condition": final["best_condition"],
                "gate_margin": public_gate_margin(final),
                "public_objective": public_objective(final, action.nodes),
                "gram_logdet": float(index),
                "gram_lambda_min": float(index + 1) / 100,
                "gram_condition": float(index + 1),
                "weak_rows": 32,
                "node_cost": action.nodes / 49,
            }
        )
    return rows


def _campaign() -> dict:
    request = build_request(_actions(), _actions())
    cases = []
    for index in range(MAX_LLM_CALLS):
        cases.append(
            {
                "case_id": f"case_{index:02d}",
                "request": request,
                "prompt_sha256": prompt_sha256(request),
                "bindings": {
                    "spectral_etdrk4": {
                        "episode_id": f"confirmation_ep_{index:06d}",
                        "public_episode_sha256": f"{index:064x}",
                    },
                    "fd_bdf": {
                        "episode_id": f"confirmation_ep_{index:06d}",
                        "public_episode_sha256": f"{index + 100:064x}",
                    },
                },
            }
        )
    payload = {
        "schema_version": f"{SCHEMA_PREFIX}.llm_campaign",
        "source_public_campaigns": {
            "spectral_etdrk4": "1" * 64,
            "fd_bdf": "2" * 64,
        },
        "call_count": len(cases),
        "cases": cases,
    }
    payload["campaign_sha256"] = digest(payload)
    return payload


class FakeProvider:
    def __init__(self) -> None:
        self.count_calls = 0
        self.generate_calls = 0

    def count_tokens(self, *, model: str, system: str, user: str) -> int:
        del model, system, user
        self.count_calls += 1
        return 123

    def generate(self, *, model: str, system: str, user: str, max_output_tokens: int) -> ProviderResponse:
        del model, system, user
        assert max_output_tokens == 64
        self.generate_calls += 1
        return ProviderResponse(
            text='{"spectral_etdrk4_action":"A00","fd_bdf_action":"A00"}',
            input_tokens=123,
            output_tokens=14,
            latency_seconds=0.01,
            response_id="fake",
            stop_reason="end_turn",
        )


def test_strict_decoder_refuses_markdown_and_extra_keys() -> None:
    valid = '{"spectral_etdrk4_action":"A00","fd_bdf_action":"A11"}'
    assert decode_response(valid)["fd_bdf_action"] == "A11"
    with pytest.raises(ValueError):
        decode_response("```json\n" + valid + "\n```")
    with pytest.raises(ValueError):
        decode_response('{"spectral_etdrk4_action":"A00","fd_bdf_action":"A11","reason":"x"}')


def test_preflight_counts_tokens_but_never_generates() -> None:
    provider = FakeProvider()
    row = preflight(_campaign(), provider, "fake-model")
    assert row["call_count"] == MAX_LLM_CALLS
    assert provider.count_calls == MAX_LLM_CALLS
    assert provider.generate_calls == 0
    assert row["hard_cap_estimated_cost_usd"] == 0.0
    assert row["maximum_authorized_cost_usd"] == 0.0


def test_claude_preflight_has_sub_dollar_hard_cap() -> None:
    provider = FakeProvider()
    row = preflight(_campaign(), provider, "claude-sonnet-4-6")
    assert 0.0 < row["hard_cap_estimated_cost_usd"] < 1.0
    assert row["maximum_authorized_cost_usd"] == 1.0
    assert row["pricing"]["snapshot_date"] == "2026-08-17"


def test_hash_bound_execution_is_zero_retry_and_emits_two_proposals_per_call(tmp_path: Path) -> None:
    campaign = _campaign()
    provider = FakeProvider()
    preflight_row = preflight(campaign, provider, "fake-model")
    authorization = authorization_template(preflight_row)
    authorization.update(
        {
            "billable_calls_authorized": True,
            "authorization_reference": "explicit-test-authorization",
        }
    )
    result = execute(
        campaign,
        preflight_row,
        authorization,
        provider,
        "fake-model",
        "fake_llm",
        tmp_path,
    )
    assert result["proposal_count"] == 2 * MAX_LLM_CALLS
    assert provider.generate_calls == MAX_LLM_CALLS
    assert all(row["proposal_valid"] for row in result["proposals"])
    assert not any("api" in json.dumps(row).lower() for row in result["proposals"])


def test_credential_parser_requires_one_key_and_key_is_not_written(tmp_path: Path) -> None:
    secret = "sk-ant-" + "A" * 32
    credential = tmp_path / "credential.rtf"
    credential.write_bytes(("{\\rtf1 " + secret + "}").encode("ascii"))
    assert read_anthropic_key(credential) == secret
    artifact = {"api_key_recorded_or_hashed": False}
    output = tmp_path / "artifact.json"
    output.write_text(json.dumps(artifact), encoding="utf-8")
    assert secret not in output.read_text(encoding="utf-8")


def test_secret_scan_reports_only_paths_and_never_the_key(tmp_path: Path) -> None:
    secret = "sk-ant-" + "B" * 32
    credential = tmp_path / "credential.rtf"
    credential.write_bytes(("{\\rtf1 " + secret + "}").encode("ascii"))
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "result.json").write_text('{"api_key_recorded_or_hashed":false}')
    result = scan(credential, clean)
    assert result["pass"] is True
    assert result["violation_count"] == 0
    assert secret not in json.dumps(result)
